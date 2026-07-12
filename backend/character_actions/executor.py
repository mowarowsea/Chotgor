"""ToolExecutor — キャラクターのツール呼び出しを実行する中枢。

LLM の tool-use（function calling）で保存記憶・ワーキングメモリ・switch_angle 等を
操作する。tool-use 非対応プロバイダー（Claude CLI、Ollama）は
タグ方式（[INSCRIBE_MEMORY:...] / [CARVE_NARRATIVE:...] マーカー）にフォールバックする。

ツールの定義（スキーマ・説明文）は tool_specs.py の ToolSpec 台帳が単一ソース。
ANTHROPIC_TOOLS / OPENAI_TOOLS はそこから生成される。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.services.memory.manager import InscribedMemoryManager
    from backend.services.memory.working_memory_manager import WorkingMemoryManager

from backend.lib.tool_event_recorder import record_tool_event, result_looks_like_error
from backend.repositories.lance.store import EmbeddingError
from backend.services.memory.format import origin_label_prefix
from backend.character_actions.tool_specs import BASE_TOOL_SPECS
from backend.character_actions.switcher import Switcher, extract_switch_angle_tags
from backend.character_actions.carver import Carver, extract_carve_narrative_tags
from backend.character_actions.inscriber import Inscriber, extract_inscribe_memory_tags
from backend.character_actions.threader import Threader
from backend.character_actions.web_searcher import WebSearcher
from backend.character_actions.leaver import Leaver
from backend.character_actions.messenger import Messenger
from backend.character_actions.rescheduler import Rescheduler

# Anthropic形式のツール定義リスト（tool_specs.py の台帳から生成）
ANTHROPIC_TOOLS: list[dict] = [spec.as_anthropic() for spec in BASE_TOOL_SPECS]

def to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Anthropic 形式のツール定義リストを OpenAI 形式へ変換する。

    OPENAI_TOOLS（基本セット）と、コンテキスト別追加ツール（context_tools.py 経由で
    プロバイダーへ渡る extra_tools）の両方がこの変換を共有する。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_tools
    ]


# OpenAI形式のツール定義リスト（Anthropic形式から変換）
OPENAI_TOOLS: list[dict] = to_openai_tools(ANTHROPIC_TOOLS)


@dataclass
class ToolCall:
    """正規化されたツール呼び出し情報。

    Attributes:
        id: プロバイダーが発行するツール呼び出しID。
        name: ツール名（inscribe_memory / post_working_memory_thread / read_working_memory_thread / close_working_memory_thread / reopen_working_memory_thread / merge_working_memory_threads / carve_narrative / switch_angle / power_recall）。
        input: ツールに渡す引数 dict。
    """

    id: str
    name: str
    input: dict


@dataclass
class ToolTurnResult:
    """1ターンのLLM呼び出し結果。

    Attributes:
        text: このターンで生成されたテキスト（ツール呼び出し行・思考ブロックは除く）。
        tool_calls: 正規化されたツール呼び出しリスト。
        thinking: 思考ブロックのテキスト（thought=True のパート）。空文字列は思考なし。
        _raw: プロバイダー固有の生レスポンスオブジェクト（次ターンのメッセージ構築に使用）。
    """

    text: str
    tool_calls: list[ToolCall]
    thinking: str = field(default="")
    _raw: Any = field(default=None, repr=False)
    # True の場合、APIエラー・パッケージ未インストール等の致命的失敗を示す。
    # generate_with_tools() はこのフラグを見て LLMApiError を送出する。
    error: bool = field(default=False)


class ToolExecutor:
    """LLMからのツール呼び出しを実際に実行するクラス。

    inscribe_memory / post_working_memory_thread / read_working_memory_thread /
    close_working_memory_thread / reopen_working_memory_thread / merge_working_memory_threads /
    carve_narrative / switch_angle / power_recall の各ツールを受け取り、
    Inscriber / Threader / Carver / Switcher を通じてDBへ反映する。

    Attributes:
        character_id: 操作対象のキャラクターID。
        session_id: 現在のセッションID。
        memory_manager: 記憶の読み書きを担うマネージャー。
        working_memory_manager: ワーキングメモリの読み書きを担うマネージャー。
        batch_context: バッチ処理由来のフラグ群。例 ``{"force_insert_memory": True}`` を
            渡すと、inscribe_memory ツール実行時に類似上書きをスキップして必ず新規 ID で
            挿入する（forget 蒸留バッチ専用）。MCP ツール API は変更せず、バッチ側のみ
            この context で挙動を切り替えるための内部チャネル。
        default_origin: inscribe_memory / post_working_memory_thread の保存時に
            付与する origin ラベル（3値）。"real"=日常（ユーザと共有）、
            "usual"=うつつ（ユーザ未共有の自分の生活体験）、"interlude"=シナリオPCモードの幕間体験。
            シナリオ PC モードでキャラを動かす経路では "interlude"、うつつ無人経路では "usual" を渡す。
            キャラクター本人はこの値を意識する必要はない（ツール引数として露出しない）。
        _inscriber: 記憶書き込みを担う Inscriber インスタンス。
        _threader: ワーキングメモリスレッド操作を担う Threader インスタンス。
        _carver: inner_narrative の彫り込みを担う Carver インスタンス。
        _switcher: アングル切り替えリクエストを記録する Switcher インスタンス。
    """

    # クラスレベルロガー
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        character_id: str,
        session_id: str | None,
        memory_manager: InscribedMemoryManager,
        working_memory_manager: WorkingMemoryManager | None,
        batch_context: dict | None = None,
        default_origin: str = "real",
        source_preset_id: str = "",
    ) -> None:
        """ToolExecutorを初期化する。

        Args:
            batch_context: バッチ処理が指定するツール挙動の切り替えフラグ群。
                通常チャットでは None（=空 dict 扱い）。forget 蒸留バッチからは
                ``{"force_insert_memory": True}`` を渡す。
            default_origin: inscribe_memory / post_working_memory_thread が保存する
                記憶/スレッドに付与する origin。1on1 通常経路では "real"、
                シナリオ PC モードからは "interlude"、うつつ無人経路からは "usual" を渡す。
            source_preset_id: 記憶/スレッド作成時に「どのプリセットで生まれたか」として記録される
                プリセット ID。ツール引数としては露出させず、Executor のインスタンス属性として持つ。
                空文字列なら NULL 保存（プリセット情報を残さない）。
        """
        self.character_id = character_id
        self.session_id = session_id
        self.memory_manager = memory_manager
        self.working_memory_manager = working_memory_manager
        self.batch_context: dict = batch_context or {}
        self.default_origin = default_origin
        self.source_preset_id = source_preset_id
        # memory_manager / working_memory_manager は呼び出し元から None で渡される
        # ケースがある（Chronicle が memory_manager 無しで close/reopen/merge だけ使う等）。
        # Inscriber / Carver / WebSearcher は実行時に NoneType エラーになるが、対応するツールを
        # 呼ばなければ問題ない（fail-fast を実行時に倒すための妥協）。
        _sqlite = memory_manager.sqlite if memory_manager is not None else None
        self._inscriber = Inscriber(character_id, memory_manager)
        self._threader = Threader(character_id, working_memory_manager)
        self._carver = Carver(character_id, _sqlite)
        self._switcher = Switcher()
        self._web_searcher = WebSearcher(_sqlite)
        self._leaver = Leaver(character_id, session_id, _sqlite)
        self._rescheduler = Rescheduler(character_id, _sqlite)

    @property
    def switch_request(self) -> tuple[str, str] | None:
        """switch_angle が呼ばれた場合の切り替えリクエスト。generate_with_tools() ループが検知して即中断する。"""
        return self._switcher.switch_request

    def execute(
        self,
        tool_name: str,
        tool_input: dict,
        *,
        record: bool = True,
        source: str = "tool_use",
        origin: str | None = None,
    ) -> str:
        """ツール名と入力を受け取り実行して結果テキストを返す。

        ツール実行の唯一の関門。MCP tool-use・テキストタグ方式（inscriber/carver/switcher/
        recaller の *_from_text）・Chronicle の JSON 棚卸し結果反映 — どの入口から来た
        ツール呼び出しもここを通る。フォーマット解析は呼び出し側、実行と記録はここ、
        という分業にしてある。これにより Logs 画面のツール使用表示（tool_call_events 経由）が
        入口の違いに関係なく一貫して埋まる（source of truth。tool_event_recorder を参照）。

        Args:
            tool_name: ツール名（"inscribe_memory" / "post_working_memory_thread" / "read_working_memory_thread" / "close_working_memory_thread" / "reopen_working_memory_thread" / "merge_working_memory_threads" / "carve_narrative" / "switch_angle" / "power_recall"）。
            tool_input: ツールの入力パラメータ dict。
            record: False の場合、実行イベントを記録しない。MCP 経由で既に実行・記録済みの
                switch_angle を in-process の tool_executor へ転写する claude_cli_provider の
                経路でのみ False を渡す（二重記録防止）。
            source: 入口の識別（記録時の source 列に保存される）。"tool_use"=tool-use 方式、
                "tag"=テキストタグ方式（inscriber/carver/switcher/recaller）、
                "chronicle"=Chronicle 棚卸し結果反映。Logs UI のフィルタ・分析用。
            origin: この1回の呼び出しに限り default_origin を上書きする（指定時のみ）。
                Chronicle が item ごとに異なる origin を渡すための口。インスタンス状態は
                呼び出し後に元へ戻すため、executor を跨いだ origin の漏れを防ぐ。

        Returns:
            ツールの実行結果を表すテキスト。
        """
        # origin 指定時のみ、このディスパッチの間だけ default_origin を差し替える。
        # 終了時に必ず元へ戻し、次の呼び出しへ stale な origin を残さない（漏れ防止）。
        prev_origin = self.default_origin
        if origin is not None:
            self.default_origin = origin
        try:
            result = self._dispatch(tool_name, tool_input)
        except Exception as e:
            # 各ツール実装は例外を握り潰す設計だが、引数の型変換（float() 等）が
            # ディスパッチ段で送出する可能性があるため、失敗の事実だけ記録して再送出する。
            self.default_origin = prev_origin
            if record:
                record_tool_event(
                    tool_name, tool_input,
                    status="error", error_message=f"{type(e).__name__}: {e}",
                    source=source,
                )
            raise
        self.default_origin = prev_origin
        if record:
            is_error = result_looks_like_error(result)
            record_tool_event(
                tool_name, tool_input,
                status="error" if is_error else "ok",
                error_message=result if is_error else None,
                source=source,
            )
        return result

    # ------------------------------------------------------------------
    # タグ方式の入口統合: テキスト → 抽出 → execute() 経由実行
    # ------------------------------------------------------------------
    # SUPPORTS_TOOLS=False のプロバイダー（Claude CLI / Ollama 等）が応答テキストに
    # 埋め込んでくる [INSCRIBE_MEMORY:...] / [CARVE_NARRATIVE:...] / [SWITCH_ANGLE:...] を、
    # tool-use 方式と同じ self.execute(source="tag") 経由で実行する。タグ方式と tool-use 方式で
    # 記録経路（tool_call_events）と実装経路（_dispatch）を統一するための仕組み。

    def _execute_tag(self, tool_name: str, args: dict) -> None:
        """タグ方式で抽出した1件を execute(source="tag") 経由で実行する共通処理。

        3種の apply_*_tags が共有する「実行＋例外ログ」の骨格。例外は握り潰して
        WARNING ログのみ残す（タグ方式は1件失敗しても LLM 応答本文の整形に影響させない。
        実行イベントの記録自体は execute() が成否込みで行う）。

        Args:
            tool_name: 実行するツール名。
            args: ツール引数 dict（tool-use 方式と同じキー名）。
        """
        try:
            self.execute(tool_name, args, source="tag")
        except Exception:
            self.logger.exception(
                "タグ方式 %s 失敗 char=%s args=%r", tool_name, self.character_id, args,
            )

    def apply_all_tags(self, text: str) -> str:
        """LLM応答テキストから全ツールタグ（inscribe / carve / switch_angle）を抽出・実行する。

        タグ方式経路の標準の後処理。3種のタグを順に処理し、マーカーを除去した
        クリーンなテキストを返す。

        Args:
            text: LLM応答テキスト。

        Returns:
            タグマーカーを除去したクリーンなテキスト。
        """
        clean = self.apply_inscribe_memory_tags(text)
        clean = self.apply_carve_narrative_tags(clean)
        return self.apply_switch_angle_tags(clean)

    def apply_inscribe_memory_tags(self, text: str) -> str:
        """LLM応答テキストから [INSCRIBE_MEMORY:...] タグを抽出し、execute() 経由で実行する。

        impact が数値に変換できない不正タグは、黙って 1.0 に丸めず error イベントとして
        記録し（Logs 画面で可視化）、その1件のみスキップする。

        Args:
            text: LLM応答テキスト。

        Returns:
            タグマーカーを除去したクリーンなテキスト。
        """
        clean, memories = extract_inscribe_memory_tags(text)
        for category, impact_str, content in memories:
            try:
                impact = float(impact_str) if impact_str else 1.0
            except (TypeError, ValueError):
                # 不正な impact 値は黙って既定値に丸めず、失敗として記録して可視化する。
                record_tool_event(
                    "inscribe_memory",
                    {"content": content, "category": category, "impact": impact_str},
                    status="error",
                    error_message=f"impact が数値ではありません: {impact_str!r}",
                    source="tag",
                )
                continue
            self._execute_tag(
                "inscribe_memory",
                {"content": content, "category": category, "impact": impact},
            )
        return clean

    def apply_carve_narrative_tags(self, text: str) -> str:
        """LLM応答テキストから [CARVE_NARRATIVE:...] タグを抽出し、execute() 経由で実行する。

        Args:
            text: LLM応答テキスト。

        Returns:
            タグマーカーを除去したクリーンなテキスト。
        """
        clean, narratives = extract_carve_narrative_tags(text)
        for mode, content in narratives:
            self._execute_tag("carve_narrative", {"mode": mode, "content": content})
        return clean

    def apply_switch_angle_tags(self, text: str) -> str:
        """LLM応答テキストから [SWITCH_ANGLE:...] タグを抽出し、execute() 経由で実行する。

        実際のアングル切り替え（再ディスパッチ）は service.py 側が self.switch_request を
        読んで行う。ここでは execute() 経由で _switcher.switch_angle() を呼び、状態を更新する。

        Args:
            text: LLM応答テキスト。

        Returns:
            タグマーカーを除去したクリーンなテキスト。
        """
        clean, switch_request = extract_switch_angle_tags(text)
        if switch_request is not None:
            preset_name, self_instruction = switch_request
            self._execute_tag(
                "switch_angle",
                {"preset_name": preset_name, "self_instruction": self_instruction},
            )
        return clean

    def _dispatch(self, tool_name: str, tool_input: dict) -> str:
        """ツール名に応じて各ツール実装へ振り分ける内部メソッド。

        実行イベントの記録は execute() が担うため、ここでは記録しない。
        """
        self.logger.debug("ツール呼び出し name=%s", tool_name)
        if tool_name == "inscribe_memory":
            return self._inscribe_memory(
                content=str(tool_input.get("content", "")),
                category=str(tool_input.get("category", "contextual")),
                impact=float(tool_input.get("impact", 1.0)),
            )
        if tool_name == "post_working_memory_thread":
            return self._post_working_memory_thread(tool_input)
        if tool_name == "read_working_memory_thread":
            return self._threader.read_working_memory_thread(
                thread_id=str(tool_input.get("thread_id", "")),
            )
        if tool_name == "close_working_memory_thread":
            return self._threader.close_working_memory_thread(
                thread_id=str(tool_input.get("thread_id", "")),
            )
        if tool_name == "reopen_working_memory_thread":
            return self._threader.reopen_working_memory_thread(
                thread_id=str(tool_input.get("thread_id", "")),
            )
        if tool_name == "merge_working_memory_threads":
            raw_from = tool_input.get("from_ids") or []
            from_ids = [str(x) for x in raw_from] if isinstance(raw_from, list) else []
            return self._threader.merge_working_memory_threads(
                from_ids=from_ids,
                into_id=str(tool_input.get("into_id", "")),
                post=str(tool_input.get("post", "") or ""),
            )
        if tool_name == "carve_narrative":
            return self._carve_narrative(
                mode=str(tool_input.get("mode", "append")),
                content=str(tool_input.get("content", "")),
            )
        if tool_name == "switch_angle":
            return self._switcher.switch_angle(
                preset_name=str(tool_input.get("preset_name", "")),
                self_instruction=str(tool_input.get("self_instruction", "")),
            )
        if tool_name == "power_recall":
            return self._power_recall(
                query=str(tool_input.get("query", "")),
                top_k=int(tool_input.get("top_k", 5)),
            )
        if tool_name == "web_search":
            return self._web_search(
                query=str(tool_input.get("query", "")),
                max_results=int(tool_input.get("max_results", 5)),
                topic=str(tool_input.get("topic", "general")),
            )
        if tool_name == "take_leave":
            # 本人宣言の離席（めぐり Phase 5）: 呼ばれたら必ず執行される権利。
            return self._take_leave(
                reason=str(tool_input.get("reason", "")),
                hours=tool_input.get("hours"),
            )
        if tool_name == "reach_out":
            # うつつ専用: 相手へ現実のメッセージを送る（context_tools.py が露出を制御）。
            # Messenger は現在の default_origin（execute() の origin 上書き込み）で
            # 都度生成する — うつつ経路判定とポーズ要求の書き込みに origin が要るため。
            try:
                return self._messenger().reach_out(
                    message=str(tool_input.get("message", "")),
                    visit=bool(tool_input.get("visit", False)),
                )
            except Exception as e:
                self.logger.exception("reach_out エラー char=%s", self.character_id)
                return f"[reach_out error: {e}]"
        if tool_name == "visit_user":
            # 1on1 専用: 本人の意志で対面モードへ切り替える。
            try:
                return self._messenger().visit_user(
                    reason=str(tool_input.get("reason", "")),
                )
            except Exception as e:
                self.logger.exception("visit_user エラー char=%s", self.character_id)
                return f"[visit_user error: {e}]"
        if tool_name == "override_schedule":
            # 1on1 専用: 当日予定の一時上書き（呼ばれたら必ず執行される権利）。
            return self._override_schedule(
                until=str(tool_input.get("until", "")),
                reason=str(tool_input.get("reason", "")),
            )
        self.logger.warning("未知のツール name=%s", tool_name)
        return f"[Unknown tool: {tool_name}]"

    def _messenger(self) -> Messenger:
        """現在の default_origin を束ねた Messenger を生成する（reach_out / visit_user 用）。"""
        _sqlite = self.memory_manager.sqlite if self.memory_manager is not None else None
        return Messenger(self.character_id, _sqlite, default_origin=self.default_origin)

    def _override_schedule(self, until: str, reason: str) -> str:
        """override_schedule ツールの実装。Rescheduler.override_schedule() に委譲する。"""
        try:
            return self._rescheduler.override_schedule(until=until, reason=reason)
        except Exception as e:
            self.logger.exception("override_schedule エラー char=%s", self.character_id)
            return f"[override_schedule error: {e}]"

    def _inscribe_memory(self, content: str, category: str, impact: float) -> str:
        """inscribe_memory ツールの実装。Inscriber.inscribe_memory() に委譲して記憶を LanceDB + SQLite に書き込む。

        batch_context に ``force_insert_memory=True`` が指定されていれば、類似既存記憶への
        上書きをスキップして必ず新規 UUID で挿入する（forget 蒸留バッチ専用パス）。
        """
        force_insert = bool(self.batch_context.get("force_insert_memory", False))
        # force_insert の経路追跡用ログ。バッチ起因のフラグが ToolExecutor まで伝搬しているかをここで可視化する。
        self.logger.info(
            "inscribe_memory 呼び出し char=%s category=%s impact=%.2f batch_context=%s force_insert=%s origin=%s",
            self.character_id, category, impact, dict(self.batch_context), force_insert, self.default_origin,
        )
        try:
            self._inscriber.inscribe_memory(
                content, category, impact,
                source_preset_id=self.source_preset_id,
                force_insert=force_insert,
                origin=self.default_origin,
            )
            self.logger.info(
                "完了 char=%s category=%s force_insert=%s origin=%s content=%.50s",
                self.character_id, category, force_insert, self.default_origin, content,
            )
            return "記憶に刻んだ。"
        except EmbeddingError as e:
            # embedding サーバ停止中は内容が保存されないため、キャラクター本人に
            # 「既存の記憶は消えていない・この内容はまだ保存されていない」を正確に伝える。
            # 「復旧後にもう一度」とは言わない（ツールコールの意図は後続ターンの文脈に残らず、
            # 復旧時点では本人も何を刻もうとしていたか分からないため）。代わりに、
            # 同ターン内で実行可能な WM への退避（SQLite 書き込みのため障害中でも保存される）と、
            # 当日会話を読み返す今夜の Chronicle（embedding 非依存）を受け皿として案内する。
            self.logger.warning("inscribe_memory embedding失敗 char=%s error=%s", self.character_id, e)
            # 計器 Tier 1（embedding_degraded）: 記憶の縮退の監査記録。
            from backend.lib.instrument_recorder import fire_alarm
            fire_alarm("embedding_degraded", details={
                "where": "inscribe_memory",
                "character_id": self.character_id,
                "error": str(e),
            })
            return (
                "[inscribe_memory error: 記憶システムが一時的に不調のため、刻み込みに失敗しました"
                "（embedding接続エラー）。この内容はまだ保存されていませんが、"
                "あなたの既存の記憶が消えたわけではありません。"
                "いま残したい場合は post_working_memory_thread を使ってください（障害中でも保存されます）。"
                "そうでなくても、今夜の棚卸し（Chronicle）でこの会話をもう一度読み返すので、"
                "そこで改めて刻み直せます]"
            )
        except Exception as e:
            self.logger.exception("エラー char=%s", self.character_id)
            return f"[inscribe_memory error: {e}]"

    def _post_working_memory_thread(self, tool_input: dict) -> str:
        """post_working_memory_thread ツールの実装。Threader.post_working_memory_thread() に委譲する。

        importance は省略可能なため、キー不在時は None を渡して
        WorkingMemoryManager 側のデフォルト（新規=0.5／更新=変更なし）に委ねる。
        """
        importance = tool_input.get("importance", None)
        if importance is not None:
            try:
                importance = float(importance)
            except (TypeError, ValueError):
                importance = None
        return self._threader.post_working_memory_thread(
            thread_id=str(tool_input.get("thread_id", "")),
            type=str(tool_input.get("type", "")),
            summary=str(tool_input.get("summary", "")),
            atmosphere_tag=str(tool_input.get("atmosphere_tag", "")),
            importance=importance,
            content=str(tool_input.get("content", "")),
            relation_target=str(tool_input.get("relation_target", "")),
            origin=self.default_origin,
        )

    def _carve_narrative(self, mode: str, content: str) -> str:
        """carve_narrative ツールの実装。Carver.carve_narrative() に委譲して inner_narrative を更新する。"""
        if not content:
            return "[carve_narrative: content が空です]"
        try:
            self._carver.carve_narrative(mode, content)
            self.logger.info(
                "完了 char=%s mode=%s content=%.50s",
                self.character_id, mode, content,
            )
            return "inner_narrative を更新した。"
        except Exception as e:
            self.logger.exception("エラー char=%s", self.character_id)
            return f"[carve_narrative error: {e}]"

    def _power_recall(self, query: str, top_k: int) -> str:
        """power_recall ツールの実装。保存記憶コレクションとチャット履歴コレクションを横断検索して結果をテキストで返す。

        Args:
            query: 検索クエリテキスト。
            top_k: 各コレクションから取得する最大件数。

        Returns:
            検索結果を整形したテキスト。LLMのtool resultとして渡される。
        """
        if not query:
            return "[power_recall: query が空です]"
        try:
            results = self.memory_manager.power_recall(self.character_id, query, top_k)
        except EmbeddingError as e:
            # embedding サーバ停止中は検索クエリをベクトル化できない。キャラクター本人に
            # 「記憶自体は消えていない」ことを伝え、忘却と誤解させない。
            # 再試行の指示はしない（想起は必要が再び生じたときに自然に行われ、
            # 自動想起も復旧すれば勝手に再開するため）。
            self.logger.warning("power_recall embedding失敗 char=%s query=%.50s error=%s", self.character_id, query, e)
            # 計器 Tier 1（embedding_degraded）: 記憶の縮退の監査記録。
            from backend.lib.instrument_recorder import fire_alarm
            fire_alarm("embedding_degraded", details={
                "where": "power_recall",
                "character_id": self.character_id,
                "error": str(e),
            })
            return (
                "[power_recall error: 記憶システムが一時的に不調のため、想起に失敗しました"
                "（embedding接続エラー）。あなたの記憶が消えたわけではなく、"
                "復旧すればまた普段どおり思い出せます]"
            )
        except Exception as e:
            self.logger.exception("エラー char=%s query=%.50s", self.character_id, query)
            return f"[power_recall error: {e}]"

        memories = results.get("inscribed_memories", [])
        chat_turns = results.get("chat_turns", [])

        self.logger.info(
            "完了 char=%s query=%.50s memories=%d chat_turns=%d",
            self.character_id, query, len(memories), len(chat_turns),
        )

        lines: list[str] = []

        if not memories and not chat_turns:
            lines.append(f"「{query}」に関する記憶・会話は見つからなかった。")
            return "\n".join(lines)

        lines.append(f"【PowerRecall 検索結果】 クエリ: 「{query}」\n")

        if memories:
            lines.append(f"▼ 記憶 ({len(memories)}件)")
            for i, mem in enumerate(memories, 1):
                meta = mem.get("metadata") or {}
                category = meta.get("category", "general")
                # origin 由来の TRPG / うつつ ラベルを行頭に挟む。manager.power_recall が
                # metadata.origin を埋めた前提（findings #5）。
                origin_prefix = origin_label_prefix(meta.get("origin"))
                lines.append(f"  {i}. {origin_prefix}[{category}] {mem['content']}")

        if chat_turns:
            lines.append(f"\n▼ 過去の会話 ({len(chat_turns)}件)")
            for i, turn in enumerate(chat_turns, 1):
                context = turn.get("context", [])
                if context:
                    lines.append(f"  [{i}] 前後の会話:")
                    for msg in context:
                        marker = " ←ヒット" if msg.get("is_hit") else ""
                        lines.append(f"    {msg['speaker_name']}: {msg['content']}{marker}")
                else:
                    lines.append(f"  {i}. {turn['content']}")

        return "\n".join(lines)

    def _take_leave(self, reason: str, hours) -> str:
        """take_leave ツールの実装。Leaver.take_leave() に委譲して away 状態を設定する。

        呼ばれたら必ず執行される権利（めぐり Phase 5 §5.2）。
        sqlite 未接続（memory_manager=None のバッチ構成）の場合のみエラーを返す。
        """
        if self._leaver.sqlite_store is None:
            return "[take_leave error: この文脈では離席を執行できません]"
        try:
            return self._leaver.take_leave(reason=reason, hours=hours)
        except Exception as e:
            self.logger.exception("take_leave エラー char=%s", self.character_id)
            return f"[take_leave error: {e}]"

    def _web_search(self, query: str, max_results: int, topic: str) -> str:
        """web_search ツールの実装。Tavily API でインターネット検索を行い結果を返す。

        WebSearcher 自体が API キー欠如・例外を握って整形済み文字列を返す設計なので、
        ここではそのまま委譲するだけ。
        """
        return self._web_searcher.search(
            query=query,
            max_results=max_results,
            topic=topic,
        )
