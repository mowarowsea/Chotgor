"""Chotgor MCPツール定義 — 記憶・ワーキングメモリ・アングル切り替えのツールスキーマとexecutor。

LLM の tool-use（function calling）で保存記憶・ワーキングメモリ・switch_angle 等を
操作するための定義を提供する。tool-use 非対応プロバイダー（Claude CLI、Ollama）は
タグ方式（[INSCRIBE_MEMORY:...] / [CARVE_NARRATIVE:...] マーカー）にフォールバックする。

各ツールのスキーマ・説明文は対応モジュールに定義する:
- inscribe_memory: inscriber.py
- carve_narrative: carver.py
- post_working_memory_thread / open_working_memory_thread: threader.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.services.memory.manager import InscribedMemoryManager
    from backend.services.memory.working_memory_manager import WorkingMemoryManager

from backend.character_actions.recaller import POWER_RECALL_SCHEMA, POWER_RECALL_TOOL_DESCRIPTION
from backend.character_actions.switcher import Switcher, SWITCH_ANGLE_SCHEMA, SWITCH_ANGLE_TOOL_DESCRIPTION
from backend.character_actions.carver import Carver, CARVE_NARRATIVE_SCHEMA, CARVE_NARRATIVE_TOOL_DESCRIPTION
from backend.character_actions.inscriber import Inscriber, INSCRIBE_MEMORY_SCHEMA, INSCRIBE_MEMORY_TOOL_DESCRIPTION
from backend.character_actions.threader import (
    Threader,
    POST_WORKING_MEMORY_THREAD_SCHEMA,
    POST_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
    OPEN_WORKING_MEMORY_THREAD_SCHEMA,
    OPEN_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
)

# Anthropic形式のツール定義リスト
ANTHROPIC_TOOLS: list[dict] = [
    {
        "name": "inscribe_memory",
        "description": INSCRIBE_MEMORY_TOOL_DESCRIPTION,
        "input_schema": INSCRIBE_MEMORY_SCHEMA,
    },
    {
        "name": "post_working_memory_thread",
        "description": POST_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
        "input_schema": POST_WORKING_MEMORY_THREAD_SCHEMA,
    },
    {
        "name": "open_working_memory_thread",
        "description": OPEN_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
        "input_schema": OPEN_WORKING_MEMORY_THREAD_SCHEMA,
    },
    {
        "name": "carve_narrative",
        "description": CARVE_NARRATIVE_TOOL_DESCRIPTION,
        "input_schema": CARVE_NARRATIVE_SCHEMA,
    },
    {
        "name": "switch_angle",
        "description": SWITCH_ANGLE_TOOL_DESCRIPTION,
        "input_schema": SWITCH_ANGLE_SCHEMA,
    },
    {
        "name": "power_recall",
        "description": POWER_RECALL_TOOL_DESCRIPTION,
        "input_schema": POWER_RECALL_SCHEMA,
    },
]

# OpenAI形式のツール定義リスト（Anthropic形式から変換）
OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in ANTHROPIC_TOOLS
]


@dataclass
class ToolCall:
    """正規化されたツール呼び出し情報。

    Attributes:
        id: プロバイダーが発行するツール呼び出しID。
        name: ツール名（inscribe_memory / post_working_memory_thread / open_working_memory_thread / carve_narrative / switch_angle / power_recall）。
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

    inscribe_memory / post_working_memory_thread / open_working_memory_thread / carve_narrative / switch_angle / power_recall の
    各ツールを受け取り、Inscriber / Threader / Carver / Switcher を通じてDBへ反映する。

    Attributes:
        character_id: 操作対象のキャラクターID。
        session_id: 現在のセッションID。
        memory_manager: 記憶の読み書きを担うマネージャー。
        working_memory_manager: ワーキングメモリの読み書きを担うマネージャー。
        batch_context: バッチ処理由来のフラグ群。例 ``{"force_insert_memory": True}`` を
            渡すと、inscribe_memory ツール実行時に類似上書きをスキップして必ず新規 ID で
            挿入する（forget 蒸留バッチ専用）。MCP ツール API は変更せず、バッチ側のみ
            この context で挙動を切り替えるための内部チャネル。
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
    ) -> None:
        """ToolExecutorを初期化する。

        Args:
            batch_context: バッチ処理が指定するツール挙動の切り替えフラグ群。
                通常チャットでは None（=空 dict 扱い）。forget 蒸留バッチからは
                ``{"force_insert_memory": True}`` を渡す。
        """
        self.character_id = character_id
        self.session_id = session_id
        self.memory_manager = memory_manager
        self.working_memory_manager = working_memory_manager
        self.batch_context: dict = batch_context or {}
        self._inscriber = Inscriber(character_id, memory_manager)
        self._threader = Threader(character_id, working_memory_manager)
        self._carver = Carver(character_id, memory_manager.sqlite)
        self._switcher = Switcher()

    @property
    def switch_request(self) -> tuple[str, str] | None:
        """switch_angle が呼ばれた場合の切り替えリクエスト。generate_with_tools() ループが検知して即中断する。"""
        return self._switcher.switch_request

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """ツール名と入力を受け取り実行して結果テキストを返す。

        Args:
            tool_name: ツール名（"inscribe_memory" / "post_working_memory_thread" / "open_working_memory_thread" / "carve_narrative" / "switch_angle" / "power_recall"）。
            tool_input: ツールの入力パラメータ dict。

        Returns:
            ツールの実行結果を表すテキスト。
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
        if tool_name == "open_working_memory_thread":
            return self._threader.open_working_memory_thread(
                thread_id=str(tool_input.get("thread_id", "")),
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
        self.logger.warning("未知のツール name=%s", tool_name)
        return f"[Unknown tool: {tool_name}]"

    def _inscribe_memory(self, content: str, category: str, impact: float) -> str:
        """inscribe_memory ツールの実装。Inscriber.inscribe_memory() に委譲して記憶を LanceDB + SQLite に書き込む。

        batch_context に ``force_insert_memory=True`` が指定されていれば、類似既存記憶への
        上書きをスキップして必ず新規 UUID で挿入する（forget 蒸留バッチ専用パス）。
        """
        force_insert = bool(self.batch_context.get("force_insert_memory", False))
        # force_insert の経路追跡用ログ。バッチ起因のフラグが ToolExecutor まで伝搬しているかをここで可視化する。
        self.logger.info(
            "inscribe_memory 呼び出し char=%s category=%s impact=%.2f batch_context=%s force_insert=%s",
            self.character_id, category, impact, dict(self.batch_context), force_insert,
        )
        try:
            self._inscriber.inscribe_memory(content, category, impact, force_insert=force_insert)
            self.logger.info(
                "完了 char=%s category=%s force_insert=%s content=%.50s",
                self.character_id, category, force_insert, content,
            )
            return "記憶に刻んだ。"
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
                category = mem.get("metadata", {}).get("category", "general")
                lines.append(f"  {i}. [{category}] {mem['content']}")

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
