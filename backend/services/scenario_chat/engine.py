"""シナリオチャット用 シーンエンジン。

SceneEngine は「セッション状態 + NPC + 履歴 + プレイヤー発話」を受け取り、
UtteranceDelta / TurnRecord の列を非同期 yield する抽象。

P1 では EnsembleEngine のみを実装する。GM が単一 LLM 呼出で
Narrator + 全 NPC を演じる方式。出力を parser.py で `@名前:` ブロックに分解し、
ストリーミング差分をそのまま流す。

将来:
    - PolyphonyEngine: キャラごとに個別 provider 呼出
    - HybridEngine: 一部 NPC を既存 Character に lift-out したセッション用
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Protocol

from backend.providers.registry import create_provider
from backend.services.scenario_chat.context import (
    format_history_for_gm,
    resolve_history_limits,
    slice_history,
)
from backend.services.scenario_chat.parser import (
    ScenarioChatParser,
    UtteranceDelta,
)
from backend.services.scenario_chat.prompt_builder import build_gm_system_prompt


@dataclass
class TurnRecord:
    """1 発話の確定レコード（speaker_end 相当）。

    話者の発話末尾（次の話者へ切り替わる直前または stream 終了時）に発行され、
    SSE の `speaker_end` イベントに対応する。
    """

    speaker_type: str
    speaker_id: Optional[str]
    speaker_name: str
    content: str
    is_known: bool


@dataclass
class EngineResult:
    """エンジンが 1 ターンを完了したときに残す副産物。

    raw_response はターン全体の生 LLM 出力で、`scenario_turns.raw_response` に
    保存される（同一ターン内の全発話レコードに同じ値が格納される想定）。
    """

    raw_response: str


class SceneEngine(Protocol):
    """シーンエンジン抽象。P1: EnsembleEngine / 将来: Polyphony/Hybrid。

    シナリオテンプレート（scenario）+ プレイインスタンス（session）の 2 層構造を取る。
    GM プロンプトの組み立てには scenario の世界観・NPC・user_alias 等を使い、
    session は履歴の所属判定や追加保存先として使われる。
    """

    async def generate_stream(
        self,
        scenario: Any,
        npcs: list[Any],
        history: list[Any],
        user_message: str,
        settings: dict,
        auto_advance: bool = False,
    ) -> AsyncIterator[Any]:
        """1 ターンの発話列をストリーミング生成する。

        Args:
            auto_advance: True なら「ユーザは無言で続きを促す」モード。
                          user_message は使わず、GM が地の文と NPC だけで進める。

        Yields:
            UtteranceDelta: ストリーミング中の発話差分。
            TurnRecord: 話者の発話末尾（speaker_end 相当）。
            EngineResult: ターン完了時の副産物（最後に 1 回だけ）。
        """
        ...


class EnsembleEngine:
    """GM が単一 LLM 呼出で全話者を代弁する P1 エンジン。

    処理の流れ:
        1. 履歴を切り出して `<話者>本文</話者>` 形式に整形する
        2. GM 用 system prompt を組み立てる
        3. preset_loader 経由でプロバイダを生成し、generate_stream_typed() を呼ぶ
        4. パーサに流して UtteranceDelta を yield
        5. 話者切替の境界で 1 つ前の話者の TurnRecord を yield
        6. ストリーム終了時に最後の TurnRecord と EngineResult を yield

    依存注入:
        preset_loader: preset_id → (provider, model, settings, preset_name) を返す関数。
                       本番では SQLiteStore.get_model_preset を使うが、テストでは
                       モックを差し込めるよう外出ししている。
        provider_factory: 標準では providers.registry.create_provider。
                          テスト時に置換しやすいよう注入可能。
    """

    def __init__(
        self,
        preset_loader: Callable[[str], Any],
        provider_factory: Callable[..., Any] = create_provider,
    ) -> None:
        """エンジンを初期化する。

        Args:
            preset_loader: preset_id を受け取って ORM 風 preset オブジェクトを返す関数。
                           `provider` / `model_id` / `name` 属性を持つ必要がある。
            provider_factory: プロバイダ生成関数。デフォルトは
                              backend.providers.registry.create_provider。
        """
        self._preset_loader = preset_loader
        self._provider_factory = provider_factory

    async def generate_stream(
        self,
        scenario: Any,
        npcs: list[Any],
        history: list[Any],
        user_message: str,
        settings: dict,
        auto_advance: bool = False,
        synopsis_auto: str = "",
        synopsis_manual: str = "",
    ) -> AsyncIterator[Any]:
        """1 ターンの発話列をストリーミング生成する。

        Args:
            scenario: Scenario ORM。user_alias / gm_preset_id を必須とする。
            npcs: 既知 NPC のリスト（ScenarioNpc ORM 風）。
            history: ScenarioTurn ORM 風オブジェクトの時系列昇順リスト（全件）。
            user_message: 今回のプレイヤー発話テキスト。
                          auto_advance=True の場合は空文字列を渡してよい（無視される）。
            settings: グローバル設定辞書（API キー等）。
            auto_advance: True なら「ユーザは無言で続きを促す」モード。
                          user_message は GM プロンプトに含まれず、代わりに
                          OOC 指示が末尾に注入される。
            synopsis_auto: セッションの自動あらすじ（メイン）。GM への system prompt に挿入される。
            synopsis_manual: セッションの手動補足メモ（補正指示）。同上。

        Yields:
            UtteranceDelta | TurnRecord | EngineResult

        Raises:
            ValueError: preset_id に対応するプリセットが見つからない場合。
        """
        # 1. 履歴切り出しと整形
        max_turns, max_chars = resolve_history_limits(scenario, settings)
        sliced = slice_history(history, max_turns=max_turns, max_chars=max_chars)
        history_text = format_history_for_gm(
            sliced, user_alias=scenario.user_alias
        )

        # 2. GM 用 system prompt
        system_prompt = build_gm_system_prompt(
            scenario=scenario,
            npcs=npcs,
            history_text=history_text,
            user_message=None if auto_advance else user_message,
            auto_advance=auto_advance,
            synopsis_auto=synopsis_auto,
            synopsis_manual=synopsis_manual,
        )

        # 3. プロバイダ生成
        preset = self._preset_loader(scenario.gm_preset_id)
        if preset is None:
            raise ValueError(
                f"GM プリセットが見つかりません: gm_preset_id={scenario.gm_preset_id}"
            )
        provider = self._provider_factory(
            preset.provider,
            model=preset.model_id,
            settings=settings,
            preset_name=preset.name,
        )

        # 4. ストリーミング受信＆パース
        # messages はユーザ発話のみ（system prompt 側にも複製して入れている）。
        # auto_advance 時はプレイヤー発話がないため、OOC 指示を user role で渡す。
        if auto_advance:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "(プレイヤーは今ターン無言。物語を前に進めてください。"
                        "時間飛ばし・場面転換も自由です。)"
                    ),
                }
            ]
        else:
            messages = [{"role": "user", "content": user_message}]
        known_map = {
            npc.name: npc.id for npc in npcs if getattr(npc, "name", None)
        }
        parser = ScenarioChatParser(
            known_npc_names=known_map,
            user_alias=scenario.user_alias,
        )

        raw_chunks: list[str] = []
        # 各話者ごとに本文を蓄積し、話者切替時 / 終端で TurnRecord を発行する。
        cur_speaker_key: Optional[tuple] = None
        cur_buffer: list[str] = []
        cur_is_known: bool = True

        def _make_record() -> Optional[TurnRecord]:
            """現在蓄積中の発話をレコード化する。空なら None。

            content の末尾改行・空白は除去する（GM が話者ブロック末尾に `\\n` を
            つけて区切るのが慣例だが、UI 表示には不要なノイズになるため）。
            """
            if cur_speaker_key is None:
                return None
            body = "".join(cur_buffer).rstrip()
            if not body:
                return None
            speaker_type, speaker_id, speaker_name = cur_speaker_key
            return TurnRecord(
                speaker_type=speaker_type,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                content=body,
                is_known=cur_is_known,
            )

        async def _flush_deltas(deltas: list[UtteranceDelta]):
            """delta 列を上位へ yield しつつ、TurnRecord 発行タイミングを管理する。"""
            nonlocal cur_speaker_key, cur_buffer, cur_is_known
            for d in deltas:
                key = (d.speaker_type, d.speaker_id, d.speaker_name)
                if d.is_speaker_change:
                    # 直前の話者を確定して record を発行
                    prev = _make_record()
                    if prev is not None:
                        yield prev
                    cur_speaker_key = key
                    cur_buffer = []
                    cur_is_known = d.is_known
                cur_buffer.append(d.content_delta)
                yield d

        # generate_stream_typed は (type, content) を yield する。
        # text 以外（thinking 等）は無視する（GM ロールには思考可視化を出さない方針）。
        async for chunk_type, content in provider.generate_stream_typed(
            system_prompt, messages
        ):
            if chunk_type != "text" or not content:
                continue
            raw_chunks.append(content)
            deltas = parser.feed(content)
            async for item in _flush_deltas(deltas):
                yield item

        # ストリーム終端: parser flush
        deltas = parser.flush()
        async for item in _flush_deltas(deltas):
            yield item

        # 最後の話者の TurnRecord を flush
        final = _make_record()
        if final is not None:
            yield final

        # ターン副産物
        yield EngineResult(raw_response="".join(raw_chunks))
