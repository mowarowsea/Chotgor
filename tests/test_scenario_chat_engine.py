"""シナリオチャット EnsembleEngine のテスト。

backend.services.scenario_chat.engine.EnsembleEngine の振る舞いを検証する。

LLM プロバイダはモック化する: provider_factory を差し替えて
固定の generate_stream_typed() を返す FakeProvider を返す関数を渡す。
これにより GM の出力テキストをテストごとに自由に設定できる。

検証する観点:
    - UtteranceDelta と TurnRecord が期待通りの順序で yield されること
    - 既知 NPC は speaker_id を持ち、未知 NPC は None になること
    - GM のユーザ代弁ブロックは UtteranceDelta も TurnRecord も発行しないこと
    - 複数話者の境界で正しく TurnRecord が確定されること
    - EngineResult.raw_response にチャンクの連結が入ること
    - preset_loader が None を返したら ValueError
    - 履歴切り出しが GM プロンプトに反映されること（履歴が長すぎる場合は切られる）
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.scenario_chat.engine import (
    EngineResult,
    EnsembleEngine,
    TurnRecord,
)
from backend.services.scenario_chat.parser import UtteranceDelta


# ─── フェイクオブジェクト ────────────────────────────────────────────────────


@dataclass
class FakeScenario:
    """ZetaScenario 風のダミー（テンプレート）。"""

    user_alias: str = "プレイヤー"
    gm_preset_id: str = "preset-001"
    scenario: Optional[str] = None
    history_max_turns: Optional[int] = None
    history_max_chars: Optional[int] = None


@dataclass
class FakeNpc:
    """ZetaNpc 風のダミー（scenario_id に紐づく）。"""

    id: str
    name: str
    description: Optional[str] = None
    image_data: Optional[str] = None
    scenario_id: str = "scenario-001"


@dataclass
class FakeTurn:
    """ZetaTurn 風のダミー。"""

    speaker_type: str
    speaker_name: str
    content: str
    speaker_id: Optional[str] = None


@dataclass
class FakePreset:
    """LLMModelPreset 風のダミー。"""

    id: str = "preset-001"
    name: str = "TestPreset"
    provider: str = "fake"
    model_id: str = "fake-model"


class FakeProvider:
    """generate_stream_typed() で固定の (type, chunk) 列を返すスタブ。"""

    def __init__(self, chunks: list[tuple[str, str]]):
        """
        Args:
            chunks: yield する (type, content) ペアのリスト。
                    type は "text" / "thinking" など。
        """
        self.chunks = chunks
        self.received_system_prompt: Optional[str] = None
        self.received_messages: Optional[list[dict]] = None

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """指定チャンクを 1 つずつ yield する。"""
        self.received_system_prompt = system_prompt
        self.received_messages = messages
        for typ, content in self.chunks:
            yield typ, content


def _make_engine(
    chunks: list[tuple[str, str]],
    preset: Optional[FakePreset] = None,
) -> tuple[EnsembleEngine, FakeProvider]:
    """指定チャンクを返す FakeProvider 付きエンジンを構築する。"""
    provider = FakeProvider(chunks)
    preset = preset or FakePreset()

    def loader(preset_id: str):
        # preset_id が一致する場合のみ返す（None を返すケースもテストで使う）
        if preset is None or preset_id != preset.id:
            return None
        return preset

    def factory(provider_id: str, model: str, settings: dict, **kwargs):
        return provider

    engine = EnsembleEngine(preset_loader=loader, provider_factory=factory)
    return engine, provider


async def _collect(generator):
    """非同期ジェネレーターの全 yield を収集する。"""
    items = []
    async for item in generator:
        items.append(item)
    return items


# ─── 基本的なターン生成 ──────────────────────────────────────────────────────


class TestBasicTurnGeneration:
    """単一・複数話者のターン生成、TurnRecord の発行タイミングを検証する。

    シナリオモードの最も基本的なフローを通して、エンジンが
    UtteranceDelta と TurnRecord を正しい順序で発行することを確認する。
    """

    @pytest.mark.asyncio
    async def test_single_known_npc_turn(self):
        """既知 NPC 1 体だけの応答が UtteranceDelta + TurnRecord で返ること。"""
        chunks = [("text", "@レイカ: ……来たんだ\n")]
        engine, _ = _make_engine(chunks)
        npcs = [FakeNpc(id="npc-r", name="レイカ")]
        items = await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=npcs,
                history=[],
                user_message="やぁ",
                settings={},
            )
        )

        deltas = [i for i in items if isinstance(i, UtteranceDelta)]
        records = [i for i in items if isinstance(i, TurnRecord)]
        results = [i for i in items if isinstance(i, EngineResult)]

        assert any(d.speaker_name == "レイカ" for d in deltas)
        assert len(records) == 1
        assert records[0].speaker_name == "レイカ"
        assert records[0].speaker_id == "npc-r"
        assert records[0].is_known is True
        assert "来たんだ" in records[0].content
        assert len(results) == 1
        assert "@レイカ" in results[0].raw_response

    @pytest.mark.asyncio
    async def test_multiple_speakers_emit_records_in_order(self):
        """複数話者の発話が順番に TurnRecord で確定されること。"""
        full = (
            "@Narrator: 雨\n"
            "@レイカ: 来た\n"
            "@トウコ: 遅いですよ\n"
        )
        chunks = [("text", full)]
        engine, _ = _make_engine(chunks)
        npcs = [
            FakeNpc(id="npc-r", name="レイカ"),
            FakeNpc(id="npc-t", name="トウコ"),
        ]
        items = await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=npcs,
                history=[],
                user_message="",
                settings={},
            )
        )
        records = [i for i in items if isinstance(i, TurnRecord)]
        names = [r.speaker_name for r in records]
        assert names == ["Narrator", "レイカ", "トウコ"]
        # Narrator の speaker_id は None
        assert records[0].speaker_type == "narrator"
        assert records[0].speaker_id is None
        # 既知 NPC の speaker_id は ORM ID
        assert records[1].speaker_id == "npc-r"
        assert records[2].speaker_id == "npc-t"

    @pytest.mark.asyncio
    async def test_unknown_npc_in_output(self):
        """GM が未登録の話者を生やしたら speaker_id=None, is_known=False で扱うこと。"""
        chunks = [("text", "@モブ店主: いらっしゃい\n")]
        engine, _ = _make_engine(chunks)
        items = await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=[],
                history=[],
                user_message="",
                settings={},
            )
        )
        records = [i for i in items if isinstance(i, TurnRecord)]
        assert len(records) == 1
        assert records[0].speaker_name == "モブ店主"
        assert records[0].speaker_id is None
        assert records[0].is_known is False

    @pytest.mark.asyncio
    async def test_chunked_stream_aggregates(self):
        """チャンクが細切れに来ても、最終的に正しい TurnRecord が組み上がること。"""
        chunks = [
            ("text", "@レ"),
            ("text", "イカ:"),
            ("text", " こん"),
            ("text", "にち"),
            ("text", "は\n"),
        ]
        engine, _ = _make_engine(chunks)
        npcs = [FakeNpc(id="npc-r", name="レイカ")]
        items = await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=npcs,
                history=[],
                user_message="",
                settings={},
            )
        )
        records = [i for i in items if isinstance(i, TurnRecord)]
        assert len(records) == 1
        assert records[0].content.replace("\n", "") == "こんにちは"


# ─── ユーザ代弁の破棄 ─────────────────────────────────────────────────────────


class TestUserAliasSuppression:
    """GM がユーザを代弁した場合、UtteranceDelta も TurnRecord も発行されないこと。"""

    @pytest.mark.asyncio
    async def test_user_alias_block_no_record(self):
        """@<user_alias>: のブロックは TurnRecord に出ないこと。"""
        chunks = [
            ("text", "@プレイヤー: 勝手な発話\n"),
            ("text", "@レイカ: 通る\n"),
        ]
        engine, _ = _make_engine(chunks)
        npcs = [FakeNpc(id="npc-r", name="レイカ")]
        items = await _collect(
            engine.generate_stream(
                scenario=FakeScenario(user_alias="プレイヤー"),
                npcs=npcs,
                history=[],
                user_message="",
                settings={},
            )
        )
        records = [i for i in items if isinstance(i, TurnRecord)]
        names = [r.speaker_name for r in records]
        assert "プレイヤー" not in names
        assert "レイカ" in names


# ─── 非 text チャンクの扱い ──────────────────────────────────────────────────


class TestNonTextChunks:
    """thinking など非 text チャンクは無視されることを検証する。

    GM ロールに思考可視化を出さない方針。
    """

    @pytest.mark.asyncio
    async def test_thinking_ignored(self):
        """thinking チャンクは UtteranceDelta も生まないこと。"""
        chunks = [
            ("thinking", "(内心) どう答えるか…"),
            ("text", "@レイカ: 答えるよ\n"),
        ]
        engine, _ = _make_engine(chunks)
        npcs = [FakeNpc(id="npc-r", name="レイカ")]
        items = await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=npcs,
                history=[],
                user_message="",
                settings={},
            )
        )
        deltas = [i for i in items if isinstance(i, UtteranceDelta)]
        # 内心テキストは UtteranceDelta に含まれない
        assert all("内心" not in d.content_delta for d in deltas)


# ─── EngineResult ────────────────────────────────────────────────────────────


class TestEngineResult:
    """EngineResult.raw_response がストリーム全テキストの結合になっていることを検証する。"""

    @pytest.mark.asyncio
    async def test_raw_response_accumulated(self):
        """text チャンクのみが raw_response に集約されること。"""
        chunks = [
            ("thinking", "ignored"),
            ("text", "@レイカ: "),
            ("text", "本文1\n"),
            ("text", "@Narrator: "),
            ("text", "情景\n"),
        ]
        engine, _ = _make_engine(chunks)
        npcs = [FakeNpc(id="npc-r", name="レイカ")]
        items = await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=npcs,
                history=[],
                user_message="",
                settings={},
            )
        )
        results = [i for i in items if isinstance(i, EngineResult)]
        assert len(results) == 1
        raw = results[0].raw_response
        assert "@レイカ" in raw
        assert "本文1" in raw
        assert "@Narrator" in raw
        assert "情景" in raw
        assert "ignored" not in raw

    @pytest.mark.asyncio
    async def test_engine_result_is_last(self):
        """EngineResult は最後に 1 つだけ yield されること。"""
        chunks = [("text", "@Narrator: x\n")]
        engine, _ = _make_engine(chunks)
        items = await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=[],
                history=[],
                user_message="",
                settings={},
            )
        )
        # 最後の要素が EngineResult
        assert isinstance(items[-1], EngineResult)
        # EngineResult は 1 つだけ
        assert sum(1 for i in items if isinstance(i, EngineResult)) == 1


# ─── プリセット未発見 ────────────────────────────────────────────────────────


class TestPresetMissing:
    """preset_loader が None を返した場合は ValueError を投げること。"""

    @pytest.mark.asyncio
    async def test_unknown_preset_raises(self):
        """gm_preset_id が未知なら ValueError を投げること。"""
        engine = EnsembleEngine(
            preset_loader=lambda pid: None,
            provider_factory=lambda *a, **kw: None,
        )
        with pytest.raises(ValueError, match="プリセット"):
            await _collect(
                engine.generate_stream(
                    scenario=FakeScenario(gm_preset_id="missing"),
                    npcs=[],
                    history=[],
                    user_message="hi",
                    settings={},
                )
            )


# ─── プロンプトへの履歴反映 ──────────────────────────────────────────────────


class TestHistoryInPrompt:
    """送信される system prompt に履歴が反映されることを検証する。

    実際の GM プロンプトの組み立てロジックは prompt_builder.py 側のテストで
    詳細検証しているため、ここでは「履歴テキストが入ること」「上限が利くこと」
    の最低限のみ確認する。
    """

    @pytest.mark.asyncio
    async def test_history_included_in_system_prompt(self):
        """渡された履歴が GM への system prompt に整形されて含まれること。"""
        chunks = [("text", "@Narrator: ok\n")]
        engine, provider = _make_engine(chunks)
        history = [
            FakeTurn(speaker_type="user", speaker_name="プレイヤー", content="導入"),
            FakeTurn(speaker_type="narrator", speaker_name="Narrator", content="夜"),
        ]
        await _collect(
            engine.generate_stream(
                scenario=FakeScenario(user_alias="プレイヤー"),
                npcs=[],
                history=history,
                user_message="続き",
                settings={},
            )
        )
        sp = provider.received_system_prompt
        assert sp is not None
        # 履歴は `@名前: 本文` 規約で整形される（GM 出力形式と一致）
        assert "@プレイヤー: 導入" in sp
        assert "@Narrator: 夜" in sp
        # 直近の流れブロックがあること
        assert "直近の流れ" in sp

    @pytest.mark.asyncio
    async def test_history_truncated_by_max_turns(self):
        """history_max_turns で履歴件数が制限されること。"""
        chunks = [("text", "@Narrator: ok\n")]
        engine, provider = _make_engine(chunks)
        history = [
            FakeTurn(speaker_type="user", speaker_name="P", content=f"発話{i}")
            for i in range(10)
        ]
        await _collect(
            engine.generate_stream(
                scenario=FakeScenario(history_max_turns=2),
                npcs=[],
                history=history,
                user_message="",
                settings={},
            )
        )
        sp = provider.received_system_prompt or ""
        # 直近 2 件のみ
        assert "発話8" in sp
        assert "発話9" in sp
        # 古い方は含まれない
        assert "発話0" not in sp
        assert "発話7" not in sp


# ─── あらすじ（synopsis）の GM プロンプト注入 ─────────────────────────────────


class TestSynopsisIntoSystemPrompt:
    """`synopsis_auto` / `synopsis_manual` 引数が GM システムプロンプトに反映されるかを検証する。

    エンジンは synopsis を受け取って prompt_builder に渡すだけのパススルー責務。
    生成内容や追記ロジックは別モジュール（synopsis.py）の責務。
    """

    @pytest.mark.asyncio
    async def test_synopsis_auto_appears_in_system_prompt(self):
        """synopsis_auto を渡すと「これまでのあらすじ」ブロックが system prompt に入ること。"""
        chunks = [("text", "@Narrator: 続きの場面。\n")]
        engine, provider = _make_engine(chunks)
        await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=[],
                history=[],
                user_message="",
                settings={},
                synopsis_auto="勇者は森でレイカと出会った。",
            )
        )
        sp = provider.received_system_prompt or ""
        assert "# これまでのあらすじ" in sp
        assert "勇者は森でレイカと出会った。" in sp

    @pytest.mark.asyncio
    async def test_synopsis_manual_appears_in_system_prompt(self):
        """synopsis_manual を渡すと補足メモブロックが system prompt に入ること。"""
        chunks = [("text", "@Narrator: 場面。\n")]
        engine, provider = _make_engine(chunks)
        await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=[],
                history=[],
                user_message="",
                settings={},
                synopsis_manual="プレイヤーは「絶対に裏切らない」と約束した。",
            )
        )
        sp = provider.received_system_prompt or ""
        assert "# プレイヤーからの補足メモ" in sp
        assert "プレイヤーは「絶対に裏切らない」と約束した。" in sp

    @pytest.mark.asyncio
    async def test_empty_synopsis_omits_block(self):
        """synopsis を渡さない場合は両方ともブロックが省略されること（後方互換）。"""
        chunks = [("text", "@Narrator: 通常ターン。\n")]
        engine, provider = _make_engine(chunks)
        await _collect(
            engine.generate_stream(
                scenario=FakeScenario(),
                npcs=[],
                history=[],
                user_message="",
                settings={},
            )
        )
        sp = provider.received_system_prompt or ""
        assert "# これまでのあらすじ" not in sp
        assert "# プレイヤーからの補足メモ" not in sp
