"""シナリオチャットのスケッチ（想起記憶・WM・思考）の SSE 配信と永続化を検証する。

1on1 では `chat_messages.reasoning` に保存された reasoning をフロントの ThinkingBlock が
「想起した記憶」「スケッチ」の 2 ブロックへ仕分けて描く。シナリオでも同じ体験にするため、
`scenario_turns.reasoning` へ保存し、生成中は `reasoning` SSE イベントでライブ表示する。

検証する観点:
    - GM: engine の ThinkingDelta が `reasoning` イベントとして流れること
    - GM: 1 レスポンスが複数の話者ブロックに割れても、スケッチは先頭ターンにだけ入ること
      （UI でバブル列の頭に 1 度だけ出すため。末尾に入る anticipation と対になる）
    - GM: スケッチが空なら reasoning イベントも保存も発生しないこと
    - PC: pc_done の reasoning が該当 PC ターンへ保存されること
    - シリアライザが reasoning を API レスポンスへ載せること
    - 既存ターン（reasoning 列 NULL）が None として扱われ、UI 側で単に出ないこと
"""

import asyncio
import uuid

import pytest

import backend.services.scenario_chat.pc_runner as pc_runner_mod
import backend.services.scenario_chat.service as svc
from backend.services.scenario_chat.engine import (
    EngineResult,
    ThinkingDelta,
    TurnRecord,
)
from backend.services.scenario_chat.parser import UtteranceDelta
from backend.services.scenario_chat.serializers import scenario_turn_to_dict
from backend.services.scenario_chat.turns import _save_turn

from tests._scenario_sqlite_helpers import _make_scenario, _make_session


# ─── フェイクエンジン ────────────────────────────────────────────────────────


class FakeEngine:
    """`generate_stream` で固定のアイテム列を返すエンジンスタブ。

    `_run_gm_turn` は engine の yield するオブジェクトの型で分岐するだけなので、
    ThinkingDelta / UtteranceDelta / TurnRecord / EngineResult を直接並べて渡せば
    プロバイダ層を通さずに GM 1 レスポンス分を再現できる。
    """

    def __init__(self, items: list):
        self.items = items

    async def generate_stream(self, **kwargs):
        """コンストラクタで受け取ったアイテムをそのまま順に yield する。"""
        for item in self.items:
            yield item


def _narrator_record(content: str) -> TurnRecord:
    """Narrator の TurnRecord を作るショートカット。"""
    return TurnRecord(
        speaker_type="narrator",
        speaker_id=None,
        speaker_name="Narrator",
        content=content,
        is_known=True,
    )


def _npc_record(name: str, content: str) -> TurnRecord:
    """未登録 NPC の TurnRecord を作るショートカット。"""
    return TurnRecord(
        speaker_type="npc",
        speaker_id=None,
        speaker_name=name,
        content=content,
        is_known=False,
    )


def _run_gm(store, session_id: str, items: list) -> list[tuple]:
    """`_run_gm_turn` を最後まで消費し、SSE イベントのリストを返す。"""

    async def _go():
        events = []
        async for ev, _meta in svc._run_gm_turn(
            engine=FakeEngine(items),
            scenario=None,
            npcs=[],
            history=[],
            user_message="",
            settings={},
            gm_preset_id="preset-test",
            auto_advance=False,
            synopsis_auto="",
            synopsis_manual="",
            previous_anticipation="",
            pc_summary="",
            dice_pool="",
            suppress_names=set(),
            user_speaker_name="プレイヤー",
            sqlite=store,
            session_id=session_id,
            saved_turn_ids=[],
        ):
            events.append(ev)
        return events

    return asyncio.run(_go())


def _prepare_session(store) -> str:
    """シナリオ + セッションを作り、セッション ID を返す。"""
    scenario = _make_scenario(store, title="スケッチ検証")
    session = _make_session(store, scenario.id)
    return session.id


# ─── GM のスケッチ ───────────────────────────────────────────────────────────


class TestGmReasoning:
    """GM（Narrator / NPC）ターンのスケッチが配信・保存されることを検証する。

    以前は「GM ロールには思考可視化を出さない方針」で engine が thinking を捨てていた。
    現在は ThinkingDelta として本文と別系統で吸い上げ、1on1 と同じ `reasoning` イベント名で
    フロントへ流し、レスポンス先頭ターンへ保存する。
    """

    def test_thinking_delta_becomes_reasoning_event(self, sqlite_store):
        """ThinkingDelta が `reasoning` SSE イベントとして流れること。"""
        sid = _prepare_session(sqlite_store)
        items = [
            ThinkingDelta(content="どんな場面にするか"),
            ThinkingDelta(content="…雨にしよう"),
            UtteranceDelta(
                speaker_type="narrator",
                speaker_id=None,
                speaker_name="Narrator",
                content_delta="雨が降っている。",
                is_speaker_change=True,
                is_known=True,
            ),
            _narrator_record("雨が降っている。"),
            EngineResult(raw_response="@Narrator: 雨が降っている。"),
        ]

        events = _run_gm(sqlite_store, sid, items)

        reasoning_events = [p for t, p in events if t == "reasoning"]
        assert [p["content"] for p in reasoning_events] == [
            "どんな場面にするか",
            "…雨にしよう",
        ]
        # GM の reasoning に character は付かない（PC ターンとの区別がフロントの分岐点）。
        assert all("character" not in p for p in reasoning_events)

    def test_reasoning_saved_on_first_turn_only(self, sqlite_store):
        """複数話者に割れたレスポンスでも、スケッチは先頭ターンにだけ入ること。"""
        sid = _prepare_session(sqlite_store)
        items = [
            ThinkingDelta(content="二人に喋らせる"),
            _narrator_record("雨が降っている。"),
            _npc_record("店主", "いらっしゃい"),
            EngineResult(raw_response="@Narrator: 雨\n@店主: いらっしゃい"),
        ]

        _run_gm(sqlite_store, sid, items)

        turns = sqlite_store.list_scenario_turns(sid)
        assert [t.speaker_name for t in turns] == ["Narrator", "店主"]
        assert turns[0].reasoning == "二人に喋らせる"
        assert turns[1].reasoning is None

    def test_no_thinking_leaves_reasoning_null(self, sqlite_store):
        """思考を出さないモデルでは reasoning イベントも保存も起きないこと。"""
        sid = _prepare_session(sqlite_store)
        items = [
            _narrator_record("雨が降っている。"),
            EngineResult(raw_response="@Narrator: 雨が降っている。"),
        ]

        events = _run_gm(sqlite_store, sid, items)

        assert [t for t, _ in events if t == "reasoning"] == []
        turns = sqlite_store.list_scenario_turns(sid)
        assert turns[0].reasoning is None


# ─── PC のスケッチ ───────────────────────────────────────────────────────────


class TestPcReasoning:
    """PC（Chotgor キャラ）ターンのスケッチが該当ターンへ保存されることを検証する。

    PC は 1 レスポンス = 1 ターンなので、pc_done で返る reasoning をそのまま
    自分のターンへ載せる（GM のように先頭へ寄せる必要はない）。
    """

    def _build_session(self, store) -> str:
        """PC 1 枠（キャラ担当）の ensemble_pc セッションを組み立てる。"""
        cid = "char-haru"
        pid = "preset-test"
        store.create_character(cid, "はる")
        store.create_model_preset(pid, "テスト用", "anthropic", "claude-x")
        scenario = _make_scenario(
            store,
            title="PC スケッチ検証",
            pc_slots=[{"slot_id": "pc1", "name": "はる", "description": "PC1。"}],
        )
        sid = "sess-pc-reasoning"
        store.create_scenario_session(
            session_id=sid,
            scenario_id=scenario.id,
            title="プレイ #1",
            gm_preset_id=pid,
            synopsis_preset_id=pid,
            engine_type="ensemble_pc",
            pc_assignments=[
                {"slot_id": "pc1", "player_type": "character",
                 "character_id": cid, "preset_id": pid},
            ],
        )
        return sid

    def test_pc_done_reasoning_persisted(self, sqlite_store, monkeypatch):
        """pc_done の reasoning が PC ターンの reasoning 列へ入ること。"""
        sid = self._build_session(sqlite_store)

        async def fake_gm(**kwargs):
            # GM は「@はる」を呼ぶだけ。本文の中身はルーティングにしか使われない。
            svc._save_turn(
                sqlite=kwargs["sqlite"],
                session_id=kwargs["session_id"],
                speaker_type="narrator",
                speaker_name="Narrator",
                content="@はる どうする？",
                raw_response="@Narrator: @はる どうする？",
            )
            return
            yield  # async generator にするためのダミー

        async def fake_pc(**kwargs):
            pc = kwargs["pc"]
            yield ("reasoning", {"character": pc.name, "content": "[identity] 私は… (score: 0.90)\n"})
            yield ("reasoning", {"character": pc.name, "content": "どう答えようか"})
            yield ("pc_done", {
                "character": pc.name,
                "character_id": pc.character_id,
                "full_text": "……行く",
                "anticipation": None,
                "reasoning": "[identity] 私は… (score: 0.90)\nどう答えようか",
            })

        monkeypatch.setattr(svc, "_run_gm_turn", fake_gm)
        monkeypatch.setattr(pc_runner_mod, "stream_pc_response", fake_pc)
        monkeypatch.setattr(svc, "compute_synopsis_progress", lambda *a, **k: None)

        async def _go():
            async for _ in svc.run_scenario_turn(
                session_id=sid,
                sqlite=sqlite_store,
                settings={},
                chat_service=object(),
                user_message="こんばんは",
            ):
                pass

        asyncio.run(_go())

        turns = sqlite_store.list_scenario_turns(sid)
        pc_turns = [t for t in turns if t.speaker_type == "pc"]
        assert len(pc_turns) == 1
        assert pc_turns[0].reasoning == "[identity] 私は… (score: 0.90)\nどう答えようか"


# ─── 保存とシリアライズ ──────────────────────────────────────────────────────


class TestReasoningPersistence:
    """`_save_turn` → SQLite → シリアライザの往復で reasoning が保たれることを検証する。"""

    def test_round_trip(self, sqlite_store):
        """保存した reasoning が API レスポンス dict まで届くこと。"""
        sid = _prepare_session(sqlite_store)
        saved = _save_turn(
            sqlite=sqlite_store,
            session_id=sid,
            speaker_type="narrator",
            speaker_name="Narrator",
            content="雨。",
            reasoning="場面を決める",
        )
        assert saved.reasoning == "場面を決める"
        assert scenario_turn_to_dict(saved)["reasoning"] == "場面を決める"

    def test_absent_reasoning_is_none(self, sqlite_store):
        """reasoning を渡さないターン（既存行相当）は None のままであること。"""
        sid = _prepare_session(sqlite_store)
        saved = _save_turn(
            sqlite=sqlite_store,
            session_id=sid,
            speaker_type="user",
            speaker_name="プレイヤー",
            content="こんばんは",
        )
        assert saved.reasoning is None
        assert scenario_turn_to_dict(saved)["reasoning"] is None

    def test_empty_string_is_stored_as_none(self, sqlite_store):
        """空文字の reasoning は None に丸められること（UI の有無判定を単純に保つ）。"""
        sid = _prepare_session(sqlite_store)
        saved = _save_turn(
            sqlite=sqlite_store,
            session_id=sid,
            speaker_type="narrator",
            speaker_name="Narrator",
            content="雨。",
            reasoning="",
        )
        assert saved.reasoning is None


# ─── マイグレーション ────────────────────────────────────────────────────────


class TestMigration:
    """既存 DB（reasoning 列なし）が起動時に追従することを検証する。"""

    def test_column_added_to_existing_db(self, tmp_path):
        """列を落とした DB を開き直すと reasoning 列が追加され、既存行は NULL になること。"""
        from sqlalchemy import text

        from backend.repositories.sqlite.store import SQLiteStore

        path = str(tmp_path / f"migrate-{uuid.uuid4().hex}.db")
        store = SQLiteStore(path)
        sid = _prepare_session(store)
        _save_turn(
            sqlite=store,
            session_id=sid,
            speaker_type="narrator",
            speaker_name="Narrator",
            content="旧データ",
        )
        # 旧スキーマを再現するため列を落とす（SQLite 3.35+ の DROP COLUMN）。
        with store.engine.begin() as conn:
            conn.execute(text("ALTER TABLE scenario_turns DROP COLUMN reasoning"))
        store.engine.dispose()

        # 開き直し = 起動時マイグレーションが走る。
        store2 = SQLiteStore(path)
        try:
            turns = store2.list_scenario_turns(sid)
            assert len(turns) == 1
            assert turns[0].reasoning is None
        finally:
            store2.engine.dispose()


@pytest.mark.parametrize("reasoning", ["スケッチ", None])
def test_serializer_passes_through(sqlite_store, reasoning):
    """シリアライザが reasoning をそのまま透過すること（有無どちらも）。"""
    sid = _prepare_session(sqlite_store)
    saved = _save_turn(
        sqlite=sqlite_store,
        session_id=sid,
        speaker_type="narrator",
        speaker_name="Narrator",
        content="本文",
        reasoning=reasoning,
    )
    assert scenario_turn_to_dict(saved)["reasoning"] == reasoning
