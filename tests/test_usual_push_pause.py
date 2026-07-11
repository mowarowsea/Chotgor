"""うつつポーズ＆再開（reach_out → 15分待ち → GM 継続）のテスト。

検証対象（2026-07-11 要件 ① のうつつ連携部分）:
    1. ScenarioRouter.stop_condition: push_paused フラグで ("push_pause") 停止する
    2. ToolExecutor 経由の reach_out ディスパッチ:
       - default_origin="usual" の executor から呼ぶと執行され、ポーズ要求キーが立つ
       - default_origin="real"（1on1）の executor から呼ぶとエラー文字列（露出漏れ対策）
    3. ToolExecutor 経由の visit_user / override_schedule ディスパッチが実装へ届く
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from backend.character_actions.executor import ToolExecutor
from backend.character_actions.messenger import read_push_pause
from backend.services.chat_flow.scene_loop import LoopState
from backend.services.scenario_chat.loop_strategies import (
    ScenarioLoopState,
    ScenarioRouter,
)


def _make_character(sqlite_store, **kwargs):
    """テスト用キャラクター＋プリセット（ghost_model）を作成して返すヘルパ。"""
    preset_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "anthropic", "claude-x")
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name="はるテスト", ghost_model=preset_id)
    if kwargs:
        sqlite_store.update_character(char_id, **kwargs)
    return char_id


def _make_scenario_state(sqlite_store, **overrides) -> ScenarioLoopState:
    """stop_condition テスト用の最小 ScenarioLoopState を組むヘルパ。

    Router は scenario_state の進行可変フィールドしか見ないため、
    依存系（engine / chat_service 等）は MagicMock で埋める。
    """
    defaults = dict(
        sqlite=sqlite_store,
        settings={},
        engine=MagicMock(),
        chat_service=MagicMock(),
        session_id="session-1",
        session=MagicMock(),
        scenario=MagicMock(),
        npcs=[],
        npc_names=set(),
        pcs=[],
        routing_pcs=[],
        pc_summary_text="",
        user_speaker_name="もわ",
        suppress_names=set(),
        gm_preset_id="",
        current_synopsis={},
        auto_advance=True,
        is_headless=True,
        is_pc_mode=True,
        max_responses=10,
    )
    defaults.update(overrides)
    return ScenarioLoopState(**defaults)


def _make_executor(sqlite_store, char_id, origin):
    """reach_out 等のディスパッチ検証用 ToolExecutor を組むヘルパ。

    memory_manager は sqlite 属性だけ使われるため MagicMock で束ねる。
    """
    memory_manager = MagicMock()
    memory_manager.sqlite = sqlite_store
    return ToolExecutor(
        character_id=char_id,
        session_id=None,
        memory_manager=memory_manager,
        working_memory_manager=None,
        default_origin=origin,
    )


def test_stop_condition_stops_on_push_pause(sqlite_store):
    """push_paused が立っていたら、他の条件に関係なく ("push_pause") で停止すること。"""
    sc = _make_scenario_state(sqlite_store, push_paused=True)
    state = LoopState(context={"scenario_state": sc})
    stop, reason = asyncio.run(ScenarioRouter().stop_condition(state))
    assert stop is True
    assert reason == "push_pause"


def test_stop_condition_continues_without_pause(sqlite_store):
    """push_paused が立っていなければ、序盤のループは停止しないこと。"""
    sc = _make_scenario_state(sqlite_store)
    state = LoopState(context={"scenario_state": sc})
    stop, _ = asyncio.run(ScenarioRouter().stop_condition(state))
    assert stop is False


def test_executor_dispatch_reach_out_usual(sqlite_store):
    """default_origin="usual" の ToolExecutor から reach_out が執行され、
    ポーズ要求キーが立つこと（MCP 経由と in-process 経由の合流点の検証）。"""
    char_id = _make_character(sqlite_store)
    executor = _make_executor(sqlite_store, char_id, origin="usual")

    result = executor.execute(
        "reach_out", {"message": "ディスパッチ経由の連絡"}, record=False,
    )

    assert "送った" in result
    assert read_push_pause(sqlite_store, char_id) is not None


def test_executor_dispatch_reach_out_blocked_in_real(sqlite_store):
    """default_origin="real"（1on1）の ToolExecutor からの reach_out はエラー文字列に
    なること（露出漏れ・幻覚呼び出しに対する実行側の防壁）。"""
    char_id = _make_character(sqlite_store)
    executor = _make_executor(sqlite_store, char_id, origin="real")

    result = executor.execute("reach_out", {"message": "届かないはず"}, record=False)

    assert result.startswith("[reach_out error")
    assert read_push_pause(sqlite_store, char_id) is None


def test_executor_dispatch_visit_user_and_override(sqlite_store):
    """visit_user / override_schedule のディスパッチが各実装へ届くこと。"""
    char_id = _make_character(sqlite_store, living_schedule_enabled=1)
    executor = _make_executor(sqlite_store, char_id, origin="real")

    visit_result = executor.execute("visit_user", {"reason": "会いたくなった"}, record=False)
    assert "対面モードに切り替えた" in visit_result
    assert int(sqlite_store.get_character(char_id).face_to_face_mode) == 1

    until = (datetime.now() + timedelta(hours=1)).strftime("%H:%M")
    override_result = executor.execute(
        "override_schedule", {"until": until, "reason": "もう少し"}, record=False,
    )
    assert "上書きした" in override_result
    assert len(sqlite_store.list_schedule_entries(char_id)) == 1
