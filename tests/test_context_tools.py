"""コンテキスト別ツール出し分け（context_tools）のテスト。

検証対象（2026-07-11 要件 ①②④ の露出マトリクス）:
    - reach_out         : うつつ（origin="usual"）専用。日次上限到達日は露出しない。
    - visit_user        : 1on1（origin="real" ＋ session_id）専用。対面中は露出しない。
    - override_schedule : 1on1 専用かつ生活カレンダー有効のみ。
    - バッチ（origin="real"・session なし）・シナリオ幕間（interlude）では追加ツールなし。
    - ヒント（resolve_context_tool_hints）はツール露出と同じ判定で並ぶ。
    - sqlite=None / キャラ不在では安全側（空リスト）に倒れる。
"""

import uuid
from datetime import datetime

from backend.character_actions.context_tools import (
    resolve_context_tool_hints,
    resolve_context_tool_names,
    resolve_context_tools,
)


def _make_character(sqlite_store, **kwargs):
    """テスト用キャラクターを作成して ID を返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name="はるテスト")
    if kwargs:
        sqlite_store.update_character(char_id, **kwargs)
    return char_id


def test_usual_exposes_reach_out_only(sqlite_store):
    """うつつ経路では reach_out だけが露出すること（1on1 専用ツールは出ない）。"""
    char_id = _make_character(sqlite_store, living_schedule_enabled=1)
    names = resolve_context_tool_names(sqlite_store, char_id, origin="usual")
    assert names == ["reach_out"]


def test_usual_hides_reach_out_when_cap_reached(sqlite_store):
    """日次上限到達日は reach_out の露出自体が消えること（プロンプトからも消える）。"""
    char_id = _make_character(sqlite_store)
    today = datetime.now().date().isoformat()
    sqlite_store.set_setting("escrow_delivery_daily_cap", "1")
    sqlite_store.set_setting(f"escrow_delivery_count_{today}", "1")
    assert resolve_context_tool_names(sqlite_store, char_id, origin="usual") == []


def test_oneonone_exposes_visit_and_override(sqlite_store):
    """1on1（real＋session）では visit_user と override_schedule が露出すること。"""
    char_id = _make_character(sqlite_store, living_schedule_enabled=1)
    names = resolve_context_tool_names(
        sqlite_store, char_id, origin="real", session_id="session-1",
    )
    assert names == ["visit_user", "override_schedule"]


def test_oneonone_hides_visit_when_face_to_face(sqlite_store):
    """対面中の 1on1 では visit_user が露出しないこと（切り替える意味がない）。"""
    char_id = _make_character(
        sqlite_store, living_schedule_enabled=1, face_to_face_mode=1,
    )
    names = resolve_context_tool_names(
        sqlite_store, char_id, origin="real", session_id="session-1",
    )
    assert names == ["override_schedule"]


def test_oneonone_hides_override_without_living_calendar(sqlite_store):
    """生活カレンダー無効キャラでは override_schedule が露出しないこと。"""
    char_id = _make_character(sqlite_store, living_schedule_enabled=0)
    names = resolve_context_tool_names(
        sqlite_store, char_id, origin="real", session_id="session-1",
    )
    assert names == ["visit_user"]


def test_batch_and_interlude_expose_nothing(sqlite_store):
    """バッチ（session なし）とシナリオ幕間（interlude）では追加ツールが無いこと。"""
    char_id = _make_character(sqlite_store, living_schedule_enabled=1)
    assert resolve_context_tool_names(sqlite_store, char_id, origin="real") == []
    assert resolve_context_tool_names(
        sqlite_store, char_id, origin="interlude", session_id="scenario-1",
    ) == []


def test_tools_and_hints_follow_same_gate(sqlite_store):
    """ツール定義とヒントが同じ判定で同数並ぶこと（露出と説明のズレを作らない）。"""
    char_id = _make_character(sqlite_store, living_schedule_enabled=1)
    tools = resolve_context_tools(
        sqlite_store, char_id, origin="real", session_id="s1",
    )
    hints = resolve_context_tool_hints(
        sqlite_store, char_id, origin="real", session_id="s1",
    )
    assert [t["name"] for t in tools] == ["visit_user", "override_schedule"]
    assert len(hints) == len(tools)
    # ツール定義は Anthropic 形式（name / description / input_schema）
    for t in tools:
        assert set(t) == {"name", "description", "input_schema"}


def test_safe_fallbacks(sqlite_store):
    """sqlite なし・キャラ不在では安全側（空リスト）に倒れること。"""
    assert resolve_context_tool_names(None, "someone", origin="usual") == []
    assert resolve_context_tool_names(sqlite_store, "存在しないID", origin="usual") == []
    assert resolve_context_tool_names(sqlite_store, "", origin="usual") == []
