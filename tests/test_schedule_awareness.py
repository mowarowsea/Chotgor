"""予定コンテキスト（schedule awareness・要件③）のテスト。

検証対象:
    1. 生活カレンダー無効キャラは空リスト（ブロック自体が出ない）
    2. いまの予定: 占有圧最大の planned エントリが「開始からの経過」付きで出る
    3. 本人上書き（④ self_override）中は「意志で上書き中」＋「本来の予定をX分超過中」
       が出る（夜更かしの自覚材料）
    4. 次の予定: 固定予定（template）だけが出る。world/adhoc（伏せ枠・発火済み突発）は
       未来の予定として**絶対に出ない**（ランダムイベントのネタバレ防止・要件の絶対条件）
    5. 素材が無ければ空リスト（ヘッダだけのブロックを作らない）
    6. 発話予約（speak_later）の相乗り: pending の間だけ「本人の段取り」行が載り、
       fired/superseded 等へ遷移したら消える。生活カレンダー無効キャラにも載る
       （自分の段取りはどの場でも本人の頭の中にあるもの）
"""

import uuid
from datetime import datetime, timedelta

from backend.character_actions.rescheduler import OVERRIDE_PAYLOAD_KIND
from backend.services.schedule.awareness import build_schedule_lines

# 固定基準時刻（金曜の夜）
_NOW = datetime(2026, 7, 10, 23, 0)


def _make_character(sqlite_store, living=1):
    """テスト用キャラクター（生活カレンダー有効）を作成して ID を返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name="はるテスト")
    sqlite_store.update_character(char_id, living_schedule_enabled=living)
    return char_id


def test_disabled_returns_empty(sqlite_store):
    """生活カレンダー無効キャラは空リスト（③のブロックが出ない）。"""
    char_id = _make_character(sqlite_store, living=0)
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW) == []


def test_no_entries_returns_empty(sqlite_store):
    """エントリ皆無なら空リスト（ヘッダだけのブロックを作らない）。"""
    char_id = _make_character(sqlite_store)
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW) == []


def test_current_entry_with_elapsed(sqlite_store):
    """いまの予定が「開始からの経過」「終わる時刻」付きで出ること。"""
    char_id = _make_character(sqlite_store)
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW - timedelta(minutes=40),
        end_at=_NOW + timedelta(minutes=80),
        state="active",
        source="haru",
        origin="template",
        occupancy=0.75,
        label="調べもの",
    )
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW)
    assert len(lines) == 1
    assert "いまの予定: 調べもの" in lines[0]
    assert "40分経過" in lines[0]


def test_next_entry_with_remaining(sqlite_store):
    """次の固定予定が「あと何分」付きで出ること（2時間後就寝→あと2時間）。"""
    char_id = _make_character(sqlite_store)
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW + timedelta(hours=2),
        end_at=_NOW + timedelta(hours=8),
        state="offline",
        source="haru",
        origin="template",
        occupancy=0.5,
        label="就寝",
    )
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW)
    assert len(lines) == 1
    assert "次の予定: 就寝" in lines[0]
    assert "あと2時間" in lines[0]


def test_override_shows_overrun_of_beaten_plan(sqlite_store):
    """本人上書き（④）中は「意志で上書き中」と「本来の予定をX分超過中」が出ること。"""
    char_id = _make_character(sqlite_store)
    # 本来の就寝（30分前に始まっているが、上書きに轢かれている）
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW - timedelta(minutes=30),
        end_at=_NOW + timedelta(hours=6),
        state="offline",
        source="haru",
        origin="template",
        occupancy=0.5,
        label="就寝",
    )
    # 本人の意志上書き（override_schedule 相当）
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW - timedelta(minutes=30),
        end_at=_NOW + timedelta(hours=1),
        state="OnTime",
        source="haru",
        origin="adhoc",
        occupancy=0.85,
        label="話の続きが気になる",
        payload={"kind": OVERRIDE_PAYLOAD_KIND, "reason": "話の続きが気になる"},
    )
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW)
    joined = "\n".join(lines)
    assert "自分の意志で" in joined and "上書きしている" in joined
    assert "本来の予定「就寝」" in joined
    assert "30分超過中" in joined


def test_hidden_and_world_events_never_shown_as_next(sqlite_store):
    """③伏せ枠（pending）も発火済み world 突発（planned/adhoc）も「次の予定」に
    出ないこと — ランダムイベントのネタバレ防止（要件の絶対条件）。"""
    char_id = _make_character(sqlite_store)
    # 伏せ枠（未発火・pending）
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW + timedelta(hours=1),
        end_at=_NOW + timedelta(hours=2),
        state="busy",
        source="world",
        origin="adhoc",
        occupancy=0.0,
        status="pending",
        label="友人からの誘い",
        payload={"kind": "sudden_event_seed", "category": "友人からの誘い"},
    )
    # 発火済みの world 突発（planned だが未来開始）
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW + timedelta(hours=3),
        end_at=_NOW + timedelta(hours=4),
        state="busy",
        source="world",
        origin="adhoc",
        occupancy=0.9,
        status="planned",
        label="突発の呼び出し",
    )
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW)
    joined = "\n".join(lines)
    assert "友人からの誘い" not in joined
    assert "突発の呼び出し" not in joined
    # 固定予定を足せばそれだけが「次の予定」に出る
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW + timedelta(hours=5),
        end_at=_NOW + timedelta(hours=11),
        state="offline",
        source="haru",
        origin="template",
        occupancy=0.5,
        label="就寝",
    )
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW)
    joined = "\n".join(lines)
    assert "次の予定: 就寝" in joined
    assert "友人からの誘い" not in joined and "突発の呼び出し" not in joined


def test_reservation_line_shown_even_without_calendar(sqlite_store):
    """発話予約の行は生活カレンダー無効キャラにも載ること（予約はセッションに
    紐づくが、注入はセッションを問わない — 本人の段取りとして常に想起される）。"""
    char_id = _make_character(sqlite_store, living=0)
    sqlite_store.create_speech_reservation(
        character_id=char_id, session_id="session-1",
        speak_at=_NOW + timedelta(minutes=30), note="調べものの結果を伝える",
    )
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW)
    assert lines == ["23:30 になったら「調べものの結果を伝える」をしようと思っている"]


def test_reservation_line_disappears_after_transition(sqlite_store):
    """fired / superseded へ遷移した予約の行は消えること（pending の間だけ載る）。"""
    char_id = _make_character(sqlite_store, living=0)
    r1 = sqlite_store.create_speech_reservation(
        character_id=char_id, session_id="s1",
        speak_at=_NOW + timedelta(minutes=30), note="発火済みの用件",
    )
    r2 = sqlite_store.create_speech_reservation(
        character_id=char_id, session_id="s2",
        speak_at=_NOW + timedelta(hours=1), note="置き直し前の用件",
    )
    sqlite_store.set_speech_reservation_status(r1.id, "fired", fired_at=_NOW)
    sqlite_store.set_speech_reservation_status(r2.id, "superseded")
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW) == []


def test_reservation_rides_with_calendar_lines(sqlite_store):
    """生活カレンダー有効キャラでは、予定の行に続けて予約の行が同じ枠に載ること。"""
    char_id = _make_character(sqlite_store)
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW + timedelta(minutes=60),
        end_at=_NOW + timedelta(hours=7),
        state="offline",
        source="haru",
        origin="template",
        occupancy=0.5,
        label="就寝",
    )
    # 日付を跨ぐ予約は M/D 付きの表記になる
    sqlite_store.create_speech_reservation(
        character_id=char_id, session_id="s1",
        speak_at=_NOW + timedelta(hours=11), note="朝いちで声をかける",
    )
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW)
    assert any("次の予定: 就寝" in line for line in lines)
    assert lines[-1] == "7/11 10:00 になったら「朝いちで声をかける」をしようと思っている"


def test_current_world_event_is_shown(sqlite_store):
    """現在進行中の world 突発は「いまの予定」として出てよい（本人が体験している
    現実であり、隠す方が不整合になる）。"""
    char_id = _make_character(sqlite_store)
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=_NOW - timedelta(minutes=10),
        end_at=_NOW + timedelta(minutes=50),
        state="busy",
        source="world",
        origin="adhoc",
        occupancy=0.9,
        status="planned",
        label="急な来客",
    )
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW)
    assert any("いまの予定: 急な来客" in line for line in lines)
