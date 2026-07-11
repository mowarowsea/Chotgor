"""Rescheduler（override_schedule）のテスト — 本人の意志による当日予定の一時上書き。

検証対象（2026-07-11 要件 ④）:
    1. parse_until_time: "HH:MM" の次回到来解決
       - 24時超え表記（"25:30" = 翌1:30）が正しく解決される
       - 過去時刻は翌日へ送られる（常に now < result <= now+24h）
       - 不正な形式は None
    2. override_schedule:
       - state=OnTime / source=haru / origin=adhoc / payload.kind=self_override の
         エントリが insert される（既存エントリは物理的に触らない）
       - 上書きで轢かれる予定（占有圧が上書き未満の planned）のラベルが結果文に載る
       - 生活カレンダー無効キャラはエラー文字列（実行側ガード）
       - until 形式不正はエラー文字列
    3. 上書き後の availability が実際に OnTime へ変わる（占有圧最大が勝つ読み取り解決）
"""

import uuid
from datetime import datetime, timedelta

from backend.character_actions.rescheduler import (
    OVERRIDE_PAYLOAD_KIND,
    Rescheduler,
    parse_until_time,
)
from backend.services.gate.availability import check_availability


def _make_character(sqlite_store, name="はるテスト", living=1):
    """テスト用キャラクター（生活カレンダー有効）を作成して返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    sqlite_store.update_character(char_id, living_schedule_enabled=living)
    return char_id


def test_parse_until_time_over_24h_notation():
    """24時超え表記（25:30）が翌日の 1:30 に解決されること。"""
    now = datetime(2026, 7, 11, 23, 30)
    assert parse_until_time("25:30", now) == datetime(2026, 7, 12, 1, 30)


def test_parse_until_time_past_rolls_to_next_occurrence():
    """now を過ぎた時刻は「次に来る同時刻」へ送られること（深夜0時台の "01:30" 等）。"""
    now = datetime(2026, 7, 11, 23, 30)
    # 23:30 時点の "01:30" は今夜（翌日日付）の 1:30
    assert parse_until_time("01:30", now) == datetime(2026, 7, 12, 1, 30)
    # 昼の "10:00" は翌日の 10:00（今日の10:00は過ぎている）
    noon = datetime(2026, 7, 11, 12, 0)
    assert parse_until_time("10:00", noon) == datetime(2026, 7, 12, 10, 0)


def test_parse_until_time_always_within_24h():
    """解決結果が常に now < result <= now+24h に収まること（当日限りの構造的保証）。"""
    for now in (datetime(2026, 7, 11, 0, 5), datetime(2026, 7, 11, 23, 55)):
        for raw in ("00:00", "06:00", "12:00", "23:59", "24:00", "25:30", "30:00"):
            result = parse_until_time(raw, now)
            assert result is not None
            assert now < result <= now + timedelta(hours=24), (now, raw, result)


def test_parse_until_time_invalid():
    """不正な形式（数値でない・範囲外）は None を返すこと。"""
    now = datetime(2026, 7, 11, 12, 0)
    assert parse_until_time("そのうち", now) is None
    assert parse_until_time("48:00", now) is None
    assert parse_until_time("12:99", now) is None
    assert parse_until_time("", now) is None


def test_override_schedule_inserts_entry_and_reports_beaten(sqlite_store):
    """override_schedule が本人上書きエントリを insert し、轢かれる予定
    （就寝 offline）のラベルが結果文に載ること。就寝エントリ自体は無傷。"""
    char_id = _make_character(sqlite_store)
    now = datetime.now()
    # 現在進行中の就寝（offline・中0.5）を仕込む
    sleep = sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=now - timedelta(minutes=30),
        end_at=now + timedelta(hours=6),
        state="offline",
        source="haru",
        origin="template",
        occupancy=0.5,
        label="就寝",
    )

    until = (now + timedelta(hours=2)).strftime("%H:%M")
    result = Rescheduler(char_id, sqlite_store).override_schedule(until, "話の続きが気になる")

    assert "上書きした" in result
    assert "就寝" in result
    entries = sqlite_store.list_schedule_entries(char_id)
    override = next(
        e for e in entries
        if (getattr(e, "payload", None) or {}).get("kind") == OVERRIDE_PAYLOAD_KIND
    )
    assert override.state == "OnTime"
    assert override.source == "haru"
    assert override.origin == "adhoc"
    assert override.occupancy > 0.5
    assert override.label == "話の続きが気になる"
    # 既存の就寝エントリは物理的に無傷（status も planned のまま）
    fresh_sleep = next(e for e in entries if e.id == sleep.id)
    assert fresh_sleep.status == "planned"


def test_override_schedule_changes_availability(sqlite_store):
    """就寝（offline）中でも、上書き後の availability が OnTime になること
    （占有圧最大が勝つ読み取り解決が効いている）。"""
    char_id = _make_character(sqlite_store)
    now = datetime.now()
    sqlite_store.create_schedule_entry(
        character_id=char_id,
        start_at=now - timedelta(minutes=10),
        end_at=now + timedelta(hours=6),
        state="offline",
        source="haru",
        origin="template",
        occupancy=0.5,
        label="就寝",
    )
    char = sqlite_store.get_character(char_id)
    before = check_availability(char, sqlite=sqlite_store)
    assert before.state == "offline"

    until = (now + timedelta(hours=1)).strftime("%H:%M")
    Rescheduler(char_id, sqlite_store).override_schedule(until, "まだ寝ない")

    after = check_availability(char, sqlite=sqlite_store)
    assert after.state == "OnTime"
    assert after.available is True


def test_override_schedule_requires_living_calendar(sqlite_store):
    """生活カレンダー無効キャラの override_schedule はエラー文字列を返すこと。"""
    char_id = _make_character(sqlite_store, living=0)
    result = Rescheduler(char_id, sqlite_store).override_schedule("25:00", "理由")
    assert result.startswith("[override_schedule error")
    assert sqlite_store.list_schedule_entries(char_id) == []


def test_override_schedule_invalid_until(sqlite_store):
    """until 形式不正はエラー文字列を返し、何も insert されないこと。"""
    char_id = _make_character(sqlite_store)
    result = Rescheduler(char_id, sqlite_store).override_schedule("そのうち", "")
    assert result.startswith("[override_schedule error")
    assert sqlite_store.list_schedule_entries(char_id) == []
