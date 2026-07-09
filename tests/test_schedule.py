"""生活カレンダー（Living Schedule）Phase 0/1 のテスト。

検証対象（docs/planned/schedule_plan.md §2/§4/§5/§7.5）:
    1. ScheduleStoreMixin: 実現層エントリの CRUD（作成・期間フィルタ読み出し・
       now を含む planned の抽出・status 更新・削除の絞り込み）。
    2. 配達値プリセットと個別上書きの解決（_resolve_delivery_values）。
    3. availability 生活カレンダー経路: 占有圧最大が勝つ・エントリなし=OnTime・
       優先順位（対面 > away > エントリ > なし）・うつつ進行中の無視（§7.5）。

いずれも純関数／決定論であり LLM を呼ばない。
"""

import uuid
from datetime import datetime, timedelta

from backend.services.gate.availability import (
    Availability,
    _resolve_delivery_values,
    check_availability,
)


def _make_character(sqlite_store, name="はるスケジュール", **kwargs):
    """テスト用キャラクターを1体作り、追加フィールドを更新して char を返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    if kwargs:
        sqlite_store.update_character(char_id, **kwargs)
    return sqlite_store.get_character(char_id)


# 月曜（2026-07-06）の各時刻を使う。生活カレンダーは絶対時刻ベース。
_MON = datetime(2026, 7, 6)


class TestScheduleStore:
    """schedule_entries CRUD（ScheduleStoreMixin）の追加・読み出し・削除を検証する。"""

    def test_create_and_get_active(self, sqlite_store):
        """作成したエントリが now を含む時だけ get_active_schedule_entries に出る。"""
        char = _make_character(sqlite_store)
        sqlite_store.create_schedule_entry(
            character_id=char.id,
            start_at=_MON.replace(hour=9),
            end_at=_MON.replace(hour=17, minute=30),
            state="active", occupancy=0.75, label="仕事",
        )
        # 時間帯の内側は取れる
        active = sqlite_store.get_active_schedule_entries(char.id, _MON.replace(hour=12))
        assert len(active) == 1 and active[0].label == "仕事"
        # 時間帯の外は取れない
        assert sqlite_store.get_active_schedule_entries(char.id, _MON.replace(hour=20)) == []

    def test_get_active_only_planned(self, sqlite_store):
        """cancelled/done のエントリは get_active に出ない（planned のみ）。"""
        char = _make_character(sqlite_store)
        e = sqlite_store.create_schedule_entry(
            character_id=char.id,
            start_at=_MON.replace(hour=9), end_at=_MON.replace(hour=12),
            state="busy", occupancy=1.0,
        )
        assert len(sqlite_store.get_active_schedule_entries(char.id, _MON.replace(hour=10))) == 1
        sqlite_store.set_schedule_entry_status(e.id, "cancelled")
        assert sqlite_store.get_active_schedule_entries(char.id, _MON.replace(hour=10)) == []

    def test_get_active_sorted_by_occupancy_desc(self, sqlite_store):
        """重なるエントリは占有圧の降順で返る（先頭が勝者候補）。"""
        char = _make_character(sqlite_store)
        span = dict(start_at=_MON.replace(hour=9), end_at=_MON.replace(hour=12))
        sqlite_store.create_schedule_entry(character_id=char.id, occupancy=0.5, state="active", **span)
        sqlite_store.create_schedule_entry(character_id=char.id, occupancy=1.0, state="busy", **span)
        got = sqlite_store.get_active_schedule_entries(char.id, _MON.replace(hour=10))
        assert [e.occupancy for e in got] == [1.0, 0.5]

    def test_list_overlap_filter(self, sqlite_store):
        """list_schedule_entries は期間と重なるエントリを取る（跨ぎを取り逃さない）。"""
        char = _make_character(sqlite_store)
        # 就寝: 前日 25:00(=翌1:00) 相当を想定し、月 01:00-07:00 に置く
        sqlite_store.create_schedule_entry(
            character_id=char.id,
            start_at=_MON.replace(hour=1), end_at=_MON.replace(hour=7),
            state="offline", occupancy=0.5, label="就寝",
        )
        # 問い合わせ窓 [06:00, 09:00) は就寝終端(07:00)と重なる → 取れる
        hits = sqlite_store.list_schedule_entries(
            char.id, since=_MON.replace(hour=6), until=_MON.replace(hour=9),
        )
        assert len(hits) == 1 and hits[0].label == "就寝"
        # 窓 [08:00, 09:00) は重ならない → 取れない
        assert sqlite_store.list_schedule_entries(
            char.id, since=_MON.replace(hour=8), until=_MON.replace(hour=9),
        ) == []

    def test_delete_by_period_and_origin(self, sqlite_store):
        """削除は (character_id, 期間, origin) で絞れる（週次再生成の掃除用）。"""
        char = _make_character(sqlite_store)
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=9),
            end_at=_MON.replace(hour=10), origin="template", occupancy=0.5,
        )
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=11),
            end_at=_MON.replace(hour=12), origin="adhoc", occupancy=0.5,
        )
        deleted = sqlite_store.delete_schedule_entries(
            character_id=char.id, origins=["template"],
        )
        assert deleted == 1
        remaining = sqlite_store.list_schedule_entries(char.id)
        assert len(remaining) == 1 and remaining[0].origin == "adhoc"

    def test_delete_guard_no_scope(self, sqlite_store):
        """character_id も entry_ids も無い削除は何もしない（全消し事故防止）。"""
        char = _make_character(sqlite_store)
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=9),
            end_at=_MON.replace(hour=10), occupancy=0.5,
        )
        assert sqlite_store.delete_schedule_entries() == 0
        assert len(sqlite_store.list_schedule_entries(char.id)) == 1


class TestDeliveryValues:
    """配達値プリセットと個別上書きの解決（_resolve_delivery_values）を検証する。"""

    def test_preset_defaults(self):
        """各 state のプリセット既定が返る（OnTime=即時 / offline=∞）。"""
        assert _resolve_delivery_values("OnTime") == (1.0, 0)
        assert _resolve_delivery_values("active") == (0.9, 5)
        assert _resolve_delivery_values("busy") == (0.4, 60)
        assert _resolve_delivery_values("offline") == (0.0, None)

    def test_unknown_state_falls_back_to_active(self):
        """未知 state はプリセット既定 active に寄せる（生成側タイポ耐性）。"""
        assert _resolve_delivery_values("garbage") == (0.9, 5)

    def test_entry_override(self):
        """エントリ個別の reply_rate / check_interval がプリセットを上書きする。"""
        class _E:
            reply_rate = 0.2
            check_interval = 90
        assert _resolve_delivery_values("active", _E()) == (0.2, 90)

    def test_entry_partial_override_keeps_preset(self):
        """個別値が None のフィールドはプリセット既定を保つ（部分上書き）。"""
        class _E:
            reply_rate = None
            check_interval = 30
        assert _resolve_delivery_values("active", _E()) == (0.9, 30)


class TestScheduledAvailability:
    """生活カレンダー経路の check_availability（§2/§7.5）を検証する。

    living_schedule_enabled=1 かつ sqlite 注入時に、実現層エントリの占有圧最大を引く。
    """

    def _enabled_char(self, sqlite_store, **kwargs):
        return _make_character(sqlite_store, living_schedule_enabled=1, **kwargs)

    def test_no_entries_is_ontime(self, sqlite_store):
        """有効キャラでもエントリが無ければ OnTime（完全リアルタイム）。"""
        char = self._enabled_char(sqlite_store)
        a = check_availability(char, _MON.replace(hour=12), sqlite=sqlite_store)
        assert a.state == "OnTime" and a.available is True
        assert a.reply_rate == 1.0 and a.check_interval == 0

    def test_active_entry_delivery_values(self, sqlite_store):
        """active エントリ内は available だが配達値はプリセット（0.9 / 5分）。"""
        char = self._enabled_char(sqlite_store)
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=9),
            end_at=_MON.replace(hour=17), state="active", occupancy=0.75, label="仕事",
        )
        a = check_availability(char, _MON.replace(hour=12), sqlite=sqlite_store)
        assert a.state == "active" and a.available is True
        assert a.occupancy == 0.75 and a.reason == "仕事"
        assert a.reply_rate == 0.9 and a.check_interval == 5

    def test_offline_entry_is_unavailable(self, sqlite_store):
        """offline エントリ内は unavailable（available=False・∞）。"""
        char = self._enabled_char(sqlite_store)
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=1),
            end_at=_MON.replace(hour=7), state="offline", occupancy=0.5, label="就寝",
        )
        a = check_availability(char, _MON.replace(hour=3), sqlite=sqlite_store)
        assert a.state == "offline" and a.available is False
        assert a.reply_rate == 0.0 and a.check_interval is None

    def test_occupancy_max_wins(self, sqlite_store):
        """重なるエントリは占有圧最大が勝つ（激強 busy が 弱 active を制す）。"""
        char = self._enabled_char(sqlite_store)
        span = dict(start_at=_MON.replace(hour=9), end_at=_MON.replace(hour=12))
        sqlite_store.create_schedule_entry(
            character_id=char.id, state="active", occupancy=0.25, label="だらだら", **span,
        )
        sqlite_store.create_schedule_entry(
            character_id=char.id, state="busy", occupancy=1.0, label="突発電話", **span,
        )
        a = check_availability(char, _MON.replace(hour=10), sqlite=sqlite_store)
        assert a.state == "busy" and a.reason == "突発電話" and a.occupancy == 1.0

    def test_face_to_face_forces_ontime(self, sqlite_store):
        """対面中は offline エントリがあっても OnTime 強制（§7 (a)）。"""
        char = self._enabled_char(sqlite_store, face_to_face_mode=1)
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=1),
            end_at=_MON.replace(hour=23), state="offline", occupancy=1.0,
        )
        a = check_availability(char, _MON.replace(hour=12), sqlite=sqlite_store)
        assert a.state == "OnTime" and a.available is True

    def test_away_overrides_entry(self, sqlite_store):
        """away_until が未来なら offline 相当（エントリより優先・§7.5）。"""
        char = self._enabled_char(
            sqlite_store,
            away_until=_MON.replace(hour=13), away_reason="疲労のため休息中",
        )
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=9),
            end_at=_MON.replace(hour=17), state="active", occupancy=0.5,
        )
        a = check_availability(char, _MON.replace(hour=12), sqlite=sqlite_store)
        assert a.state == "offline" and a.available is False
        assert a.reason == "疲労のため休息中"

    def test_usual_scene_running_ignored_in_scheduled_path(self, sqlite_store):
        """生活カレンダー経路では「うつつ進行中=unavailable」は無視される（§7.5で削除）。"""
        char = self._enabled_char(sqlite_store)
        a = check_availability(
            char, _MON.replace(hour=12), usual_scene_running=True, sqlite=sqlite_store,
        )
        assert a.state == "OnTime" and a.available is True

    def test_enabled_but_no_sqlite_uses_legacy(self, sqlite_store):
        """有効キャラでも sqlite 未注入なら従来経路（うつつ進行中で unavailable）。"""
        char = self._enabled_char(sqlite_store)
        a = check_availability(char, _MON.replace(hour=12), usual_scene_running=True)
        assert a.available is False and a.reason == "usual_scene"


class TestLegacyAvailabilityFields:
    """従来経路でも Availability の連続量フィールドが埋まることを検証する（二値互換）。"""

    class _FakeChar:
        """living_schedule_enabled を持たない従来キャラのスタブ。"""

        def __init__(self, schedule=None):
            self.face_to_face_mode = 0
            self.away_until = None
            self.away_reason = None
            self.availability_schedule = schedule

    def test_available_maps_to_ontime(self):
        """従来経路の available は OnTime（reply 1.0 / interval 0）に写る。"""
        a = check_availability(self._FakeChar(), _MON.replace(hour=12))
        assert a.available is True and a.state == "OnTime"
        assert a.reply_rate == 1.0 and a.check_interval == 0

    def test_unavailable_maps_to_offline(self):
        """従来経路の unavailable（時間割ヒット）は offline に写る（reason 保持）。"""
        schedule = {"mon": [{"from": "09:00", "to": "18:00", "label": "仕事"}]}
        a = check_availability(self._FakeChar(schedule), _MON.replace(hour=12))
        assert a.available is False and a.state == "offline"
        assert a.reason == "仕事" and a.reply_rate == 0.0 and a.check_interval is None
