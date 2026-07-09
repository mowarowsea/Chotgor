"""生活カレンダー（Living Schedule）Phase 0〜3 のテスト。

検証対象（docs/planned/schedule_plan.md §2〜§5/§7.5/§14）:
    1. ScheduleStoreMixin: 実現層エントリの CRUD（作成・期間フィルタ読み出し・
       now を含む planned の抽出・status 更新・削除の絞り込み）。
    2. 配達値プリセットと個別上書きの解決（_resolve_delivery_values）。
    3. availability 生活カレンダー経路: 占有圧最大が勝つ・エントリなし=OnTime・
       優先順位（対面 > away > エントリ > なし）・うつつ進行中の無視（§7.5）。
    4. 配達の一般化（Phase 2）: チェック間隔格子＋決定論 reply_rate 判定
       （resolve_delivery_due）— offline は配達なし・OnTime は即時・
       チェック点到来まで待つ・決定論（同じ入力なら同じ結果）。
    5. [PLAN] パーサ（Phase 3）: 行抽出・24時超え表記・日跨ぎ・曜日/状態/圧の
       正規化・不正行スキップ・テンプレ層の裸変換・整形の往復一致。
    6. 週次バッチ（Phase 3）: 層フォールバック（GM/本人 → 前週 → テンプレ裸/なし）・
       template 入れ替えと adhoc 温存・冪等キー（コールドスタート・日曜夜の翌週分）。

LLM 呼び出しはすべてモック（monkeypatch）し、純関数／決定論の性質を検証する。
"""

import uuid
from datetime import date, datetime, timedelta
from types import SimpleNamespace

from backend.services.gate.availability import (
    Availability,
    _resolve_delivery_values,
    check_availability,
)
from backend.services.gate.delivery import resolve_delivery_due
from backend.services.schedule import (
    entries_from_template,
    format_plan_lines,
    layer_has_offline,
    parse_plan_lines,
    week_key,
    week_start_of,
)
from backend.services.schedule import weekly_batch as weekly_batch_module
from backend.services.schedule.weekly_batch import (
    _generate_haru_layer,
    _generate_world_layer,
    run_pending_weekly_batches,
    run_weekly_schedule_batch,
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


class TestResolveDeliveryDue:
    """配達一般化（Phase 2）の純関数 resolve_delivery_due を検証する。

    チェック間隔格子（0時起点・check_interval 分刻み）のうち、最古の預かり到着以降の
    チェック点を決定論乱数（seed = session＋チェック点時刻）で reply_rate 判定し、
    now までに成功点があれば配達（True）。状態を持たず、同じ入力なら常に同じ結果。
    """

    _SID = "session-due-test"

    def test_offline_never_due(self):
        """check_interval=None（offline・∞）は何時間経っても配達しない。"""
        arrived = _MON.replace(hour=1)
        assert resolve_delivery_due(
            self._SID, arrived, 0.0, None, arrived + timedelta(days=2)
        ) is False

    def test_ontime_immediately_due(self):
        """check_interval=0（OnTime）は到着済みなら即配達。"""
        arrived = _MON.replace(hour=12, minute=3)
        assert resolve_delivery_due(self._SID, arrived, 1.0, 0, arrived) is True

    def test_waits_for_first_checkpoint(self):
        """reply_rate=1.0 でも、到着後の最初のチェック点が来るまでは配達しない。

        60分格子（0時起点）で 12:03 到着 → 最初のチェック点は 13:00。
        12:30 時点では False、13:00 以降は True（率 1.0 なので必ず成功）。
        """
        arrived = _MON.replace(hour=12, minute=3)
        assert resolve_delivery_due(
            self._SID, arrived, 1.0, 60, _MON.replace(hour=12, minute=30)
        ) is False
        assert resolve_delivery_due(
            self._SID, arrived, 1.0, 60, _MON.replace(hour=13)
        ) is True

    def test_zero_reply_rate_never_due(self):
        """reply_rate=0.0 はチェック点が何度来ても配達しない（見たけど流す）。"""
        arrived = _MON.replace(hour=9)
        assert resolve_delivery_due(
            self._SID, arrived, 0.0, 60, arrived + timedelta(days=3)
        ) is False

    def test_deterministic_and_monotonic(self):
        """同じ入力は常に同じ結果。一度 due になったら以降の now でも due のまま。

        reply_rate=0.4 の確率判定でも乱数は世界（決定論 seed）に置かれるため、
        再評価でぶれない。成功チェック点は now が進んでも残るので単調。
        """
        arrived = _MON.replace(hour=8)
        results = [
            resolve_delivery_due(
                self._SID, arrived, 0.4, 60, arrived + timedelta(hours=hrs)
            )
            for hrs in range(1, 24)
        ]
        # 決定論: もう一度評価しても完全一致
        again = [
            resolve_delivery_due(
                self._SID, arrived, 0.4, 60, arrived + timedelta(hours=hrs)
            )
            for hrs in range(1, 24)
        ]
        assert results == again
        # 単調: True になった以降に False へ戻らない
        if True in results:
            first = results.index(True)
            assert all(results[first:])
        # 率 0.4 × 23 チェック点なら、実質必ずどこかで成功している
        assert True in results


# 対象週（2026-07-06 の月曜開始）。パーサ・週次バッチの基準に使う。
_WEEK = date(2026, 7, 6)


class TestPlanParser:
    """[PLAN] 行パーサ（Phase 3・schedule_plan.md §3 生成プロトコル）を検証する。

    行単位タグ方式（自由文と共存）・24時超え表記・from>to 日跨ぎ・表記ゆらぎの
    正規化・不正行スキップ（部分成功の安全側縮退）を網羅する。
    """

    def test_basic_line(self):
        """基本形: 曜日・時間帯・ラベル・状態・圧が絶対時刻エントリへ展開される。"""
        got = parse_plan_lines("[PLAN: 月 | 09:00-17:30 | 仕事 | active | 強]", _WEEK)
        assert len(got) == 1
        e = got[0]
        assert e.start_at == datetime(2026, 7, 6, 9, 0)
        assert e.end_at == datetime(2026, 7, 6, 17, 30)
        assert e.label == "仕事" and e.state == "active" and e.occupancy == 0.75

    def test_over_24h_notation(self):
        """24時超え表記（25:00-31:30）は翌日の時刻へ展開される（25:00 = 翌1:00）。"""
        got = parse_plan_lines("[PLAN: 月 | 25:00-31:30 | 就寝 | offline | 中]", _WEEK)
        assert len(got) == 1
        assert got[0].start_at == datetime(2026, 7, 7, 1, 0)
        assert got[0].end_at == datetime(2026, 7, 7, 7, 30)
        assert got[0].state == "offline" and got[0].occupancy == 0.5

    def test_from_greater_than_to_crosses_day(self):
        """from > to は日跨ぎとして終了を翌日に送る（既存 _schedule_block と同じ流儀）。"""
        got = parse_plan_lines("[PLAN: 火 | 23:00-06:00 | 夜更かし | active | 弱]", _WEEK)
        assert len(got) == 1
        assert got[0].start_at == datetime(2026, 7, 7, 23, 0)
        assert got[0].end_at == datetime(2026, 7, 8, 6, 0)

    def test_english_day_and_state_case(self):
        """曜日の英語表記（sun）と状態の大文字小文字ゆらぎ（ONTIME）を受理する。"""
        got = parse_plan_lines("[PLAN: sun | 14:00-18:00 | ゲーム | ONTIME | 激強]", _WEEK)
        assert len(got) == 1
        assert got[0].start_at == datetime(2026, 7, 12, 14, 0)
        assert got[0].state == "OnTime" and got[0].occupancy == 1.0

    def test_invalid_lines_skipped_valid_kept(self):
        """不正行（フィールド不足・未知曜日・時刻不正・未知圧）はスキップし有効行だけ採用。"""
        text = "\n".join([
            "今週の段取りはこんな感じ。",                          # 自由文（無視）
            "[PLAN: 月 | 09:00-17:30 | 仕事 | active | 強]",       # 有効
            "[PLAN: 月 | 09:00-17:30 | フィールド不足 | active]",   # 4フィールド
            "[PLAN: 曜 | 09:00-17:30 | 未知曜日 | active | 強]",    # 未知曜日
            "[PLAN: 火 | 9時-17時 | 時刻不正 | active | 強]",       # 時刻書式
            "[PLAN: 水 | 48:00-50:00 | 範囲外 | active | 強]",      # 時が48以上
            "[PLAN: 木 | 10:00-11:00 | 未知圧 | active | 超強]",    # 未知圧ラベル
            "以上。",
        ])
        got = parse_plan_lines(text, _WEEK)
        assert [e.label for e in got] == ["仕事"]

    def test_unknown_state_falls_back_to_active(self):
        """未知の状態表記は行ごと捨てず active に寄せる（タイポ耐性）。"""
        got = parse_plan_lines("[PLAN: 金 | 10:00-11:00 | 打合せ | ACTIV | 中]", _WEEK)
        assert len(got) == 1 and got[0].state == "active"

    def test_empty_text(self):
        """空文字列・None 相当は空リスト。"""
        assert parse_plan_lines("", _WEEK) == []

    def test_format_round_trip(self):
        """format_plan_lines の出力を再パースすると同じ時刻・値に戻る（往復一致）。"""
        original = parse_plan_lines(
            "\n".join([
                "[PLAN: 月 | 09:00-17:30 | 仕事 | active | 強]",
                "[PLAN: 月 | 25:00-31:30 | 就寝 | offline | 中]",
                "[PLAN: 土 | 23:00-06:00 | 夜更かし | active | 弱]",
            ]),
            _WEEK,
        )
        reparsed = parse_plan_lines(format_plan_lines(original), _WEEK)
        assert [(e.start_at, e.end_at, e.label, e.state, e.occupancy) for e in original] == \
               [(e.start_at, e.end_at, e.label, e.state, e.occupancy) for e in reparsed]


class TestEntriesFromTemplate:
    """テンプレ層（availability_schedule）の裸変換（§3 最終フォールバック）を検証する。"""

    def test_blocks_expand_with_state(self):
        """ブロックの state 指定（既定 active）と占有圧の初期値（offline=中/他=強）。"""
        schedule = {
            "mon": [
                {"from": "09:00", "to": "18:00", "label": "仕事"},
                {"from": "23:00", "to": "07:00", "label": "就寝", "state": "offline"},
            ],
        }
        got = entries_from_template(schedule, _WEEK)
        assert len(got) == 2
        work, sleep = got
        assert work.state == "active" and work.occupancy == 0.75
        assert sleep.state == "offline" and sleep.occupancy == 0.5
        # from > to は日跨ぎ（月 23:00 → 火 07:00）
        assert sleep.start_at == datetime(2026, 7, 6, 23, 0)
        assert sleep.end_at == datetime(2026, 7, 7, 7, 0)
        assert layer_has_offline(got) is True

    def test_empty_or_invalid_schedule(self):
        """空・不正なテンプレは空リスト（週次バッチは「エントリなし=OnTime」へ縮退）。"""
        assert entries_from_template(None, _WEEK) == []
        assert entries_from_template({"mon": "broken"}, _WEEK) == []


def _make_state(sqlite_store):
    """週次バッチ用の app.state スタブ（sqlite だけ持つ）を作るヘルパ。"""
    return SimpleNamespace(
        sqlite=sqlite_store, memory_manager=None, working_memory_manager=None,
    )


# ①②のモック応答（就寝 offline を含む有効な [PLAN] テキスト）
_GM_TEXT = "\n".join([
    "[PLAN: 月 | 09:00-17:30 | 仕事 | active | 強]",
    "[PLAN: 月 | 25:00-31:00 | 就寝 | offline | 中]",
])
_HARU_TEXT = "\n".join([
    "今週はゲーム漬けの日曜にしたい。",
    "[PLAN: 日 | 14:00-18:00 | ゲームに没頭 | active | 強]",
    "[PLAN: 日 | 25:00-31:00 | 就寝 | offline | 中]",
])


class TestWorldLayerFallback:
    """①世界固定予定の層フォールバック（GM → 前週① → テンプレ裸変換）を検証する。

    層失敗 = offline（就寝）が1件も取れなかった（§3 2026-07-09 裁定）。
    """

    async def test_gm_success(self, sqlite_store, monkeypatch):
        """GM 出力に offline が含まれれば GM 層をそのまま採用する。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        monkeypatch.setattr(
            weekly_batch_module, "_ask_gm_for_world_plan", _async_return(_GM_TEXT),
        )
        entries, mode = await _generate_world_layer(_make_state(sqlite_store), char, _WEEK)
        assert mode == "gm" and len(entries) == 2

    async def test_no_offline_falls_back_to_prev_week(self, sqlite_store, monkeypatch):
        """GM 出力に offline が無ければ層失敗 → 前週①を +7日シフトして採用する。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        # 前週（6/29 週）の world/template エントリを仕込む
        sqlite_store.create_schedule_entry(
            character_id=char.id,
            start_at=datetime(2026, 6, 30, 1, 0), end_at=datetime(2026, 6, 30, 7, 0),
            state="offline", source="world", origin="template",
            occupancy=0.5, label="就寝",
        )
        monkeypatch.setattr(
            weekly_batch_module, "_ask_gm_for_world_plan",
            _async_return("[PLAN: 月 | 09:00-17:30 | 仕事 | active | 強]"),  # offline なし
        )
        entries, mode = await _generate_world_layer(_make_state(sqlite_store), char, _WEEK)
        assert mode == "prev_week" and len(entries) == 1
        assert entries[0].start_at == datetime(2026, 7, 7, 1, 0)  # +7日
        assert entries[0].state == "offline"

    async def test_gm_failure_falls_back_to_template(self, sqlite_store, monkeypatch):
        """GM 失敗（None）かつ前週なしなら availability_schedule を裸で変換する。"""
        char = _make_character(
            sqlite_store,
            living_schedule_enabled=1,
            availability_schedule={
                "mon": [{"from": "23:00", "to": "07:00", "label": "就寝", "state": "offline"}],
            },
        )
        monkeypatch.setattr(
            weekly_batch_module, "_ask_gm_for_world_plan", _async_return(None),
        )
        entries, mode = await _generate_world_layer(_make_state(sqlite_store), char, _WEEK)
        assert mode == "template" and len(entries) == 1
        assert entries[0].state == "offline"


class TestHaruLayerFallback:
    """②はる固定予定の層フォールバック（本人 → 前週② → はる層なし）を検証する。"""

    async def test_haru_success(self, sqlite_store, monkeypatch):
        """本人の応答に offline が含まれれば本人層をそのまま採用する。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        monkeypatch.setattr(
            weekly_batch_module, "_ask_character_for_week_plan", _async_return(_HARU_TEXT),
        )
        entries, mode = await _generate_haru_layer(
            _make_state(sqlite_store), char, _WEEK, [],
        )
        assert mode == "haru" and len(entries) == 2

    async def test_failure_without_prev_week_adopts_world_only(self, sqlite_store, monkeypatch):
        """本人失敗（None）かつ前週②なしなら空リスト＝①をそのまま採用（はる層なしの週）。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        monkeypatch.setattr(
            weekly_batch_module, "_ask_character_for_week_plan", _async_return(None),
        )
        entries, mode = await _generate_haru_layer(
            _make_state(sqlite_store), char, _WEEK, [],
        )
        assert mode == "none" and entries == []


class TestRunWeeklyScheduleBatch:
    """週次バッチ本体（①②生成 → template 入れ替え保存）を検証する。"""

    async def test_persists_both_layers_and_preserves_adhoc(self, sqlite_store, monkeypatch):
        """world/haru 両層が template として保存され、既存 template は入れ替え・adhoc は温存。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        # 前回生成の残骸（template）と、③④由来の adhoc を対象週に仕込む
        sqlite_store.create_schedule_entry(
            character_id=char.id,
            start_at=datetime(2026, 7, 6, 10, 0), end_at=datetime(2026, 7, 6, 11, 0),
            origin="template", occupancy=0.5, label="旧予定",
        )
        sqlite_store.create_schedule_entry(
            character_id=char.id,
            start_at=datetime(2026, 7, 8, 11, 0), end_at=datetime(2026, 7, 8, 15, 0),
            origin="adhoc", occupancy=1.0, label="突発仕事",
        )
        monkeypatch.setattr(
            weekly_batch_module, "_ask_gm_for_world_plan", _async_return(_GM_TEXT),
        )
        monkeypatch.setattr(
            weekly_batch_module, "_ask_character_for_week_plan", _async_return(_HARU_TEXT),
        )
        result = await run_weekly_schedule_batch(_make_state(sqlite_store), char, _WEEK)
        assert result["world"] == 2 and result["haru"] == 2
        assert result["world_mode"] == "gm" and result["haru_mode"] == "haru"

        entries = sqlite_store.list_schedule_entries(char.id)
        labels = {e.label for e in entries}
        assert "旧予定" not in labels          # template は入れ替え
        assert "突発仕事" in labels             # adhoc は温存
        assert {e.source for e in entries if e.origin == "template"} == {"world", "haru"}
        # 保存されたエントリで availability が引ける（月曜 10:00 は仕事中 = active）
        a = check_availability(char, datetime(2026, 7, 6, 10, 0), sqlite=sqlite_store)
        assert a.state == "active" and a.reason == "仕事"


class TestWeeklyBatchScheduling:
    """スケジューラ入口（run_pending_weekly_batches）の冪等キー判定を検証する。

    コールドスタート（未生成の当週を即時生成）・日曜夜の翌週分生成・
    冪等（同週の二重実行なし）・無効キャラのスキップ。
    """

    def _patch_batch_recorder(self, monkeypatch):
        """run_weekly_schedule_batch を記録用スタブへ差し替え、呼び出しリストを返す。"""
        calls: list[tuple[str, date]] = []

        async def _fake(state, char, week_start):
            calls.append((char.id, week_start))
            return {}

        monkeypatch.setattr(weekly_batch_module, "run_weekly_schedule_batch", _fake)
        return calls

    async def test_cold_start_runs_current_week_once(self, sqlite_store, monkeypatch):
        """有効化直後（マーカーなし）は当週分を即時生成し、二度目は走らない。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        calls = self._patch_batch_recorder(monkeypatch)
        now = datetime(2026, 7, 8, 12, 0)  # 水曜昼
        await run_pending_weekly_batches(_make_state(sqlite_store), now)
        assert calls == [(char.id, date(2026, 7, 6))]
        await run_pending_weekly_batches(_make_state(sqlite_store), now)
        assert len(calls) == 1  # 冪等

    async def test_sunday_night_generates_next_week(self, sqlite_store, monkeypatch):
        """日曜 20:00 以降は翌週分も生成する（当週未生成ならまとめて2週分）。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        calls = self._patch_batch_recorder(monkeypatch)
        now = datetime(2026, 7, 12, 20, 30)  # 日曜夜
        await run_pending_weekly_batches(_make_state(sqlite_store), now)
        assert calls == [(char.id, date(2026, 7, 6)), (char.id, date(2026, 7, 13))]
        # 冪等キーは翌週まで進んでいる
        assert sqlite_store.get_setting(f"weekly_schedule_done_{char.id}", "") == "2026-W29"

    async def test_sunday_before_batch_time_only_current(self, sqlite_store, monkeypatch):
        """日曜でも定時（既定 20:00）前は翌週分を生成しない。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        calls = self._patch_batch_recorder(monkeypatch)
        await run_pending_weekly_batches(
            _make_state(sqlite_store), datetime(2026, 7, 12, 19, 59),
        )
        assert calls == [(char.id, date(2026, 7, 6))]

    async def test_disabled_character_skipped(self, sqlite_store, monkeypatch):
        """生活カレンダー無効キャラは走査対象外（従来挙動のまま）。"""
        _make_character(sqlite_store)  # living_schedule_enabled=0
        calls = self._patch_batch_recorder(monkeypatch)
        await run_pending_weekly_batches(
            _make_state(sqlite_store), datetime(2026, 7, 8, 12, 0),
        )
        assert calls == []

    def test_week_key_ordering(self):
        """週キーは辞書順 = 時系列順（年跨ぎ含む — 冪等判定の前提）。"""
        assert week_key(date(2026, 7, 6)) == "2026-W28"
        assert week_key(date(2026, 12, 28)) < week_key(date(2027, 1, 4))
        assert week_start_of(date(2026, 7, 9)) == date(2026, 7, 6)


def _async_return(value):
    """固定値を返す async スタブを作る（monkeypatch 用ヘルパ）。"""
    async def _stub(*args, **kwargs):
        return value
    return _stub


# ---------------------------------------------------------------------------
# Phase 4: うつつ②導出（シーン選出純関数）
# ---------------------------------------------------------------------------

from backend.services.schedule import (  # noqa: E402
    EventEntry,
    format_scene_framing,
    max_overlapping_occupancy,
    parse_event_line,
    place_weekly_hidden_events,
    run_pending_sudden_events,
    select_daily_scenes,
)
from backend.services.schedule import events as events_module  # noqa: E402


def _fake_entry(entry_id, hour, state="active", occupancy=0.5, label="予定", day=None):
    """select_daily_scenes 用の軽量エントリスタブを作る。"""
    base = day or _MON
    return SimpleNamespace(
        id=entry_id,
        start_at=base.replace(hour=hour),
        end_at=base.replace(hour=hour) + timedelta(hours=2),
        state=state,
        occupancy=occupancy,
        label=label,
    )


class TestSceneSelection:
    """②導出のシーン選出純関数（select_daily_scenes・§8）を検証する。"""

    def test_offline_excluded(self):
        """offline（就寝）は体験の題材にならないので選出対象外。"""
        entries = [
            _fake_entry("a", 2, state="offline", occupancy=0.5, label="就寝"),
            _fake_entry("b", 14, state="active", occupancy=0.5, label="ゲーム"),
        ]
        got = select_daily_scenes(
            entries, character_id="c1", day=_MON.date(), scenes_per_day=5,
        )
        assert [s.label for s in got] == ["ゲーム"]

    def test_all_when_within_cap(self):
        """候補数が枠以下なら全採用（起動時刻昇順）。"""
        entries = [
            _fake_entry("b", 14, label="午後"),
            _fake_entry("a", 9, label="午前"),
        ]
        got = select_daily_scenes(
            entries, character_id="c1", day=_MON.date(), scenes_per_day=3,
        )
        assert [s.label for s in got] == ["午前", "午後"]  # fire_at 昇順
        assert all(s.fire_at >= e.start_at for s, e in zip(
            sorted(got, key=lambda s: s.entry_id), sorted(entries, key=lambda e: e.id)
        ))

    def test_zero_scenes_per_day_empty(self):
        """scenes_per_day=0 は空リスト（生活カレンダー有効でもシーン導出しない）。"""
        entries = [_fake_entry("a", 9)]
        assert select_daily_scenes(
            entries, character_id="c1", day=_MON.date(), scenes_per_day=0,
        ) == []

    def test_top_occupancy_prioritised_and_deterministic(self):
        """枠超過時は占有圧上位が必ず入り、選出は決定論（同入力＝同結果）。"""
        entries = [
            _fake_entry("hi", 9, occupancy=1.0, label="激強"),
            _fake_entry("mid", 11, occupancy=0.5, label="中"),
            _fake_entry("lo1", 13, occupancy=0.25, label="弱1"),
            _fake_entry("lo2", 15, occupancy=0.25, label="弱2"),
        ]
        got1 = select_daily_scenes(
            entries, character_id="c1", day=_MON.date(), scenes_per_day=2,
        )
        got2 = select_daily_scenes(
            entries, character_id="c1", day=_MON.date(), scenes_per_day=2,
        )
        labels1 = [s.label for s in got1]
        assert labels1 == [s.label for s in got2]  # 決定論
        assert "激強" in labels1  # 占有圧上位（枠の 50%=1枠切り上げ）は必ず入る
        assert len(got1) == 2

    def test_other_day_entries_ignored(self):
        """対象日以外に始まるエントリは候補にしない。"""
        entries = [_fake_entry("tue", 14, day=_MON + timedelta(days=1))]
        assert select_daily_scenes(
            entries, character_id="c1", day=_MON.date(), scenes_per_day=3,
        ) == []

    def test_format_scene_framing(self):
        """framing OOC はラベルとキャラ名を含む。空ラベルは空文字列。"""
        text = format_scene_framing("はる", "ゲームに没頭")
        assert "はる" in text and "ゲームに没頭" in text
        assert format_scene_framing("はる", "") == ""


# ---------------------------------------------------------------------------
# Phase 5: ③突発（[EVENT] パーサ・伏せ枠配置・轢き判定・発火）
# ---------------------------------------------------------------------------

class TestEventParser:
    """[EVENT] 行パーサ（parse_event_line・§5）を検証する。"""

    _DAY = date(2026, 7, 6)

    def test_basic_event(self):
        """基本形: 時間帯・ラベル・状態・圧が絶対時刻イベントへ展開される。"""
        e = parse_event_line("[EVENT: 11:00-15:00 | 客先トラブル | busy | 激強]", self._DAY)
        assert e is not None
        assert e.start_at == datetime(2026, 7, 6, 11, 0)
        assert e.end_at == datetime(2026, 7, 6, 15, 0)
        assert e.state == "busy" and e.occupancy == 1.0
        assert e.reply_rate is None and e.check_interval is None

    def test_delivery_overrides(self):
        """末尾の reply= / check= が配達値の個別上書きになる。"""
        e = parse_event_line(
            "[EVENT: 20:00-21:00 | 友人が来た | active | 強 | reply=0.2 | check=90]",
            self._DAY,
        )
        assert e.reply_rate == 0.2 and e.check_interval == 90

    def test_free_text_coexists(self):
        """自由文と混在してよい（最初の [EVENT] 行だけ採用）。"""
        text = "急に電話が鳴った。\n[EVENT: 13:00-14:00 | 電話対応 | busy | 強]\n以上。"
        e = parse_event_line(text, self._DAY)
        assert e is not None and e.label == "電話対応"

    def test_no_event_line(self):
        """[EVENT] 行が無ければ None。"""
        assert parse_event_line("ただの説明文", self._DAY) is None

    def test_invalid_pressure_skipped(self):
        """未知の圧ラベルはパース不能（None）。"""
        assert parse_event_line("[EVENT: 11:00-15:00 | x | busy | 超激強]", self._DAY) is None


class TestHiddenEventPlacement:
    """③伏せ枠の確率配置（place_weekly_hidden_events・§3）を検証する。"""

    def _usual_scenario(self, sqlite_store, char, prob, cats):
        """イベント設定付きのうつつシナリオを作る。"""
        sqlite_store.create_scenario(
            scenario_id=str(uuid.uuid4()),
            title="うつつ",
            owner_character_id=char.id,
            usual_config={
                "enabled": True,
                "event_categories": cats,
                "event_probability": prob,
            },
        )

    def test_places_pending_seeds_not_in_availability(self, sqlite_store):
        """伏せ枠は status=pending で置かれ、availability（planned のみ）には出ない。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        self._usual_scenario(sqlite_store, char, prob=1.0, cats=["残業", "来客"])
        placed = place_weekly_hidden_events(sqlite_store, char, _WEEK)
        assert placed == 7  # prob=1.0 なので全曜日に1件
        # pending なので get_active には出ない（availability を侵さない）
        all_rows = sqlite_store.list_schedule_entries(char.id)
        assert all(r.status == "pending" and r.origin == "adhoc" for r in all_rows)
        # 週内のどこかの伏せ枠時刻を取り、その時刻の get_active が空であることを確認
        seed = all_rows[0]
        assert sqlite_store.get_active_schedule_entries(char.id, seed.start_at) == []

    def test_deterministic(self, sqlite_store):
        """同じキャラ・同じ週なら配置は決定論（再配置で同じ時刻集合）。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        self._usual_scenario(sqlite_store, char, prob=0.5, cats=["残業"])
        place_weekly_hidden_events(sqlite_store, char, _WEEK)
        first = sorted(r.start_at for r in sqlite_store.list_schedule_entries(char.id))
        # 再配置（未発火 pending を消して置き直す）— 同じ結果
        place_weekly_hidden_events(sqlite_store, char, _WEEK)
        second = sorted(r.start_at for r in sqlite_store.list_schedule_entries(char.id))
        assert first == second

    def test_no_categories_places_nothing(self, sqlite_store):
        """カテゴリ／確率が無ければ0件（伏せ枠を置かない）。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        self._usual_scenario(sqlite_store, char, prob=0.0, cats=[])
        assert place_weekly_hidden_events(sqlite_store, char, _WEEK) == 0

    def test_replacement_preserves_fired_events(self, sqlite_store):
        """再配置は未発火 pending だけ消し、発火済み（planned）実イベントは温存する。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        self._usual_scenario(sqlite_store, char, prob=1.0, cats=["残業"])
        # 発火済みの実イベント（planned adhoc）を週内に置く
        sqlite_store.create_schedule_entry(
            character_id=char.id,
            start_at=datetime(2026, 7, 8, 11, 0), end_at=datetime(2026, 7, 8, 15, 0),
            origin="adhoc", status="planned", occupancy=1.0, label="発火済み突発",
        )
        place_weekly_hidden_events(sqlite_store, char, _WEEK)
        labels = {r.label for r in sqlite_store.list_schedule_entries(char.id)}
        assert "発火済み突発" in labels  # planned は消えない


class TestMaxOverlappingOccupancy:
    """轢き判定の下請け（max_overlapping_occupancy・§4）を検証する。"""

    def test_returns_max_of_overlaps(self, sqlite_store):
        """重なる planned の占有圧最大を返す。重なりなしは 0.0。"""
        char = _make_character(sqlite_store)
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=9),
            end_at=_MON.replace(hour=12), status="planned", occupancy=0.5,
        )
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=10),
            end_at=_MON.replace(hour=11), status="planned", occupancy=0.75,
        )
        assert max_overlapping_occupancy(
            sqlite_store, char.id, _MON.replace(hour=10, minute=30),
            _MON.replace(hour=10, minute=45),
        ) == 0.75
        # 重ならない時間帯は 0.0
        assert max_overlapping_occupancy(
            sqlite_store, char.id, _MON.replace(hour=20), _MON.replace(hour=21),
        ) == 0.0


def _make_event_state(sqlite_store):
    """③発火テスト用の app.state スタブ（sqlite + chat_service スタブ）。"""
    return SimpleNamespace(
        sqlite=sqlite_store, memory_manager=None, working_memory_manager=None,
        chat_service=SimpleNamespace(),
    )


class TestFireSuddenEvent:
    """③発火（run_pending_sudden_events）の轢き判定・上限・聖域化を検証する。

    GM 具体化・シーン実行・玉突き裁定はモックし、insert 可否（§4）を純粋に検証する。
    """

    def _seed(self, sqlite_store, char, fire_at, category="残業"):
        """未発火の伏せ枠を1件仕込む。"""
        return sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=fire_at,
            end_at=fire_at + timedelta(hours=1),
            origin="adhoc", source="world", status="pending", occupancy=0.0,
            label=category,
            payload={"kind": "sudden_event_seed", "category": category},
        )

    def _patch_gm_and_scene(self, monkeypatch, event_text):
        """GM 具体化と後続（シーン・玉突き）をモックする。"""
        monkeypatch.setattr(
            events_module, "_ask_gm_to_concretize", _async_return(event_text),
        )
        monkeypatch.setattr(
            events_module, "_run_event_scene", _async_return({"error": None}),
        )
        import backend.services.schedule.dilemma as dilemma_module
        monkeypatch.setattr(
            dilemma_module, "run_collision_ruling",
            _async_return({"status": "skipped"}),
        )

    async def test_override_inserts_event(self, sqlite_store, monkeypatch):
        """占有圧が既存を上回れば突発を insert する（轢く）。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        fire = _MON.replace(hour=13)
        self._seed(sqlite_store, char, fire)
        # 既存の中(0.5)予定に、激強(1.0)の突発が入る
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=12),
            end_at=_MON.replace(hour=15), status="planned", occupancy=0.5, label="ゲーム",
        )
        self._patch_gm_and_scene(
            monkeypatch, "[EVENT: 13:00-15:00 | 客先トラブル | busy | 激強]",
        )
        await run_pending_sudden_events(_make_event_state(sqlite_store), fire)
        planned = sqlite_store.list_schedule_entries(char.id, statuses=["planned"])
        labels = {e.label for e in planned}
        assert "客先トラブル" in labels  # 轢いて insert された

    async def test_equal_or_lower_does_not_insert(self, sqlite_store, monkeypatch):
        """占有圧が既存以下なら insert しない（本人の予定が僅差で守られる）。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        fire = _MON.replace(hour=13)
        self._seed(sqlite_store, char, fire)
        # 既存の強(0.75)予定に、中(0.5)の突発は轢けない
        sqlite_store.create_schedule_entry(
            character_id=char.id, start_at=_MON.replace(hour=12),
            end_at=_MON.replace(hour=15), status="planned", occupancy=0.75, label="仕事",
        )
        self._patch_gm_and_scene(
            monkeypatch, "[EVENT: 13:00-15:00 | 雑談の誘い | active | 中]",
        )
        await run_pending_sudden_events(_make_event_state(sqlite_store), fire)
        planned = sqlite_store.list_schedule_entries(char.id, statuses=["planned"])
        assert "雑談の誘い" not in {e.label for e in planned}
        # 伏せ枠は発火試行済み（done）
        seeds = sqlite_store.list_schedule_entries(char.id, statuses=["pending"])
        assert seeds == []

    async def test_face_to_face_holds_seed(self, sqlite_store, monkeypatch):
        """対面中は発火を保留（伏せ枠は pending のまま・done にしない）。"""
        char = _make_character(
            sqlite_store, living_schedule_enabled=1, face_to_face_mode=1,
        )
        fire = _MON.replace(hour=13)
        self._seed(sqlite_store, char, fire)
        self._patch_gm_and_scene(
            monkeypatch, "[EVENT: 13:00-15:00 | x | busy | 激強]",
        )
        await run_pending_sudden_events(_make_event_state(sqlite_store), fire)
        # 保留: pending のまま残る
        assert len(sqlite_store.list_schedule_entries(char.id, statuses=["pending"])) == 1

    async def test_daily_cap_discards(self, sqlite_store, monkeypatch):
        """日次上限に達したら発火せず捨てる（done・insert なし）。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        fire = _MON.replace(hour=13)
        self._seed(sqlite_store, char, fire)
        sqlite_store.set_setting("sudden_event_daily_cap", "0")
        self._patch_gm_and_scene(
            monkeypatch, "[EVENT: 13:00-15:00 | x | busy | 激強]",
        )
        await run_pending_sudden_events(_make_event_state(sqlite_store), fire)
        assert sqlite_store.list_schedule_entries(char.id, statuses=["planned"]) == []
        assert sqlite_store.list_schedule_entries(char.id, statuses=["pending"]) == []

    async def test_grace_window_discards_stale(self, sqlite_store, monkeypatch):
        """発火猶予を超えた古い伏せ枠は発火せず捨てる（生活は流れる）。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        fire = _MON.replace(hour=1)
        self._seed(sqlite_store, char, fire)
        self._patch_gm_and_scene(
            monkeypatch, "[EVENT: 01:00-02:00 | x | busy | 激強]",
        )
        # 10時間後に評価 → 猶予(6h)超過
        await run_pending_sudden_events(
            _make_event_state(sqlite_store), fire + timedelta(hours=10),
        )
        assert sqlite_store.list_schedule_entries(char.id, statuses=["planned"]) == []


# ---------------------------------------------------------------------------
# Phase 6: 玉突き裁定
# ---------------------------------------------------------------------------

from backend.services.schedule import dilemma as dilemma_module  # noqa: E402
from backend.services.schedule.dilemma import (  # noqa: E402
    _find_disrupted_entries,
    _parse_ruling,
    run_collision_ruling,
)


class TestCollisionRulingHelpers:
    """玉突き裁定の下請け（轢かれた予定の抽出・タグパース）を検証する。"""

    def test_find_disrupted_excludes_offline_and_higher(self, sqlite_store):
        """轢かれた予定 = 重なる template・低占有圧・非 offline のみ。"""
        char = _make_character(sqlite_store)
        span = dict(start_at=_MON.replace(hour=12), end_at=_MON.replace(hour=15))
        # 轢かれる: 中(0.5) の active 予定
        low = sqlite_store.create_schedule_entry(
            character_id=char.id, origin="template", status="planned",
            occupancy=0.5, state="active", label="ゲーム", **span,
        )
        # 轢かれない: 高占有圧（激強と同等）
        sqlite_store.create_schedule_entry(
            character_id=char.id, origin="template", status="planned",
            occupancy=1.0, state="busy", label="重要", **span,
        )
        # 轢かれない: offline（就寝は裁定対象外）
        sqlite_store.create_schedule_entry(
            character_id=char.id, origin="template", status="planned",
            occupancy=0.25, state="offline", label="就寝", **span,
        )
        event = SimpleNamespace(
            id="ev", start_at=_MON.replace(hour=13), end_at=_MON.replace(hour=14),
            occupancy=1.0,
        )
        disrupted = _find_disrupted_entries(sqlite_store, char.id, event)
        assert [e.label for e in disrupted] == ["ゲーム"]
        assert disrupted[0].id == low.id

    def test_parse_ruling_filters_unknown_ids(self):
        """設問で提示していない id のタグは捨てる（id は UUID 形の16進）。"""
        text = (
            "[GIVE_UP: aa11bb]\n[RESCHEDULE: cc22dd | 19:00-21:00]\n"
            "[DISSATISFIED: aa11bb | もやもやする]\n[GIVE_UP: ee33ff]"
        )
        parsed = _parse_ruling(text, {"aa11bb", "cc22dd"})
        assert parsed["give_up"] == ["aa11bb"]
        assert parsed["reschedule"] == [{"id": "cc22dd", "range": "19:00-21:00"}]
        assert parsed["dissatisfied"] == [{"id": "aa11bb", "words": "もやもやする"}]


class TestRunCollisionRuling:
    """玉突き裁定本体（run_collision_ruling・§6）を検証する。"""

    def _setup(self, sqlite_store):
        """轢かれた予定1件＋轢いた突発イベントを用意する。"""
        char = _make_character(sqlite_store, ghost_model="preset-1")
        disrupted = sqlite_store.create_schedule_entry(
            character_id=char.id, origin="template", status="planned",
            occupancy=0.5, state="active", label="ゲーム",
            start_at=_MON.replace(hour=12), end_at=_MON.replace(hour=15),
        )
        event = sqlite_store.create_schedule_entry(
            character_id=char.id, origin="adhoc", source="world", status="planned",
            occupancy=1.0, state="busy", label="客先トラブル",
            start_at=_MON.replace(hour=13), end_at=_MON.replace(hour=14),
        )
        return char, disrupted, event

    async def test_give_up_cancels(self, sqlite_store, monkeypatch):
        """[GIVE_UP] で轢かれた予定が cancelled になる（見送り）。"""
        char, disrupted, event = self._setup(sqlite_store)
        monkeypatch.setattr(
            dilemma_module, "ask_character_with_tools",
            _async_return(f"しゃーない。[GIVE_UP: {disrupted.id}]"),
        )
        result = await run_collision_ruling(
            _make_event_state(sqlite_store), char, event,
        )
        assert result["given_up"] == 1
        rows = sqlite_store.list_schedule_entries(char.id, statuses=["cancelled"])
        assert disrupted.id in {r.id for r in rows}

    async def test_reschedule_inserts_new(self, sqlite_store, monkeypatch):
        """[RESCHEDULE] で旧を cancel し新しい時間帯へ④ adhoc を insert する。"""
        char, disrupted, event = self._setup(sqlite_store)
        monkeypatch.setattr(
            dilemma_module, "ask_character_with_tools",
            _async_return(f"夜にずらす。[RESCHEDULE: {disrupted.id} | 20:00-22:00]"),
        )
        result = await run_collision_ruling(
            _make_event_state(sqlite_store), char, event,
        )
        assert result["rescheduled"] == 1
        planned = sqlite_store.list_schedule_entries(char.id, statuses=["planned"])
        moved = [e for e in planned if e.origin == "adhoc" and e.label == "ゲーム"]
        assert len(moved) == 1 and moved[0].start_at == _MON.replace(hour=20)
        # 旧予定は cancelled
        assert disrupted.id in {
            r.id for r in sqlite_store.list_schedule_entries(char.id, statuses=["cancelled"])
        }

    async def test_dissatisfied_sours_intent(self, sqlite_store, monkeypatch):
        """[DISSATISFIED] で意図を作って soured にする（不満圧へ転化）。"""
        char, disrupted, event = self._setup(sqlite_store)
        monkeypatch.setattr(
            dilemma_module, "ask_character_with_tools",
            _async_return(f"納得いかない。[DISSATISFIED: {disrupted.id} | 楽しみにしてたのに]"),
        )
        result = await run_collision_ruling(
            _make_event_state(sqlite_store), char, event,
        )
        assert result["soured"] == 1
        soured = sqlite_store.list_intents(char.id, status="soured")
        assert len(soured) == 1

    async def test_no_disrupted_skips(self, sqlite_store, monkeypatch):
        """轢かれた予定が無ければ問い合わせずスキップ（LLM を呼ばない）。"""
        char = _make_character(sqlite_store, ghost_model="preset-1")
        event = sqlite_store.create_schedule_entry(
            character_id=char.id, origin="adhoc", status="planned",
            occupancy=1.0, state="busy", label="突発",
            start_at=_MON.replace(hour=13), end_at=_MON.replace(hour=14),
        )
        called = {"n": 0}

        async def _spy(*a, **k):
            called["n"] += 1
            return ""

        monkeypatch.setattr(dilemma_module, "ask_character_with_tools", _spy)
        result = await run_collision_ruling(
            _make_event_state(sqlite_store), char, event,
        )
        assert result["status"] == "skipped" and called["n"] == 0


# ---------------------------------------------------------------------------
# Phase 8: 計器（weekly_batch_heartbeat）
# ---------------------------------------------------------------------------

from backend.services.instruments.tier1 import (  # noqa: E402
    _check_weekly_batch_heartbeat,
)


class TestWeeklyBatchHeartbeat:
    """Tier 1 計器 weekly_batch_heartbeat（§13）を検証する。"""

    def test_fires_when_no_current_week_entries(self, sqlite_store):
        """有効キャラに当週の template エントリが無ければ発火する。"""
        _make_character(sqlite_store, living_schedule_enabled=1)
        fired = _check_weekly_batch_heartbeat(sqlite_store)
        assert len(fired) == 1

    def test_silent_when_entries_exist(self, sqlite_store):
        """当週に template planned エントリがあれば発火しない。"""
        char = _make_character(sqlite_store, living_schedule_enabled=1)
        now = datetime.now()
        week_start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sqlite_store.create_schedule_entry(
            character_id=char.id,
            start_at=week_start.replace(hour=9),
            end_at=week_start.replace(hour=17),
            origin="template", status="planned", occupancy=0.75, label="仕事",
        )
        assert _check_weekly_batch_heartbeat(sqlite_store) == []

    def test_disabled_char_ignored(self, sqlite_store):
        """生活カレンダー無効キャラは対象外（発火しない）。"""
        _make_character(sqlite_store)  # living_schedule_enabled=0
        assert _check_weekly_batch_heartbeat(sqlite_store) == []
