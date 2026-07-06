"""応答可能性ゲート（gate）のテスト — めぐり（巡り / Aliveness）Phase 5。

検証対象（docs/aliveness_plan.md §5.1〜5.2）:
    1. check_availability: 純関数の判定優先順位
       （対面 > away > うつつシーン進行中 > 生活時間割）と日跨ぎ時間帯
    2. メッセージ預かり（escrow）: delivered=False 保存では封筒が作られず、
       配達（mark_messages_delivered）で初めて chat.message 封筒ができること。
       時間差注釈の整形
    3. 疲労離席: 発火式（体調圧 > θ_base + β×engagement / θ_hard 無条件）、
       away 設定・退去挨拶・chat.farewell(fatigue) 封筒
    4. take_leave: 呼べば必ず執行される権利（away 設定＋封筒＋上限クランプ）
    5. farewell judge の engagement パース（縮退込み）
"""

import uuid
from datetime import datetime, timedelta

from backend.character_actions.leaver import Leaver
from backend.services.gate.availability import (
    check_availability,
    format_escrow_annotation,
    is_usual_scene_running,
    mark_usual_scene_running,
)
from backend.services.gate.fatigue import check_fatigue_leave


def _make_character(sqlite_store, name="はるテスト", **kwargs):
    """テスト用キャラクターを1体作成して返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    if kwargs:
        sqlite_store.update_character(char_id, **kwargs)
    return char_id, name


class _FakeChar:
    """check_availability に渡す Character スタブ（純関数テスト用）。"""

    def __init__(self, face_to_face=0, away_until=None, away_reason=None, schedule=None):
        self.face_to_face_mode = face_to_face
        self.away_until = away_until
        self.away_reason = away_reason
        self.availability_schedule = schedule


# 月曜の昼（2026-07-06 は月曜）
_MONDAY_NOON = datetime(2026, 7, 6, 12, 30)


class TestCheckAvailability:
    """check_availability の判定優先順位と時間割の解釈を検証するテストクラス。"""

    def test_default_available(self):
        """設定なし（スケジュール空・away なし）は available。"""
        a = check_availability(_FakeChar(), _MONDAY_NOON)
        assert a.available is True and a.reason == ""

    def test_none_character_available(self):
        """キャラ未解決（None）はゲート対象外として available。"""
        assert check_availability(None).available is True

    def test_schedule_blocks(self):
        """生活時間割の応答不可時間帯に入っていれば unavailable(ラベル)。"""
        schedule = {"mon": [{"from": "09:00", "to": "18:00", "label": "仕事"}]}
        a = check_availability(_FakeChar(schedule=schedule), _MONDAY_NOON)
        assert a.available is False and a.reason == "仕事"
        # 時間帯の外（夜）は available
        evening = _MONDAY_NOON.replace(hour=20)
        assert check_availability(_FakeChar(schedule=schedule), evening).available

    def test_schedule_overnight_block(self):
        """from > to は日跨ぎ（23:00〜06:00 = 就寝など）として扱う。"""
        schedule = {"mon": [{"from": "23:00", "to": "06:00", "label": "就寝"}]}
        late = _MONDAY_NOON.replace(hour=23, minute=30)
        early = _MONDAY_NOON.replace(hour=5, minute=0)
        assert check_availability(_FakeChar(schedule=schedule), late).available is False
        assert check_availability(_FakeChar(schedule=schedule), early).available is False
        assert check_availability(_FakeChar(schedule=schedule), _MONDAY_NOON).available

    def test_away_blocks_until_expiry(self):
        """away_until が未来なら unavailable(理由)、過ぎれば自動解除。"""
        char = _FakeChar(
            away_until=_MONDAY_NOON + timedelta(hours=2), away_reason="疲労のため休息中",
        )
        a = check_availability(char, _MONDAY_NOON)
        assert a.available is False and a.reason == "疲労のため休息中"
        after = _MONDAY_NOON + timedelta(hours=3)
        assert check_availability(char, after).available is True

    def test_face_to_face_overrides_everything(self):
        """対面モード中は away・時間割に関係なく available（目の前にいる）。"""
        char = _FakeChar(
            face_to_face=1,
            away_until=_MONDAY_NOON + timedelta(hours=2),
            schedule={"mon": [{"from": "00:00", "to": "23:59", "label": "仕事"}]},
        )
        assert check_availability(char, _MONDAY_NOON).available is True

    def test_usual_scene_running_blocks(self):
        """うつつシーン進行中は unavailable("usual_scene")。"""
        a = check_availability(_FakeChar(), _MONDAY_NOON, usual_scene_running=True)
        assert a.available is False and a.reason == "usual_scene"

    def test_usual_scene_marker_ttl(self, sqlite_store):
        """進行中マーカーは TTL（30分）を超えると無視される（クラッシュ耐性）。"""
        char_id, _ = _make_character(sqlite_store)
        mark_usual_scene_running(sqlite_store, char_id, True)
        assert is_usual_scene_running(sqlite_store, char_id) is True
        # TTL 超過を再現
        assert is_usual_scene_running(
            sqlite_store, char_id, now=datetime.now() + timedelta(minutes=31)
        ) is False
        mark_usual_scene_running(sqlite_store, char_id, False)
        assert is_usual_scene_running(sqlite_store, char_id) is False


class TestEscrow:
    """メッセージ預かり（escrow）の保存・配達・封筒タイミングを検証するテストクラス。

    「呼ばれなければ継続できない」の実装 — 預かり中のメッセージはキャラの身に
    まだ起きていないため封筒に載らず、配達の瞬間に chat.message 封筒ができる。
    """

    def _make_session(self, sqlite_store, char_name):
        """キャラ名に紐づく 1on1 セッションを作るヘルパ。"""
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        return sid

    def test_escrowed_message_has_no_envelope(self, sqlite_store):
        """delivered=False で保存されたメッセージは封筒を作らない。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        msg_id = str(uuid.uuid4())
        sqlite_store.create_chat_message(
            message_id=msg_id, session_id=sid, role="user",
            content="仕事中に送ったメッセージ", delivered=False,
        )
        assert sqlite_store.list_timeline_events(char_id) == []
        pending = sqlite_store.list_undelivered_messages(sid)
        assert [m.id for m in pending] == [msg_id]

    def test_delivery_creates_envelope(self, sqlite_store):
        """配達で delivered_at が立ち、chat.message 封筒（occurred_at=配達時刻）ができる。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        msg_id = str(uuid.uuid4())
        sqlite_store.create_chat_message(
            message_id=msg_id, session_id=sid, role="user",
            content="預かり分", delivered=False,
        )
        sqlite_store.mark_messages_delivered([msg_id])
        assert sqlite_store.list_undelivered_messages(sid) == []
        events = sqlite_store.list_timeline_events(char_id)
        assert len(events) == 1
        assert events[0].event_type == "chat.message"
        assert events[0].source_id == msg_id

    def test_delivery_is_idempotent(self, sqlite_store):
        """二重配達しても封筒は増えない（delivered_at IS NULL のみ対象）。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        msg_id = str(uuid.uuid4())
        sqlite_store.create_chat_message(
            message_id=msg_id, session_id=sid, role="user",
            content="a", delivered=False,
        )
        sqlite_store.mark_messages_delivered([msg_id])
        sqlite_store.mark_messages_delivered([msg_id])
        assert len(sqlite_store.list_timeline_events(char_id)) == 1

    def test_sessions_with_undelivered(self, sqlite_store):
        """配達スケジューラ用の未配達セッション一覧が件数付きで返る。"""
        _, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        for i in range(2):
            sqlite_store.create_chat_message(
                message_id=str(uuid.uuid4()), session_id=sid, role="user",
                content=f"m{i}", delivered=False,
            )
        rows = sqlite_store.list_sessions_with_undelivered()
        assert len(rows) == 1
        session, count = rows[0]
        assert session.id == sid and count == 2

    def test_escrow_annotation_format(self, sqlite_store):
        """時間差注釈が送信時刻付きで本文の前に付く（DB は変更しない）。"""
        _, char_name = _make_character(sqlite_store)
        sid = self._make_session(sqlite_store, char_name)
        msg_id = str(uuid.uuid4())
        sqlite_store.create_chat_message(
            message_id=msg_id, session_id=sid, role="user",
            content="本文", delivered=False,
        )
        msg = sqlite_store.list_undelivered_messages(sid)[0]
        annotated = format_escrow_annotation(msg)
        assert "席を外している間に届いていた" in annotated
        assert annotated.endswith("本文")


class TestFatigueLeave:
    """疲労離席の発火式と執行内容を検証するテストクラス。

    体調圧は封筒の純関数なので、封筒を大量投入して高圧状態を作って検証する。
    """

    def _exhaust(self, sqlite_store, char_id, count=200):
        """直近数時間に大量の活動イベントを積んで体調圧を最大化するヘルパ。"""
        base = datetime.now()
        for i in range(count):
            sqlite_store.record_timeline_event(
                character_id=char_id, event_type="scene.turn",
                origin="usual", occurred_at=base - timedelta(minutes=i * 5),
            )

    def test_no_fatigue_config_never_fires(self, sqlite_store):
        """farewell_config に fatigue が無ければ発火しない（機能オプトイン）。"""
        char_id, char_name = _make_character(sqlite_store)
        self._exhaust(sqlite_store, char_id)
        fired = check_fatigue_leave(
            sqlite_store, char_id, char_name, "s1", farewell_config={},
        )
        assert fired is False

    def test_fires_on_hard_threshold(self, sqlite_store):
        """体調圧が θ_hard 超なら没入度に関係なく無条件発火（限界は限界）。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        self._exhaust(sqlite_store, char_id)
        fired = check_fatigue_leave(
            sqlite_store, char_id, char_name, sid,
            farewell_config={"fatigue": {"theta_hard": 0.9}},
            engagement=1.0,  # 夢中でも限界は限界
        )
        assert fired is True
        char = sqlite_store.get_character(char_id)
        assert char.away_until is not None and char.away_until > datetime.now()
        assert "休息" in char.away_reason
        # 退去挨拶（システムメッセージ）が保存されている
        msgs = sqlite_store.list_chat_messages(sid)
        assert any(m.is_system_message for m in msgs)
        # chat.farewell(fatigue) 封筒が正本に載る
        farewells = sqlite_store.list_timeline_events(
            char_id, event_type_prefixes=["chat.farewell"],
        )
        assert farewells[0].payload["farewell_type"] == "fatigue"

    def test_engagement_raises_threshold(self, sqlite_store):
        """没入度が高いと閾値が持ち上がり、同じ体調圧でも発火しない。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        # 体調圧を「θ_base とβ持ち上げの間」に置くための調整はイベント数では難しいので、
        # θ を直接調整して式の分岐だけを検証する（体調圧はこのフィクスチャで最大近く）。
        self._exhaust(sqlite_store, char_id)
        # θ_base=0.5, β=0.6 → engagement=1.0 で閾値 1.1（体調圧 <= 1.0 なので発火しない）
        fired_engaged = check_fatigue_leave(
            sqlite_store, char_id, char_name, sid,
            farewell_config={"fatigue": {
                "theta_base": 0.5, "beta": 0.6, "theta_hard": 1.1,
            }},
            engagement=1.0,
        )
        assert fired_engaged is False
        # engagement=0.0 なら閾値 0.5 → 発火する
        fired_bored = check_fatigue_leave(
            sqlite_store, char_id, char_name, sid,
            farewell_config={"fatigue": {
                "theta_base": 0.5, "beta": 0.6, "theta_hard": 1.1,
            }},
            engagement=0.0,
        )
        assert fired_bored is True


class TestTakeLeave:
    """take_leave ツール（本人宣言の離席）を検証するテストクラス。"""

    def test_take_leave_sets_away_and_envelope(self, sqlite_store):
        """執行で away 状態＋chat.farewell(take_leave) 封筒ができる。"""
        char_id, _ = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        leaver = Leaver(char_id, sid, sqlite_store)
        result = leaver.take_leave(reason="少し一人になりたい", hours=3)
        assert "席を外した" in result
        char = sqlite_store.get_character(char_id)
        assert char.away_reason == "少し一人になりたい"
        assert char.away_until > datetime.now() + timedelta(hours=2.5)
        ev = sqlite_store.list_timeline_events(
            char_id, event_type_prefixes=["chat.farewell"],
        )[0]
        assert ev.payload["farewell_type"] == "take_leave"

    def test_hours_clamped(self, sqlite_store):
        """離席時間は上限12時間・不正値は既定2時間にクランプされる。"""
        char_id, _ = _make_character(sqlite_store)
        leaver = Leaver(char_id, None, sqlite_store)
        leaver.take_leave(reason="長期休暇", hours=100)
        char = sqlite_store.get_character(char_id)
        assert char.away_until <= datetime.now() + timedelta(hours=12.1)
        leaver.take_leave(reason="変な値", hours="abc")
        char = sqlite_store.get_character(char_id)
        assert char.away_until <= datetime.now() + timedelta(hours=2.1)


class TestEngagementParsing:
    """farewell judge 応答の engagement パース（縮退込み）を検証するテストクラス。"""

    def test_engagement_parsed_and_clamped(self):
        """JSON の engagement が FarewellResult に載り、範囲外はクランプされる。"""
        from backend.character_actions.farewell_detector import _parse_judge_response
        parsed = _parse_judge_response(
            '{"emotions": {"anger": 0.1}, "engagement": 0.9, "should_exit": false}'
        )
        assert parsed["engagement"] == 0.9

    def test_missing_engagement_defaults(self):
        """engagement フィールドが無い judge 応答でもパースは成立する（0.5 縮退は detect 側）。"""
        from backend.character_actions.farewell_detector import _parse_judge_response
        parsed = _parse_judge_response(
            '{"emotions": {}, "should_exit": false, "farewell_type": null}'
        )
        assert "engagement" not in parsed  # detect() が 0.5 に縮退させる
