"""speak_later ツール（LaterSpeaker・仕掛け側）のテスト。

検証対象（docs/planned/speak_later_plan.md §①）:
    1. parse_speak_at 純関数:
       - "HH:MM" は次にその時刻が来る時点（今日 or 明日）に解決される
       - 24時超え表記（"25:30" = 翌1:30）を override_schedule と同じ流儀で受ける
       - "YYYY-MM-DD HH:MM"（"T" 区切りも可）はその時刻そのまま
       - 形式不正・範囲外は None
    2. 実行側ガード（露出とのタイムラグ対策の二重ガード）:
       - トグル OFF / origin 不一致（usual）/ session なし → エラー文字列
    3. バリデーション:
       - note 空・at 不正・過去時刻・72時間 horizon 超過 → エラー文字列
       - 指定時刻が unavailable → 理由付きエラー。生活カレンダー有効キャラには
         「予定を動かせば置ける（override_schedule）」のヒントが添えられる
    4. 置き直し（superseded 遷移・pending 一意性）:
       - 同一セッションへの2回目の仕掛けで旧行が superseded になり pending は常に1件
       - 履歴は行として残る（削除しない）
"""

import uuid
from datetime import datetime, timedelta

from backend.character_actions.later_speaker import (
    LaterSpeaker,
    parse_speak_at,
)

# パース純関数用の固定基準時刻（2026-07-20 20:00 月曜）
_NOW = datetime(2026, 7, 20, 20, 0)


def _make_char_session(sqlite_store, name="はる発話予約", **char_kwargs):
    """speak_later 有効キャラ＋セッション ID を作るヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    kwargs = {"speak_later_enabled": 1, **char_kwargs}
    sqlite_store.update_character(char_id, **kwargs)
    return char_id, f"session-{uuid.uuid4()}"


def _speaker(sqlite_store, char_id, session_id, origin="real"):
    """LaterSpeaker を組み立てるヘルパ。"""
    return LaterSpeaker(char_id, session_id, sqlite_store, default_origin=origin)


class TestParseSpeakAt:
    """parse_speak_at 純関数のテスト — 時刻解釈の全パターン。"""

    def test_hhmm_resolves_to_next_occurrence_today(self):
        """20:00 時点の "21:00" は今日の 21:00 に解決される。"""
        assert parse_speak_at("21:00", _NOW) == datetime(2026, 7, 20, 21, 0)

    def test_hhmm_past_time_rolls_to_tomorrow(self):
        """20:00 時点の "19:00" は既に過ぎているため翌日の 19:00 になる。"""
        assert parse_speak_at("19:00", _NOW) == datetime(2026, 7, 21, 19, 0)

    def test_hhmm_over_24_notation(self):
        """"25:30" は翌1:30（override_schedule と同じ深夜表記）。"""
        assert parse_speak_at("25:30", _NOW) == datetime(2026, 7, 21, 1, 30)

    def test_explicit_datetime_forms(self):
        """"YYYY-MM-DD HH:MM"（スペース・"T" 区切りの両方）はその時刻そのまま。"""
        expected = datetime(2026, 7, 22, 9, 0)
        assert parse_speak_at("2026-07-22 09:00", _NOW) == expected
        assert parse_speak_at("2026-07-22T09:00", _NOW) == expected

    def test_invalid_inputs_return_none(self):
        """形式不正・範囲外・空文字は None（呼び出し側がエラー文字列にする）。"""
        assert parse_speak_at("", _NOW) is None
        assert parse_speak_at("そのうち", _NOW) is None
        assert parse_speak_at("99:00", _NOW) is None
        assert parse_speak_at("21:99", _NOW) is None


class TestSpeakLaterGuards:
    """実行側ガード（露出とのタイムラグに備えた二重ガード）のテスト。"""

    def test_toggle_off_rejected(self, sqlite_store):
        """speak_later_enabled=0 のキャラは実行側でも弾かれる。"""
        char_id, sid = _make_char_session(sqlite_store, speak_later_enabled=0)
        result = _speaker(sqlite_store, char_id, sid).speak_later("21:00", "結果を伝える")
        assert "error" in result and "有効" in result
        assert sqlite_store.list_pending_speech_reservations() == []

    def test_usual_origin_rejected(self, sqlite_store):
        """うつつ（origin=usual）からの実行は弾かれる（1on1 専用）。"""
        char_id, sid = _make_char_session(sqlite_store)
        result = _speaker(sqlite_store, char_id, sid, origin="usual").speak_later(
            "21:00", "結果を伝える",
        )
        assert "error" in result
        assert sqlite_store.list_pending_speech_reservations() == []

    def test_missing_session_rejected(self, sqlite_store):
        """session_id 無し（バッチ経路等）は弾かれる。"""
        char_id, _ = _make_char_session(sqlite_store)
        result = _speaker(sqlite_store, char_id, None).speak_later("21:00", "伝える")
        assert "error" in result


class TestSpeakLaterValidation:
    """時刻・note のバリデーションのテスト。"""

    def test_empty_note_rejected(self, sqlite_store):
        """note 空は「何を話すつもりか残して」のエラーになる。"""
        char_id, sid = _make_char_session(sqlite_store)
        result = _speaker(sqlite_store, char_id, sid).speak_later("21:00", "  ")
        assert "note" in result
        assert sqlite_store.list_pending_speech_reservations() == []

    def test_unparseable_at_rejected(self, sqlite_store):
        """at がパース不能ならエラー文字列（形式の案内付き）。"""
        char_id, sid = _make_char_session(sqlite_store)
        result = _speaker(sqlite_store, char_id, sid).speak_later("そのうち", "伝える")
        assert "error" in result and "HH:MM" in result

    def test_past_explicit_datetime_rejected(self, sqlite_store):
        """明示日付形式で過去の時刻は弾かれる。"""
        char_id, sid = _make_char_session(sqlite_store)
        result = _speaker(sqlite_store, char_id, sid).speak_later(
            "2020-01-01 10:00", "伝える",
        )
        assert "error" in result and "過去" in result

    def test_beyond_horizon_rejected(self, sqlite_store):
        """72時間 horizon を超える時刻は弾かれる。"""
        char_id, sid = _make_char_session(sqlite_store)
        far = datetime.now() + timedelta(hours=80)
        result = _speaker(sqlite_store, char_id, sid).speak_later(
            far.strftime("%Y-%m-%d %H:%M"), "伝える",
        )
        assert "error" in result and "72" in result

    def test_unavailable_time_rejected_with_reason(self, sqlite_store):
        """指定時刻が unavailable（away 中）なら理由付きエラーで、予約は作られない。

        availability の上書きはしない — 就寝中に発話→返信したら「寝てます」になる
        矛盾を仕掛け時に防ぐ（ユーザ裁定）。
        """
        char_id, sid = _make_char_session(
            sqlite_store,
            away_until=datetime.now() + timedelta(days=3),
            away_reason="旅行",
        )
        result = _speaker(sqlite_store, char_id, sid).speak_later("21:00", "伝える")
        assert "error" in result and "旅行" in result
        # 生活カレンダー無効キャラにはヒントを出さない（override_schedule が使えないため）
        assert "override_schedule" not in result
        assert sqlite_store.list_pending_speech_reservations() == []

    def test_unavailable_with_living_calendar_adds_hint(self, sqlite_store):
        """生活カレンダー有効キャラの offline 時間帯にはヒント（予定を動かせば置ける）。"""
        char_id, sid = _make_char_session(sqlite_store, living_schedule_enabled=1)
        target = (datetime.now() + timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0,
        )
        sqlite_store.create_schedule_entry(
            character_id=char_id,
            start_at=target - timedelta(hours=1),
            end_at=target + timedelta(hours=1),
            state="offline",
            source="haru",
            origin="template",
            occupancy=0.5,
            label="遠出",
        )
        result = _speaker(sqlite_store, char_id, sid).speak_later(
            target.strftime("%Y-%m-%d %H:%M"), "伝える",
        )
        assert "error" in result and "遠出" in result
        assert "override_schedule" in result


class TestSpeakLaterReservation:
    """予約 insert と置き直し（superseded）のテスト。"""

    def test_success_creates_pending(self, sqlite_store):
        """成功時は pending 予約ができ、確認文言（「予約」と言わない）が返る。"""
        char_id, sid = _make_char_session(sqlite_store)
        result = _speaker(sqlite_store, char_id, sid).speak_later(
            "21:00", "調べものの結果を伝える",
        )
        assert "心づもり" in result and "予約" not in result
        rows = sqlite_store.list_pending_speech_reservations(character_id=char_id)
        assert len(rows) == 1
        assert rows[0].session_id == sid
        assert rows[0].note == "調べものの結果を伝える"
        assert rows[0].speak_at > datetime.now()

    def test_replace_supersedes_previous_pending(self, sqlite_store):
        """置き直しで旧行が superseded になり、pending は常に1件（履歴は残る）。"""
        char_id, sid = _make_char_session(sqlite_store)
        speaker = _speaker(sqlite_store, char_id, sid)
        speaker.speak_later("21:00", "結果を伝える")
        first = sqlite_store.get_pending_speech_reservation(sid)

        result = speaker.speak_later("22:00", "やっぱり22時にする")
        assert "取り下げた" in result

        pending = sqlite_store.list_pending_speech_reservations(character_id=char_id)
        assert len(pending) == 1
        assert pending[0].note == "やっぱり22時にする"
        # 旧行は消えず superseded として残る
        from backend.repositories.sqlite.models import SpeechReservation
        with sqlite_store.get_session() as session:
            old = session.get(SpeechReservation, first.id)
            assert old is not None and old.status == "superseded"

    def test_reservations_in_other_sessions_untouched(self, sqlite_store):
        """置き直しは同一セッション内だけ — 別セッションの pending は無傷。"""
        char_id, sid1 = _make_char_session(sqlite_store)
        sid2 = f"session-{uuid.uuid4()}"
        speaker1 = _speaker(sqlite_store, char_id, sid1)
        speaker2 = _speaker(sqlite_store, char_id, sid2)
        speaker1.speak_later("21:00", "会話1の用件")
        speaker2.speak_later("21:30", "会話2の用件")
        speaker1.speak_later("22:00", "会話1を置き直し")

        pending = sqlite_store.list_pending_speech_reservations(character_id=char_id)
        notes = sorted(r.note for r in pending)
        assert notes == ["会話1を置き直し", "会話2の用件"]
