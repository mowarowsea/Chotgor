"""発話予約（speak_later）発火ランナーのテスト — docs/planned/speak_later_plan.md §②。

検証対象（backend/services/gate/speech_reservation.py）:
    1. 発火判定（run_pending_speech_reservations）:
       - 定刻到来で発火し、fired 遷移・決定ログ・日次カウンタ消費（reach_out と共有）が揃う
       - unavailable 中は発火せず pending 維持 → 復帰後に遅延発火（遅延文言の合成注釈）
       - speak_at + 24h を過ぎたら expired ＋ declined 記録（黙って消さない）
       - 日次 cap 到達日は skipped 記録（日1回だけ）→ pending 維持
       - セッション削除 / estranged / 退席済み / トグル OFF → cancelled ＋記録
    2. 発火時の生成（_deliver_session 共用）:
       - 合成注釈が最終ユーザターン相当として LLM に渡り、**DB には保存されない**
         （画面にはキャラの発話だけが増える）
       - 未配達メッセージが残っていたら escrow 配達と同じ手順で併せて配達し、
         合成注釈を末尾に添えて1ターンに併合する（2ターン発生させない）

    LLM 本体と ChatRequest 構築はフェイクに差し替える（test_escrow_delivery.py と同型）。
"""

import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from backend.services.gate.speech_reservation import (
    build_reservation_annotation,
    run_pending_speech_reservations,
)


class _FakeFlow:
    """ChatFlow の代役 — 固定イベント列を流すだけの execute_stream を持つ。"""

    def __init__(self, events=None):
        self.events = events if events is not None else [
            ("text", "約束どおり、"),
            ("text", "声をかけにきたよ。"),
        ]
        self.calls = 0

    async def execute_stream(self, request):
        """呼び出し回数を記録して固定イベントを流す。"""
        self.calls += 1
        for ev in self.events:
            yield ev


def _make_state(sqlite_store, flow=None):
    """app.state 相当のフェイクを組み立てるヘルパ。"""
    return SimpleNamespace(
        sqlite=sqlite_store,
        chat_service=flow or _FakeFlow(),
        vector_store=None,
        uploads_dir="",
    )


def _make_char_session(sqlite_store, name="はる発話予約", **char_kwargs):
    """speak_later 有効キャラ＋1on1 セッションを作るヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    kwargs = {"speak_later_enabled": 1, **char_kwargs}
    sqlite_store.update_character(char_id, **kwargs)
    sid = str(uuid.uuid4())
    sqlite_store.create_chat_session(session_id=sid, model_id=f"{name}@d")
    return char_id, name, sid


def _reserve(sqlite_store, char_id, sid, *, speak_at, note="結果を伝える"):
    """pending 予約を1件作るヘルパ。"""
    return sqlite_store.create_speech_reservation(
        character_id=char_id, session_id=sid, speak_at=speak_at, note=note,
    )


def _status_of(sqlite_store, reservation_id):
    """予約の現在 status を読むヘルパ。"""
    from backend.repositories.sqlite.models import SpeechReservation
    with sqlite_store.get_session() as session:
        return session.get(SpeechReservation, reservation_id).status


@pytest.fixture
def _patched_llm(monkeypatch):
    """LLM 周辺（リクエスト構築・インデックス・計器）をフェイク化する共通フィクスチャ。

    build_1on1_chat_request が受け取った (history, user_content) を captured に
    記録するので、合成注釈の渡り方をテスト側で検証できる。
    """
    captured = {}

    async def _fake_build(state, session, history, user_content, model_id=None):
        captured["history"] = history
        captured["user_content"] = user_content
        return SimpleNamespace(character_name="fake", character_id=None)

    import backend.api.chat as chat_module
    monkeypatch.setattr(chat_module, "build_1on1_chat_request", _fake_build)
    monkeypatch.setattr(
        "backend.services.gate.delivery.index_message_sync",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "backend.services.instruments.tier2.record_response_smells",
        lambda *a, **kw: None,
    )
    return captured


class TestReservationFiring:
    """定刻発火・遅延発火・合成注釈の内容を検証するテストクラス。"""

    @pytest.mark.asyncio
    async def test_fires_on_time(self, sqlite_store, _patched_llm):
        """定刻到来で発火: fired 遷移・キャラ発話保存・決定ログ・カウンタ消費が揃う。"""
        char_id, char_name, sid = _make_char_session(sqlite_store)
        now = datetime.now()
        r = _reserve(sqlite_store, char_id, sid, speak_at=now - timedelta(minutes=1))
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)

        await run_pending_speech_reservations(state, now=now)

        assert flow.calls == 1
        assert _status_of(sqlite_store, r.id) == "fired"
        # 定刻文言の合成注釈が最終ユーザターン相当として渡る
        assert "あなたはこの時間に『結果を伝える』をやろうとしていた" in _patched_llm["user_content"]
        # キャラ発話が保存される
        msgs = sqlite_store.list_chat_messages(sid)
        char_msgs = [m for m in msgs if m.role == "character"]
        assert len(char_msgs) == 1
        assert char_msgs[0].content == "約束どおり、声をかけにきたよ。"
        # 決定ログ（fired）と日次カウンタ（reach_out と共有）
        decisions = sqlite_store.list_scheduler_decisions(scheduler="speech_reservation")
        assert [d.outcome for d in decisions] == ["fired"]
        today = now.date().isoformat()
        assert int(sqlite_store.get_setting(f"spontaneous_initiative_count_{today}", "0") or 0) == 1

    @pytest.mark.asyncio
    async def test_annotation_wrapped_in_turn_context(self, sqlite_store, _patched_llm):
        """合成注釈が <turn_context> で包まれて渡ること。

        単独発火では合成注釈だけが最終ユーザターン相当になる。タグ無しのままだと
        claude_cli の会話整形がユーザ発言として <user_label> で包んでしまい、
        「声をかける番」という Chotgor の添え書きがユーザの発言に見える（発話者の
        取り違え対策、2026-09-23）。
        """
        char_id, _, sid = _make_char_session(sqlite_store)
        now = datetime.now()
        _reserve(sqlite_store, char_id, sid, speak_at=now - timedelta(minutes=1))
        state = _make_state(sqlite_store)

        await run_pending_speech_reservations(state, now=now)

        content = _patched_llm["user_content"]
        text = content if isinstance(content, str) else "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
        assert text.startswith("<turn_context>\n")
        assert text.rstrip().endswith("</turn_context>")

    @pytest.mark.asyncio
    async def test_annotation_not_saved_to_db(self, sqlite_store, _patched_llm):
        """合成注釈は DB に保存されない — 画面にはキャラの発話だけが増える。"""
        char_id, _, sid = _make_char_session(sqlite_store)
        now = datetime.now()
        _reserve(sqlite_store, char_id, sid, speak_at=now - timedelta(minutes=1))
        state = _make_state(sqlite_store)

        await run_pending_speech_reservations(state, now=now)

        msgs = sqlite_store.list_chat_messages(sid)
        assert [m.role for m in msgs] == ["character"]  # user 行は増えていない
        assert all("やろうとしていた" not in m.content for m in msgs)

    @pytest.mark.asyncio
    async def test_unavailable_keeps_pending_then_fires_delayed(
        self, sqlite_store, _patched_llm,
    ):
        """unavailable 中は pending 維持。復帰後に遅延文言の合成注釈で発火する。"""
        char_id, _, sid = _make_char_session(
            sqlite_store, away_until=datetime.now() + timedelta(hours=1),
        )
        now = datetime.now()
        r = _reserve(sqlite_store, char_id, sid, speak_at=now - timedelta(hours=2))
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)

        await run_pending_speech_reservations(state, now=now)
        assert flow.calls == 0
        assert _status_of(sqlite_store, r.id) == "pending"

        # away が明けた（復帰）→ 遅れて発火。遅延文言になる
        sqlite_store.update_character(char_id, away_until=None)
        await run_pending_speech_reservations(state, now=now)
        assert flow.calls == 1
        assert _status_of(sqlite_store, r.id) == "fired"
        assert "都合がつかず今になった" in _patched_llm["user_content"]

    def test_annotation_wording_on_time_vs_delayed(self):
        """合成注釈の純関数: 10分未満の遅れは定刻文言、以上は遅延文言。"""
        now = datetime(2026, 7, 20, 21, 5)
        on_time = SimpleNamespace(
            speak_at=datetime(2026, 7, 20, 21, 0), note="結果を伝える",
        )
        assert "あなたはこの時間に" in build_reservation_annotation(on_time, now)
        delayed = SimpleNamespace(
            speak_at=datetime(2026, 7, 20, 18, 0), note="結果を伝える",
        )
        text = build_reservation_annotation(delayed, now)
        assert "本当は 18:00 に" in text and "都合がつかず今になった" in text


class TestReservationGate:
    """expired・cap・cancelled 系の判定を検証するテストクラス。"""

    @pytest.mark.asyncio
    async def test_expired_after_24h(self, sqlite_store, _patched_llm):
        """speak_at + 24h を過ぎたら expired ＋ declined 記録（発火しない）。"""
        char_id, _, sid = _make_char_session(sqlite_store)
        now = datetime.now()
        r = _reserve(sqlite_store, char_id, sid, speak_at=now - timedelta(hours=25))
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)

        await run_pending_speech_reservations(state, now=now)

        assert flow.calls == 0
        assert _status_of(sqlite_store, r.id) == "expired"
        decisions = sqlite_store.list_scheduler_decisions(scheduler="speech_reservation")
        assert [d.outcome for d in decisions] == ["declined"]

    @pytest.mark.asyncio
    async def test_daily_cap_skips_and_records_once(self, sqlite_store, _patched_llm):
        """cap 到達日は発火せず pending 維持。skipped 記録は日1回だけ。"""
        char_id, _, sid = _make_char_session(sqlite_store)
        now = datetime.now()
        r = _reserve(sqlite_store, char_id, sid, speak_at=now - timedelta(minutes=1))
        sqlite_store.set_setting("spontaneous_initiative_daily_cap", "0")
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)

        await run_pending_speech_reservations(state, now=now)
        await run_pending_speech_reservations(state, now=now)  # 2周しても記録は増えない

        assert flow.calls == 0
        assert _status_of(sqlite_store, r.id) == "pending"
        decisions = sqlite_store.list_scheduler_decisions(scheduler="speech_reservation")
        assert [d.outcome for d in decisions] == ["skipped"]

    @pytest.mark.asyncio
    async def test_cancelled_cases(self, sqlite_store, _patched_llm):
        """セッション削除 / estranged / 退席済み / トグル OFF → cancelled ＋記録。"""
        now = datetime.now()
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)
        # セッション削除済み（存在しない session_id）
        char1, _, _ = _make_char_session(sqlite_store, name="消えた会話")
        r1 = _reserve(sqlite_store, char1, "no-such-session",
                      speak_at=now - timedelta(minutes=1))
        # estranged
        char2, _, sid2 = _make_char_session(
            sqlite_store, name="別れたキャラ", relationship_status="estranged",
        )
        r2 = _reserve(sqlite_store, char2, sid2, speak_at=now - timedelta(minutes=1))
        # 退席済み
        char3, name3, sid3 = _make_char_session(sqlite_store, name="退席キャラ")
        sqlite_store.update_chat_session(
            sid3,
            exited_chars=[{"char_name": name3, "reason": "", "farewell_type": "neutral"}],
        )
        r3 = _reserve(sqlite_store, char3, sid3, speak_at=now - timedelta(minutes=1))
        # 仕掛け後にトグル OFF（課金ガードなので発火側でも見る）
        char4, _, sid4 = _make_char_session(sqlite_store, name="トグルOFF")
        r4 = _reserve(sqlite_store, char4, sid4, speak_at=now - timedelta(minutes=1))
        sqlite_store.update_character(char4, speak_later_enabled=0)

        await run_pending_speech_reservations(state, now=now)

        assert flow.calls == 0
        for r in (r1, r2, r3, r4):
            assert _status_of(sqlite_store, r.id) == "cancelled"
        decisions = sqlite_store.list_scheduler_decisions(scheduler="speech_reservation")
        assert sorted(d.outcome for d in decisions) == ["skipped"] * 4


class TestReservationMerge:
    """未配達メッセージとの合流（1ターン併合）を検証するテストクラス。"""

    @pytest.mark.asyncio
    async def test_merges_with_undelivered_messages(self, sqlite_store, _patched_llm):
        """未配達分を escrow と同じ手順で併せて配達し、合成注釈を末尾に添える。

        別々に2ターン発生させない（LLM 呼び出しは1回）。
        """
        char_id, _, sid = _make_char_session(sqlite_store)
        now = datetime.now()
        # 預かり（未配達）のユーザメッセージが残っている状況
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid, role="user",
            content="席を外してる間に送った", delivered=False,
        )
        r = _reserve(sqlite_store, char_id, sid, speak_at=now - timedelta(minutes=1))
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)

        await run_pending_speech_reservations(state, now=now)

        assert flow.calls == 1  # 1ターンに併合
        assert _status_of(sqlite_store, r.id) == "fired"
        # 時間差注釈つきの預かり分の末尾に、合成注釈が添えられている
        user_content = _patched_llm["user_content"]
        assert "席を外している間に届いていた" in user_content
        assert user_content.index("やろうとしていた") > user_content.index("送った")
        # mark_messages_delivered 済み（未配達が残らない）
        assert sqlite_store.list_undelivered_messages(sid) == []
