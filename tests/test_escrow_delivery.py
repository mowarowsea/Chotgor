"""預かり（escrow）メッセージの能動配達スケジューラのテスト — めぐり §5.1 フォローアップ。

検証対象（backend/services/gate/delivery.py）:
    1. 判定フェーズ（_maybe_deliver_session 経由の run_pending_escrow_deliveries）:
       - unavailable 中は配達せず、既存の ready マーカー（復帰観測時刻）は破棄される
         （窓が閉じたら仕切り直し）
       - available を最初に観測したターンは ready マーカーを立てるだけで
         配達しない（決定論ジッター待ち — 復帰直後の機械的な即レスを避ける）
       - ジッター経過後のターンで配達され、マーカーは消費される
       - estranged / 退席済みセッションは配達対象外
       - 日次コストガード（escrow_delivery_daily_cap）到達時は配達しない
    2. 配達フェーズ（_deliver_session）:
       - LLM に渡る user_content / 履歴コピーに時間差注釈が付く
         （DB 本文は変更されない）
       - delivered_at が立ち、chat.message 封筒ができ、返信がキャラクター
         メッセージとして保存される
       - LLM 応答が空のときは返信を保存しない（配達マークは残る＝再送ループなし）
       - 走査とユーザターンのレースで pending が消えていたら何もしない

    LLM 本体（ChatFlow）と ChatRequest 構築（build_1on1_chat_request）はフェイクに
    差し替える。ジッターは monkeypatch で制御し、決定論性だけ実物で検証する。
"""

import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from backend.services.gate.delivery import (
    _delivery_jitter_seconds,
    _deliver_session,
    run_pending_escrow_deliveries,
)


class _FakeFlow:
    """ChatFlow の代役 — 固定イベント列を流すだけの execute_stream を持つ。"""

    def __init__(self, events=None):
        self.events = events if events is not None else [
            ("text", "おかえり。届いてたよ、"),
            ("text", "ぜんぶ読んだ。"),
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


def _make_char_session(sqlite_store, name="はる能動配達", **char_kwargs):
    """キャラ＋1on1 セッションを作り (char_id, char_name, session_id) を返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    if char_kwargs:
        sqlite_store.update_character(char_id, **char_kwargs)
    sid = str(uuid.uuid4())
    sqlite_store.create_chat_session(session_id=sid, model_id=f"{name}@d")
    return char_id, name, sid


def _escrow(sqlite_store, sid, content="仕事中に送った"):
    """預かり（delivered=False）のユーザメッセージを1件作るヘルパ。"""
    msg_id = str(uuid.uuid4())
    sqlite_store.create_chat_message(
        message_id=msg_id, session_id=sid, role="user",
        content=content, delivered=False,
    )
    return msg_id


@pytest.fixture
def _patched_llm(monkeypatch):
    """LLM 周辺（リクエスト構築・インデックス・計器）をフェイク化する共通フィクスチャ。

    build_1on1_chat_request が受け取った (history, user_content) を captured に
    記録するので、注釈の付き方をテスト側で検証できる。
    """
    captured = {}

    async def _fake_build(state, session, history, user_content, model_id=None):
        captured["history"] = history
        captured["user_content"] = user_content
        return SimpleNamespace(character_name="fake", character_id=None)

    import backend.api.chat as chat_module
    monkeypatch.setattr(chat_module, "build_1on1_chat_request", _fake_build)
    # インデックスと計器は本流に影響しないよう素通しにする
    monkeypatch.setattr(
        "backend.services.gate.delivery.index_message_sync",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "backend.services.instruments.tier2.record_response_smells",
        lambda *a, **kw: None,
    )
    return captured


class TestDeliveryGate:
    """判定フェーズ（availability・ジッター・ガード）を検証するテストクラス。"""

    @pytest.mark.asyncio
    async def test_unavailable_skips_and_clears_ready(self, sqlite_store, _patched_llm):
        """unavailable 中は配達せず、既存の ready マーカーは破棄される。

        復帰を観測した後・配達前に再び窓が閉じたケース（例: 復帰5分後に
        うつつシーン開始）で、古い観測時刻からジッター起算しないための仕切り直し。
        """
        _, _, sid = _make_char_session(
            sqlite_store, away_until=datetime.now() + timedelta(hours=2),
        )
        _escrow(sqlite_store, sid)
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)
        # 事前に ready マーカーが立っていた状況を再現する
        sqlite_store.set_setting(f"escrow_ready_{sid}", datetime.now().isoformat())

        await run_pending_escrow_deliveries(state)

        assert flow.calls == 0
        assert sqlite_store.get_setting(f"escrow_ready_{sid}", "") == ""
        assert len(sqlite_store.list_undelivered_messages(sid)) == 1  # 預かりは残る

    @pytest.mark.asyncio
    async def test_first_observation_waits_for_jitter(
        self, sqlite_store, _patched_llm, monkeypatch,
    ):
        """available 初観測では ready マーカーを立てるだけで配達しない（ジッター待ち）。"""
        monkeypatch.setattr(
            "backend.services.gate.delivery._delivery_jitter_seconds",
            lambda sid, ready_at: 300,
        )
        _, _, sid = _make_char_session(sqlite_store)
        _escrow(sqlite_store, sid)
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)

        now = datetime.now()
        await run_pending_escrow_deliveries(state, now=now)

        assert flow.calls == 0
        assert sqlite_store.get_setting(f"escrow_ready_{sid}", "") == now.isoformat()
        assert len(sqlite_store.list_undelivered_messages(sid)) == 1

    @pytest.mark.asyncio
    async def test_delivers_after_jitter_elapsed(
        self, sqlite_store, _patched_llm, monkeypatch,
    ):
        """ジッター経過後のターンで配達され、マーカーは消費される。"""
        monkeypatch.setattr(
            "backend.services.gate.delivery._delivery_jitter_seconds",
            lambda sid, ready_at: 300,
        )
        char_id, char_name, sid = _make_char_session(sqlite_store)
        _escrow(sqlite_store, sid)
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)

        now = datetime.now()
        await run_pending_escrow_deliveries(state, now=now)          # 観測ターン
        await run_pending_escrow_deliveries(
            state, now=now + timedelta(seconds=301),                 # ジッター経過後
        )

        assert flow.calls == 1
        assert sqlite_store.get_setting(f"escrow_ready_{sid}", "") == ""
        assert sqlite_store.list_undelivered_messages(sid) == []
        # 返信がキャラクターメッセージとして保存されている
        msgs = sqlite_store.list_chat_messages(sid)
        char_msgs = [m for m in msgs if m.role == "character"]
        assert len(char_msgs) == 1
        assert char_msgs[0].content == "おかえり。届いてたよ、ぜんぶ読んだ。"
        assert char_msgs[0].character_name == char_name

    @pytest.mark.asyncio
    async def test_estranged_and_exited_skipped(
        self, sqlite_store, _patched_llm, monkeypatch,
    ):
        """estranged キャラ・退席済みセッションは能動配達の対象外。"""
        monkeypatch.setattr(
            "backend.services.gate.delivery._delivery_jitter_seconds",
            lambda sid, ready_at: 0,
        )
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)
        # estranged キャラ
        _, _, sid1 = _make_char_session(
            sqlite_store, name="別れたキャラ", relationship_status="estranged",
        )
        _escrow(sqlite_store, sid1)
        # 退席済みセッション
        _, name2, sid2 = _make_char_session(sqlite_store, name="退席キャラ")
        _escrow(sqlite_store, sid2)
        sqlite_store.update_chat_session(
            sid2, exited_chars=[{"char_name": name2, "reason": "", "farewell_type": "neutral"}],
        )

        await run_pending_escrow_deliveries(state)
        await run_pending_escrow_deliveries(state)  # 2周しても同じ

        assert flow.calls == 0
        assert len(sqlite_store.list_undelivered_messages(sid1)) == 1
        assert len(sqlite_store.list_undelivered_messages(sid2)) == 1

    @pytest.mark.asyncio
    async def test_daily_cap_blocks_delivery(
        self, sqlite_store, _patched_llm, monkeypatch,
    ):
        """日次コストガード到達時はジッター経過後でも配達しない。"""
        monkeypatch.setattr(
            "backend.services.gate.delivery._delivery_jitter_seconds",
            lambda sid, ready_at: 0,
        )
        _, _, sid = _make_char_session(sqlite_store)
        _escrow(sqlite_store, sid)
        sqlite_store.set_setting("escrow_delivery_daily_cap", "0")
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)

        await run_pending_escrow_deliveries(state)

        assert flow.calls == 0
        assert len(sqlite_store.list_undelivered_messages(sid)) == 1

    def test_jitter_is_deterministic(self):
        """同じセッション・同じ観測時刻なら常に同じジッター（乱数は世界に置く）。"""
        ready = datetime(2026, 7, 7, 12, 0)
        a = _delivery_jitter_seconds("session-x", ready)
        b = _delivery_jitter_seconds("session-x", ready)
        assert a == b
        assert 0 <= a <= 10 * 60
        # セッションが違えば（ほぼ確実に）違う待ち時間になり得る — 範囲だけ検証
        c = _delivery_jitter_seconds("session-y", ready)
        assert 0 <= c <= 10 * 60


class TestDeliverSession:
    """配達フェーズ（注釈・封筒・保存・縮退）を検証するテストクラス。"""

    @pytest.mark.asyncio
    async def test_annotation_on_llm_copy_not_db(self, sqlite_store, _patched_llm):
        """LLM に渡るコピーにだけ時間差注釈が付き、DB 本文は変更されない。

        預かり2件のうち、最後の1件が「今回のユーザ発話」（user_content）、
        先行分は履歴側で注釈付きになる。
        """
        char_id, _, sid = _make_char_session(sqlite_store)
        _escrow(sqlite_store, sid, content="1通目")
        _escrow(sqlite_store, sid, content="2通目")
        state = _make_state(sqlite_store)
        session = sqlite_store.get_chat_session(sid)
        char = sqlite_store.get_character(char_id)

        await _deliver_session(state, session, char)

        # user_content = 最後の預かり分（注釈付き）
        assert "席を外している間に届いていた" in _patched_llm["user_content"]
        assert _patched_llm["user_content"].endswith("2通目")
        # 履歴側の先行預かり分も注釈付きコピー
        annotated = [
            m for m in _patched_llm["history"]
            if "席を外している間に届いていた" in m.content
        ]
        assert len(annotated) == 1 and annotated[0].content.endswith("1通目")
        # DB 本文は素のまま
        db_contents = [m.content for m in sqlite_store.list_chat_messages(sid)
                       if m.role == "user"]
        assert "1通目" in db_contents and "2通目" in db_contents

    @pytest.mark.asyncio
    async def test_delivery_marks_and_creates_envelopes(self, sqlite_store, _patched_llm):
        """配達で delivered_at が立ち、chat.message 封筒（user 分＋返信分）ができる。"""
        char_id, _, sid = _make_char_session(sqlite_store)
        _escrow(sqlite_store, sid)
        state = _make_state(sqlite_store)
        session = sqlite_store.get_chat_session(sid)
        char = sqlite_store.get_character(char_id)

        await _deliver_session(state, session, char)

        assert sqlite_store.list_undelivered_messages(sid) == []
        events = sqlite_store.list_timeline_events(char_id)
        actors = sorted(e.actor for e in events if e.event_type == "chat.message")
        # 配達された user 発話 + キャラの返信の2通
        assert actors == ["character", "user"]

    @pytest.mark.asyncio
    async def test_empty_response_marks_but_saves_nothing(
        self, sqlite_store, _patched_llm,
    ):
        """LLM 応答が空なら返信は保存しない。配達マークは残る（再送ループなし）。"""
        char_id, _, sid = _make_char_session(sqlite_store)
        _escrow(sqlite_store, sid)
        state = _make_state(sqlite_store, _FakeFlow(events=[("text", "")]))
        session = sqlite_store.get_chat_session(sid)
        char = sqlite_store.get_character(char_id)

        await _deliver_session(state, session, char)

        assert sqlite_store.list_undelivered_messages(sid) == []
        msgs = sqlite_store.list_chat_messages(sid)
        assert [m for m in msgs if m.role == "character"] == []

    @pytest.mark.asyncio
    async def test_no_pending_race_is_noop(self, sqlite_store, _patched_llm):
        """走査後にユーザターンが先に配達していたら何もしない（レース安全）。"""
        char_id, _, sid = _make_char_session(sqlite_store)
        msg_id = _escrow(sqlite_store, sid)
        sqlite_store.mark_messages_delivered([msg_id])  # 先に通常経路で配達済み
        flow = _FakeFlow()
        state = _make_state(sqlite_store, flow)
        session = sqlite_store.get_chat_session(sid)
        char = sqlite_store.get_character(char_id)

        await _deliver_session(state, session, char)

        assert flow.calls == 0
        assert [m for m in sqlite_store.list_chat_messages(sid)
                if m.role == "character"] == []
