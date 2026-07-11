"""Messenger（reach_out / visit_user）のテスト — 本人発のプッシュ連絡・対面切替。

検証対象（2026-07-11 要件 ①②）:
    1. reach_out（うつつ経路 origin="usual"）:
       - 新規セッション＋キャラ発メッセージが作られる（行動権 push と同一経路）
       - 日次カウンタ（escrow_delivery_count_{date}）を消費する（預かり配達と予算共有）
       - うつつポーズ要求キー（usual_push_pause_{character_id}）が立つ
         （sent_at / resume_at=15分後 / visit の3点セット）
       - visit=true で characters.face_to_face_mode が ON になる
    2. reach_out の実行ガード:
       - うつつ以外（origin="real"）からはエラー文字列を返し、何も作られない
       - 日次上限（escrow_delivery_daily_cap）到達時はエラー文字列を返す
       - message 空はエラー文字列
    3. visit_user（1on1）:
       - face_to_face_mode が OFF→ON になる
       - すでに ON なら「すでに対面」の案内だけ返し、状態は変えない
    4. ポーズ要求ヘルパ（read_push_pause / clear_push_pause）の往復
"""

import uuid
from datetime import datetime

from backend.character_actions.messenger import (
    PUSH_PAUSE_MINUTES,
    Messenger,
    clear_push_pause,
    delivery_cap_reached,
    read_push_pause,
)


def _make_character(sqlite_store, name="はるテスト", **kwargs):
    """テスト用キャラクター＋プリセット（ghost_model）を作成して返すヘルパ。"""
    preset_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "anthropic", "claude-x")
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name, ghost_model=preset_id)
    if kwargs:
        sqlite_store.update_character(char_id, **kwargs)
    return char_id, name


def test_reach_out_usual_creates_session_and_pause(sqlite_store):
    """うつつ経路の reach_out で、新規セッション・キャラ発メッセージ・
    日次カウンタ消費・ポーズ要求キーの4点がすべて揃うこと。"""
    char_id, name = _make_character(sqlite_store)
    messenger = Messenger(char_id, sqlite_store, default_origin="usual")

    before = datetime.now()
    result = messenger.reach_out("おーい、生きてる？")

    assert "送った" in result
    # 新規セッション＋キャラ発メッセージ（行動権 push と同一経路）
    sessions = sqlite_store.list_chat_sessions()
    assert any(s.title == f"{name}より" for s in sessions)
    push_session = next(s for s in sessions if s.title == f"{name}より")
    messages = sqlite_store.list_chat_messages(push_session.id)
    assert len(messages) == 1
    assert messages[0].role == "character"
    assert "おーい、生きてる？" in messages[0].content
    # 日次カウンタ消費（預かり配達と共有の予算。get_setting は数値文字列を int に戻す）
    today = datetime.now().date().isoformat()
    assert int(sqlite_store.get_setting(f"escrow_delivery_count_{today}", "0")) == 1
    # ポーズ要求キー（15分後再開）
    pause = read_push_pause(sqlite_store, char_id)
    assert pause is not None
    assert pause["visit"] is False
    sent_at = datetime.fromisoformat(pause["sent_at"])
    resume_at = datetime.fromisoformat(pause["resume_at"])
    assert sent_at >= before
    assert (resume_at - sent_at).total_seconds() == PUSH_PAUSE_MINUTES * 60


def test_reach_out_visit_turns_face_to_face_on(sqlite_store):
    """visit=true の reach_out で対面モードが ON になり、ポーズ要求にも visit が残ること。"""
    char_id, _ = _make_character(sqlite_store)
    messenger = Messenger(char_id, sqlite_store, default_origin="usual")

    result = messenger.reach_out("会いに行くわ", visit=True)

    assert "会いに行く" in result
    char = sqlite_store.get_character(char_id)
    assert int(char.face_to_face_mode) == 1
    pause = read_push_pause(sqlite_store, char_id)
    assert pause is not None and pause["visit"] is True


def test_reach_out_rejected_outside_usual(sqlite_store):
    """うつつ以外（origin="real"）からの reach_out はエラー文字列を返し、
    セッションもポーズ要求も作られないこと（うつつ専用ツールの実行側ガード）。"""
    char_id, _ = _make_character(sqlite_store)
    messenger = Messenger(char_id, sqlite_store, default_origin="real")

    result = messenger.reach_out("これは届かないはず")

    assert result.startswith("[reach_out error")
    assert sqlite_store.list_chat_sessions() == []
    assert read_push_pause(sqlite_store, char_id) is None


def test_reach_out_rejected_when_cap_reached(sqlite_store):
    """日次上限（escrow_delivery_daily_cap）到達時、reach_out はエラー文字列を返すこと。"""
    char_id, _ = _make_character(sqlite_store)
    today = datetime.now().date().isoformat()
    sqlite_store.set_setting("escrow_delivery_daily_cap", "2")
    sqlite_store.set_setting(f"escrow_delivery_count_{today}", "2")
    assert delivery_cap_reached(sqlite_store) is True

    messenger = Messenger(char_id, sqlite_store, default_origin="usual")
    result = messenger.reach_out("上限越えの連絡")

    assert result.startswith("[reach_out error")
    assert sqlite_store.list_chat_sessions() == []


def test_reach_out_rejected_when_message_empty(sqlite_store):
    """message 空の reach_out はエラー文字列を返すこと（空プッシュ事故の防止）。"""
    char_id, _ = _make_character(sqlite_store)
    messenger = Messenger(char_id, sqlite_store, default_origin="usual")
    assert messenger.reach_out("   ").startswith("[reach_out")


def test_visit_user_turns_face_to_face_on(sqlite_store):
    """visit_user で対面モードが OFF→ON になること（1on1「突然会いに来た」）。"""
    char_id, _ = _make_character(sqlite_store)
    messenger = Messenger(char_id, sqlite_store, default_origin="real")

    result = messenger.visit_user("チャットめんどいから")

    assert "対面モードに切り替えた" in result
    assert int(sqlite_store.get_character(char_id).face_to_face_mode) == 1


def test_visit_user_when_already_on(sqlite_store):
    """すでに対面中の visit_user は状態を変えず案内だけ返すこと。"""
    char_id, _ = _make_character(sqlite_store, face_to_face_mode=1)
    messenger = Messenger(char_id, sqlite_store, default_origin="real")

    result = messenger.visit_user()

    assert "すでに対面" in result
    assert int(sqlite_store.get_character(char_id).face_to_face_mode) == 1


def test_push_pause_roundtrip_and_clear(sqlite_store):
    """ポーズ要求の read / clear ヘルパが往復すること（壊れた値は None）。"""
    char_id, _ = _make_character(sqlite_store)
    assert read_push_pause(sqlite_store, char_id) is None

    messenger = Messenger(char_id, sqlite_store, default_origin="usual")
    messenger.reach_out("往復テスト")
    assert read_push_pause(sqlite_store, char_id) is not None

    clear_push_pause(sqlite_store, char_id)
    assert read_push_pause(sqlite_store, char_id) is None

    # 壊れた JSON は None に倒れる（永久ポーズはスケジューラ側で破棄される前提）
    from backend.character_actions.messenger import push_pause_key
    sqlite_store.set_setting(push_pause_key(char_id), "{壊れてる")
    assert read_push_pause(sqlite_store, char_id) is None
