"""タイムライン投影（projector）のテスト — めぐり（巡り / Aliveness）Phase 1。

検証対象（docs/aliveness_plan.md §2.4〜2.5）:
    1. resolve_disclosure: 観測者×イベント×origin → 開示レベルのポリシー表
       - self は全 content / world_frame は chat.*=envelope・memory.*等=hidden・
         intent.*=hidden・action.performed(対ユーザ)=envelope
       - user_ui ダイヤル 0〜3 の累積修飾
    2. project(): hidden の除外・content の中身取得（source JOIN / payload 完結型）・
       予算（イベント数・文字数）が新しいイベント優先で効くこと
    3. format_real_contact_block: chat.message 封筒のセッション×日付集約と
       「中身を載せない」ことの確認（GM 注入ブロック）
"""

import uuid
from datetime import datetime, timedelta

import pytest

from backend.services.timeline import (
    Budget,
    format_real_contact_block,
    project,
    resolve_disclosure,
)


def _make_character(sqlite_store, name="はるテスト"):
    """テスト用キャラクターを1体作成して返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    return char_id, name


class TestResolveDisclosure:
    """観測者ポリシー表（resolve_disclosure）を検証するテストクラス。

    docs/aliveness_plan.md §2.4 の表と設計判断（GM への chat.* envelope 止め・
    intent.* hidden・action.performed 対ユーザ envelope）を1項目ずつ確認する。
    """

    def test_self_sees_everything(self):
        """self（キャラ本人）は全イベント content。"""
        for ev_type in [
            "chat.message", "scene.turn", "memory.inscribed",
            "intent.created", "night.chronicle", "action.performed",
        ]:
            assert resolve_disclosure("self", ev_type, "real") == "content"

    def test_world_frame_chat_is_envelope(self):
        """GM への chat.*（real）は封筒止め（中身は「ユーザが」の材料になるため）。"""
        assert resolve_disclosure("world_frame", "chat.message", "real") == "envelope"
        assert resolve_disclosure("world_frame", "chat.farewell", "real") == "envelope"

    def test_world_frame_scene_is_content(self):
        """GM への scene.*（usual/interlude）は content（世界を回す側の当然の材料）。"""
        assert resolve_disclosure("world_frame", "scene.turn", "usual") == "content"
        assert resolve_disclosure("world_frame", "scene.closed", "interlude") == "content"

    def test_world_frame_memory_night_intent_hidden(self):
        """GM への memory.* / night.* / intent.* は存在ごと hidden（秘密は GM にも適用）。"""
        for ev_type in [
            "memory.inscribed", "memory.carved", "night.chronicle",
            "night.forget", "intent.created", "intent.soured",
        ]:
            assert resolve_disclosure("world_frame", ev_type, "real") == "hidden"

    def test_world_frame_action_to_user_is_envelope(self):
        """action.performed は対ユーザのみ GM に envelope（因果的一貫性の担保）、他は hidden。"""
        assert resolve_disclosure(
            "world_frame", "action.performed", "real", counterpart="user"
        ) == "envelope"
        assert resolve_disclosure(
            "world_frame", "action.performed", "real", counterpart=None
        ) == "hidden"

    def test_world_frame_unknown_event_hidden(self):
        """未知の event_type は安全側（hidden）に倒す。"""
        assert resolve_disclosure("world_frame", "future.event", "real") == "hidden"

    def test_user_ui_dial_0_all_content(self):
        """ダイヤル0（全開・開発期）は全 content。"""
        for ev_type in ["chat.message", "scene.turn", "memory.inscribed", "intent.created"]:
            assert resolve_disclosure("user_ui", ev_type, "usual", user_dial=0) == "content"

    def test_user_ui_dial_1_hides_usual_life(self):
        """ダイヤル1（生活の秘匿）: scene.*（usual）だけ envelope、他は content。"""
        assert resolve_disclosure("user_ui", "scene.turn", "usual", user_dial=1) == "envelope"
        # interlude のシナリオはユーザも同席しているので秘匿対象外
        assert resolve_disclosure("user_ui", "scene.turn", "interlude", user_dial=1) == "content"
        assert resolve_disclosure("user_ui", "memory.inscribed", "usual", user_dial=1) == "content"

    def test_user_ui_dial_2_hides_inner_world(self):
        """ダイヤル2（内面の秘匿）: ＋ memory.* / intent.* / night.* が hidden。"""
        assert resolve_disclosure("user_ui", "memory.inscribed", "real", user_dial=2) == "hidden"
        assert resolve_disclosure("user_ui", "intent.created", "real", user_dial=2) == "hidden"
        assert resolve_disclosure("user_ui", "night.chronicle", "real", user_dial=2) == "hidden"
        assert resolve_disclosure("user_ui", "scene.turn", "usual", user_dial=2) == "envelope"
        assert resolve_disclosure("user_ui", "chat.message", "real", user_dial=2) == "content"

    def test_user_ui_dial_3_chat_only(self):
        """ダイヤル3（最終形）: chat.message content 以外すべて hidden。"""
        assert resolve_disclosure("user_ui", "chat.message", "real", user_dial=3) == "content"
        for ev_type in ["chat.farewell", "scene.turn", "memory.inscribed", "action.performed"]:
            assert resolve_disclosure("user_ui", ev_type, "real", user_dial=3) == "hidden"

    def test_unknown_observer_raises(self):
        """未知の観測者クラスはフェイルファスト（ValueError）。"""
        with pytest.raises(ValueError):
            resolve_disclosure("npc", "chat.message", "real")


class TestProject:
    """project() の投影動作を検証するテストクラス。

    実 SQLiteStore（dual-write 済み封筒）に対して観測者別の投影を行い、
    hidden の除外・封筒/中身の出し分け・予算の効き方を確認する。
    """

    def _seed(self, sqlite_store):
        """chat / memory イベントが混在するタイムラインを作るヘルパ。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid,
            role="user", content="おはよう",
        )
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid,
            role="character", content="おはよー", character_name=char_name,
        )
        sqlite_store.create_inscribed_memory(
            memory_id=str(uuid.uuid4()), character_id=char_id, content="朝の挨拶",
        )
        return char_id, char_name, sid

    def test_self_sees_content(self, sqlite_store):
        """self 投影は全イベント content で、chat の中身テキストが読める。"""
        char_id, _, _ = self._seed(sqlite_store)
        events = project(char_id, "self", sqlite_store)
        assert [e.event_type for e in events] == [
            "chat.message", "chat.message", "memory.inscribed",
        ]
        assert all(e.disclosure == "content" for e in events)
        assert events[0].content == "おはよう"
        assert events[2].content == "朝の挨拶"

    def test_world_frame_envelope_and_hidden(self, sqlite_store):
        """world_frame 投影は chat が封筒止め（content=None）、memory は存在ごと消える。"""
        char_id, _, _ = self._seed(sqlite_store)
        events = project(char_id, "world_frame", sqlite_store)
        assert [e.event_type for e in events] == ["chat.message", "chat.message"]
        assert all(e.disclosure == "envelope" and e.content is None for e in events)

    def test_payload_complete_content(self, sqlite_store):
        """payload 完結型（memory.carved）は payload から中身を復元する。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.carve_inner_narrative(char_id, "append", "わたしの指針")
        events = project(char_id, "self", sqlite_store, types=["memory.carved"])
        assert events[0].content == "わたしの指針"

    def test_types_namespace_filter(self, sqlite_store):
        """types の名前空間指定（"chat.*"）が memory を除外する。"""
        char_id, _, _ = self._seed(sqlite_store)
        events = project(char_id, "self", sqlite_store, types=["chat.*"])
        assert {e.event_type for e in events} == {"chat.message"}

    def test_budget_max_events_keeps_newest(self, sqlite_store):
        """イベント数予算は新しい方を優先して残す。"""
        char_id, _ = _make_character(sqlite_store)
        base = datetime(2026, 7, 1, 10, 0, 0)
        for i in range(5):
            sqlite_store.record_timeline_event(
                character_id=char_id, event_type="scene.closed",
                origin="usual", occurred_at=base + timedelta(hours=i),
                payload={"n": i},
            )
        events = project(
            char_id, "self", sqlite_store, budget=Budget(max_events=2),
        )
        assert [e.payload["n"] for e in events] == [3, 4]

    def test_budget_max_chars_degrades_to_envelope(self, sqlite_store):
        """文字数予算が尽きた古いイベントは封筒止めへ格下げされる（存在は残る）。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid,
            role="user", content="あ" * 100,
        )
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid,
            role="user", content="い" * 50,
        )
        events = project(
            char_id, "self", sqlite_store, budget=Budget(max_chars=60),
        )
        # 新しい方（50文字）だけ content、古い方（100文字）は envelope に落ちる
        assert events[0].disclosure == "envelope" and events[0].content is None
        assert events[1].disclosure == "content" and events[1].content == "い" * 50


class TestFormatRealContactBlock:
    """GM 注入用「現実の接触の記録」ブロックの整形を検証するテストクラス。

    chat.message 封筒の集約（セッション×日付 → 時間帯と発言数）と、
    中身のテキストが決して混入しないことを確認する。
    """

    def test_aggregates_and_never_leaks_content(self, sqlite_store):
        """封筒が時間帯+発言数へ集約され、発言の中身はブロックに現れない。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        secret = "ひみつのはなし"
        for i in range(3):
            sqlite_store.create_chat_message(
                message_id=str(uuid.uuid4()), session_id=sid,
                role="user" if i % 2 == 0 else "character",
                content=f"{secret}{i}",
            )
        events = project(char_id, "world_frame", sqlite_store, origins=["real"])
        block = format_real_contact_block(events, char_name, "もわ")
        assert "もわ" in block
        assert "3発言分" in block
        assert secret not in block  # 中身は絶対に漏れない

    def test_empty_when_no_events_or_label(self, sqlite_store):
        """素材ゼロ・呼称なしでは空文字列（ブロック非生成）。"""
        char_id, char_name = _make_character(sqlite_store)
        assert format_real_contact_block([], char_name, "もわ") == ""
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid, role="user", content="a",
        )
        events = project(char_id, "world_frame", sqlite_store)
        assert format_real_contact_block(events, char_name, "") == ""
