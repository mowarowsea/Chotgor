"""
SQLiteStore の ChatSession / ChatMessage CRUD テスト。

fixtures の sqlite_store は conftest.py が提供するインメモリ一時DBを使用する。
各テストは独立した一時DBで動作するため、テスト間の干渉はない。
"""

import uuid

import pytest


# --- ChatSession CRUD ---

class TestChatSessionCRUD:
    """ChatSession の作成・取得・一覧・更新・削除を検証するテストスイート。"""

    def test_create_and_get(self, sqlite_store):
        """セッションを作成後、IDで取得できること。"""
        sid = str(uuid.uuid4())
        s = sqlite_store.create_chat_session(session_id=sid, model_id="alice@gemini")
        assert s.id == sid
        assert s.model_id == "alice@gemini"
        assert s.title == "新しいチャット"

        fetched = sqlite_store.get_chat_session(sid)
        assert fetched is not None
        assert fetched.id == sid

    def test_create_with_custom_title(self, sqlite_store):
        """カスタムタイトルを指定して作成できること。"""
        sid = str(uuid.uuid4())
        s = sqlite_store.create_chat_session(session_id=sid, model_id="bob@anthropic", title="朝の挨拶")
        assert s.title == "朝の挨拶"

    def test_get_nonexistent_returns_none(self, sqlite_store):
        """存在しないIDを取得した場合 None を返すこと。"""
        result = sqlite_store.get_chat_session("no-such-id")
        assert result is None

    def test_list_sessions_empty(self, sqlite_store):
        """セッションがない場合は空リストを返すこと。"""
        assert sqlite_store.list_chat_sessions() == []

    def test_list_sessions_ordered_by_updated_at_desc(self, sqlite_store):
        """list_chat_sessions は更新日時の降順で返すこと。

        全セッションを作成後、最後のセッションだけを明示的に更新することで
        updated_at の差を確実につける（同一ミリ秒での同着を防ぐ）。
        """
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for i, sid in enumerate(ids):
            sqlite_store.create_chat_session(session_id=sid, model_id=f"char{i}@gemini")

        # 最後のセッションのみ明示的に更新して updated_at を確実に最新にする
        sqlite_store.update_chat_session(ids[-1], title="最新")

        sessions = sqlite_store.list_chat_sessions()
        assert sessions[0].id == ids[-1]

    def test_update_session_title(self, sqlite_store):
        """タイトルを更新できること。"""
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id="alice@gemini")
        updated = sqlite_store.update_chat_session(sid, title="更新済みタイトル")
        assert updated is not None
        assert updated.title == "更新済みタイトル"

    def test_update_nonexistent_returns_none(self, sqlite_store):
        """存在しないセッションの更新は None を返すこと。"""
        result = sqlite_store.update_chat_session("no-such-id", title="x")
        assert result is None

    def test_delete_session(self, sqlite_store):
        """セッションを削除後、取得できなくなること。"""
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id="alice@gemini")
        ok = sqlite_store.delete_chat_session(sid)
        assert ok is True
        assert sqlite_store.get_chat_session(sid) is None

    def test_delete_nonexistent_returns_false(self, sqlite_store):
        """存在しないセッションの削除は False を返すこと。"""
        result = sqlite_store.delete_chat_session("no-such-id")
        assert result is False


# --- ChatMessage CRUD ---

class TestChatMessageCRUD:
    """ChatMessage の作成・取得・一覧を検証するテストスイート。"""

    def _make_session(self, store, model_id="alice@gemini"):
        """テスト用セッションを作成して ID を返すヘルパー。"""
        sid = str(uuid.uuid4())
        store.create_chat_session(session_id=sid, model_id=model_id)
        return sid

    def test_create_and_list_message(self, sqlite_store):
        """メッセージを作成後、セッション内の一覧に含まれること。"""
        sid = self._make_session(sqlite_store)
        mid = str(uuid.uuid4())
        msg = sqlite_store.create_chat_message(
            message_id=mid, session_id=sid, role="user", content="こんにちは"
        )
        assert msg.id == mid
        assert msg.role == "user"
        assert msg.content == "こんにちは"

        messages = sqlite_store.list_chat_messages(sid)
        assert len(messages) == 1
        assert messages[0].id == mid

    def test_list_messages_empty_for_new_session(self, sqlite_store):
        """新しいセッションはメッセージが空であること。"""
        sid = self._make_session(sqlite_store)
        assert sqlite_store.list_chat_messages(sid) == []

    def test_messages_ordered_chronologically(self, sqlite_store):
        """list_chat_messages は作成日時の昇順（古い順）で返すこと。"""
        sid = self._make_session(sqlite_store)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        roles = ["user", "character", "user"]
        contents = ["はじめまして", "よろしく", "ありがとう"]
        for mid, role, content in zip(ids, roles, contents):
            sqlite_store.create_chat_message(
                message_id=mid, session_id=sid, role=role, content=content
            )

        messages = sqlite_store.list_chat_messages(sid)
        assert len(messages) == 3
        assert [m.id for m in messages] == ids

    def test_messages_from_different_sessions_are_isolated(self, sqlite_store):
        """異なるセッションのメッセージは互いに見えないこと。"""
        sid_a = self._make_session(sqlite_store)
        sid_b = self._make_session(sqlite_store)
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid_a, role="user", content="A"
        )
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid_b, role="user", content="B"
        )

        assert len(sqlite_store.list_chat_messages(sid_a)) == 1
        assert len(sqlite_store.list_chat_messages(sid_b)) == 1

    def test_delete_session_also_deletes_messages(self, sqlite_store):
        """セッションを削除すると紐づくメッセージも削除されること。"""
        sid = self._make_session(sqlite_store)
        for _ in range(3):
            sqlite_store.create_chat_message(
                message_id=str(uuid.uuid4()), session_id=sid, role="user", content="msg"
            )
        assert len(sqlite_store.list_chat_messages(sid)) == 3

        sqlite_store.delete_chat_session(sid)
        assert sqlite_store.list_chat_messages(sid) == []

    def test_character_role_message(self, sqlite_store):
        """roleが"character"のメッセージを保存・取得できること。"""
        sid = self._make_session(sqlite_store)
        mid = str(uuid.uuid4())
        msg = sqlite_store.create_chat_message(
            message_id=mid, session_id=sid, role="character", content="はじめまして！"
        )
        assert msg.role == "character"
        fetched = sqlite_store.list_chat_messages(sid)
        assert fetched[0].role == "character"


# --- delete_chat_messages_from ---

class TestDeleteChatMessagesFrom:
    """delete_chat_messages_from のテストスイート。

    ユーザメッセージ編集・キャラクター応答再生成で使用する、指定メッセージ以降を
    バッサリ削除する機能の正常系・異常系・境界値を網羅する。
    """

    def _make_session(self, store, model_id="alice@gemini"):
        """テスト用セッションを作成して ID を返すヘルパー。"""
        sid = str(uuid.uuid4())
        store.create_chat_session(session_id=sid, model_id=model_id)
        return sid

    def _add_messages(self, store, session_id, count):
        """指定件数のユーザメッセージを追加し、メッセージIDリストを返すヘルパー。

        時刻の重複を避けるため、メッセージ間に import time.sleep を使わず
        uuid 生成順に SQLite の autoincrement 的な挿入順序で管理する。
        実際の created_at はデフォルト datetime.now() だが、
        テストでは list_chat_messages の返却順（created_at 昇順）に依存する。
        """
        ids = []
        for i in range(count):
            mid = str(uuid.uuid4())
            store.create_chat_message(
                message_id=mid,
                session_id=session_id,
                role="user" if i % 2 == 0 else "character",
                content=f"msg-{i}",
            )
            ids.append(mid)
        return ids

    def test_delete_from_first_message_removes_all(self, sqlite_store):
        """先頭メッセージを起点に削除すると全メッセージが削除されること。"""
        sid = self._make_session(sqlite_store)
        ids = self._add_messages(sqlite_store, sid, 4)

        ok = sqlite_store.delete_chat_messages_from(sid, ids[0])

        assert ok is True
        assert sqlite_store.list_chat_messages(sid) == []

    def test_delete_from_middle_removes_from_pivot_onwards(self, sqlite_store):
        """中間メッセージを起点に削除すると、それ以前のメッセージは残ること。

        メッセージを4件追加し、3件目（index=2）から削除した場合、
        先頭2件（index 0, 1）のみ残ることを検証する。
        """
        sid = self._make_session(sqlite_store)
        ids = self._add_messages(sqlite_store, sid, 4)

        ok = sqlite_store.delete_chat_messages_from(sid, ids[2])

        assert ok is True
        remaining = sqlite_store.list_chat_messages(sid)
        assert len(remaining) == 2
        assert [m.id for m in remaining] == [ids[0], ids[1]]

    def test_delete_from_last_message_removes_only_last(self, sqlite_store):
        """最後のメッセージを起点に削除すると、最後の1件のみ削除されること。"""
        sid = self._make_session(sqlite_store)
        ids = self._add_messages(sqlite_store, sid, 3)

        ok = sqlite_store.delete_chat_messages_from(sid, ids[-1])

        assert ok is True
        remaining = sqlite_store.list_chat_messages(sid)
        assert len(remaining) == 2
        assert ids[-1] not in [m.id for m in remaining]

    def test_returns_false_for_nonexistent_message_id(self, sqlite_store):
        """存在しないメッセージIDを指定した場合は False を返すこと。"""
        sid = self._make_session(sqlite_store)
        self._add_messages(sqlite_store, sid, 2)

        ok = sqlite_store.delete_chat_messages_from(sid, "no-such-message-id")

        assert ok is False

    def test_returns_false_for_message_in_different_session(self, sqlite_store):
        """別セッションに属するメッセージIDを指定した場合は False を返し、
        対象外セッションのメッセージは削除されないこと。"""
        sid_a = self._make_session(sqlite_store)
        sid_b = self._make_session(sqlite_store)
        ids_a = self._add_messages(sqlite_store, sid_a, 2)
        ids_b = self._add_messages(sqlite_store, sid_b, 2)

        # sid_b のメッセージIDを sid_a に対して指定する（クロスセッション）
        ok = sqlite_store.delete_chat_messages_from(sid_a, ids_b[0])

        assert ok is False
        # sid_b のメッセージは無傷であること
        assert len(sqlite_store.list_chat_messages(sid_b)) == 2

    def test_messages_in_other_sessions_are_unaffected(self, sqlite_store):
        """削除操作が他のセッションのメッセージに影響しないこと。"""
        sid_a = self._make_session(sqlite_store)
        sid_b = self._make_session(sqlite_store)
        ids_a = self._add_messages(sqlite_store, sid_a, 3)
        self._add_messages(sqlite_store, sid_b, 3)

        sqlite_store.delete_chat_messages_from(sid_a, ids_a[1])

        # sid_b の全メッセージは残っていること
        assert len(sqlite_store.list_chat_messages(sid_b)) == 3
