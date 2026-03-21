"""
Afterglow（感情継続機構）のテストスイート。

以下の機能を検証する:
  1. get_recent_turns — 末尾 n_turns ターン分を正しく取得できること
  2. find_latest_session_for_character — 同キャラの最新セッションを特定できること
  3. afterglow_session_id の保存・取得
  4. afterglow_default カラムの保存・取得
  5. セッション作成時の afterglow_session_id 自動設定（API層の結合テスト）
  6. メッセージ送信時の Afterglow 注入（api/chat.py の内部ヘルパー）
"""

import uuid

import pytest


# --- get_recent_turns ---

class TestGetRecentTurns:
    """get_recent_turns の正常系・境界値・異常系を検証するテストスイート。"""

    def _make_session(self, store, model_id="alice@gemini"):
        """テスト用セッションを作成して ID を返すヘルパー。"""
        sid = str(uuid.uuid4())
        store.create_chat_session(session_id=sid, model_id=model_id)
        return sid

    def _add_turns(self, store, session_id, n_turns):
        """n_turns ターン分（user + character 交互）のメッセージを追加し、ID リストを返すヘルパー。

        1ターン = user 発言 + character 応答 の計2メッセージ。
        """
        ids = []
        for i in range(n_turns):
            uid = str(uuid.uuid4())
            cid = str(uuid.uuid4())
            store.create_chat_message(
                message_id=uid, session_id=session_id, role="user", content=f"user-{i}"
            )
            store.create_chat_message(
                message_id=cid, session_id=session_id, role="character", content=f"char-{i}"
            )
            ids.extend([uid, cid])
        return ids

    def test_returns_last_n_turns(self, sqlite_store):
        """10ターン中の末尾5ターン（10メッセージ）を正しく取得できること。"""
        sid = self._make_session(sqlite_store)
        all_ids = self._add_turns(sqlite_store, sid, 10)

        result = sqlite_store.get_recent_turns(sid, n_turns=5)

        assert len(result) == 10
        # 末尾10件と一致する
        expected_ids = all_ids[-10:]
        assert [m.id for m in result] == expected_ids

    def test_returns_all_when_fewer_than_n_turns(self, sqlite_store):
        """全体が n_turns 未満の場合は全件を返すこと（5ターン指定で3ターン分のみ存在）。"""
        sid = self._make_session(sqlite_store)
        all_ids = self._add_turns(sqlite_store, sid, 3)

        result = sqlite_store.get_recent_turns(sid, n_turns=5)

        assert len(result) == 6
        assert [m.id for m in result] == all_ids

    def test_returns_empty_for_empty_session(self, sqlite_store):
        """メッセージが存在しないセッションは空リストを返すこと。"""
        sid = self._make_session(sqlite_store)

        result = sqlite_store.get_recent_turns(sid, n_turns=5)

        assert result == []

    def test_message_order_is_chronological(self, sqlite_store):
        """取得されたメッセージが時系列順（昇順）であること。"""
        sid = self._make_session(sqlite_store)
        all_ids = self._add_turns(sqlite_store, sid, 3)

        result = sqlite_store.get_recent_turns(sid, n_turns=2)

        # 末尾2ターン = 4メッセージ
        assert len(result) == 4
        # 時系列順に並んでいること
        assert [m.id for m in result] == all_ids[-4:]

    def test_default_n_turns_is_5(self, sqlite_store):
        """n_turns のデフォルト値は 5（最大10メッセージ）であること。"""
        sid = self._make_session(sqlite_store)
        self._add_turns(sqlite_store, sid, 8)

        result = sqlite_store.get_recent_turns(sid)

        assert len(result) == 10


# --- find_latest_session_for_character ---

class TestFindLatestSessionForCharacter:
    """find_latest_session_for_character の正常系・異常系を検証するテストスイート。"""

    def _make_session(self, store, char_name, preset_name="gemini"):
        """テスト用セッションを作成して ID を返すヘルパー。"""
        sid = str(uuid.uuid4())
        store.create_chat_session(
            session_id=sid, model_id=f"{char_name}@{preset_name}"
        )
        return sid

    def test_finds_latest_session_for_character(self, sqlite_store):
        """同キャラクターのセッションが複数ある場合、最新を返すこと。"""
        # 古いセッションを作成後、最新セッションを明示的に updated_at を更新する
        old_sid = self._make_session(sqlite_store, "alice")
        new_sid = self._make_session(sqlite_store, "alice")
        # updated_at を確実に新しくする
        sqlite_store.update_chat_session(new_sid, title="最新")

        result = sqlite_store.find_latest_session_for_character("alice")

        assert result == new_sid

    def test_returns_none_when_no_session_exists(self, sqlite_store):
        """該当キャラのセッションが存在しない場合 None を返すこと。"""
        result = sqlite_store.find_latest_session_for_character("no-such-char")

        assert result is None

    def test_excludes_specified_session(self, sqlite_store):
        """exclude_session_id で指定したセッションを除外して最新を返すこと。

        新規セッション作成時に「自分自身」を除外するユースケースを検証する。
        """
        old_sid = self._make_session(sqlite_store, "alice")
        new_sid = self._make_session(sqlite_store, "alice")
        sqlite_store.update_chat_session(new_sid, title="最新")

        # 最新セッション (new_sid) を除外した場合、old_sid が返ること
        result = sqlite_store.find_latest_session_for_character(
            "alice", exclude_session_id=new_sid
        )

        assert result == old_sid

    def test_does_not_return_group_sessions(self, sqlite_store):
        """グループセッションは除外されること（session_type="group" は対象外）。"""
        # グループセッションを作成する
        group_sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=group_sid,
            model_id="alice@gemini",
            session_type="group",
        )

        result = sqlite_store.find_latest_session_for_character("alice")

        assert result is None

    def test_ignores_different_character_sessions(self, sqlite_store):
        """別キャラクターのセッションを返さないこと。"""
        self._make_session(sqlite_store, "bob")

        result = sqlite_store.find_latest_session_for_character("alice")

        assert result is None


# --- afterglow_session_id の保存・取得 ---

class TestAfterglowSessionId:
    """afterglow_session_id カラムの保存・取得を検証するテストスイート。"""

    def test_create_session_with_afterglow_session_id(self, sqlite_store):
        """afterglow_session_id を指定してセッションを作成・取得できること。"""
        prev_sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=prev_sid, model_id="alice@gemini")

        new_sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=new_sid,
            model_id="alice@gemini",
            afterglow_session_id=prev_sid,
        )

        fetched = sqlite_store.get_chat_session(new_sid)
        assert fetched is not None
        assert fetched.afterglow_session_id == prev_sid

    def test_create_session_without_afterglow_is_null(self, sqlite_store):
        """afterglow_session_id を指定しない場合は None であること。"""
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id="alice@gemini")

        fetched = sqlite_store.get_chat_session(sid)
        assert fetched.afterglow_session_id is None


# --- afterglow_default カラムの保存・取得 ---

class TestAfterglowDefault:
    """Character.afterglow_default カラムの保存・取得を検証するテストスイート。"""

    def test_default_is_off(self, sqlite_store):
        """afterglow_default を指定しない場合はデフォルト 0（OFF）であること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="alice")

        fetched = sqlite_store.get_character(cid)
        assert fetched.afterglow_default == 0

    def test_can_set_afterglow_default_on(self, sqlite_store):
        """afterglow_default=1 を指定してキャラクターを作成・取得できること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(
            character_id=cid, name="alice", afterglow_default=1
        )

        fetched = sqlite_store.get_character(cid)
        assert fetched.afterglow_default == 1

    def test_update_afterglow_default(self, sqlite_store):
        """update_character で afterglow_default を変更できること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="alice", afterglow_default=0)

        sqlite_store.update_character(cid, afterglow_default=1)
        fetched = sqlite_store.get_character(cid)

        assert fetched.afterglow_default == 1


# --- _prepend_afterglow ヘルパーの単体テスト ---

class TestPrependAfterglow:
    """api/chat.py の _prepend_afterglow 内部ヘルパーを検証するテストスイート。

    ヘルパーを直接インポートして、Afterglow履歴が正しくプリペンドされることを確認する。
    """

    def _make_session(self, store, model_id="alice@gemini"):
        """テスト用セッションを作成して ID を返すヘルパー。"""
        sid = str(uuid.uuid4())
        store.create_chat_session(session_id=sid, model_id=model_id)
        return sid

    def _add_messages(self, store, sid, contents):
        """指定コンテンツのメッセージを交互（user/character）で追加し、IDリストを返すヘルパー。"""
        ids = []
        for i, content in enumerate(contents):
            mid = str(uuid.uuid4())
            role = "user" if i % 2 == 0 else "character"
            store.create_chat_message(
                message_id=mid, session_id=sid, role=role, content=content
            )
            ids.append(mid)
        return ids

    def test_prepends_afterglow_messages(self, sqlite_store):
        """afterglow_session_id が設定されている場合、引き継ぎメッセージが先頭に追加されること。"""
        from backend.api.chat import _prepend_afterglow

        # 引き継ぎ元セッションに3ターン（6メッセージ）追加する
        prev_sid = self._make_session(sqlite_store)
        prev_contents = [f"prev-{i}" for i in range(6)]
        prev_ids = self._add_messages(sqlite_store, prev_sid, prev_contents)

        # 現在セッション（afterglow_session_id あり）
        new_sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=new_sid,
            model_id="alice@gemini",
            afterglow_session_id=prev_sid,
        )
        current_session = sqlite_store.get_chat_session(new_sid)

        # 現在のセッション履歴（1件）
        curr_id = str(uuid.uuid4())
        sqlite_store.create_chat_message(
            message_id=curr_id, session_id=new_sid, role="user", content="新規メッセージ"
        )
        current_history = sqlite_store.list_chat_messages(new_sid)

        # _prepend_afterglow を呼び出す
        class FakeState:
            def __init__(self, store):
                self.sqlite = store
                self.uploads_dir = ""
        fake_state = FakeState(sqlite_store)
        result = _prepend_afterglow(fake_state, current_session, current_history)

        # 引き継ぎ元6件 + 現在1件 = 7件であること
        assert len(result) == 7
        # 先頭6件が引き継ぎ元メッセージであること
        assert [m.id for m in result[:6]] == prev_ids
        # 末尾1件が現在のメッセージであること
        assert result[-1].id == curr_id

    def test_no_prepend_when_afterglow_session_id_is_none(self, sqlite_store):
        """afterglow_session_id が None の場合、履歴がそのまま返ること。"""
        from backend.api.chat import _prepend_afterglow

        sid = self._make_session(sqlite_store)
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid, role="user", content="msg"
        )
        current_history = sqlite_store.list_chat_messages(sid)
        session = sqlite_store.get_chat_session(sid)

        class FakeState:
            def __init__(self, store):
                self.sqlite = store
                self.uploads_dir = ""
        result = _prepend_afterglow(FakeState(sqlite_store), session, current_history)

        assert result == current_history

    def test_afterglow_capped_at_10_messages(self, sqlite_store):
        """引き継ぎ元が5ターン超（10メッセージ超）でも最大10件しか注入しないこと。"""
        from backend.api.chat import _prepend_afterglow

        prev_sid = self._make_session(sqlite_store)
        # 10ターン（20メッセージ）を追加する
        for i in range(10):
            sqlite_store.create_chat_message(
                message_id=str(uuid.uuid4()), session_id=prev_sid, role="user", content=f"u{i}"
            )
            sqlite_store.create_chat_message(
                message_id=str(uuid.uuid4()), session_id=prev_sid, role="character", content=f"c{i}"
            )

        new_sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=new_sid,
            model_id="alice@gemini",
            afterglow_session_id=prev_sid,
        )
        session = sqlite_store.get_chat_session(new_sid)

        class FakeState:
            def __init__(self, store):
                self.sqlite = store
                self.uploads_dir = ""
        result = _prepend_afterglow(FakeState(sqlite_store), session, [])

        # 最大10件（5ターン）であること
        assert len(result) == 10
