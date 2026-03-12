"""
グループチャット関連のSQLiteStore拡張テスト。

テスト対象:
  - session_type / group_config カラムを持つグループセッションの作成・取得
  - character_name カラムを持つグループメッセージの作成・取得
  - 既存の1on1セッション・メッセージ機能が破壊されていないこと（後方互換性）

fixtures の sqlite_store は conftest.py が提供するインメモリ一時DBを使用する。
"""

import json
import uuid

import pytest


# ---------------------------------------------------------------------------
# グループセッション CRUD
# ---------------------------------------------------------------------------

class TestGroupChatSession:
    """グループチャットセッションの作成・取得を検証するテストスイート。"""

    def _group_config(self, **overrides) -> str:
        """デフォルトのグループ設定JSONテキストを生成するヘルパー。"""
        config = {
            "participants": [
                {"char_name": "はる", "preset_name": "Sonnet"},
                {"char_name": "Chotgor君", "preset_name": "Gemini"},
            ],
            "director_model": "qwen2.5:latest",
            "max_auto_turns": 3,
            "turn_timeout_sec": 30,
        }
        config.update(overrides)
        return json.dumps(config, ensure_ascii=False)

    def test_create_group_session(self, sqlite_store):
        """グループセッションを session_type='group' で作成できること。"""
        sid = str(uuid.uuid4())
        cfg = self._group_config()
        session = sqlite_store.create_chat_session(
            session_id=sid,
            model_id="group",
            title="テストグループ",
            session_type="group",
            group_config=cfg,
        )
        assert session.id == sid
        assert session.model_id == "group"
        assert session.session_type == "group"
        assert session.group_config == cfg

    def test_get_group_session_preserves_config(self, sqlite_store):
        """保存したグループ設定を取得後も完全に一致すること。"""
        sid = str(uuid.uuid4())
        cfg = self._group_config()
        sqlite_store.create_chat_session(
            session_id=sid,
            model_id="group",
            session_type="group",
            group_config=cfg,
        )
        fetched = sqlite_store.get_chat_session(sid)
        assert fetched is not None
        assert fetched.group_config == cfg
        # JSONとしてデシリアライズできること
        parsed = json.loads(fetched.group_config)
        assert len(parsed["participants"]) == 2
        assert parsed["max_auto_turns"] == 3

    def test_1on1_session_has_default_type(self, sqlite_store):
        """従来の1on1セッションは session_type='1on1' がデフォルトになること（後方互換性）。"""
        sid = str(uuid.uuid4())
        session = sqlite_store.create_chat_session(
            session_id=sid,
            model_id="alice@gemini",
        )
        # session_type のデフォルト値が "1on1" であること
        assert session.session_type == "1on1"

    def test_1on1_session_group_config_is_none(self, sqlite_store):
        """従来の1on1セッションは group_config が None であること（後方互換性）。"""
        sid = str(uuid.uuid4())
        session = sqlite_store.create_chat_session(
            session_id=sid,
            model_id="alice@gemini",
        )
        assert session.group_config is None

    def test_group_and_1on1_sessions_coexist(self, sqlite_store):
        """グループセッションと1on1セッションが同一DBに共存できること。"""
        sid_1on1 = str(uuid.uuid4())
        sid_group = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid_1on1, model_id="alice@gemini")
        sqlite_store.create_chat_session(
            session_id=sid_group,
            model_id="group",
            session_type="group",
            group_config=self._group_config(),
        )

        sessions = sqlite_store.list_chat_sessions()
        session_ids = {s.id for s in sessions}
        assert sid_1on1 in session_ids
        assert sid_group in session_ids


# ---------------------------------------------------------------------------
# グループメッセージ CRUD
# ---------------------------------------------------------------------------

class TestGroupChatMessage:
    """グループチャットメッセージの character_name カラムを検証するテストスイート。"""

    def _make_session(self, store):
        """テスト用グループセッションを作成してIDを返すヘルパー。"""
        sid = str(uuid.uuid4())
        store.create_chat_session(
            session_id=sid,
            model_id="group",
            session_type="group",
        )
        return sid

    def test_create_message_with_character_name(self, sqlite_store):
        """character_name を指定してメッセージを作成・取得できること。"""
        sid = self._make_session(sqlite_store)
        mid = str(uuid.uuid4())
        msg = sqlite_store.create_chat_message(
            message_id=mid,
            session_id=sid,
            role="character",
            content="やあ！",
            character_name="はる",
        )
        assert msg.character_name == "はる"

        fetched = sqlite_store.list_chat_messages(sid)
        assert len(fetched) == 1
        assert fetched[0].character_name == "はる"

    def test_user_message_character_name_is_none(self, sqlite_store):
        """ユーザーメッセージは character_name が None であること。"""
        sid = self._make_session(sqlite_store)
        mid = str(uuid.uuid4())
        msg = sqlite_store.create_chat_message(
            message_id=mid,
            session_id=sid,
            role="user",
            content="こんにちは",
        )
        assert msg.character_name is None

    def test_multiple_characters_in_same_session(self, sqlite_store):
        """同一セッションに複数キャラクターのメッセージが混在できること。"""
        sid = self._make_session(sqlite_store)
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()),
            session_id=sid,
            role="user",
            content="みんなに質問",
        )
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()),
            session_id=sid,
            role="character",
            content="はるの返答",
            character_name="はる",
        )
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()),
            session_id=sid,
            role="character",
            content="Chotgor君の返答",
            character_name="Chotgor君",
        )

        messages = sqlite_store.list_chat_messages(sid)
        assert len(messages) == 3
        char_names = [m.character_name for m in messages if m.character_name]
        assert "はる" in char_names
        assert "Chotgor君" in char_names

    def test_1on1_message_character_name_is_none(self, sqlite_store):
        """既存の1on1チャットメッセージは character_name が None であること（後方互換性）。"""
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id="alice@gemini")
        msg = sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()),
            session_id=sid,
            role="character",
            content="こんにちは！",
        )
        assert msg.character_name is None

    def test_messages_ordered_chronologically_in_group(self, sqlite_store):
        """グループセッションのメッセージも時系列順で返ること。"""
        sid = self._make_session(sqlite_store)
        ids_and_chars = [
            (str(uuid.uuid4()), "user", None),
            (str(uuid.uuid4()), "character", "はる"),
            (str(uuid.uuid4()), "character", "Chotgor君"),
        ]
        for mid, role, char_name in ids_and_chars:
            sqlite_store.create_chat_message(
                message_id=mid,
                session_id=sid,
                role=role,
                content="テスト",
                character_name=char_name,
            )

        messages = sqlite_store.list_chat_messages(sid)
        assert [m.id for m in messages] == [i[0] for i in ids_and_chars]
