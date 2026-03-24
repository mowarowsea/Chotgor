"""
backend.api.chat のエンドポイント統合テスト。

FastAPI TestClient を使い、モック化した app.state (sqlite / chat_service / memory_manager)
で各エンドポイントの HTTP 振る舞いを検証する。
LLM への実際のネットワーク呼び出しは一切行わない。
"""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import chat as chat_module


# ---------------------------------------------------------------------------
# テスト用 FastAPI アプリのセットアップ
# ---------------------------------------------------------------------------

def _make_app(sqlite_mock, chat_service_mock=None, memory_manager_mock=None) -> FastAPI:
    """モック state を持つ最小 FastAPI アプリを生成するファクトリ。"""
    app = FastAPI()
    app.include_router(chat_module.router)
    app.state.sqlite = sqlite_mock
    app.state.chat_service = chat_service_mock or MagicMock()
    app.state.memory_manager = memory_manager_mock or MagicMock()
    app.state.chroma = MagicMock()
    app.state.uploads_dir = "/tmp"
    return app


def _fake_session(sid=None, model_id="alice@gemini", title="新しいチャット"):
    """ChatSession 風の MagicMock を返すヘルパー。"""
    s = MagicMock()
    s.id = sid or str(uuid.uuid4())
    s.model_id = model_id
    s.title = title
    s.session_type = "1on1"
    s.group_config = None
    s.created_at = datetime(2026, 3, 10, 12, 0, 0)
    s.updated_at = datetime(2026, 3, 10, 12, 0, 0)
    return s


def _fake_message(mid=None, session_id="sid", role="user", content="hello"):
    """ChatMessage 風の MagicMock を返すヘルパー。"""
    m = MagicMock()
    m.id = mid or str(uuid.uuid4())
    m.session_id = session_id
    m.role = role
    m.content = content
    m.reasoning = None
    m.images = None
    m.character_name = None
    m.preset_name = None
    m.created_at = datetime(2026, 3, 10, 12, 0, 0)
    return m


# ---------------------------------------------------------------------------
# GET /api/chat/settings/user-name
# ---------------------------------------------------------------------------

class TestGetUserName:
    """ユーザ名取得エンドポイントのテスト。"""

    def test_returns_username_from_settings(self):
        """設定に保存されたユーザ名を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_setting.return_value = "テストユーザ"
        client = TestClient(_make_app(sqlite))

        res = client.get("/api/chat/settings/user-name")
        assert res.status_code == 200
        assert res.json() == {"user_name": "テストユーザ"}

    def test_defaults_to_user_if_not_set(self):
        """設定がない場合はデフォルト「ユーザ」を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_setting.return_value = "ユーザ"
        client = TestClient(_make_app(sqlite))

        res = client.get("/api/chat/settings/user-name")
        assert res.json()["user_name"] == "ユーザ"


# ---------------------------------------------------------------------------
# GET /api/chat/sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    """セッション一覧エンドポイントのテスト。"""

    def test_returns_empty_list_when_no_sessions(self):
        """セッションがない場合は空リストを返すこと。"""
        sqlite = MagicMock()
        sqlite.list_chat_sessions.return_value = []
        client = TestClient(_make_app(sqlite))

        res = client.get("/api/chat/sessions")
        assert res.status_code == 200
        assert res.json() == []

    def test_returns_session_list(self):
        """セッション一覧を正しい形式で返すこと。"""
        s = _fake_session(sid="sess-1", model_id="alice@gemini", title="朝の会話")
        sqlite = MagicMock()
        sqlite.list_chat_sessions.return_value = [s]
        client = TestClient(_make_app(sqlite))

        res = client.get("/api/chat/sessions")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["id"] == "sess-1"
        assert data[0]["model_id"] == "alice@gemini"
        assert data[0]["title"] == "朝の会話"


# ---------------------------------------------------------------------------
# POST /api/chat/sessions
# ---------------------------------------------------------------------------

class TestCreateSession:
    """セッション作成エンドポイントのテスト。"""

    def test_creates_session_returns_201(self):
        """セッション作成に成功し HTTP 201 を返すこと。"""
        s = _fake_session(model_id="bob@anthropic", title="新しいチャット")
        sqlite = MagicMock()
        sqlite.create_chat_session.return_value = s
        client = TestClient(_make_app(sqlite))

        res = client.post("/api/chat/sessions", json={"model_id": "bob@anthropic"})
        assert res.status_code == 201
        assert res.json()["model_id"] == "bob@anthropic"

    def test_custom_title_is_used(self):
        """カスタムタイトルを指定した場合に反映されること。"""
        s = _fake_session(title="夜の雑談")
        sqlite = MagicMock()
        sqlite.create_chat_session.return_value = s
        client = TestClient(_make_app(sqlite))

        res = client.post("/api/chat/sessions", json={"model_id": "alice@gemini", "title": "夜の雑談"})
        assert res.status_code == 201
        assert res.json()["title"] == "夜の雑談"


# ---------------------------------------------------------------------------
# GET /api/chat/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestGetSession:
    """セッション詳細取得エンドポイントのテスト。"""

    def test_returns_session_with_messages(self):
        """セッションとそのメッセージ一覧を返すこと。"""
        sid = "sess-1"
        s = _fake_session(sid=sid)
        msgs = [
            _fake_message(session_id=sid, role="user", content="hi"),
            _fake_message(session_id=sid, role="character", content="hello"),
        ]
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = s
        sqlite.list_chat_messages.return_value = msgs
        client = TestClient(_make_app(sqlite))

        res = client.get(f"/api/chat/sessions/{sid}")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == sid
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "character"

    def test_returns_404_when_not_found(self):
        """存在しないセッションIDは 404 を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = None
        client = TestClient(_make_app(sqlite))

        res = client.get("/api/chat/sessions/no-such-id")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/chat/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestUpdateSession:
    """セッションタイトル更新エンドポイントのテスト。"""

    def test_updates_title(self):
        """タイトルを更新してセッション情報を返すこと。"""
        s = _fake_session(sid="sess-1", title="新しいタイトル")
        sqlite = MagicMock()
        sqlite.update_chat_session.return_value = s
        client = TestClient(_make_app(sqlite))

        res = client.patch("/api/chat/sessions/sess-1", json={"title": "新しいタイトル"})
        assert res.status_code == 200
        assert res.json()["title"] == "新しいタイトル"

    def test_returns_404_when_not_found(self):
        """存在しないセッションの更新は 404 を返すこと。"""
        sqlite = MagicMock()
        sqlite.update_chat_session.return_value = None
        client = TestClient(_make_app(sqlite))

        res = client.patch("/api/chat/sessions/no-such-id", json={"title": "x"})
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/chat/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSession:
    """セッション削除エンドポイントのテスト。"""

    def test_deletes_session_returns_204(self):
        """セッション削除に成功し HTTP 204 を返すこと。"""
        sqlite = MagicMock()
        sqlite.delete_chat_session.return_value = True
        client = TestClient(_make_app(sqlite))

        res = client.delete("/api/chat/sessions/sess-1")
        assert res.status_code == 204

    def test_returns_404_when_not_found(self):
        """存在しないセッションの削除は 404 を返すこと。"""
        sqlite = MagicMock()
        sqlite.delete_chat_session.return_value = False
        client = TestClient(_make_app(sqlite))

        res = client.delete("/api/chat/sessions/no-such-id")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/chat/sessions/{session_id}/messages/stream  (SSEストリーミング)
# ---------------------------------------------------------------------------

class TestStreamMessage:
    """SSEストリーミングエンドポイントのテスト。"""

    def _make_sqlite_for_stream(self, sid):
        """ストリーミングテスト用のモック sqlite を構築する内部ヘルパー。

        キャラクター・プリセット・設定の取得をすべてモックし、
        実際の DB アクセスが発生しないようにする。
        """
        session = _fake_session(sid=sid, model_id="alice@gemini")
        user_msg = _fake_message(session_id=sid, role="user", content="こんにちは")
        char_msg = _fake_message(session_id=sid, role="character", content="やあ！")

        character = MagicMock()
        character.id = "char-1"
        character.name = "alice"
        character.system_prompt_block1 = ""
        character.meta_instructions = ""
        character.enabled_providers = {"preset-1": {"additional_instructions": ""}}

        preset = MagicMock()
        preset.id = "preset-1"
        preset.provider = "anthropic"
        preset.model_id = "claude-sonnet-4-6"
        preset.thinking_level = "default"

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.create_chat_message.side_effect = [user_msg, char_msg]
        sqlite.list_chat_messages.return_value = [user_msg]
        sqlite.get_character_by_name.return_value = character
        sqlite.get_character.return_value = character
        sqlite.get_model_preset_by_name.return_value = preset
        sqlite.get_model_preset.return_value = preset
        sqlite.get_all_settings.return_value = {"enable_time_awareness": "false"}
        sqlite.get_chat_session.return_value = session
        sqlite.update_chat_session.return_value = session
        sqlite.set_setting.return_value = None

        return sqlite, user_msg, char_msg

    def test_stream_returns_text_event_stream_content_type(self):
        """Content-Type が text/event-stream であること。"""
        sid = str(uuid.uuid4())
        sqlite, _, _ = self._make_sqlite_for_stream(sid)

        # execute_stream() は型付きタプルをyieldする
        async def fake_stream(_request):
            yield ("text", "chunk1")
            yield ("text", "chunk2")

        chat_service = MagicMock()
        chat_service.execute_stream = fake_stream

        memory_manager = MagicMock()
        memory_manager.recall_memory.return_value = []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("backend.core.debug_logger.ChotgorLogger.log_front_output", lambda *_: None)
            client = TestClient(_make_app(sqlite, chat_service, memory_manager))
            res = client.post(
                f"/api/chat/sessions/{sid}/messages/stream",
                json={"content": "こんにちは"},
            )

        assert "text/event-stream" in res.headers.get("content-type", "")

    def test_stream_returns_404_for_missing_session(self):
        """存在しないセッションへのストリーミング送信は 404 を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = None
        client = TestClient(_make_app(sqlite))

        res = client.post(
            "/api/chat/sessions/no-such-id/messages/stream",
            json={"content": "test"},
        )
        assert res.status_code == 404

    def test_stream_emits_chunk_and_done_events(self):
        """SSEストリームがchunkイベントとdoneイベントを含むこと。"""
        sid = str(uuid.uuid4())
        sqlite, user_msg, char_msg = self._make_sqlite_for_stream(sid)

        # execute_stream() は型付きタプルをyieldする
        async def fake_stream(_request):
            yield ("text", "やあ！")

        chat_service = MagicMock()
        chat_service.execute_stream = fake_stream

        memory_manager = MagicMock()
        memory_manager.recall_memory.return_value = []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("backend.core.debug_logger.ChotgorLogger.log_front_output", lambda *_: None)
            client = TestClient(_make_app(sqlite, chat_service, memory_manager))
            res = client.post(
                f"/api/chat/sessions/{sid}/messages/stream",
                json={"content": "こんにちは"},
            )

        body = res.text
        events = []
        for line in body.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        types = [e["type"] for e in events]
        assert "chunk" in types
        assert "done" in types

        done_event = next(e for e in events if e["type"] == "done")
        assert "user_message" in done_event
        assert "character_message" in done_event

    def _make_sqlite_for_auto_title(self, sid, session_title="新しいチャット", pre_existing_messages=None):
        """自動タイトルテスト専用のモック sqlite を構築する内部ヘルパー。

        create_chat_message が渡された message_id を id として持つオブジェクトを返すよう
        side_effect を設定する。これにより、エンドポイント内の
        ``history = [m for m in history_before if m.id != user_msg_id]``
        のフィルタが正しく機能し、初回メッセージ時に history が空になる。

        :param sid: セッション ID
        :param session_title: セッションの初期タイトル
        :param pre_existing_messages: 事前に存在するメッセージのリスト（2回目以降の検証用）
        """
        session = _fake_session(sid=sid, model_id="alice@gemini", title=session_title)

        # 保存済みメッセージを追跡するリスト（pre_existing_messages で初期化）
        saved_messages = list(pre_existing_messages or [])

        def fake_create_message(message_id, session_id, role, content, **kwargs):
            """呼び出し時に渡された message_id を id として持つメッセージモックを返す。"""
            msg = _fake_message(mid=message_id, session_id=session_id, role=role, content=content)
            saved_messages.append(msg)
            return msg

        def fake_list_messages(session_id):
            """create_chat_message で保存されたメッセージをそのまま返す。"""
            return list(saved_messages)

        character = MagicMock()
        character.id = "char-1"
        character.name = "alice"
        character.system_prompt_block1 = ""
        character.meta_instructions = ""
        character.enabled_providers = {"gemini": {"additional_instructions": ""}}

        preset = MagicMock()
        preset.id = "preset-1"
        preset.provider = "anthropic"
        preset.model_id = "claude-sonnet-4-6"
        preset.thinking_level = "default"

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.create_chat_message.side_effect = fake_create_message
        sqlite.list_chat_messages.side_effect = fake_list_messages
        sqlite.get_character_by_name.return_value = character
        sqlite.get_character.return_value = character
        sqlite.get_model_preset_by_name.return_value = preset
        sqlite.get_model_preset.return_value = preset
        sqlite.get_all_settings.return_value = {"enable_time_awareness": "false"}
        sqlite.update_chat_session.return_value = session
        sqlite.set_setting.return_value = None

        return sqlite

    def _run_stream(self, sid, content, sqlite):
        """ストリーミングエンドポイントを呼び出す共通ヘルパー。"""
        async def fake_stream(_request):
            yield ("text", "応答")

        chat_service = MagicMock()
        chat_service.execute_stream = fake_stream
        memory_manager = MagicMock()
        memory_manager.recall_memory.return_value = []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("backend.core.debug_logger.ChotgorLogger.log_front_output", lambda *_: None)
            client = TestClient(_make_app(sqlite, chat_service, memory_manager))
            client.post(
                f"/api/chat/sessions/{sid}/messages/stream",
                json={"content": content},
            )

    def test_auto_title_set_on_first_message(self):
        """初回メッセージ送信時にセッションタイトルが自動設定されること。

        バグ修正確認: sse_generator 完了後の update_chat_session が
        古い session.title（"新しいチャット"）で上書きせず、
        effective_title（auto_title）を使うこと。
        修正前は session オブジェクトのキャプチャにより "新しいチャット" が
        最終更新で再セットされていた。
        """
        sid = str(uuid.uuid4())
        sqlite = self._make_sqlite_for_auto_title(sid)

        self._run_stream(sid, "おはよう、今日もよろしく！", sqlite)

        all_calls = sqlite.update_chat_session.call_args_list
        assert len(all_calls) >= 1

        # どの呼び出しにも "新しいチャット" が title として渡されていないこと
        for call in all_calls:
            title = call.kwargs.get("title")
            if title is not None:
                assert title != "新しいチャット", (
                    f"update_chat_session に古い title が渡された: {call}"
                )

    def test_auto_title_truncated_to_30_chars(self):
        """30文字を超えるメッセージは先頭30文字でタイトルが設定されること。"""
        sid = str(uuid.uuid4())
        sqlite = self._make_sqlite_for_auto_title(sid)

        long_message = "あ" * 50  # 50文字のメッセージ
        self._run_stream(sid, long_message, sqlite)

        # 自動タイトル設定呼び出し（最初の update_chat_session）を確認する
        first_call = sqlite.update_chat_session.call_args_list[0]
        title = first_call.kwargs.get("title")
        assert title is not None
        assert len(title) == 30
        assert title == "あ" * 30

    def test_auto_title_replaces_newlines(self):
        """メッセージ中の改行はスペースに置換されてタイトルに設定されること。"""
        sid = str(uuid.uuid4())
        sqlite = self._make_sqlite_for_auto_title(sid)

        self._run_stream(sid, "一行目\n二行目\n三行目", sqlite)

        first_call = sqlite.update_chat_session.call_args_list[0]
        title = first_call.kwargs.get("title")
        assert title is not None
        assert "\n" not in title
        assert title == "一行目 二行目 三行目"

    def test_auto_title_not_set_on_second_message(self):
        """2回目以降のメッセージ送信ではタイトルが変更されないこと。

        既にタイトルが設定済みのセッション（history に既存メッセージあり）では
        自動タイトルのロジックが発動せず、既存タイトルが保持されることを確認する。
        """
        sid = str(uuid.uuid4())
        # 既存メッセージ1件（キャラクター応答）を事前にセットしておく
        existing_msg = _fake_message(
            mid="existing-char-msg",
            session_id=sid,
            role="character",
            content="前の返事",
        )
        sqlite = self._make_sqlite_for_auto_title(
            sid,
            session_title="昨日の話の続き",
            pre_existing_messages=[existing_msg],
        )

        self._run_stream(sid, "続きの話題を始めよう", sqlite)

        # 全 update_chat_session 呼び出しで title が既存タイトルのまま保たれること
        for call in sqlite.update_chat_session.call_args_list:
            title = call.kwargs.get("title")
            if title is not None:
                assert title == "昨日の話の続き", (
                    f"2回目のメッセージでタイトルが変わってしまった: {call}"
                )


# ---------------------------------------------------------------------------
# DELETE /api/chat/sessions/{session_id}/messages/from/{message_id}
# ---------------------------------------------------------------------------

class TestDeleteMessagesFrom:
    """指定メッセージ以降削除エンドポイントのテスト。

    ユーザメッセージ編集・キャラクター応答再生成の前処理として呼ばれる
    DELETE エンドポイントの正常系・異常系を検証する。
    """

    def test_returns_204_on_success(self):
        """削除成功時に HTTP 204 を返すこと。"""
        session = _fake_session(sid="sess-1")
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.delete_chat_messages_from.return_value = True
        client = TestClient(_make_app(sqlite))

        res = client.delete("/api/chat/sessions/sess-1/messages/from/msg-1")

        assert res.status_code == 204

    def test_calls_delete_with_correct_args(self):
        """sqlite.delete_chat_messages_from が正しい引数で呼ばれること。"""
        session = _fake_session(sid="sess-1")
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.delete_chat_messages_from.return_value = True
        client = TestClient(_make_app(sqlite))

        client.delete("/api/chat/sessions/sess-1/messages/from/msg-abc")

        sqlite.delete_chat_messages_from.assert_called_once_with("sess-1", "msg-abc")

    def test_returns_404_when_session_not_found(self):
        """セッションが存在しない場合は 404 を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = None
        client = TestClient(_make_app(sqlite))

        res = client.delete("/api/chat/sessions/no-such-sess/messages/from/msg-1")

        assert res.status_code == 404
        # セッションが存在しない場合は delete_chat_messages_from を呼ばないこと
        sqlite.delete_chat_messages_from.assert_not_called()

    def test_returns_404_when_message_not_found(self):
        """メッセージが存在しない場合は 404 を返すこと。"""
        session = _fake_session(sid="sess-1")
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.delete_chat_messages_from.return_value = False
        client = TestClient(_make_app(sqlite))

        res = client.delete("/api/chat/sessions/sess-1/messages/from/no-such-msg")

        assert res.status_code == 404
