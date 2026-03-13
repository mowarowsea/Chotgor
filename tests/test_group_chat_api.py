"""
グループチャット API (api/group_chat.py) の統合テスト。

FastAPI TestClient を使い、モック化した app.state で各エンドポイントの
HTTP 振る舞いを検証する。LLMへの実際のネットワーク呼び出しは一切行わない。

テスト対象エンドポイント:
  - POST /api/group/sessions       : グループセッション作成
  - GET  /api/group/sessions/{id}  : グループセッション取得
  - POST /api/group/sessions/{id}/messages/stream : SSEストリーミング送信
"""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import group_chat as group_chat_module


# ---------------------------------------------------------------------------
# テスト用ヘルパー
# ---------------------------------------------------------------------------

def _make_app(sqlite_mock, memory_manager_mock=None) -> FastAPI:
    """モック state を持つ最小 FastAPI アプリを生成するファクトリ。"""
    app = FastAPI()
    app.include_router(group_chat_module.router)
    app.state.sqlite = sqlite_mock
    app.state.memory_manager = memory_manager_mock or MagicMock()
    app.state.chat_service = MagicMock()
    app.state.uploads_dir = "/tmp"
    return app


def _fake_group_session(
    sid=None,
    title="はる、Chotgor君 のグループチャット",
    group_config: dict | None = None,
):
    """グループセッション風 MagicMock を返すヘルパー。"""
    cfg = group_config or {
        "participants": [
            {"char_name": "はる", "preset_name": "Sonnet"},
            {"char_name": "Chotgor君", "preset_name": "Gemini"},
        ],
        "director_model_id": "はる@Sonnet",
        "max_auto_turns": 3,
        "turn_timeout_sec": 30,
    }
    s = MagicMock()
    s.id = sid or str(uuid.uuid4())
    s.model_id = "group"
    s.title = title
    s.session_type = "group"
    s.group_config = json.dumps(cfg, ensure_ascii=False)
    s.created_at = datetime(2026, 3, 11, 12, 0, 0)
    s.updated_at = datetime(2026, 3, 11, 12, 0, 0)
    return s


def _fake_message(mid=None, session_id="sid", role="user", content="hello", character_name=None):
    """ChatMessage 風 MagicMock を返すヘルパー。"""
    m = MagicMock()
    m.id = mid or str(uuid.uuid4())
    m.session_id = session_id
    m.role = role
    m.content = content
    m.character_name = character_name
    m.reasoning = None
    m.images = None
    m.created_at = datetime(2026, 3, 11, 12, 0, 0)
    return m


def _make_valid_sqlite(sid=None):
    """グループセッション作成リクエストを受け付けるモック sqlite を構築するヘルパー。

    キャラクター・プリセット・セッション作成のすべてをモック化する。
    """
    sid = sid or str(uuid.uuid4())
    session = _fake_group_session(sid=sid)

    char_haru = MagicMock()
    char_haru.id = "char-haru"
    char_haru.name = "はる"

    char_chotgor = MagicMock()
    char_chotgor.id = "char-chotgor"
    char_chotgor.name = "Chotgor君"

    preset_sonnet = MagicMock()
    preset_sonnet.id = "preset-sonnet"
    preset_sonnet.name = "Sonnet"

    preset_gemini = MagicMock()
    preset_gemini.id = "preset-gemini"
    preset_gemini.name = "Gemini"

    def get_char_by_name(name):
        return char_haru if name == "はる" else char_chotgor if name == "Chotgor君" else None

    def get_preset_by_name(name):
        return preset_sonnet if name == "Sonnet" else preset_gemini if name == "Gemini" else None

    sqlite = MagicMock()
    sqlite.get_character_by_name.side_effect = get_char_by_name
    sqlite.get_character.side_effect = get_char_by_name
    sqlite.get_model_preset_by_name.side_effect = get_preset_by_name
    sqlite.get_model_preset.side_effect = get_preset_by_name
    sqlite.create_chat_session.return_value = session
    sqlite.get_chat_session.return_value = session
    sqlite.list_chat_messages.return_value = []
    sqlite.update_chat_session.return_value = session
    return sqlite, sid


# ---------------------------------------------------------------------------
# POST /api/group/sessions
# ---------------------------------------------------------------------------

class TestCreateGroupSession:
    """グループセッション作成エンドポイントのテスト。"""

    def test_creates_session_and_returns_201(self):
        """有効なリクエストでセッションが作成され 201 が返ること。"""
        sqlite, sid = _make_valid_sqlite()
        client = TestClient(_make_app(sqlite))

        res = client.post("/api/group/sessions", json={
            "participants": [
                {"model_id": "はる@Sonnet"},
                {"model_id": "Chotgor君@Gemini"},
            ],
            "director_model_id": "はる@Sonnet",
            "max_auto_turns": 3,
        })

        assert res.status_code == 201
        data = res.json()
        assert data["session_type"] == "group"
        assert data["model_id"] == "group"

    def test_returns_400_for_single_participant(self):
        """参加者が1名のみのリクエストは 422（バリデーションエラー）を返すこと。

        Pydantic の min_length=2 バリデーションが機能していることを確認する。
        """
        sqlite, _ = _make_valid_sqlite()
        client = TestClient(_make_app(sqlite))

        res = client.post("/api/group/sessions", json={
            "participants": [{"model_id": "はる@Sonnet"}],
        })

        assert res.status_code == 422

    def test_returns_404_for_unknown_character(self):
        """存在しないキャラクターを指定した場合、404 を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_character_by_name.return_value = None
        sqlite.get_character.return_value = None
        client = TestClient(_make_app(sqlite))

        res = client.post("/api/group/sessions", json={
            "participants": [
                {"model_id": "存在しない@Preset"},
                {"model_id": "はる@Sonnet"},
            ],
            "director_model_id": "存在しない@Preset",
        })

        assert res.status_code == 404

    def test_returns_400_for_invalid_model_id_format(self):
        """'@' を含まない model_id は 400 を返すこと。"""
        sqlite = MagicMock()
        client = TestClient(_make_app(sqlite))

        res = client.post("/api/group/sessions", json={
            "participants": [
                {"model_id": "はるだけ"},
                {"model_id": "Chotgor君@Gemini"},
            ],
            "director_model_id": "はる@Sonnet",
        })

        assert res.status_code == 400

    def test_warning_when_max_auto_turns_is_5_or_more(self):
        """max_auto_turns >= 5 のとき、warning フィールドが含まれること。"""
        sqlite, _ = _make_valid_sqlite()
        client = TestClient(_make_app(sqlite))

        res = client.post("/api/group/sessions", json={
            "participants": [
                {"model_id": "はる@Sonnet"},
                {"model_id": "Chotgor君@Gemini"},
            ],
            "director_model_id": "はる@Sonnet",
            "max_auto_turns": 5,
        })

        assert res.status_code == 201
        assert "warning" in res.json()

    def test_no_warning_when_max_auto_turns_is_4(self):
        """max_auto_turns < 5 のとき、warning フィールドが含まれないこと。"""
        sqlite, _ = _make_valid_sqlite()
        client = TestClient(_make_app(sqlite))

        res = client.post("/api/group/sessions", json={
            "participants": [
                {"model_id": "はる@Sonnet"},
                {"model_id": "Chotgor君@Gemini"},
            ],
            "director_model_id": "はる@Sonnet",
            "max_auto_turns": 4,
        })

        assert res.status_code == 201
        assert "warning" not in res.json()


# ---------------------------------------------------------------------------
# GET /api/group/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestGetGroupSession:
    """グループセッション取得エンドポイントのテスト。"""

    def test_returns_session_with_messages(self):
        """セッションとメッセージ一覧が返ること。"""
        sid = str(uuid.uuid4())
        session = _fake_group_session(sid=sid)
        msg = _fake_message(session_id=sid, role="user", content="こんにちは")

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.list_chat_messages.return_value = [msg]
        client = TestClient(_make_app(sqlite))

        res = client.get(f"/api/group/sessions/{sid}")

        assert res.status_code == 200
        data = res.json()
        assert data["id"] == sid
        assert data["session_type"] == "group"
        assert len(data["messages"]) == 1

    def test_returns_404_for_unknown_session(self):
        """存在しないセッションIDは 404 を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = None
        client = TestClient(_make_app(sqlite))

        res = client.get("/api/group/sessions/no-such-id")

        assert res.status_code == 404

    def test_returns_404_for_1on1_session(self):
        """1on1セッションのIDをグループエンドポイントで指定した場合、404 を返すこと。"""
        session = MagicMock()
        session.session_type = "1on1"
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        client = TestClient(_make_app(sqlite))

        res = client.get("/api/group/sessions/some-1on1-id")

        assert res.status_code == 404

    def test_message_with_character_name_is_included(self):
        """character_name を持つメッセージのフィールドが含まれること。"""
        sid = str(uuid.uuid4())
        session = _fake_group_session(sid=sid)
        msg = _fake_message(
            session_id=sid, role="character", content="やあ！", character_name="はる"
        )

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.list_chat_messages.return_value = [msg]
        client = TestClient(_make_app(sqlite))

        res = client.get(f"/api/group/sessions/{sid}")

        msgs = res.json()["messages"]
        assert msgs[0].get("character_name") == "はる"


# ---------------------------------------------------------------------------
# POST /api/group/sessions/{id}/messages/stream
# ---------------------------------------------------------------------------

class TestStreamGroupMessage:
    """グループチャット SSE ストリーミングエンドポイントのテスト。"""

    def _make_user_msg(self, sid):
        """テスト用ユーザーメッセージモックを返すヘルパー。"""
        return _fake_message(session_id=sid, role="user", content="こんにちは")

    def test_content_type_is_event_stream(self):
        """レスポンスの Content-Type が text/event-stream であること。"""
        sid = str(uuid.uuid4())
        session = _fake_group_session(sid=sid)
        user_msg = self._make_user_msg(sid)

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.create_chat_message.return_value = user_msg
        sqlite.list_chat_messages.return_value = [user_msg]
        sqlite.update_chat_session.return_value = session
        sqlite.get_all_settings.return_value = {}

        async def fake_group_turn(*args, **kwargs):
            yield ("user_turn", {"auto_turns_used": 0})

        with patch("backend.api.group_chat.run_group_turn", side_effect=fake_group_turn):
            client = TestClient(_make_app(sqlite))
            res = client.post(
                f"/api/group/sessions/{sid}/messages/stream",
                json={"content": "こんにちは"},
            )

        assert "text/event-stream" in res.headers.get("content-type", "")

    def test_emits_user_saved_event(self):
        """SSEストリームが user_saved イベントを含むこと。"""
        sid = str(uuid.uuid4())
        session = _fake_group_session(sid=sid)
        user_msg = self._make_user_msg(sid)

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.create_chat_message.return_value = user_msg
        sqlite.list_chat_messages.return_value = [user_msg]
        sqlite.update_chat_session.return_value = session
        sqlite.get_all_settings.return_value = {}

        async def fake_group_turn(*args, **kwargs):
            yield ("user_turn", {"auto_turns_used": 0})

        with patch("backend.api.group_chat.run_group_turn", side_effect=fake_group_turn):
            client = TestClient(_make_app(sqlite))
            res = client.post(
                f"/api/group/sessions/{sid}/messages/stream",
                json={"content": "こんにちは"},
            )

        events = _parse_sse(res.text)
        types = [e["type"] for e in events]
        assert "user_saved" in types

    def test_emits_done_event(self):
        """SSEストリームが done イベントを含むこと。"""
        sid = str(uuid.uuid4())
        session = _fake_group_session(sid=sid)
        user_msg = self._make_user_msg(sid)

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.create_chat_message.return_value = user_msg
        sqlite.list_chat_messages.return_value = [user_msg]
        sqlite.update_chat_session.return_value = session
        sqlite.get_all_settings.return_value = {}

        async def fake_group_turn(*args, **kwargs):
            yield ("speaker_decided", {"speakers": ["はる"]})
            yield ("user_turn", {"auto_turns_used": 1})

        with patch("backend.api.group_chat.run_group_turn", side_effect=fake_group_turn):
            client = TestClient(_make_app(sqlite))
            res = client.post(
                f"/api/group/sessions/{sid}/messages/stream",
                json={"content": "こんにちは"},
            )

        events = _parse_sse(res.text)
        types = [e["type"] for e in events]
        assert "done" in types

    def test_emits_speaker_decided_and_character_message_events(self):
        """speaker_decided と character_message イベントが正しく含まれること。"""
        sid = str(uuid.uuid4())
        session = _fake_group_session(sid=sid)
        user_msg = self._make_user_msg(sid)
        char_msg = _fake_message(
            session_id=sid, role="character", content="やあ！", character_name="はる"
        )

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.create_chat_message.return_value = user_msg
        sqlite.list_chat_messages.return_value = [user_msg]
        sqlite.update_chat_session.return_value = session
        sqlite.get_all_settings.return_value = {}

        char_msg_dict = {
            "id": char_msg.id,
            "session_id": char_msg.session_id,
            "role": "character",
            "content": "やあ！",
            "character_name": "はる",
            "created_at": char_msg.created_at.isoformat(),
        }

        async def fake_group_turn(*args, **kwargs):
            yield ("speaker_decided", {"speakers": ["はる"]})
            yield ("character_message", {"character": "はる", "message": char_msg_dict})
            yield ("user_turn", {"auto_turns_used": 1})

        with patch("backend.api.group_chat.run_group_turn", side_effect=fake_group_turn):
            client = TestClient(_make_app(sqlite))
            res = client.post(
                f"/api/group/sessions/{sid}/messages/stream",
                json={"content": "こんにちは"},
            )

        events = _parse_sse(res.text)
        types = [e["type"] for e in events]
        assert "speaker_decided" in types
        assert "character_message" in types

        speaker_event = next(e for e in events if e["type"] == "speaker_decided")
        assert speaker_event["speakers"] == ["はる"]

        char_event = next(e for e in events if e["type"] == "character_message")
        assert char_event["character"] == "はる"
        assert char_event["message"]["content"] == "やあ！"

    def test_returns_404_for_missing_session(self):
        """存在しないセッションは 404 を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = None
        client = TestClient(_make_app(sqlite))

        res = client.post(
            "/api/group/sessions/no-such-id/messages/stream",
            json={"content": "こんにちは"},
        )

        assert res.status_code == 404

    def test_returns_404_for_1on1_session(self):
        """1on1セッションのIDをグループエンドポイントで指定した場合、404 を返すこと。"""
        session = MagicMock()
        session.session_type = "1on1"
        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        client = TestClient(_make_app(sqlite))

        res = client.post(
            "/api/group/sessions/some-1on1-id/messages/stream",
            json={"content": "こんにちは"},
        )

        assert res.status_code == 404

    def test_emits_error_event_when_group_turn_raises(self):
        """run_group_turn が例外を送出したとき、error イベントを含むこと。"""
        sid = str(uuid.uuid4())
        session = _fake_group_session(sid=sid)
        user_msg = self._make_user_msg(sid)

        sqlite = MagicMock()
        sqlite.get_chat_session.return_value = session
        sqlite.create_chat_message.return_value = user_msg
        sqlite.list_chat_messages.return_value = [user_msg]
        sqlite.update_chat_session.return_value = session
        sqlite.get_all_settings.return_value = {}

        async def failing_group_turn(*args, **kwargs):
            raise RuntimeError("テスト用エラー")
            yield  # ジェネレーターとして認識させるためのダミー

        with patch("backend.api.group_chat.run_group_turn", side_effect=failing_group_turn):
            client = TestClient(_make_app(sqlite))
            res = client.post(
                f"/api/group/sessions/{sid}/messages/stream",
                json={"content": "こんにちは"},
            )

        events = _parse_sse(res.text)
        types = [e["type"] for e in events]
        assert "error" in types


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _parse_sse(body: str) -> list[dict]:
    """SSEレスポンスボディから data: 行を解析してイベントリストを返す。

    Args:
        body: SSEレスポンスのテキストボディ。

    Returns:
        各 data イベントをパースした辞書のリスト。
    """
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events
