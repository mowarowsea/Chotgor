"""Tests for adapters.openai.router and adapters.openai.schemas."""

import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.adapters.openai import router as adapter_module
from backend.adapters.openai.router import _derive_session_id, _format_completion, _sse_chunk
from backend.lib.utils import format_time_delta as _format_time_delta
from backend.adapters.openai.schemas import OAIChatMessage, OAIChatRequest
from backend.services.chat.models import Message


# --- _derive_session_id ---

class TestDeriveSessionId:
    """_derive_session_id の振る舞いを検証するテストスイート。

    OpenWebUI はsession_idを送ってこないため、会話の先頭userメッセージ +
    character_idからMD5ハッシュで擬似session_idを導出する関数のテスト。
    """

    def _make_messages(self, *contents) -> list:
        """テスト用の Message リストを作成するヘルパー。(role="user" で交互に入力)"""
        return [Message(role="user", content=c) for c in contents]

    def test_returns_md5_hex_string(self):
        """返り値が32文字の16進数文字列（MD5ハッシュ）であること。"""
        messages = self._make_messages("こんにちは")
        result = _derive_session_id("char-1", messages)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_same_inputs(self):
        """同じ入力に対して常に同じIDが返ること（決定論的）。"""
        messages = self._make_messages("最初のメッセージ")
        id1 = _derive_session_id("char-1", messages)
        id2 = _derive_session_id("char-1", messages)
        assert id1 == id2

    def test_different_first_message_gives_different_id(self):
        """先頭userメッセージが異なれば別のIDになること。"""
        id1 = _derive_session_id("char-1", self._make_messages("メッセージA"))
        id2 = _derive_session_id("char-1", self._make_messages("メッセージB"))
        assert id1 != id2

    def test_different_character_gives_different_id(self):
        """同じメッセージでもcharacter_idが異なれば別のIDになること。"""
        messages = self._make_messages("同じメッセージ")
        id1 = _derive_session_id("char-1", messages)
        id2 = _derive_session_id("char-2", messages)
        assert id1 != id2

    def test_only_first_user_message_is_used(self):
        """先頭userメッセージのみがIDに使われること。

        2ターン目以降のメッセージが異なっても、先頭が同じなら同じIDになることを確認する。
        これにより同一会話スレッドが一貫したIDを持つ。
        """
        msgs_turn1 = [Message(role="user", content="最初の質問")]
        msgs_turn2 = [
            Message(role="user", content="最初の質問"),
            Message(role="assistant", content="返答"),
            Message(role="user", content="2つ目の質問"),
        ]
        id1 = _derive_session_id("char-1", msgs_turn1)
        id2 = _derive_session_id("char-1", msgs_turn2)
        assert id1 == id2

    def test_returns_empty_string_when_no_user_messages(self):
        """userメッセージが一件もない場合は空文字列を返すこと。"""
        messages = [Message(role="assistant", content="systemメッセージ")]
        result = _derive_session_id("char-1", messages)
        assert result == ""

    def test_returns_empty_string_for_empty_messages(self):
        """メッセージが空リストの場合は空文字列を返すこと。"""
        result = _derive_session_id("char-1", [])
        assert result == ""


# --- _format_time_delta ---

def test_format_time_delta_seconds():
    assert _format_time_delta(timedelta(seconds=30)) == "数分以内"


def test_format_time_delta_minutes():
    result = _format_time_delta(timedelta(minutes=45))
    assert result == "約 45 分"


def test_format_time_delta_hours():
    result = _format_time_delta(timedelta(hours=3))
    assert result == "約 3.0 時間"


def test_format_time_delta_days():
    result = _format_time_delta(timedelta(days=2, hours=3))
    assert result == "約 2 日"


def test_format_time_delta_boundary_one_hour():
    # ちょうど1時間は「時間」表記になること
    result = _format_time_delta(timedelta(hours=1))
    assert "時間" in result


def test_format_time_delta_boundary_one_day():
    # ちょうど24時間は「日」表記になること
    result = _format_time_delta(timedelta(hours=24))
    assert "日" in result


# --- Model string parsing ---

def test_model_string_split():
    model = "some-char-uuid@google"
    char_id, provider = model.rsplit("@", 1)
    assert char_id == "some-char-uuid"
    assert provider == "google"


def test_model_string_split_with_at_in_char_id():
    # char_id は複数 @ を含まないが念のため rsplit(...,1) を検証
    model = "char-id@anthropic"
    char_id, provider = model.rsplit("@", 1)
    assert provider == "anthropic"


# --- OAIChatRequest → Message 変換 ---

def test_oai_request_to_messages():
    body = OAIChatRequest(
        model="char@claude_cli",
        messages=[
            OAIChatMessage(role="user", content="hello"),
            OAIChatMessage(role="assistant", content="hi"),
        ],
    )
    messages = [Message(role=m.role, content=m.content) for m in body.messages]
    assert messages[0].role == "user"
    assert messages[0].content == "hello"
    assert messages[1].role == "assistant"


# --- _sse_chunk ---

def test_sse_chunk_format():
    chunk = _sse_chunk("hello world")
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    payload = json.loads(chunk[6:])
    assert payload["choices"][0]["delta"]["content"] == "hello world"
    assert payload["object"] == "chat.completion.chunk"


def test_sse_chunk_unicode():
    chunk = _sse_chunk("こんにちは")
    payload = json.loads(chunk[6:])
    assert payload["choices"][0]["delta"]["content"] == "こんにちは"


# --- _format_completion ---

def test_format_completion_structure():
    result = _format_completion("char@google", "response text")
    assert result["object"] == "chat.completion"
    assert result["model"] == "char@google"
    assert result["choices"][0]["message"]["content"] == "response text"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["id"].startswith("chatcmpl-")


# ---------------------------------------------------------------------------
# アダプターの available_presets 構築テスト
# ---------------------------------------------------------------------------

def _make_preset(preset_id: str, name: str, provider: str = "google", model_id: str = "gemini-2.0-flash"):
    """テスト用 LLMModelPreset 風 MagicMock を生成するヘルパー。"""
    p = MagicMock()
    p.id = preset_id
    p.name = name
    p.provider = provider
    p.model_id = model_id
    p.thinking_level = "default"
    return p


def _make_character(char_id: str, char_name: str, switch_angle_enabled: int, enabled_providers: dict):
    """テスト用 Character 風 MagicMock を生成するヘルパー。"""
    c = MagicMock()
    c.id = char_id
    c.name = char_name
    c.switch_angle_enabled = switch_angle_enabled
    c.enabled_providers = enabled_providers
    c.system_prompt_block1 = ""
    c.self_history = ""
    c.relationship_state = ""
    c.inner_narrative = ""
    c.afterglow_default = 0
    c.created_at = None
    return c


def _make_adapter_app(sqlite_mock, chat_service_mock=None) -> FastAPI:
    """モック state を持つアダプター用 FastAPI アプリを生成するヘルパー。"""
    app = FastAPI()
    app.include_router(adapter_module.router)
    app.state.sqlite = sqlite_mock
    app.state.chat_service = chat_service_mock or MagicMock()
    return app


class TestAdapterAvailablePresets:
    """アダプター /v1/chat/completions が available_presets を正しく構築することを検証する。

    Bug Fix: アダプターが ChatRequest に available_presets をセットしていなかったため、
    OpenWebUI 経由の switch_angle が常に「プリセット未発見」で失敗していた。
    """

    def _setup(self, switch_angle_enabled: int, enabled_providers: dict, all_presets: list):
        """テスト共通のモック環境をセットアップするヘルパー。

        Args:
            switch_angle_enabled: キャラクターの switch_angle_enabled 設定値（0 or 1）。
            enabled_providers: キャラクターの enabled_providers dict（preset_id をキーとする）。
            all_presets: sqlite.list_model_presets() が返すプリセットリスト。

        Returns:
            (client, captured_requests): TestClient と、chat_service に渡された ChatRequest のリスト。
        """
        current_preset = _make_preset("preset-gemini", "Gemini3Flash", provider="google")
        character = _make_character(
            "char-1", "TestChar",
            switch_angle_enabled=switch_angle_enabled,
            enabled_providers=enabled_providers,
        )

        sqlite_mock = MagicMock()
        sqlite_mock.get_character_by_name.return_value = character
        sqlite_mock.get_model_preset_by_name.return_value = current_preset
        sqlite_mock.get_all_settings.return_value = {}
        sqlite_mock.get_setting.return_value = None
        sqlite_mock.set_setting.return_value = None
        sqlite_mock.list_model_presets.return_value = all_presets
        sqlite_mock.find_latest_session_for_character.return_value = None

        captured_requests = []

        async def fake_execute_stream(chat_request):
            """渡された ChatRequest を記録して 1 チャンクだけ返す。"""
            captured_requests.append(chat_request)
            yield ("text", "応答テキスト")

        chat_service_mock = MagicMock()
        chat_service_mock.execute_stream = fake_execute_stream

        app = _make_adapter_app(sqlite_mock, chat_service_mock)
        client = TestClient(app, raise_server_exceptions=True)
        return client, captured_requests

    def test_available_presets_populated_when_switch_enabled(self):
        """switch_angle_enabled=1 かつ 2 プリセット有効のとき available_presets が構築される。"""
        current_preset = _make_preset("preset-gemini", "Gemini3Flash", provider="google")
        switch_preset = _make_preset("preset-qwen", "Qwen3.5", provider="openai", model_id="qwen-max")

        enabled_providers = {
            "preset-gemini": {"additional_instructions": "", "when_to_switch": ""},
            "preset-qwen": {"additional_instructions": "簡潔に", "when_to_switch": "軽い話題"},
        }
        all_presets = [current_preset, switch_preset]

        # current_preset を get_model_preset_by_name で返す（require_model_config のため enabled_providers に存在する）
        client, captured = self._setup(
            switch_angle_enabled=1,
            enabled_providers=enabled_providers,
            all_presets=all_presets,
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "TestChar@Gemini3Flash",
            "messages": [{"role": "user", "content": "こんにちは"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert len(captured) == 1

        presets = captured[0].available_presets
        assert len(presets) == 1
        assert presets[0]["preset_name"] == "Qwen3.5"
        assert presets[0]["preset_id"] == "preset-qwen"
        assert presets[0]["provider"] == "openai"
        assert presets[0]["additional_instructions"] == "簡潔に"
        assert presets[0]["when_to_switch"] == "軽い話題"

    def test_available_presets_empty_when_switch_disabled(self):
        """switch_angle_enabled=0 のとき available_presets は空リスト。"""
        current_preset = _make_preset("preset-gemini", "Gemini3Flash", provider="google")
        switch_preset = _make_preset("preset-qwen", "Qwen3.5", provider="openai")

        enabled_providers = {
            "preset-gemini": {"additional_instructions": "", "when_to_switch": ""},
            "preset-qwen": {"additional_instructions": "", "when_to_switch": ""},
        }
        all_presets = [current_preset, switch_preset]

        client, captured = self._setup(
            switch_angle_enabled=0,
            enabled_providers=enabled_providers,
            all_presets=all_presets,
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "TestChar@Gemini3Flash",
            "messages": [{"role": "user", "content": "こんにちは"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert captured[0].available_presets == []

    def test_available_presets_empty_when_only_one_provider(self):
        """有効プリセットが 1 件のとき available_presets は空リスト（切り替え先がない）。"""
        current_preset = _make_preset("preset-gemini", "Gemini3Flash", provider="google")

        # enabled_providers に現在のプリセットのみ
        enabled_providers = {
            "preset-gemini": {"additional_instructions": "", "when_to_switch": ""},
        }
        all_presets = [current_preset]

        client, captured = self._setup(
            switch_angle_enabled=1,
            enabled_providers=enabled_providers,
            all_presets=all_presets,
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "TestChar@Gemini3Flash",
            "messages": [{"role": "user", "content": "こんにちは"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert captured[0].available_presets == []
