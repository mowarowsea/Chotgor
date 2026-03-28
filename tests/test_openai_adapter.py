"""Tests for adapters.openai.router and adapters.openai.schemas."""

import json
from datetime import timedelta

import pytest

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
