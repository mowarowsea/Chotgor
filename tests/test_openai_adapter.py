"""Tests for adapters.openai.router and adapters.openai.schemas."""

import json

import pytest

from backend.adapters.openai.router import _format_completion, _sse_chunk
from backend.adapters.openai.schemas import OAIChatMessage, OAIChatRequest
from backend.core.chat.models import Message


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
