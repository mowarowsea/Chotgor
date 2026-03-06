"""Tests for core.chat.service and core.chat.models."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.chat.models import ChatRequest, Message
from backend.core.chat.service import ChatService, extract_text_content


# --- extract_text_content (moved from test_llm_service_multimodal) ---

def test_extract_text_content_string():
    assert extract_text_content("hello") == "hello"

def test_extract_text_content_none():
    assert extract_text_content(None) == ""

def test_extract_text_content_list():
    content = [
        {"type": "text", "text": "Hello "},
        {"type": "image_url", "image_url": {"url": "..."}},
        "world",
    ]
    assert extract_text_content(content) == "Hello world"

def test_extract_text_content_empty_list():
    assert extract_text_content([]) == ""


# --- ChatRequest construction ---

def test_chat_request_defaults():
    req = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="hi")],
    )
    assert req.character_system_prompt == ""
    assert req.meta_instructions == ""
    assert req.provider_additional_instructions == ""
    assert req.settings == {}


# --- ChatService.execute ---

@pytest.mark.asyncio
async def test_chat_service_execute_returns_text():
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.generate = AsyncMock(return_value="Hi there!")

    with (
        patch("backend.core.chat.service.create_provider", return_value=fake_provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=lambda text, *_: text),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "Hi there!"


@pytest.mark.asyncio
async def test_chat_service_execute_provider_error_returns_error_string():
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="boom")],
    )

    fake_provider = AsyncMock()
    fake_provider.generate = AsyncMock(side_effect=RuntimeError("oops"))

    with (
        patch("backend.core.chat.service.create_provider", return_value=fake_provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result.startswith("[Error: RuntimeError: oops")
