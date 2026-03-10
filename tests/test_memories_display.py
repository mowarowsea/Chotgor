"""Tests for Issue #26 — 想起した記憶のフロント表示対応。

ChatService.execute_stream() が記憶を先頭で ("memories", list) としてyieldすること、
および router ヘルパー関数の動作を検証する。
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.adapters.openai.router import (
    _format_memories_display,
    _sse_chunk_reasoning,
)
from backend.core.chat.models import ChatRequest, Message
from backend.core.chat.service import ChatService


# ---------------------------------------------------------------------------
# _sse_chunk_reasoning()
# ---------------------------------------------------------------------------


def test_sse_chunk_reasoning_format():
    """reasoning_content フィールドに格納された SSE チャンクが正しい形式であること。"""
    chunk = _sse_chunk_reasoning("思考テキスト")
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    payload = json.loads(chunk[6:])
    assert payload["object"] == "chat.completion.chunk"
    delta = payload["choices"][0]["delta"]
    assert "reasoning_content" in delta
    assert "content" not in delta
    assert delta["reasoning_content"] == "思考テキスト"


def test_sse_chunk_reasoning_unicode():
    """日本語・絵文字が ensure_ascii=False でそのまま格納されること。"""
    chunk = _sse_chunk_reasoning("📚 想起した記憶")
    payload = json.loads(chunk[6:])
    assert payload["choices"][0]["delta"]["reasoning_content"] == "📚 想起した記憶"


def test_sse_chunk_reasoning_no_finish_reason():
    """ストリーミング中チャンクの finish_reason は None であること。"""
    chunk = _sse_chunk_reasoning("x")
    payload = json.loads(chunk[6:])
    assert payload["choices"][0]["finish_reason"] is None


# ---------------------------------------------------------------------------
# _format_memories_display()
# ---------------------------------------------------------------------------


def _make_memory(content: str, category: str = "general", score: float = 0.8) -> dict:
    """テスト用の記憶辞書を生成するヘルパー。"""
    return {
        "id": "mem-1",
        "content": content,
        "metadata": {"category": category},
        "hybrid_score": score,
    }


def test_format_memories_display_empty():
    """記憶リストが空の場合は空文字列を返す。"""
    assert _format_memories_display([]) == ""


def test_format_memories_display_single():
    """1件の記憶が正しくフォーマットされること。"""
    mems = [_make_memory("田中さんと会った", category="general", score=0.75)]
    result = _format_memories_display(mems)
    assert "想起した記憶" in result
    assert "田中さんと会った" in result
    assert "general" in result
    assert "0.75" in result


def test_format_memories_display_multiple():
    """複数件の場合、件数ヘッダーと全件の内容が含まれること。"""
    mems = [
        _make_memory("記憶A", "identity", 0.9),
        _make_memory("記憶B", "semantic_knowledge", 0.6),
        _make_memory("記憶C", "user_info", 0.4),
    ]
    result = _format_memories_display(mems)
    assert "3件" in result
    assert "記憶A" in result
    assert "記憶B" in result
    assert "記憶C" in result
    assert "identity" in result
    assert "semantic_knowledge" in result


def test_format_memories_display_missing_metadata():
    """metadata キーがない記憶でも KeyError が発生せず category="general" として扱われること。"""
    mems = [{"id": "x", "content": "テスト", "hybrid_score": 0.5}]
    result = _format_memories_display(mems)
    assert "テスト" in result
    assert "general" in result


def test_format_memories_display_header_icon():
    """ヘッダーに絵文字アイコン📚が含まれること。"""
    mems = [_make_memory("何か")]
    result = _format_memories_display(mems)
    assert "📚" in result


# ---------------------------------------------------------------------------
# ChatService.execute_stream() — memories チャンク
# ---------------------------------------------------------------------------


def _make_request(**kwargs) -> ChatRequest:
    """テスト用の最小限 ChatRequest を生成するヘルパー。"""
    defaults = dict(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )
    defaults.update(kwargs)
    return ChatRequest(**defaults)


async def _collect_stream(service, request):
    """execute_stream() の全チャンクをリストに収集するヘルパー。"""
    chunks = []
    async for item in service.execute_stream(request):
        chunks.append(item)
    return chunks


def _fake_provider_with_text(text: str):
    """指定テキストを ("text", text) としてyieldするフェイクプロバイダーを返す。"""
    async def _typed_stream(system_prompt, messages):
        yield ("text", text)

    provider = MagicMock()
    provider.generate_stream_typed = _typed_stream
    return provider


@pytest.mark.asyncio
async def test_execute_stream_yields_memories_first():
    """記憶がある場合、最初のチャンクが ("memories", list) であること。"""
    recalled = [_make_memory("記憶あり")]
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = recalled

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    with (
        patch("backend.core.chat.service.create_provider", return_value=_fake_provider_with_text("応答")),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=lambda text, *_: text),
    ):
        chunks = await _collect_stream(service, request)

    first_type, first_content = chunks[0]
    assert first_type == "memories"
    assert isinstance(first_content, list)
    assert first_content == recalled


@pytest.mark.asyncio
async def test_execute_stream_no_memories_chunk_when_empty():
    """記憶が0件の場合、("memories", ...) チャンクはyieldされないこと。"""
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    with (
        patch("backend.core.chat.service.create_provider", return_value=_fake_provider_with_text("応答")),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=lambda text, *_: text),
    ):
        chunks = await _collect_stream(service, request)

    types = [t for t, _ in chunks]
    assert "memories" not in types


@pytest.mark.asyncio
async def test_execute_stream_yields_text_last():
    """テキストチャンクは memories / thinking の後にyieldされること。"""
    recalled = [_make_memory("記憶")]
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = recalled

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    async def _typed_with_thinking(system_prompt, messages):
        yield ("thinking", "考え中")
        yield ("text", "最終応答")

    provider = MagicMock()
    provider.generate_stream_typed = _typed_with_thinking

    with (
        patch("backend.core.chat.service.create_provider", return_value=provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=lambda text, *_: text),
    ):
        chunks = await _collect_stream(service, request)

    types = [t for t, _ in chunks]
    # 順序: memories → thinking → text
    assert types.index("memories") < types.index("thinking")
    assert types.index("thinking") < types.index("text")


@pytest.mark.asyncio
async def test_execute_stream_thinking_chunks_yielded():
    """thinking チャンクがリアルタイムでyieldされること。"""
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    async def _typed(system_prompt, messages):
        yield ("thinking", "思考A")
        yield ("thinking", "思考B")
        yield ("text", "応答")

    provider = MagicMock()
    provider.generate_stream_typed = _typed

    with (
        patch("backend.core.chat.service.create_provider", return_value=provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=lambda text, *_: text),
    ):
        chunks = await _collect_stream(service, request)

    thinking_chunks = [(t, c) for t, c in chunks if t == "thinking"]
    assert len(thinking_chunks) == 2
    assert ("thinking", "思考A") in thinking_chunks
    assert ("thinking", "思考B") in thinking_chunks


@pytest.mark.asyncio
async def test_execute_stream_text_is_carved():
    """carve() によって [MEMORY:...] マーカーが取り除かれたテキストがyieldされること。"""
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    raw = "応答テキスト[MEMORY:general|記憶内容]"

    async def _typed(system_prompt, messages):
        yield ("text", raw)

    provider = MagicMock()
    provider.generate_stream_typed = _typed

    def _fake_carve(text, char_id, manager):
        # [MEMORY:...] を除去する簡易実装
        import re
        return re.sub(r"\[MEMORY:[^\]]*\]", "", text)

    with (
        patch("backend.core.chat.service.create_provider", return_value=provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=_fake_carve),
    ):
        chunks = await _collect_stream(service, request)

    text_chunks = [c for t, c in chunks if t == "text"]
    assert len(text_chunks) == 1
    assert "[MEMORY:" not in text_chunks[0]
    assert "応答テキスト" in text_chunks[0]


@pytest.mark.asyncio
async def test_execute_stream_provider_error_yields_error_text():
    """プロバイダーがエラーを送出した場合、("text", エラーメッセージ) でストリームが終了する。"""
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    async def _failing_stream(system_prompt, messages):
        raise RuntimeError("provider boom")
        yield  # noqa: unreachable — AsyncGenerator の型を満たすために必要

    provider = MagicMock()
    provider.generate_stream_typed = _failing_stream

    with (
        patch("backend.core.chat.service.create_provider", return_value=provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
    ):
        chunks = await _collect_stream(service, request)

    text_chunks = [c for t, c in chunks if t == "text"]
    assert any("Error" in c for c in text_chunks)
