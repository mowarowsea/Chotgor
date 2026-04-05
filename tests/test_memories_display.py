"""Issue #26 — 想起した記憶のフロント表示対応のテスト。

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
from backend.services.chat.models import ChatRequest, Message
from backend.services.chat.service import ChatService


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
    """1件の記憶が正しくフォーマットされること。
    ヘッダー行は含まず、[category] content  (score: X.XX) 形式で出力される。
    """
    mems = [_make_memory("田中さんと会った", category="general", score=0.75)]
    result = _format_memories_display(mems)
    assert "田中さんと会った" in result
    assert "general" in result
    assert "0.75" in result


def test_format_memories_display_multiple():
    """複数件の場合、全件の内容とカテゴリが含まれること。
    ヘッダー行（件数表示）は出力に含まれない。
    """
    mems = [
        _make_memory("記憶A", "identity", 0.9),
        _make_memory("記憶B", "semantic_knowledge", 0.6),
        _make_memory("記憶C", "user_info", 0.4),
    ]
    result = _format_memories_display(mems)
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


def test_format_memories_display_no_header():
    """📚 ヘッダー行は出力に含まれないこと。
    フロントエンド側が「📚 想起した記憶」セクションタイトルを描画するため、
    バックエンドはヘッダーなしで [category] 行のみを返す。
    """
    mems = [_make_memory("何か")]
    result = _format_memories_display(mems)
    assert "📚" not in result


def test_format_memories_display_ends_with_newline():
    """思考ブロックとの間に空行を入れるため、末尾に改行が付くこと。"""
    mems = [_make_memory("何か")]
    result = _format_memories_display(mems)
    assert result.endswith("\n")


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
    provider.SUPPORTS_TOOLS = False
    provider.generate_stream_typed = _typed_stream
    return provider


def _mock_inscriber_passthrough():
    """テキストをそのまま返す Inscriber モックを生成するヘルパー。"""
    mock = MagicMock()
    mock.inscribe_memory_from_text.side_effect = lambda text, *_: text
    return mock


def _mock_carver_passthrough():
    """テキストをそのまま返す Carver モックを生成するヘルパー。"""
    mock = MagicMock()
    mock.carve_narrative_from_text.side_effect = lambda text: text
    return mock


@pytest.mark.asyncio
async def test_execute_stream_yields_memories_first():
    """記憶がある場合、最初のチャンクが ("memories", list) であること。
    recall_with_identity が (identity_list, other_list) を返し、
    identity + others を結合したリストが先頭チャンクとしてyieldされることを確認する。
    """
    recalled = [_make_memory("記憶あり")]
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = (recalled, [])

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    with (
        patch("backend.services.chat.service.create_provider", return_value=_fake_provider_with_text("応答")),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_mock_inscriber_passthrough()),
        patch("backend.services.chat.service.Carver", return_value=_mock_carver_passthrough()),
    ):
        chunks = await _collect_stream(service, request)

    first_type, first_content = chunks[0]
    assert first_type == "memories"
    assert isinstance(first_content, list)
    assert first_content == recalled


@pytest.mark.asyncio
async def test_execute_stream_no_memories_chunk_when_empty():
    """記憶が0件の場合、("memories", ...) チャンクはyieldされないこと。
    identity・others の両方が空リストのとき memories チャンクが発生しないことを確認する。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    with (
        patch("backend.services.chat.service.create_provider", return_value=_fake_provider_with_text("応答")),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_mock_inscriber_passthrough()),
        patch("backend.services.chat.service.Carver", return_value=_mock_carver_passthrough()),
    ):
        chunks = await _collect_stream(service, request)

    types = [t for t, _ in chunks]
    assert "memories" not in types


@pytest.mark.asyncio
async def test_execute_stream_yields_text_last():
    """テキストチャンクは memories / thinking の後にyieldされること。
    recall_with_identity が非空リストを返したとき memories チャンクが thinking より先にくることを確認する。
    """
    recalled = [_make_memory("記憶")]
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = (recalled, [])

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    async def _typed_with_thinking(system_prompt, messages):
        yield ("thinking", "考え中")
        yield ("text", "最終応答")

    provider = MagicMock()
    provider.SUPPORTS_TOOLS = False
    provider.generate_stream_typed = _typed_with_thinking

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_mock_inscriber_passthrough()),
        patch("backend.services.chat.service.Carver", return_value=_mock_carver_passthrough()),
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
    memory_manager.recall_with_identity.return_value = ([], [])

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    async def _typed(system_prompt, messages):
        yield ("thinking", "思考A")
        yield ("thinking", "思考B")
        yield ("text", "応答")

    provider = MagicMock()
    provider.SUPPORTS_TOOLS = False
    provider.generate_stream_typed = _typed

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_mock_inscriber_passthrough()),
        patch("backend.services.chat.service.Carver", return_value=_mock_carver_passthrough()),
    ):
        chunks = await _collect_stream(service, request)

    thinking_chunks = [(t, c) for t, c in chunks if t == "thinking"]
    assert len(thinking_chunks) == 2
    assert ("thinking", "思考A") in thinking_chunks
    assert ("thinking", "思考B") in thinking_chunks


@pytest.mark.asyncio
async def test_execute_stream_text_is_cleaned_by_inscribe_memory_from_text():
    """Inscriber.inscribe_memory_from_text() によって [INSCRIBE_MEMORY:...] マーカーが
    取り除かれたテキストがyieldされること。

    旧テスト名: test_execute_stream_text_is_carved（carve() 時代の名称から改名）
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    raw = "応答テキスト[INSCRIBE_MEMORY:contextual|1.0|記憶内容]"

    async def _typed(system_prompt, messages):
        yield ("text", raw)

    provider = MagicMock()
    provider.SUPPORTS_TOOLS = False
    provider.generate_stream_typed = _typed

    def _fake_inscribe(text, *_):
        """[INSCRIBE_MEMORY:...] を除去する簡易実装。"""
        import re
        return re.sub(r"\[INSCRIBE_MEMORY:[^\]]*\]", "", text)

    mock_inscriber = MagicMock()
    mock_inscriber.inscribe_memory_from_text.side_effect = _fake_inscribe

    mock_carver = MagicMock()
    mock_carver.carve_narrative_from_text.side_effect = lambda text: text

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=mock_inscriber),
        patch("backend.services.chat.service.Carver", return_value=mock_carver),
    ):
        chunks = await _collect_stream(service, request)

    text_chunks = [c for t, c in chunks if t == "text"]
    assert len(text_chunks) == 1
    assert "[INSCRIBE_MEMORY:" not in text_chunks[0]
    assert "応答テキスト" in text_chunks[0]


@pytest.mark.asyncio
async def test_execute_stream_provider_error_yields_error_text():
    """プロバイダーがエラーを送出した場合、("text", エラーメッセージ) でストリームが終了する。"""
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    async def _failing_stream(system_prompt, messages):
        raise RuntimeError("provider boom")
        yield  # noqa: unreachable — AsyncGenerator の型を満たすために必要

    provider = MagicMock()
    provider.SUPPORTS_TOOLS = False
    provider.generate_stream_typed = _failing_stream

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        chunks = await _collect_stream(service, request)

    text_chunks = [c for t, c in chunks if t == "text"]
    assert any("Error" in c for c in text_chunks)


# ---------------------------------------------------------------------------
# execute_stream — SUPPORTS_TOOLS=True パス
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_stream_with_tools_yields_text():
    """SUPPORTS_TOOLS=True のプロバイダーは generate_with_tools を呼び、
    その戻り値を ("text", text) としてyieldすること。
    generate_stream_typed は呼ばれない。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    fake_provider = MagicMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value=("ツール経由の返答", ""))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        chunks = await _collect_stream(service, request)

    text_chunks = [c for t, c in chunks if t == "text"]
    assert len(text_chunks) == 1
    assert text_chunks[0] == "ツール経由の返答"
    fake_provider.generate_with_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_stream_with_tools_error_yields_error_text():
    """SUPPORTS_TOOLS=True でプロバイダーが例外を送出した場合、
    ("text", エラーメッセージ) でストリームが終了すること。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = _make_request()
    service = ChatService(memory_manager=memory_manager)

    fake_provider = MagicMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(side_effect=RuntimeError("tools boom"))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        chunks = await _collect_stream(service, request)

    text_chunks = [c for t, c in chunks if t == "text"]
    assert any("Error" in c for c in text_chunks)
