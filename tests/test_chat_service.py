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
    fake_provider.SUPPORTS_TOOLS = False
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


# --- ChatService.execute (SUPPORTS_TOOLS=True パス) ---

@pytest.mark.asyncio
async def test_chat_service_execute_with_tools_returns_text():
    """SUPPORTS_TOOLS=True のプロバイダーは generate_with_tools を呼び、その戻り値を返すこと。

    carve や _apply_drifts は呼ばれない（ツール側が直接記憶・DRIFTを操作する）。
    """
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
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value="Hi via tools!")

    with (
        patch("backend.core.chat.service.create_provider", return_value=fake_provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "Hi via tools!"
    fake_provider.generate_with_tools.assert_awaited_once()
    fake_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_chat_service_execute_with_tools_error_returns_error_string():
    """SUPPORTS_TOOLS=True でプロバイダーが例外を送出した場合、エラー文字列を返すこと。"""
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
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(side_effect=RuntimeError("tools oops"))

    with (
        patch("backend.core.chat.service.create_provider", return_value=fake_provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result.startswith("[Error: RuntimeError: tools oops")


# --- ChatService.execute — ToolExecutor 統合テスト ---

def _make_tool_provider(turn_results: list):
    """指定した ToolTurnResult リストを順に返すフェイクプロバイダーを生成する。

    SUPPORTS_TOOLS=True のプロバイダーとして振る舞い、
    generate_with_tools のツールループ（BaseLLMProvider 実装）をそのまま使う。

    Args:
        turn_results: _tool_turn() が順に返す ToolTurnResult のリスト。

    Returns:
        BaseLLMProvider サブクラスのインスタンス。
    """
    from backend.core.providers.base import BaseLLMProvider
    from backend.core.tools import ToolTurnResult

    class FakeToolProvider(BaseLLMProvider):
        """テスト用ツール対応プロバイダー。"""

        SUPPORTS_TOOLS = True

        def __init__(self):
            """フェイクプロバイダーを初期化する。"""
            self._turns = iter(turn_results)

        async def _tool_turn(self, system_prompt, messages):
            """事前に設定した ToolTurnResult を順に返す。"""
            return next(self._turns)

        def _extend_messages_with_results(self, messages, turn_result, results):
            """ツール結果をダミーメッセージとして追加する。"""
            return messages + [{"role": "tool_result_dummy", "content": str(results)}]

    return FakeToolProvider()


@pytest.mark.asyncio
async def test_execute_with_tools_calls_memory_manager():
    """SUPPORTS_TOOLS=True のプロバイダーが carve_memory ツールを呼び出したとき、
    memory_manager.write_memory が実際に呼ばれること。

    サービス → generate_with_tools → ToolExecutor → memory_manager という
    一連の呼び出しチェーンをサービスレベルで統合検証する。
    """
    from backend.core.tools import ToolCall, ToolTurnResult

    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="覚えてね")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-1", name="carve_memory", input={"content": "ユーザは猫が好き", "category": "user", "impact": 1.0})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="覚えたよ", tool_calls=[]),
    ])

    with (
        patch("backend.core.chat.service.create_provider", return_value=provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "覚えたよ"
    memory_manager.write_memory.assert_called_once()
    call_kwargs = memory_manager.write_memory.call_args
    assert call_kwargs.kwargs.get("content") == "ユーザは猫が好き"


@pytest.mark.asyncio
async def test_execute_with_tools_calls_drift_manager():
    """SUPPORTS_TOOLS=True のプロバイダーが drift ツールを呼び出したとき、
    drift_manager.add_drift が実際に呼ばれること。
    """
    from backend.core.tools import ToolCall, ToolTurnResult

    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []
    drift_manager = MagicMock()

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="もっとクールにして")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-2", name="drift", input={"content": "クールに話す"})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="了解", tool_calls=[]),
    ])

    with (
        patch("backend.core.chat.service.create_provider", return_value=provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        result = await service.execute(request)

    assert result == "了解"
    drift_manager.add_drift.assert_called_once_with("session-abc", "char-1", "クールに話す")


@pytest.mark.asyncio
async def test_execute_with_tools_calls_drift_reset():
    """SUPPORTS_TOOLS=True のプロバイダーが drift_reset ツールを呼び出したとき、
    drift_manager.reset_drifts が実際に呼ばれること。
    """
    from backend.core.tools import ToolCall, ToolTurnResult

    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []
    drift_manager = MagicMock()

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="指針リセットして")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-3", name="drift_reset", input={})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="リセットした", tool_calls=[]),
    ])

    with (
        patch("backend.core.chat.service.create_provider", return_value=provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        result = await service.execute(request)

    assert result == "リセットした"
    drift_manager.reset_drifts.assert_called_once_with("session-abc", "char-1")


@pytest.mark.asyncio
async def test_execute_without_tools_does_not_call_memory_via_tool_executor():
    """SUPPORTS_TOOLS=False のプロバイダーは generate_with_tools を呼ばず、
    ToolExecutor 経由では memory_manager.write_memory が呼ばれないこと。

    マーカー方式（carve）の呼び出しをモックして、ToolExecutorが介入しないことを確認する。
    """
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="claude_cli",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(return_value="Hi!")

    with (
        patch("backend.core.chat.service.create_provider", return_value=fake_provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=lambda text, *_: text),
        patch("backend.core.chat.service.ToolExecutor") as mock_tool_executor_cls,
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "Hi!"
    mock_tool_executor_cls.assert_not_called()
    fake_provider.generate_with_tools = MagicMock()  # これが呼ばれていないことは上のパッチで保証済み
