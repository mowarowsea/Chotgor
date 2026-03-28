"""backend.services.chat.service と backend.services.chat.models のテスト。

ChatService のフロー全体を検証する:
- SUPPORTS_TOOLS=False パス: Inscriber.inscribe_memory_from_text / Carver.carve_narrative_from_text 経由
- SUPPORTS_TOOLS=True パス: generate_with_tools / ToolExecutor 経由
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.chat.models import ChatRequest, Message
from backend.services.chat.service import ChatService, extract_text_content


# --- extract_text_content (moved from test_llm_service_multimodal) ---

def test_extract_text_content_string():
    """文字列をそのまま返すこと。"""
    assert extract_text_content("hello") == "hello"

def test_extract_text_content_none():
    """None は空文字列を返すこと。"""
    assert extract_text_content(None) == ""

def test_extract_text_content_list():
    """リスト形式の content からテキスト部分だけ抽出すること。"""
    content = [
        {"type": "text", "text": "Hello "},
        {"type": "image_url", "image_url": {"url": "..."}},
        "world",
    ]
    assert extract_text_content(content) == "Hello world"

def test_extract_text_content_empty_list():
    """空リストは空文字列を返すこと。"""
    assert extract_text_content([]) == ""


# --- ChatRequest のデフォルト値確認 ---

def test_chat_request_defaults():
    """ChatRequest のデフォルト値が正しいこと。inner_narrative フィールドが存在すること。"""
    req = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="hi")],
    )
    assert req.character_system_prompt == ""
    assert req.inner_narrative == ""
    assert req.provider_additional_instructions == ""
    assert req.settings == {}


def test_chat_request_has_no_meta_instructions_field():
    """ChatRequest に旧フィールド名 meta_instructions が存在しないこと（改名済み確認）。"""
    req = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[],
    )
    assert not hasattr(req, "meta_instructions"), (
        "meta_instructions フィールドが残っています。inner_narrative に改名済みのはずです。"
    )


# --- ChatService.execute (SUPPORTS_TOOLS=False パス) ---

@pytest.mark.asyncio
async def test_chat_service_execute_returns_text():
    """SUPPORTS_TOOLS=False のプロバイダーは generate() を呼び、クリーンテキストを返すこと。

    Inscriber.inscribe_memory_from_text と Carver.carve_narrative_from_text はパス・スルーでモックする。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

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

    mock_inscriber = MagicMock()
    mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
    mock_carver = MagicMock()
    mock_carver.carve_narrative_from_text.side_effect = lambda text: text

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=mock_inscriber),
        patch("backend.services.chat.service.Carver", return_value=mock_carver),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "Hi there!"


@pytest.mark.asyncio
async def test_chat_service_execute_provider_error_returns_error_string():
    """SUPPORTS_TOOLS=False でプロバイダーが例外を送出した場合、エラー文字列を返すこと。"""
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="boom")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(side_effect=RuntimeError("oops"))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result.startswith("[Error: RuntimeError: oops")


# --- ChatService.execute (SUPPORTS_TOOLS=True パス) ---

@pytest.mark.asyncio
async def test_chat_service_execute_with_tools_returns_text():
    """SUPPORTS_TOOLS=True のプロバイダーは generate_with_tools を呼び、その戻り値を返すこと。

    Inscriber.inscribe_memory_from_text と Carver.carve_narrative_from_text は呼ばれない
    （ツール側が直接 inscribe_memory / carve_narrative を操作する）。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

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
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
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
    memory_manager.recall_with_identity.return_value = ([], [])

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
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
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
    from backend.providers.base import BaseLLMProvider
    from backend.character_actions.executor import ToolTurnResult

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
async def test_execute_with_tools_calls_memory_manager_via_inscribe_memory():
    """SUPPORTS_TOOLS=True のプロバイダーが inscribe_memory ツールを呼び出したとき、
    memory_manager.write_memory が実際に呼ばれること。

    サービス → generate_with_tools → ToolExecutor → inscribe_memory → memory_manager という
    一連の呼び出しチェーンをサービスレベルで統合検証する。
    """
    from backend.character_actions.executor import ToolCall, ToolTurnResult

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="覚えてね")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-1", name="inscribe_memory", input={"content": "ユーザは猫が好き", "category": "user", "impact": 1.0})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="覚えたよ", tool_calls=[]),
    ])

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "覚えたよ"
    memory_manager.write_memory.assert_called_once()
    call_kwargs = memory_manager.write_memory.call_args
    assert call_kwargs.kwargs.get("content") == "ユーザは猫が好き"


@pytest.mark.asyncio
async def test_execute_with_tools_calls_carve_narrative_via_tool_executor():
    """SUPPORTS_TOOLS=True のプロバイダーが carve_narrative ツールを呼び出したとき、
    sqlite_store.update_character が実際に呼ばれること。

    サービス → generate_with_tools → ToolExecutor → carve_narrative → Carver → sqlite という
    一連の呼び出しチェーンをサービスレベルで統合検証する。
    """
    from backend.character_actions.executor import ToolCall, ToolTurnResult

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])
    mock_char = MagicMock()
    mock_char.inner_narrative = ""
    memory_manager.sqlite.get_character.return_value = mock_char

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="自分を書き直したい")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-2", name="carve_narrative", input={"mode": "append", "content": "知的好奇心を大切にする"})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="指針を彫り込んだ", tool_calls=[]),
    ])

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "指針を彫り込んだ"
    memory_manager.sqlite.update_character.assert_called_once()


@pytest.mark.asyncio
async def test_execute_with_tools_calls_drift_manager():
    """SUPPORTS_TOOLS=True のプロバイダーが drift ツールを呼び出したとき、
    drift_manager.add_drift が実際に呼ばれること。
    """
    from backend.character_actions.executor import ToolCall, ToolTurnResult

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])
    drift_manager = MagicMock()

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="もっとクールにして")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-3", name="drift", input={"content": "クールに話す"})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="了解", tool_calls=[]),
    ])

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
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
    from backend.character_actions.executor import ToolCall, ToolTurnResult

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])
    drift_manager = MagicMock()

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="指針リセットして")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-4", name="drift_reset", input={})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="リセットした", tool_calls=[]),
    ])

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        result = await service.execute(request)

    assert result == "リセットした"
    drift_manager.reset_drifts.assert_called_once_with("session-abc", "char-1")


@pytest.mark.asyncio
async def test_execute_without_tools_does_not_use_tool_executor():
    """SUPPORTS_TOOLS=False のプロバイダーは generate_with_tools を呼ばず、
    ToolExecutor は使用されないこと。

    マーカー方式（Inscriber.inscribe_memory_from_text / Carver.carve_narrative_from_text）がパスすることを確認する。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

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

    mock_inscriber = MagicMock()
    mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
    mock_carver = MagicMock()
    mock_carver.carve_narrative_from_text.side_effect = lambda text: text

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=mock_inscriber),
        patch("backend.services.chat.service.Carver", return_value=mock_carver),
        patch("backend.services.chat.service.ToolExecutor") as mock_tool_executor_cls,
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "Hi!"
    mock_tool_executor_cls.assert_not_called()
