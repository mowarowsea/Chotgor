"""switch_angle テスト群の共有ヘルパー。

ToolExecutor / ChatRequest の組み立てとサンプルプリセット定義を提供する。
test_switch_angle_*.py から共通利用する。
ファイル名先頭がアンダースコアのため pytest のテスト収集対象にはならない。
"""

from unittest.mock import MagicMock

from backend.services.chat.models import ChatRequest, Message
from backend.character_actions.executor import ToolExecutor


def _make_executor(memory_manager=None, working_memory_manager=None, session_id="sess-1"):
    """テスト用 ToolExecutor を生成するヘルパー。"""
    return ToolExecutor(
        character_id="char-1",
        session_id=session_id,
        memory_manager=memory_manager or MagicMock(),
        working_memory_manager=working_memory_manager or MagicMock(),
    )


def _make_request(**kwargs) -> ChatRequest:
    """テスト用 ChatRequest を生成するヘルパー。"""
    defaults = dict(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="こんにちは")],
        session_id="sess-1",
    )
    defaults.update(kwargs)
    return ChatRequest(**defaults)


_SAMPLE_PRESETS = [
    {
        "preset_id": "preset-b",
        "preset_name": "fastModel",
        "provider": "google",
        "model_id": "gemini-2.0-flash-lite",
        "additional_instructions": "簡潔に",
        "thinking_level": "default",
        "when_to_switch": "軽い雑談のとき",
    },
    {
        "preset_id": "preset-c",
        "preset_name": "deepModel",
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "additional_instructions": "",
        "thinking_level": "high",
        "when_to_switch": "",
    },
]

