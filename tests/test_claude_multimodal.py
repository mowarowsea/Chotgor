import pytest
from unittest.mock import patch, MagicMock
from backend.providers.claude_cli_provider import ClaudeCliProvider, _format_conversation

def test_format_conversation_multimodal():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "image_url", "image_url": {"url": "..."}}
            ]
        },
        {
            "role": "assistant",
            "content": "Hi there"
        },
        {
            "role": "user",
            "content": "Next message"
        }
    ]
    
    formatted = _format_conversation(messages, "Ghost")
    assert "<human>Hello </human>" in formatted
    assert "<Ghost>Hi there</Ghost>" in formatted
    assert formatted.endswith("Next message")
