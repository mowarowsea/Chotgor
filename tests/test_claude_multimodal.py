import pytest
from unittest.mock import patch, MagicMock
from backend.providers.claude_cli_provider import ClaudeCliProvider, _format_conversation

@pytest.mark.asyncio
async def test_claude_cli_provider_image_note():
    provider = ClaudeCliProvider(character_name="Ghost")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "See this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }
    ]
    
    with patch("backend.providers.claude_cli_provider._run_claude") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=b'{"type":"assistant","message":{"content":[{"type":"text","text":"I see text"}]}}')
        
        # We want to check if the generated system_prompt contains the note.
        # However, generate() creates a temp file and passes the path to _run_claude.
        # So we mock _run_claude to capture the content of the file before it's deleted.
        
        def side_effect(sys_path, msg_path, extra_env=None):
            with open(sys_path, "r", encoding="utf-8") as f:
                mock_run.captured_sys_prompt = f.read()
            return mock_run.return_value

        mock_run.side_effect = side_effect
        
        await provider.generate("Base system prompt", messages)
        assert "[SYSTEM NOTE: The user has provided one or more images" in mock_run.captured_sys_prompt

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
