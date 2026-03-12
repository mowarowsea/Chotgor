"""Tests for core/providers/registry.py and each provider's from_config()."""

import pytest

from backend.core.providers.anthropic_provider import AnthropicProvider
from backend.core.providers.claude_cli_provider import (
    ClaudeCliProvider,
    invoke_claude_cli,
    _parse_stream_json,
    _clean_env,
)
from backend.core.providers.google_provider import GoogleProvider
from backend.core.providers.ollama_provider import OllamaProvider
from backend.core.providers.openai_provider import OpenAIProvider, XAIProvider
from backend.core.providers.registry import (
    PROVIDER_LABELS,
    PROVIDER_ORDER,
    PROVIDER_REGISTRY,
    create_provider,
    get_default_model,
)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_all_expected_providers_registered(self):
        assert set(PROVIDER_REGISTRY.keys()) == {
            "claude_cli", "anthropic", "openai", "xai", "google", "ollama"
        }

    def test_provider_classes_are_correct(self):
        assert PROVIDER_REGISTRY["claude_cli"] is ClaudeCliProvider
        assert PROVIDER_REGISTRY["anthropic"] is AnthropicProvider
        assert PROVIDER_REGISTRY["openai"] is OpenAIProvider
        assert PROVIDER_REGISTRY["xai"] is XAIProvider
        assert PROVIDER_REGISTRY["google"] is GoogleProvider
        assert PROVIDER_REGISTRY["ollama"] is OllamaProvider

    def test_provider_order_matches_registry(self):
        for pid in PROVIDER_ORDER:
            assert pid in PROVIDER_REGISTRY

    def test_provider_labels_present_for_all(self):
        for pid in PROVIDER_REGISTRY:
            assert pid in PROVIDER_LABELS
            assert PROVIDER_LABELS[pid]  # non-empty string


# ---------------------------------------------------------------------------
# get_default_model
# ---------------------------------------------------------------------------

class TestGetDefaultModel:
    def test_known_providers_return_non_empty_or_empty(self):
        assert get_default_model("claude_cli") == ""  # configured via OAuth
        assert get_default_model("anthropic") != ""
        assert get_default_model("openai") != ""
        assert get_default_model("xai") != ""
        assert get_default_model("google") != ""

    def test_unknown_provider_returns_empty(self):
        assert get_default_model("does_not_exist") == ""

    def test_matches_class_default_model(self):
        for pid, cls in PROVIDER_REGISTRY.items():
            assert get_default_model(pid) == cls.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# create_provider factory
# ---------------------------------------------------------------------------

class TestCreateProvider:
    def test_creates_correct_type_for_each_provider(self):
        settings = {
            "anthropic_api_key": "a",
            "openai_api_key": "b",
            "xai_api_key": "c",
            "google_api_key": "d",
        }
        assert isinstance(create_provider("claude_cli", "", settings), ClaudeCliProvider)
        assert isinstance(create_provider("anthropic", "", settings), AnthropicProvider)
        assert isinstance(create_provider("openai", "", settings), OpenAIProvider)
        assert isinstance(create_provider("xai", "", settings), XAIProvider)
        assert isinstance(create_provider("google", "", settings), GoogleProvider)

    def test_unknown_provider_falls_back_to_claude_cli(self):
        p = create_provider("totally_unknown", "", {})
        assert isinstance(p, ClaudeCliProvider)

    def test_model_is_forwarded(self):
        p = create_provider("anthropic", "claude-opus-4-6", {"anthropic_api_key": "k"})
        assert p.model == "claude-opus-4-6"

    def test_default_model_used_when_model_empty(self):
        p = create_provider("anthropic", "", {"anthropic_api_key": "k"})
        assert p.model == AnthropicProvider.DEFAULT_MODEL

    def test_character_name_forwarded_to_claude_cli(self):
        p = create_provider("claude_cli", "", {}, character_name="織羽")
        assert p.character_name == "織羽"

    def test_extra_kwargs_ignored_by_non_cli_providers(self):
        # character_name is a ClaudeCliProvider-specific kwarg;
        # other providers should accept **kwargs and silently ignore it.
        p = create_provider("anthropic", "", {"anthropic_api_key": "k"}, character_name="X")
        assert isinstance(p, AnthropicProvider)


# ---------------------------------------------------------------------------
# from_config on each provider class directly
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_anthropic_from_config(self):
        p = AnthropicProvider.from_config("claude-opus-4-6", {"anthropic_api_key": "secret"})
        assert isinstance(p, AnthropicProvider)
        assert p.api_key == "secret"
        assert p.model == "claude-opus-4-6"

    def test_anthropic_from_config_missing_key(self):
        p = AnthropicProvider.from_config("", {})
        assert p.api_key == ""

    def test_openai_from_config(self):
        p = OpenAIProvider.from_config("gpt-4-turbo", {"openai_api_key": "sk-123"})
        assert isinstance(p, OpenAIProvider)
        assert p.api_key == "sk-123"
        assert p.model == "gpt-4-turbo"

    def test_xai_from_config(self):
        p = XAIProvider.from_config("grok-3", {"xai_api_key": "xai-abc"})
        assert isinstance(p, XAIProvider)
        assert p.api_key == "xai-abc"
        assert p.model == "grok-3"
        assert p.base_url == XAIProvider.BASE_URL

    def test_xai_from_config_uses_default_model_when_empty(self):
        p = XAIProvider.from_config("", {"xai_api_key": "xai-abc"})
        assert p.model == XAIProvider.DEFAULT_MODEL

    def test_google_from_config(self):
        p = GoogleProvider.from_config("gemini-2.0-flash", {"google_api_key": "g-key"})
        assert isinstance(p, GoogleProvider)
        assert p.api_key == "g-key"
        assert p.model == "gemini-2.0-flash"

    def test_claude_cli_from_config(self):
        p = ClaudeCliProvider.from_config("", {}, character_name="Noa")
        assert isinstance(p, ClaudeCliProvider)
        assert p.character_name == "Noa"

    def test_claude_cli_from_config_no_api_key_needed(self):
        # should not raise even with empty settings
        p = ClaudeCliProvider.from_config("", {})
        assert isinstance(p, ClaudeCliProvider)


# ---------------------------------------------------------------------------
# XAIProvider specifics
# ---------------------------------------------------------------------------

class TestXAIProvider:
    def test_is_subclass_of_openai(self):
        assert issubclass(XAIProvider, OpenAIProvider)

    def test_has_own_provider_id(self):
        assert XAIProvider.PROVIDER_ID == "xai"
        assert OpenAIProvider.PROVIDER_ID == "openai"

    def test_base_url_is_set(self):
        p = XAIProvider(api_key="k")
        assert p.base_url == "https://api.x.ai/v1"

    def test_default_model_differs_from_openai(self):
        assert XAIProvider.DEFAULT_MODEL != OpenAIProvider.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# ClaudeCliProvider helpers
# ---------------------------------------------------------------------------

class TestCleanEnv:
    def test_excludes_claudecode_and_api_key(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-xxx")
        monkeypatch.setenv("SOME_OTHER_VAR", "keep")
        env = _clean_env()
        assert "CLAUDECODE" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert env.get("SOME_OTHER_VAR") == "keep"


class TestParseStreamJson:
    def test_extracts_assistant_text(self):
        raw = (
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}\n'
            '{"type":"result","result":""}'
        )
        assert _parse_stream_json(raw) == "Hello"

    def test_falls_back_to_result_when_no_assistant_block(self):
        raw = '{"type":"result","result":"Fallback"}'
        assert _parse_stream_json(raw) == "Fallback"

    def test_ignores_malformed_lines(self):
        raw = "not json\n" '{"type":"assistant","message":{"content":[{"type":"text","text":"OK"}]}}'
        assert _parse_stream_json(raw) == "OK"

    def test_empty_input_returns_empty(self):
        assert _parse_stream_json("") == ""

    def test_multiple_assistant_blocks_concatenated(self):
        raw = (
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Foo"}]}}\n'
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Bar"}]}}'
        )
        assert _parse_stream_json(raw) == "FooBar"


class TestInvokeClaudeCliImportable:
    def test_invoke_claude_cli_is_callable(self):
        import asyncio
        assert callable(invoke_claude_cli)
