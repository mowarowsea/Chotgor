"""Provider registry and factory.

Usage
-----
from core.providers.registry import create_provider, PROVIDER_REGISTRY

provider = create_provider("google", model="gemini-2.0-flash", settings=settings_dict)
response = await provider.generate(system_prompt, messages)

Adding a new provider
---------------------
1. Create a class in its own module, subclassing BaseLLMProvider.
2. Set PROVIDER_ID, DEFAULT_MODEL, REQUIRES_API_KEY.
3. Implement from_config(model, settings, **kwargs) and generate().
4. Register it in PROVIDER_REGISTRY below — that's all.
"""

from .anthropic_provider import AnthropicProvider
from .base import BaseLLMProvider
from .claude_cli_provider import ClaudeCliProvider
from .google_provider import GoogleProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .xai_provider import XAIProvider

PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    ClaudeCliProvider.PROVIDER_ID: ClaudeCliProvider,
    AnthropicProvider.PROVIDER_ID: AnthropicProvider,
    OpenAIProvider.PROVIDER_ID: OpenAIProvider,
    XAIProvider.PROVIDER_ID: XAIProvider,
    GoogleProvider.PROVIDER_ID: GoogleProvider,
    OllamaProvider.PROVIDER_ID: OllamaProvider,
}

# Human-readable labels (used by the UI)
PROVIDER_LABELS: dict[str, str] = {
    "claude_cli": "Claude Code CLI (OAuth)",
    "anthropic": "Anthropic API",
    "openai": "OpenAI",
    "xai": "xAI / Grok",
    "google": "Google Gemini",
    "ollama": "Ollama (ローカル)",
}

# Display order in the UI
PROVIDER_ORDER: list[str] = ["claude_cli", "anthropic", "openai", "xai", "google", "ollama"]


def get_default_model(provider_id: str) -> str:
    """Return the default model string for a given provider ID."""
    cls = PROVIDER_REGISTRY.get(provider_id)
    return cls.DEFAULT_MODEL if cls is not None else ""


def create_provider(provider_id: str, model: str, settings: dict, **kwargs) -> BaseLLMProvider:
    """Instantiate the provider for *provider_id* via its from_config() factory.

    Falls back to ClaudeCliProvider for unknown provider IDs.
    Extra **kwargs (e.g. character_name) are forwarded to from_config().
    """
    cls = PROVIDER_REGISTRY.get(provider_id, ClaudeCliProvider)
    return cls.from_config(model=model, settings=settings, **kwargs)
