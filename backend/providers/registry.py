"""プロバイダーレジストリとファクトリ。

使い方
------
from backend.providers.registry import create_provider, PROVIDER_REGISTRY

provider = create_provider("google", model="gemini-2.0-flash", settings=settings_dict)
response = await provider.generate(system_prompt, messages)

新しいプロバイダーの追加
------------------------
1. BaseLLMProvider を継承したクラスをモジュール内に作成する。
2. PROVIDER_ID / DEFAULT_MODEL / REQUIRES_API_KEY を設定する。
3. from_config(model, settings, **kwargs) と generate() を実装する。
4. 下記の PROVIDER_REGISTRY に登録するだけで完了。
"""

from backend.providers.anthropic_provider import AnthropicProvider
from backend.providers.base import BaseLLMProvider
from backend.providers.claude_cli_provider import ClaudeCliProvider
from backend.providers.google_provider import GoogleProvider
from backend.providers.ollama_provider import OllamaProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.providers.sakura_provider import SakuraProvider
from backend.providers.xai_provider import XAIProvider

PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    ClaudeCliProvider.PROVIDER_ID: ClaudeCliProvider,
    AnthropicProvider.PROVIDER_ID: AnthropicProvider,
    OpenAIProvider.PROVIDER_ID: OpenAIProvider,
    XAIProvider.PROVIDER_ID: XAIProvider,
    GoogleProvider.PROVIDER_ID: GoogleProvider,
    OpenRouterProvider.PROVIDER_ID: OpenRouterProvider,
    SakuraProvider.PROVIDER_ID: SakuraProvider,
    OllamaProvider.PROVIDER_ID: OllamaProvider,
}

# UI で表示するプロバイダーラベル
PROVIDER_LABELS: dict[str, str] = {
    "claude_cli": "Claude Code CLI (OAuth)",
    "anthropic": "Anthropic API",
    "openai": "OpenAI",
    "xai": "xAI / Grok",
    "google": "Google Gemini",
    "openrouter": "OpenRouter",
    "sakura": "さくらの AI Engine",
    "ollama": "Ollama (ローカル)",
}

# UI での表示順
PROVIDER_ORDER: list[str] = ["claude_cli", "anthropic", "openai", "xai", "google", "openrouter", "sakura", "ollama"]


def get_default_model(provider_id: str) -> str:
    """Return the default model string for a given provider ID."""
    cls = PROVIDER_REGISTRY.get(provider_id)
    return cls.DEFAULT_MODEL if cls is not None else ""


def create_provider(provider_id: str, model: str, settings: dict, **kwargs) -> BaseLLMProvider:
    """Instantiate the provider for *provider_id* via its from_config() factory.

    Falls back to ClaudeCliProvider for unknown provider IDs.
    Extra **kwargs (e.g. character_name) are forwarded to from_config().

    Args:
        provider_id: プロバイダーID（"google", "ollama" 等）。
        model: モデルID。
        settings: グローバル設定 dict。
        preset_name: デバッグログのファイル名に使用するプリセット名（kwargs 経由）。
                     省略時はログに PROVIDER_ID が使われる。
        **kwargs: from_config() に転送する追加引数。
    """
    preset_name: str = kwargs.pop("preset_name", "")
    cls = PROVIDER_REGISTRY.get(provider_id, ClaudeCliProvider)
    instance = cls.from_config(model=model, settings=settings, **kwargs)
    if preset_name:
        instance.preset_name = preset_name
    return instance
