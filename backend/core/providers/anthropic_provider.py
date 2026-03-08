"""Anthropic API provider (direct API, not CLI)."""

import asyncio

from .base import BaseLLMProvider

DEFAULT_MODEL = "claude-sonnet-4-6"

# thinking_level → budget_tokens (for 4.5 and older models)
_BUDGET_TOKENS = {
    "low": 1024,
    "medium": 5000,
    "high": 16000,
}


def _is_new_api(model: str) -> bool:
    """claude-opus-4-6 / claude-sonnet-4-6 use the effort-based thinking API."""
    return "-4-6" in model


class AnthropicProvider(BaseLLMProvider):
    PROVIDER_ID = "anthropic"
    DEFAULT_MODEL = DEFAULT_MODEL
    REQUIRES_API_KEY = True

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.thinking_level = thinking_level

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "AnthropicProvider":
        return cls(api_key=settings.get("anthropic_api_key", ""), model=model, thinking_level=thinking_level)

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        try:
            import anthropic
        except ImportError:
            return (
                "[Error: anthropic パッケージがインストールされていません。"
                "pip install anthropic を実行してください]"
            )

        if not self.api_key:
            return "[Error: anthropic_api_key が設定されていません。Settings ページで設定してください]"

        client = anthropic.Anthropic(api_key=self.api_key)
        api_messages = [m for m in messages if m.get("role") in ("user", "assistant")]

        def run():
            params: dict = {
                "model": self.model,
                "system": system_prompt,
                "messages": api_messages,
                "max_tokens": 4096,
            }

            if self.thinking_level != "default":
                if _is_new_api(self.model):
                    # claude-4-6: effort-based adaptive thinking
                    params["thinking"] = {"type": "adaptive"}
                    params["output_config"] = {"effort": self.thinking_level}
                    params["temperature"] = 1
                    params["max_tokens"] = 20000
                else:
                    # claude-4-5 and older: budget_tokens
                    budget = _BUDGET_TOKENS[self.thinking_level]
                    params["thinking"] = {"type": "enabled", "budget_tokens": budget}
                    params["temperature"] = 1
                    params["max_tokens"] = budget + 4096

            response = client.messages.create(**params)
            # Filter out thinking blocks; return only text blocks
            return "".join(b.text for b in response.content if b.type == "text")

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return f"[Anthropic API error: {e}]"
