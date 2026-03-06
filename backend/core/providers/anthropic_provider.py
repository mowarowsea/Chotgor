"""Anthropic API provider (direct API, not CLI)."""

import asyncio

from .base import BaseLLMProvider

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(BaseLLMProvider):
    PROVIDER_ID = "anthropic"
    DEFAULT_MODEL = DEFAULT_MODEL
    REQUIRES_API_KEY = True

    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    @classmethod
    def from_config(cls, model: str, settings: dict, **kwargs) -> "AnthropicProvider":
        return cls(api_key=settings.get("anthropic_api_key", ""), model=model)

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
            response = client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=api_messages,
                max_tokens=4096,
            )
            return response.content[0].text

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return f"[Anthropic API error: {e}]"
