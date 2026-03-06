"""OpenAI-compatible provider (OpenAI and xAI/Grok).

XAIProvider is a thin subclass of OpenAIProvider that fixes the base URL and
default model, so the registry treats xAI as a first-class provider rather than
a special-cased variant of OpenAI.
"""

import asyncio
from typing import Optional

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    PROVIDER_ID = "openai"
    DEFAULT_MODEL = "gpt-4o"
    REQUIRES_API_KEY = True

    def __init__(self, api_key: str, model: str = "", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url

    @classmethod
    def from_config(cls, model: str, settings: dict, **kwargs) -> "OpenAIProvider":
        return cls(api_key=settings.get("openai_api_key", ""), model=model)

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return (
                "[Error: openai パッケージがインストールされていません。"
                "pip install openai を実行してください]"
            )

        if not self.api_key:
            provider = "xai_api_key" if self.base_url else "openai_api_key"
            return f"[Error: {provider} が設定されていません。Settings ページで設定してください]"

        kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages += [m for m in messages if m.get("role") in ("user", "assistant")]

        def run():
            response = client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                max_tokens=4096,
            )
            return response.choices[0].message.content

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return f"[OpenAI API error: {e}]"


class XAIProvider(OpenAIProvider):
    """xAI / Grok — OpenAI-compatible API at a different base URL."""

    PROVIDER_ID = "xai"
    DEFAULT_MODEL = "grok-2-latest"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, api_key: str, model: str = ""):
        super().__init__(api_key=api_key, model=model or self.DEFAULT_MODEL, base_url=self.BASE_URL)

    @classmethod
    def from_config(cls, model: str, settings: dict, **kwargs) -> "XAIProvider":
        return cls(api_key=settings.get("xai_api_key", ""), model=model)
