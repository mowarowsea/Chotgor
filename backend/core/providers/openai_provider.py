"""OpenAI-compatible provider (OpenAI and xAI/Grok).

xAI uses the same OpenAI SDK with a different base_url.
"""

import asyncio
from typing import Optional

from .base import BaseLLMProvider

DEFAULT_MODEL = "gpt-4o"
DEFAULT_XAI_MODEL = "grok-2-latest"


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model or (DEFAULT_XAI_MODEL if base_url else DEFAULT_MODEL)
        self.base_url = base_url

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
