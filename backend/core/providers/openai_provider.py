"""OpenAI-compatible provider (OpenAI and xAI/Grok).

XAIProvider is a thin subclass of OpenAIProvider that fixes the base URL and
default model, so the registry treats xAI as a first-class provider rather than
a special-cased variant of OpenAI.
"""

import asyncio
from typing import Optional

from ..debug_logger import log_provider_request, log_provider_response
from .base import BaseLLMProvider

# thinking_level → reasoning_effort
_OPENAI_REASONING = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# xAI has no "medium"; map it to "low"
_XAI_REASONING = {
    "low": "low",
    "medium": "low",
    "high": "high",
}


class OpenAIProvider(BaseLLMProvider):
    PROVIDER_ID = "openai"
    DEFAULT_MODEL = "gpt-4o"
    REQUIRES_API_KEY = True

    _REASONING_MAP = _OPENAI_REASONING

    def __init__(self, api_key: str, model: str = "", base_url: Optional[str] = None, thinking_level: str = "default"):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url
        self.thinking_level = thinking_level

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "OpenAIProvider":
        return cls(api_key=settings.get("openai_api_key", ""), model=model, thinking_level=thinking_level)

    def _reasoning_effort(self) -> Optional[str]:
        return self._REASONING_MAP.get(self.thinking_level)

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

        init_kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            init_kwargs["base_url"] = self.base_url
        client = OpenAI(**init_kwargs)

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages += [m for m in messages if m.get("role") in ("user", "assistant")]

        effort = self._reasoning_effort()

        def run():
            call_kwargs: dict = {"model": self.model, "messages": api_messages}
            if effort:
                # o-series models use reasoning_effort + max_completion_tokens
                call_kwargs["reasoning_effort"] = effort
                call_kwargs["max_completion_tokens"] = 16000
            else:
                call_kwargs["max_tokens"] = 4096
            log_provider_request(self.PROVIDER_ID, call_kwargs)
            response = client.chat.completions.create(**call_kwargs)
            log_provider_response(self.PROVIDER_ID, response.model_dump())
            return response.choices[0].message.content

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return f"[OpenAI API error: {e}]"

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """OpenAI APIからテキストチャンクをストリーミングで取得する。

        XAIProvider はこのメソッドを継承して使用する。
        """
        try:
            from openai import OpenAI
        except ImportError:
            yield "[Error: openai パッケージがインストールされていません]"
            return

        if not self.api_key:
            provider = "xai_api_key" if self.base_url else "openai_api_key"
            yield f"[Error: {provider} が設定されていません。Settings ページで設定してください]"
            return

        import threading

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        init_kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            init_kwargs["base_url"] = self.base_url
        client = OpenAI(**init_kwargs)

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages += [m for m in messages if m.get("role") in ("user", "assistant")]
        effort = self._reasoning_effort()

        def run():
            """同期SDKストリーミングをスレッド内で実行し、キューへ送信する。"""
            accumulated = []
            try:
                call_kwargs: dict = {"model": self.model, "messages": api_messages, "stream": True}
                if effort:
                    # o-seriesモデルはreasoning_effort + max_completion_tokensを使用
                    call_kwargs["reasoning_effort"] = effort
                    call_kwargs["max_completion_tokens"] = 16000
                else:
                    call_kwargs["max_tokens"] = 4096
                log_provider_request(self.PROVIDER_ID, call_kwargs)
                response = client.chat.completions.create(**call_kwargs)
                for chunk in response:
                    content = chunk.choices[0].delta.content if chunk.choices else None
                    if content:
                        accumulated.append(content)
                        loop.call_soon_threadsafe(queue.put_nowait, content)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(e)))
            finally:
                log_provider_response(self.PROVIDER_ID, "".join(accumulated))
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield f"[OpenAI API error: {item}]"
                break
            yield item


class XAIProvider(OpenAIProvider):
    """xAI / Grok — OpenAI-compatible API at a different base URL."""

    PROVIDER_ID = "xai"
    DEFAULT_MODEL = "grok-2-latest"
    BASE_URL = "https://api.x.ai/v1"

    _REASONING_MAP = _XAI_REASONING

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        super().__init__(api_key=api_key, model=model or self.DEFAULT_MODEL, base_url=self.BASE_URL, thinking_level=thinking_level)

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "XAIProvider":
        return cls(api_key=settings.get("xai_api_key", ""), model=model, thinking_level=thinking_level)
