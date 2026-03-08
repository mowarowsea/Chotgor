"""Google Gemini provider via google-genai SDK."""

import asyncio
import base64
import re

from .base import BaseLLMProvider

DEFAULT_MODEL = "gemini-2.0-flash"

_THINKING_BUDGET = {
    "low": 1024,
    "medium": 5000,
    "high": 16000,
}


class GoogleProvider(BaseLLMProvider):
    PROVIDER_ID = "google"
    DEFAULT_MODEL = DEFAULT_MODEL
    REQUIRES_API_KEY = True

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.thinking_level = thinking_level

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "GoogleProvider":
        return cls(api_key=settings.get("google_api_key", ""), model=model, thinking_level=thinking_level)

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return (
                "[Error: google-genai パッケージがインストールされていません。"
                "pip install google-genai を実行してください]"
            )

        if not self.api_key:
            return "[Error: google_api_key が設定されていません。Settings ページで設定してください]"

        client = genai.Client(api_key=self.api_key)

        # Gemma models don't support system_instruction — prepend to first user turn instead
        supports_system_instruction = not self.model.lower().startswith("gemma")

        contents = []
        system_injected = False
        for m in messages:
            role = m.get("role")
            content = m.get("content")

            parts = []

            # プレーンテキストの場合
            if isinstance(content, str):
                text_to_add = content
                if role == "user" and not supports_system_instruction and not system_injected and system_prompt:
                    text_to_add = f"{system_prompt}\n\n---\n\n{text_to_add}"
                    system_injected = True
                parts.append(types.Part(text=text_to_add))

            # リスト形式（マルチモーダル）の場合
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append(types.Part(text=item))
                    elif isinstance(item, dict):
                        itype = item.get("type")
                        if itype == "text":
                            parts.append(types.Part(text=item.get("text", "")))
                        elif itype == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:image/"):
                                # data:image/jpeg;base64,xxxx
                                match = re.match(r"data:image/(\w+);base64,(.+)", url)
                                if match:
                                    mime_type = f"image/{match.group(1)}"
                                    b64_data = match.group(2)
                                    parts.append(
                                        types.Part.from_bytes(
                                            data=base64.b64decode(b64_data),
                                            mime_type=mime_type
                                        )
                                    )

            if parts:
                contents.append(
                    types.Content(role="model" if role == "assistant" else "user", parts=parts)
                )

        def run():
            config_kwargs = {"max_output_tokens": 4096}
            if supports_system_instruction:
                config_kwargs["system_instruction"] = system_prompt
            if self.thinking_level != "default":
                budget = _THINKING_BUDGET[self.thinking_level]
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response.text

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return f"[Google API error: {e}]"
