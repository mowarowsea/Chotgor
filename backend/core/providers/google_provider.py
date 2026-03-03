"""Google Gemini provider via google-genai SDK."""

import asyncio

from .base import BaseLLMProvider

DEFAULT_MODEL = "gemini-2.0-flash"


class GoogleProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL

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
            content = m.get("content", "")
            if role == "user":
                if not supports_system_instruction and not system_injected and system_prompt:
                    content = f"{system_prompt}\n\n---\n\n{content}"
                    system_injected = True
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=content)])
                )
            elif role == "assistant":
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=content)])
                )

        def run():
            config_kwargs = {"max_output_tokens": 4096}
            if supports_system_instruction:
                config_kwargs["system_instruction"] = system_prompt
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
