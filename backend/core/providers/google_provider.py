"""Google Gemini provider via google-genai SDK."""

import asyncio
import base64
import re

from ..debug_logger import log_provider_request, log_provider_response
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

    def _build_contents(self, system_prompt: str, messages: list[dict]):
        """Google Gemini 用の contents リストを構築する内部ヘルパー。

        Gemmaモデルはsystem_instructionをサポートしないため、
        最初のユーザーターンの先頭にシステムプロンプトを挿入する。
        """
        from google.genai import types

        # Gemmaモデルはsystem_instructionを使えないのでフラグを立てる
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
        return contents, supports_system_instruction

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Google Gemini APIから応答テキストを一括生成する。"""
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
        contents, supports_system_instruction = self._build_contents(system_prompt, messages)

        def run():
            """同期APIを実行して応答テキストを返す内部関数。"""
            config_kwargs = {
                "max_output_tokens": 4096,
                "safety_settings": [
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ],
            }
            if supports_system_instruction:
                config_kwargs["system_instruction"] = system_prompt
            if self.thinking_level != "default":
                budget = _THINKING_BUDGET[self.thinking_level]
                # include_thoughts=True がないと思考ブロックが返ってこない
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=budget, include_thoughts=True
                )
            config = types.GenerateContentConfig(**config_kwargs)
            log_provider_request("google", {"model": self.model, "contents": contents, "config": config})
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            log_provider_response("google", response.model_dump() if hasattr(response, "model_dump") else str(response))
            return response.text

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return f"[Google API error: {e}]"

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """Google Gemini APIからテキストチャンクをストリーミングで取得する。"""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            yield "[Error: google-genai パッケージがインストールされていません]"
            return

        if not self.api_key:
            yield "[Error: google_api_key が設定されていません。Settings ページで設定してください]"
            return

        import threading

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        client = genai.Client(api_key=self.api_key)
        contents, supports_system_instruction = self._build_contents(system_prompt, messages)

        def run():
            """同期SDKストリーミングをスレッド内で実行し、キューへ送信する。"""
            try:
                config_kwargs = {
                    "max_output_tokens": 4096,
                    "safety_settings": [
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ],
                }
                if supports_system_instruction:
                    config_kwargs["system_instruction"] = system_prompt
                if self.thinking_level != "default":
                    budget = _THINKING_BUDGET[self.thinking_level]
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=budget, include_thoughts=True
                    )
                config = types.GenerateContentConfig(**config_kwargs)
                log_provider_request("google", {"model": self.model, "contents": contents, "config": config})
                for chunk in client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                ):
                    if chunk.text:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield f"[Google API error: {item}]"
                break
            yield item

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """Google Gemini APIから思考ブロックを含む型付きチャンクをストリーミングで取得する。

        thinking_level != "default" のとき include_thoughts=True を設定し、
        各チャンクの parts を走査して part.thought で思考ブロックを判別する。
        thinking_level == "default" のときは ("text", ...) のみyieldする。

        Yields:
            tuple[str, str]: (type, content) 形式。
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            yield ("text", "[Error: google-genai パッケージがインストールされていません]")
            return

        if not self.api_key:
            yield ("text", "[Error: google_api_key が設定されていません。Settings ページで設定してください]")
            return

        import threading

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        client = genai.Client(api_key=self.api_key)
        contents, supports_system_instruction = self._build_contents(system_prompt, messages)

        def run():
            """同期SDKストリーミングを走査し、思考ブロックと通常テキストを区別してキューへ送信する。

            chunk.candidates[0].content.parts を直接走査して part.thought を確認する。
            parts が取得できない場合は chunk.text にフォールバックして ("text", ...) として送信する。
            """
            try:
                config_kwargs: dict = {
                    "max_output_tokens": 4096,
                    "safety_settings": [
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ],
                }
                if supports_system_instruction:
                    config_kwargs["system_instruction"] = system_prompt
                if self.thinking_level != "default":
                    budget = _THINKING_BUDGET[self.thinking_level]
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=budget, include_thoughts=True
                    )
                config = types.GenerateContentConfig(**config_kwargs)
                log_provider_request("google", {"model": self.model, "contents": contents, "config": config})

                for chunk in client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                ):
                    # candidates[0].content.parts を直接走査して思考ブロックを抽出する
                    try:
                        parts = chunk.candidates[0].content.parts or []
                    except (AttributeError, IndexError):
                        # parts が取得できない場合は chunk.text にフォールバックする
                        if chunk.text:
                            loop.call_soon_threadsafe(queue.put_nowait, ("text", chunk.text))
                        continue

                    for part in parts:
                        if not part.text:
                            continue
                        if getattr(part, "thought", False):
                            loop.call_soon_threadsafe(queue.put_nowait, ("thinking", part.text))
                        else:
                            loop.call_soon_threadsafe(queue.put_nowait, ("text", part.text))

            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield ("text", f"[Google API error: {item}]")
                break
            yield item
