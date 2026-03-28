"""OpenAI-compatible provider (OpenAI API)。

xAI / Grok は xai_provider.py に分離されている。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from backend.character_actions.executor import OPENAI_TOOLS, ToolCall, ToolTurnResult
from backend.providers.base import BaseLLMProvider, _api_guard, _api_guard_tool_turn

# thinking_level → reasoning_effort
_OPENAI_REASONING = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI APIを呼び出すプロバイダー。"""

    PROVIDER_ID = "openai"
    DEFAULT_MODEL = "gpt-4o"
    REQUIRES_API_KEY = True
    SUPPORTS_TOOLS = True
    _API_SETTINGS_KEY = "openai_api_key"

    _REASONING_MAP = _OPENAI_REASONING

    def __init__(self, api_key: str, model: str = "", base_url: Optional[str] = None, thinking_level: str = "default"):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url
        self.thinking_level = thinking_level

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "OpenAIProvider":
        return cls(api_key=settings.get("openai_api_key", ""), model=model, thinking_level=thinking_level)

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """OpenAI API からチャット向けモデル一覧を取得して返す。"""
        import httpx
        api_key = settings.get(cls._API_SETTINGS_KEY, "")
        base_url = getattr(cls, "BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            return []
        # チャット向けでないモデルを除外するキーワード
        _EXCLUDE = (
            "embedding", "whisper", "tts", "dall-e", "moderation",
            "text-similarity", "text-search", "babbage", "davinci",
            "ada", "curie", "instruct",
        )
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
            models = [
                {"id": m["id"], "name": m.get("name", m["id"])}
                for m in data.get("data", [])
                if not any(kw in m["id"] for kw in _EXCLUDE)
            ]
            return sorted(models, key=lambda m: m["id"])
        except Exception:
            return []

    def _reasoning_effort(self) -> Optional[str]:
        """thinking_level に対応する reasoning_effort 文字列を返す。"""
        return self._REASONING_MAP.get(self.thinking_level)

    def _make_openai_client(self):
        """OpenAIクライアントを初期化して返す内部ヘルパー。"""
        from openai import OpenAI
        init_kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            init_kwargs["base_url"] = self.base_url
        return OpenAI(**init_kwargs)

    @_api_guard("openai")
    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """OpenAI APIから応答テキストを一括生成する。"""
        client = self._make_openai_client()
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
            self._log_request(call_kwargs)
            response = client.chat.completions.create(**call_kwargs)
            self._log_response(response.model_dump())
            return response.choices[0].message.content

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return f"[OpenAI API error: {e}]"

    @_api_guard("openai")
    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """OpenAI APIからテキストチャンクをストリーミングで取得する。

        XAIProvider はこのメソッドを継承して使用する。
        """
        import threading

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        client = self._make_openai_client()

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
                self._log_request(call_kwargs)
                response = client.chat.completions.create(**call_kwargs)
                for chunk in response:
                    content = chunk.choices[0].delta.content if chunk.choices else None
                    if content:
                        accumulated.append(content)
                        loop.call_soon_threadsafe(queue.put_nowait, content)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(e)))
            finally:
                self._log_response("".join(accumulated))
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

    @_api_guard_tool_turn("openai")
    async def _tool_turn(self, system_prompt: str, messages: list[dict]) -> ToolTurnResult:
        """OpenAI APIを1ターン呼び出し、テキストと正規化ツール呼び出しを返す。

        Args:
            system_prompt: 構築済みのシステムプロンプト。
            messages: 現在の会話メッセージリスト（OpenAI形式）。

        Returns:
            テキスト・正規化ツール呼び出し・生レスポンスを含む ToolTurnResult。
        """
        import json

        client = self._make_openai_client()

        # OpenAI形式ではsystemメッセージを先頭に追加する
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        effort = self._reasoning_effort()

        def run() -> ToolTurnResult:
            """同期APIを呼び出してToolTurnResultを返す内部関数。"""
            call_kwargs: dict = {
                "model": self.model,
                "messages": all_messages,
                "tools": OPENAI_TOOLS,
                "tool_choice": "auto",
            }
            if effort:
                call_kwargs["reasoning_effort"] = effort
                call_kwargs["max_completion_tokens"] = 16000
            else:
                call_kwargs["max_tokens"] = 4096

            self._log_request(call_kwargs)
            response = client.chat.completions.create(**call_kwargs)
            self._log_response(response.model_dump())

            message = response.choices[0].message
            text = message.content or ""
            tool_calls = []
            for tc in message.tool_calls or []:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except Exception:
                    tool_input = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=tool_input))
            return ToolTurnResult(text=text, tool_calls=tool_calls, _raw=message)

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return ToolTurnResult(text=f"[OpenAI API error: {e}]", tool_calls=[])

    def _extend_messages_with_results(
        self,
        messages: list[dict],
        turn_result: ToolTurnResult,
        results: dict[str, str],
    ) -> list[dict]:
        """OpenAI形式でツール実行結果をメッセージリストに追加する。

        アシスタントメッセージ（tool_calls含む）と role: tool メッセージを追加して返す。

        Args:
            messages: 現在の会話メッセージリスト。
            turn_result: 直前の _tool_turn() の結果。
            results: {tool_call_id: result_text} 形式のツール実行結果 dict。

        Returns:
            拡張後の新しいメッセージリスト。
        """
        message = turn_result._raw
        new_messages = list(messages)

        # アシスタントメッセージ（tool_calls フィールドを含む）を追加する
        new_messages.append(message.model_dump(exclude_unset=False))

        # 各ツール結果を role: tool メッセージとして追加する
        for tc in turn_result.tool_calls:
            new_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": results[tc.id],
            })

        return new_messages
