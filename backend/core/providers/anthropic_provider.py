"""Anthropic API provider (direct API, not CLI)。"""

from __future__ import annotations

import asyncio

from ..tools import ANTHROPIC_TOOLS, ToolCall, ToolTurnResult
from .base import BaseLLMProvider, _api_guard, _api_guard_tool_turn

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


def _build_thinking_params(model: str, thinking_level: str) -> dict:
    """thinking_level に応じた追加パラメータ dict を返す内部ヘルパー。"""
    if thinking_level == "default":
        return {}
    if _is_new_api(model):
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": thinking_level},
            "temperature": 1,
            "max_tokens": 20000,
        }
    budget = _BUDGET_TOKENS[thinking_level]
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "temperature": 1,
        "max_tokens": budget + 4096,
    }


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API を直接呼び出すプロバイダー。"""

    PROVIDER_ID = "anthropic"
    DEFAULT_MODEL = DEFAULT_MODEL
    REQUIRES_API_KEY = True
    SUPPORTS_TOOLS = True
    _API_SETTINGS_KEY = "anthropic_api_key"

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.thinking_level = thinking_level

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "AnthropicProvider":
        return cls(api_key=settings.get("anthropic_api_key", ""), model=model, thinking_level=thinking_level)

    @_api_guard("anthropic")
    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Anthropic APIから応答テキストを一括生成する。"""
        import anthropic
        from ..debug_logger import log_provider_request, log_provider_response

        client = anthropic.Anthropic(api_key=self.api_key)
        api_messages = [m for m in messages if m.get("role") in ("user", "assistant")]

        def run():
            params: dict = {
                "model": self.model,
                "system": system_prompt,
                "messages": api_messages,
                "max_tokens": 4096,
                **_build_thinking_params(self.model, self.thinking_level),
            }
            log_provider_request("anthropic", params)
            response = client.messages.create(**params)
            log_provider_response("anthropic", response.model_dump())
            # Filter out thinking blocks; return only text blocks
            return "".join(b.text for b in response.content if b.type == "text")

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return f"[Anthropic API error: {e}]"

    @_api_guard("anthropic")
    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """Anthropic APIからテキストチャンクをストリーミングで取得する。

        同期SDKのストリーミングをthreading + asyncio.Queueで非同期ジェネレータに変換する。
        """
        import threading

        import anthropic

        from ..debug_logger import log_provider_request, log_provider_response

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        client = anthropic.Anthropic(api_key=self.api_key)
        api_messages = [m for m in messages if m.get("role") in ("user", "assistant")]

        params: dict = {
            "model": self.model,
            "system": system_prompt,
            "messages": api_messages,
            "max_tokens": 4096,
            **_build_thinking_params(self.model, self.thinking_level),
        }
        log_provider_request("anthropic", params)

        def run():
            """同期SDKストリーミングをスレッド内で実行し、キューへ送信する。"""
            accumulated = []
            try:
                with client.messages.stream(**params) as stream:
                    for text in stream.text_stream:
                        accumulated.append(text)
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(e)))
            finally:
                log_provider_response("anthropic", "".join(accumulated))
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield f"[Anthropic API error: {item}]"
                break
            yield item

    @_api_guard("anthropic")
    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """Anthropic APIから思考ブロックを含む型付きチャンクをストリーミングで取得する。

        thinking_level が "default" 以外の場合、ThinkingBlockを ("thinking", text) として
        通常テキストを ("text", text) としてyieldする。
        thinking_level == "default" のときは ("text", text) のみ。

        Yields:
            tuple[str, str]: (type, content) 形式。
        """
        import threading

        import anthropic

        from ..debug_logger import log_provider_request, log_provider_response

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        client = anthropic.Anthropic(api_key=self.api_key)
        api_messages = [m for m in messages if m.get("role") in ("user", "assistant")]

        params: dict = {
            "model": self.model,
            "system": system_prompt,
            "messages": api_messages,
            "max_tokens": 4096,
            **_build_thinking_params(self.model, self.thinking_level),
        }
        log_provider_request("anthropic", params)

        def run():
            """同期SDKのイベントストリームを逐次読み取り、型付きチャンクをキューへ送信する。

            content_block_delta イベントの delta.type を見て
            thinking_delta と text_delta を区別する。
            """
            accumulated = []
            try:
                with client.messages.stream(**params) as stream:
                    for event in stream:
                        etype = getattr(event, "type", None)
                        if etype == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta is None:
                                continue
                            dtype = getattr(delta, "type", None)
                            if dtype == "thinking_delta":
                                text = getattr(delta, "thinking", "")
                                if text:
                                    accumulated.append(text)
                                    loop.call_soon_threadsafe(queue.put_nowait, ("thinking", text))
                            elif dtype == "text_delta":
                                text = getattr(delta, "text", "")
                                if text:
                                    accumulated.append(text)
                                    loop.call_soon_threadsafe(queue.put_nowait, ("text", text))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(e)))
            finally:
                log_provider_response("anthropic", "".join(accumulated))
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield ("text", f"[Anthropic API error: {item}]")
                break
            yield item

    @_api_guard_tool_turn("anthropic")
    async def _tool_turn(self, system_prompt: str, messages: list[dict]) -> ToolTurnResult:
        """Anthropic APIを1ターン呼び出し、テキストと正規化ツール呼び出しを返す。

        Args:
            system_prompt: 構築済みのシステムプロンプト。
            messages: 現在の会話メッセージリスト（Anthropic形式）。

        Returns:
            テキスト・正規化ツール呼び出し・生レスポンスを含む ToolTurnResult。
        """
        import anthropic

        from ..debug_logger import log_provider_request, log_provider_response

        client = anthropic.Anthropic(api_key=self.api_key)

        params: dict = {
            "model": self.model,
            "system": system_prompt,
            "tools": ANTHROPIC_TOOLS,
            "messages": messages,
            "max_tokens": 4096,
            **_build_thinking_params(self.model, self.thinking_level),
        }

        def run() -> ToolTurnResult:
            """同期APIを呼び出してToolTurnResultを返す内部関数。"""
            log_provider_request("anthropic", params)
            response = client.messages.create(**params)
            log_provider_response("anthropic", response.model_dump())

            text = "".join(b.text for b in response.content if b.type == "text")
            tool_calls = [
                ToolCall(id=b.id, name=b.name, input=dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ]
            return ToolTurnResult(text=text, tool_calls=tool_calls, _raw=response)

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            return ToolTurnResult(text=f"[Anthropic API error: {e}]", tool_calls=[])

    def _extend_messages_with_results(
        self,
        messages: list[dict],
        turn_result: ToolTurnResult,
        results: dict[str, str],
    ) -> list[dict]:
        """Anthropic形式でツール実行結果をメッセージリストに追加する。

        アシスタントメッセージ（ツール呼び出しブロック含む）と
        tool_result ユーザーメッセージを追加して返す。

        Args:
            messages: 現在の会話メッセージリスト。
            turn_result: 直前の _tool_turn() の結果。
            results: {tool_call_id: result_text} 形式のツール実行結果 dict。

        Returns:
            拡張後の新しいメッセージリスト。
        """
        response = turn_result._raw
        new_messages = list(messages)

        # アシスタントターン（ツール呼び出しブロックを含む生コンテンツ）を追加する
        new_messages.append({
            "role": "assistant",
            "content": [b.model_dump() for b in response.content],
        })

        # ツール結果をuserメッセージとして追加する
        new_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": results[tc.id],
                }
                for tc in turn_result.tool_calls
            ],
        })

        return new_messages
