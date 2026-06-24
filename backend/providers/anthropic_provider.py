"""Anthropic API プロバイダー（direct API、CLI 非使用）。"""

from __future__ import annotations

import asyncio

from backend.character_actions.executor import ANTHROPIC_TOOLS, ToolCall, ToolTurnResult
from backend.providers.base import BaseLLMProvider, _api_guard, _api_guard_tool_turn, safe_loop_call

DEFAULT_MODEL = "claude-sonnet-4-6"

# thinking_level → budget_tokens（4.5以前のモデル向け）
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

    def _record_usage_from_response(self, usage) -> None:
        """Anthropic SDK の Usage オブジェクトからトークン使用量を記録する。

        usage は ``input_tokens`` / ``output_tokens`` /
        ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` を持つ。
        usage が None の場合は何もしない。
        記録失敗はチャット本流に影響しない（usage_recorder 側で握り潰す）。
        """
        from backend.lib.usage_recorder import record_usage

        if usage is None:
            return
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        record_usage(
            provider=self.PROVIDER_ID,
            model=self.model,
            preset_name=self.preset_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
        )

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """Anthropic API からモデル一覧を取得して返す。"""
        import httpx
        api_key = settings.get("anthropic_api_key", "")
        if not api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                )
                resp.raise_for_status()
                data = resp.json()
            models = [
                {"id": m["id"], "name": m.get("display_name", m["id"])}
                for m in data.get("data", [])
            ]
            return sorted(models, key=lambda m: m["id"])
        except Exception:
            return []

    @_api_guard("anthropic")
    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Anthropic APIから応答テキストを一括生成する。"""
        import anthropic

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
            self._log_request(params)
            response = client.messages.create(**params)
            self._log_response(response.model_dump())
            self._record_usage_from_response(getattr(response, "usage", None))
            # thinking ブロックを除外してテキストブロックのみ返す
            return "".join(b.text for b in response.content if b.type == "text")

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            err = f"[Anthropic API error: {e}]"
            self._log_error(err)
            return err

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """Anthropic APIからテキストチャンクをストリーミングで取得する。

        generate_stream_typed() を呼び出し、思考ブロック ("thinking", ...) を除いた
        通常テキスト ("text", ...) のみ文字列としてyieldする。
        エラー ("error", ...) は表示互換性のため文字列としてそのまま流す
        （旧呼び出し側はエラーをテキスト扱いしていたため）。
        """
        async for chunk_type, text in self.generate_stream_typed(system_prompt, messages):
            if chunk_type in ("text", "error"):
                yield text

    @_api_guard("anthropic")
    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """Anthropic APIから思考ブロックを含む型付きチャンクをストリーミングで取得する。

        thinking_level が "default" 以外の場合、ThinkingBlockを ("thinking", text) として
        通常テキストを ("text", text) としてyieldする。
        thinking_level == "default" のときは ("text", text) のみ。
        SDK 例外などのエラーは ("error", msg) として yield する。呼び出し側は
        ("error", ...) を出力に積まず、UI 表示・蒸留スキップなどの分岐を行う。

        Yields:
            tuple[str, str]: (type, content) 形式。
        """
        import threading

        import anthropic

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
        self._log_request(params)

        def run():
            """同期SDKのイベントストリームを逐次読み取り、型付きチャンクをキューへ送信する。

            content_block_delta イベントの delta.type を見て
            thinking_delta と text_delta を区別する。
            ストリーム終了時に get_final_message().usage から使用量を記録する。
            """
            accumulated = []
            final_usage = None
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
                                    safe_loop_call(loop, queue.put_nowait, ("thinking", text))
                            elif dtype == "text_delta":
                                text = getattr(delta, "text", "")
                                if text:
                                    accumulated.append(text)
                                    safe_loop_call(loop, queue.put_nowait, ("text", text))
                    # ストリーム正常終了。最終メッセージから usage を取得。
                    try:
                        final_message = stream.get_final_message()
                        final_usage = getattr(final_message, "usage", None)
                    except Exception:
                        # SDKバージョン差で取れない場合は記録しない（本流影響なし）。
                        final_usage = None
            except Exception as e:
                accumulated.append(f"\n[Anthropic API error: {e}]")
                safe_loop_call(loop, queue.put_nowait, RuntimeError(str(e)))
            finally:
                self._log_response("".join(accumulated))
                self._record_usage_from_response(final_usage)
                safe_loop_call(loop, queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                # SDK 例外は「エラー」型 chunk として通知。呼び出し側は積まずに分岐する。
                yield ("error", f"[Anthropic API error: {item}]")
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
            self._log_request(params)
            response = client.messages.create(**params)
            self._log_response(response.model_dump())
            self._record_usage_from_response(getattr(response, "usage", None))

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
            return ToolTurnResult(text=f"[Anthropic API error: {e}]", tool_calls=[], error=True)

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
