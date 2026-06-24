"""OpenAI-compatible provider (OpenAI API)。

xAI / Grok は xai_provider.py に分離されている。
"""

from __future__ import annotations

import asyncio
from backend.character_actions.executor import OPENAI_TOOLS, ToolCall, ToolTurnResult
from backend.providers.base import BaseLLMProvider, _api_guard, _api_guard_tool_turn, safe_loop_call

# thinking_level → reasoning_effort 変換テーブル
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

    def __init__(self, api_key: str, model: str = "", base_url: str | None = None, thinking_level: str = "default"):
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

    def _reasoning_effort(self) -> str | None:
        """thinking_level に対応する reasoning_effort 文字列を返す。"""
        return self._REASONING_MAP.get(self.thinking_level)

    def _make_openai_client(self):
        """OpenAIクライアントを初期化して返す内部ヘルパー。"""
        from openai import OpenAI
        init_kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            init_kwargs["base_url"] = self.base_url
        return OpenAI(**init_kwargs)

    def _record_usage_from_response(self, usage) -> None:
        """OpenAI互換レスポンスの usage オブジェクトからトークン使用量を記録する。

        usage は ChatCompletion / ChatCompletionChunk が持つ
        ``prompt_tokens`` / ``completion_tokens`` / ``prompt_tokens_details.cached_tokens``
        を含むオブジェクト。usage が None の場合は何もしない。
        記録失敗はチャット本流に影響しない（usage_recorder 側で握り潰す）。
        """
        from backend.lib.usage_recorder import record_usage

        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        record_usage(
            provider=self.PROVIDER_ID,
            model=self.model,
            preset_name=self.preset_name,
            input_tokens=prompt,
            output_tokens=completion,
            cache_read_input_tokens=cached,
        )

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
                # o系モデルは reasoning_effort + max_completion_tokens を使用する
                call_kwargs["reasoning_effort"] = effort
                call_kwargs["max_completion_tokens"] = 16000
            else:
                call_kwargs["max_tokens"] = 4096
            self._log_request(call_kwargs)
            response = client.chat.completions.create(**call_kwargs)
            self._log_response(response.model_dump())
            self._record_usage_from_response(getattr(response, "usage", None))
            return response.choices[0].message.content

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            err = f"[OpenAI API error: {e}]"
            self._log_error(err)
            return err

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """OpenAI APIからテキストチャンクをストリーミングで取得する。

        generate_stream_typed() を呼び出し、文字列としてのみ返す互換ラッパー。
        ("text", ...) と ("error", ...) の双方を文字列として流す
        （非 typed 経路の旧呼び出し側がエラーをテキスト扱いしていた挙動を維持）。
        XAIProvider / OpenRouterProvider はこのメソッドを継承して使用する。
        """
        async for chunk_type, text in self.generate_stream_typed(system_prompt, messages):
            if chunk_type in ("text", "error"):
                yield text

    @_api_guard("openai")
    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """OpenAI APIから型付きチャンクをストリーミングで取得する。

        通常テキストは ("text", text)、SDK 例外などのエラーは ("error", msg) として
        yield する。呼び出し側は ("error", ...) を出力に積まず、UI 表示・蒸留スキップ
        などの分岐を行う。XAIProvider / OpenRouterProvider はこのメソッドを継承して使用する。

        Yields:
            tuple[str, str]: (type, content) 形式。
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
            last_usage = None
            try:
                call_kwargs: dict = {
                    "model": self.model,
                    "messages": api_messages,
                    "stream": True,
                    # ストリーミング末尾で usage 入りチャンクを送らせるためのフラグ。
                    # OpenAI互換サーバの一部は本オプション非対応だが、無視されるのが一般的。
                    "stream_options": {"include_usage": True},
                }
                if effort:
                    # o-seriesモデルはreasoning_effort + max_completion_tokensを使用
                    call_kwargs["reasoning_effort"] = effort
                    call_kwargs["max_completion_tokens"] = 16000
                else:
                    call_kwargs["max_tokens"] = 4096
                self._log_request(call_kwargs)
                response = client.chat.completions.create(**call_kwargs)
                for chunk in response:
                    # include_usage で末尾に来る usage 入りチャンクは choices が空。
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        last_usage = usage
                    content = chunk.choices[0].delta.content if chunk.choices else None
                    if content:
                        accumulated.append(content)
                        safe_loop_call(loop, queue.put_nowait, ("text", content))
            except Exception as e:
                accumulated.append(f"\n[OpenAI API error: {e}]")
                safe_loop_call(loop, queue.put_nowait, RuntimeError(str(e)))
            finally:
                self._log_response("".join(accumulated))
                self._record_usage_from_response(last_usage)
                safe_loop_call(loop, queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                # SDK 例外を「エラー」型 chunk として通知。蒸留・履歴蓄積に混入させない。
                yield ("error", f"[OpenAI API error: {item}]")
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
            self._record_usage_from_response(getattr(response, "usage", None))

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
            err = f"[OpenAI API error: {e}]"
            self._log_error(err)
            return ToolTurnResult(text=err, tool_calls=[], error=True)

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
