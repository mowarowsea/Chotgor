"""Google Gemini provider via google-genai SDK。

tool-use（function calling）に対応している。SUPPORTS_TOOLS = True を設定しており、
_tool_turn / _extend_messages_with_results を実装することで
inscribe_memory・drift・carve_narrative 等の操作を tool-use で行う。

multi-turn の tool-use ループでは types.Content オブジェクトのリストを
messages として受け渡す（最初のターンのみ dict → Content 変換を行う）。
"""

import asyncio
import base64
import re

from backend.character_actions.executor import OPENAI_TOOLS, ToolCall, ToolTurnResult
from backend.providers.base import BaseLLMProvider

try:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]
    _GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    _GOOGLE_GENAI_AVAILABLE = False

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
    SUPPORTS_TOOLS = True

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.thinking_level = thinking_level

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "GoogleProvider":
        return cls(api_key=settings.get("google_api_key", ""), model=model, thinking_level=thinking_level)

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """Google Generative Language API からモデル一覧を取得して返す。

        generateContent をサポートするモデルのみ返す。
        """
        import httpx
        api_key = settings.get("google_api_key", "")
        if not api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": api_key},
                )
                resp.raise_for_status()
                data = resp.json()
            models = []
            for m in data.get("models", []):
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                # "models/gemini-2.0-flash" → "gemini-2.0-flash"
                model_id = m["name"].removeprefix("models/")
                models.append({"id": model_id, "name": m.get("displayName", model_id)})
            return sorted(models, key=lambda m: m["id"])
        except Exception:
            return []

    @classmethod
    async def list_embedding_models(cls, settings: dict) -> list[dict]:
        """Google Generative Language API から Embedding モデル一覧を取得して返す。

        embedContent をサポートするモデルのみ返す。
        """
        import httpx
        api_key = settings.get("google_api_key", "")
        if not api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": api_key},
                )
                resp.raise_for_status()
                data = resp.json()
            models = []
            for m in data.get("models", []):
                if "embedContent" not in m.get("supportedGenerationMethods", []):
                    continue
                model_id = m["name"].removeprefix("models/")
                models.append({"id": model_id, "name": m.get("displayName", model_id)})
            return sorted(models, key=lambda m: m["id"])
        except Exception:
            return []

    def _build_contents(self, messages: list[dict]) -> list:
        """Google Gemini 用の contents リストを構築する内部ヘルパー。

        Gemma4以降はGeminiと同等にsystem_instructionをサポートするため、
        システムプロンプトの埋め込み処理は行わない。
        """
        contents = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            parts = []
            # プレーンテキストの場合
            if isinstance(content, str):
                parts.append(types.Part(text=content))
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
        return contents

    def _build_generate_config(self, system_prompt: str, tools=None):
        """GenerateContentConfig を構築する共通ヘルパー。

        ContextVar のカウンターが正しく機能するよう、config 構築と
        _log_request 呼び出しは必ず asyncio コンテキスト（スレッド外）で行う。

        Args:
            system_prompt: システムプロンプト文字列。
            tools: tool-use 用の Tool リスト。指定時は thinking_config を設定しない。

        Returns:
            構築済みの GenerateContentConfig オブジェクト。
        """
        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "max_output_tokens": 4096,
            "safety_settings": [
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        }
        if tools is not None:
            # tool-use 時は thinking_config と共存できないため排他
            config_kwargs["tools"] = tools
        elif self.thinking_level != "default":
            budget = _THINKING_BUDGET[self.thinking_level]
            # include_thoughts=True がないと思考ブロックが返ってこない
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget, include_thoughts=True
            )
        return types.GenerateContentConfig(**config_kwargs)

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Google Gemini APIから応答テキストを一括生成する。"""
        if not _GOOGLE_GENAI_AVAILABLE:
            return (
                "[Error: google-genai パッケージがインストールされていません。"
                "pip install google-genai を実行してください]"
            )

        if not self.api_key:
            return "[Error: google_api_key が設定されていません。Settings ページで設定してください]"

        client = genai.Client(api_key=self.api_key)
        contents = self._build_contents(messages)

        # _log_request / _log_response は ContextVar カウンターを使うため
        # asyncio コンテキスト（スレッド外）で呼び出す
        config = self._build_generate_config(system_prompt)
        self._log_request({"model": self.model, "contents": contents, "config": config})

        def run():
            """同期APIを実行して応答オブジェクトを返す内部関数。"""
            return client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        try:
            response = await asyncio.to_thread(run)
            self._log_response(response.model_dump() if hasattr(response, "model_dump") else str(response))
            # thought=True のパートは思考ブロック（Gemma4/Gemini共通）。回答テキストのみ結合して返す。
            text_parts = []
            for candidate in (response.candidates or []):
                if not candidate.content:
                    continue
                for part in (candidate.content.parts or []):
                    if part.text and getattr(part, "thought", None) is not True:
                        text_parts.append(part.text)
            return "".join(text_parts)
        except Exception as e:
            err = f"[Google API error: {e}]"
            self._log_error(err)
            return err

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """Google Gemini APIからテキストチャンクをストリーミングで取得する。

        generate_stream_typed() を呼び出し、思考ブロック ("thinking", ...) を除いた
        通常テキスト ("text", ...) のみ文字列としてyieldする。
        """
        async for chunk_type, text in self.generate_stream_typed(system_prompt, messages):
            if chunk_type == "text":
                yield text

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """Google Gemini APIから思考ブロックを含む型付きチャンクをストリーミングで取得する。

        thinking_level != "default" のとき include_thoughts=True を設定し、
        各チャンクの parts を走査して part.thought で思考ブロックを判別する。
        thinking_level == "default" のときは ("text", ...) のみyieldする。

        Yields:
            tuple[str, str]: (type, content) 形式。
        """
        if not _GOOGLE_GENAI_AVAILABLE:
            yield ("text", "[Error: google-genai パッケージがインストールされていません]")
            return

        if not self.api_key:
            yield ("text", "[Error: google_api_key が設定されていません。Settings ページで設定してください]")
            return

        import threading

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        client = genai.Client(api_key=self.api_key)
        contents = self._build_contents(messages)

        # _log_request は asyncio コンテキストで呼び出す（スレッドに入る前）
        config = self._build_generate_config(system_prompt)
        self._log_request({"model": self.model, "contents": contents, "config": config})

        # 累積テキストをスレッド外に渡すためのコンテナ
        result_holder: list[str] = []

        def run():
            """同期SDKストリーミングを走査し、思考ブロックと通常テキストを区別してキューへ送信する。

            chunk.candidates[0].content.parts を直接走査して part.thought を確認する。
            parts が取得できない場合は chunk.text にフォールバックして ("text", ...) として送信する。
            """
            accumulated = []
            try:
                for chunk in client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                ):
                    # candidates[0].content.parts を直接走査して思考ブロックを抽出する
                    try:
                        parts = chunk.candidates[0].content.parts or []
                    except (AttributeError, IndexError):
                        # parts が取得できない場合は chunk.text にフォールバックする
                        if chunk.text:
                            accumulated.append(chunk.text)
                            loop.call_soon_threadsafe(queue.put_nowait, ("text", chunk.text))
                        continue

                    for part in parts:
                        if not part.text:
                            continue
                        accumulated.append(part.text)
                        # thought=True が思考ブロック（Gemma4/Gemini共通）
                        # thought=null/None/False はいずれも通常テキスト
                        if getattr(part, "thought", None) is True:
                            loop.call_soon_threadsafe(queue.put_nowait, ("thinking", part.text))
                        else:
                            loop.call_soon_threadsafe(queue.put_nowait, ("text", part.text))

            except Exception as e:
                # エラー文字列を accumulated に追加しておくことで、
                # finally の _log_response でエラー内容も Response ファイルに記録される
                accumulated.append(f"\n[Google API error: {e}]")
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(e)))
            finally:
                # 累積テキストを asyncio 側に渡す（None の前にセットされることが保証される）
                result_holder.append("".join(accumulated))
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        # エラーは None sentinel を受け取るまでドレインしてから yield する
        # （result_holder への書き込みが None の前に実行されることを保証するため）
        error_item: RuntimeError | None = None
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                error_item = item
            else:
                yield item

        # _log_response は asyncio コンテキストで呼び出す（スレッド終了後）
        self._log_response(result_holder[0] if result_holder else "")

        if error_item is not None:
            yield ("text", f"[Google API error: {error_item}]")

    async def _tool_turn(self, system_prompt: str, messages: list) -> ToolTurnResult:
        """Google Gemini APIを1ターン呼び出し、テキストと正規化ツール呼び出しを返す。

        最初のターンは dict メッセージを _build_contents() で types.Content に変換する。
        2ターン目以降は _extend_messages_with_results() が返した types.Content リストを
        そのまま使用する（hasattr(messages[0], "parts") で判別）。

        Args:
            system_prompt: 構築済みのシステムプロンプト。
            messages: dict メッセージリスト（初回）または types.Content リスト（2回目以降）。

        Returns:
            テキスト・正規化ツール呼び出し・生レスポンスを含む ToolTurnResult。
            _raw には (contents, response) タプルを格納する。
        """
        if not _GOOGLE_GENAI_AVAILABLE:
            return ToolTurnResult(
                text="[Error: google-genai パッケージがインストールされていません]",
                tool_calls=[],
            )

        if not self.api_key:
            return ToolTurnResult(
                text="[Error: google_api_key が設定されていません。Settings ページで設定してください]",
                tool_calls=[],
            )

        client = genai.Client(api_key=self.api_key)

        # 2ターン目以降は types.Content オブジェクトのリストが渡される
        if messages and hasattr(messages[0], "parts"):
            contents = messages
        else:
            contents = self._build_contents(messages)

        # OPENAI_TOOLS 形式の function["parameters"] は JSON Schema 形式であり
        # Gemini の function_declarations でもそのまま利用できる
        function_declarations = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in OPENAI_TOOLS
        ]

        # tools 指定時は thinking_config が除外される（_build_generate_config 内で排他制御）
        config = self._build_generate_config(
            system_prompt,
            tools=[types.Tool(function_declarations=function_declarations)],
        )

        # _log_request は asyncio コンテキストで呼び出す（スレッドに入る前）
        self._log_request({"model": self.model, "contents": contents, "config": config})

        def run():
            """同期APIを呼び出してレスポンスオブジェクトを返す内部関数。"""
            return client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        try:
            response = await asyncio.to_thread(run)
        except Exception as e:
            err = f"[Google API error: {e}]"
            self._log_error(err)
            return ToolTurnResult(text=err, tool_calls=[])

        # _log_response は asyncio コンテキストで呼び出す（スレッド終了後）
        self._log_response(response.model_dump() if hasattr(response, "model_dump") else str(response))

        text = ""
        thinking = ""
        tool_calls: list[ToolCall] = []
        for candidate in (response.candidates or []):
            if not candidate.content:
                continue
            for part in (candidate.content.parts or []):
                if part.text:
                    if getattr(part, "thought", None) is True:
                        # thought=True のパートは思考ブロックとして分離する
                        thinking += part.text
                    else:
                        text += part.text
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    # プロバイダー固有のIDがないため連番で生成する
                    call_id = f"call_{len(tool_calls)}_{fc.name}"
                    tool_calls.append(ToolCall(
                        id=call_id,
                        name=fc.name,
                        input=dict(fc.args) if fc.args else {},
                    ))

        # _raw に (contents, response) を格納し _extend_messages_with_results で利用する
        return ToolTurnResult(text=text, thinking=thinking, tool_calls=tool_calls, _raw=(contents, response))

    def _extend_messages_with_results(
        self,
        messages: list,
        turn_result: ToolTurnResult,
        results: dict[str, str],
    ) -> list:
        """Google形式でツール実行結果を types.Content リストに追加して返す。

        _tool_turn の _raw に格納した (contents, response) から
        モデルのレスポンス Content を取り出し、各ツール結果を
        function_response パートとして追加する。

        戻り値は types.Content のリストであり、次の _tool_turn に渡される。

        Args:
            messages: 今ターン前のメッセージリスト（未使用。_raw の contents を使う）。
            turn_result: 直前の _tool_turn の結果（_raw に (contents, response) を格納）。
            results: {tool_call_id: result_text} 形式のツール実行結果 dict。

        Returns:
            types.Content オブジェクトのリスト（次ターンの入力として使用）。
        """
        contents, response = turn_result._raw
        # モデルが返した Content（function_call パートを含む）を取り出す
        model_content = response.candidates[0].content

        new_contents = list(contents)
        new_contents.append(model_content)

        # 各ツール呼び出しの結果を function_response パートとして追加する
        for tc in turn_result.tool_calls:
            new_contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tc.name,
                                response={"content": results[tc.id]},
                            )
                        )
                    ],
                )
            )

        return new_contents
