"""LLMプロバイダーの基底クラスと共通デコレータ。"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any

from backend.lib.debug_logger import logger

if TYPE_CHECKING:
    from backend.character_actions.executor import ToolExecutor, ToolTurnResult


def _api_guard(package: str):
    """パッケージのインポートとAPIキーの確認を行うデコレータ。

    - async generator → エラーをyieldしてreturn
    - 通常のasync関数 → エラー文字列を返す

    クラス属性 `_API_SETTINGS_KEY` からAPIキーの設定名を動的に取得する。
    これによりサブクラスがクラス属性を上書きするだけで正しいキー名を使える。
    """
    def decorator(fn):
        if inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            async def gen_wrapper(self, *args, **kwargs):
                try:
                    __import__(package)
                except ImportError:
                    yield (
                        "text",
                        f"[Error: {package} パッケージがインストールされていません。"
                        f"pip install {package} を実行してください]",
                    )
                    return
                if hasattr(self, "api_key") and not self.api_key:
                    key = getattr(self, "_API_SETTINGS_KEY", "api_key")
                    yield ("text", f"[Error: {key} が設定されていません。Settings ページで設定してください]")
                    return
                async for item in fn(self, *args, **kwargs):
                    yield item
            return gen_wrapper
        else:
            @functools.wraps(fn)
            async def wrapper(self, *args, **kwargs):
                try:
                    __import__(package)
                except ImportError:
                    return (
                        f"[Error: {package} パッケージがインストールされていません。"
                        f"pip install {package} を実行してください]"
                    )
                if hasattr(self, "api_key") and not self.api_key:
                    key = getattr(self, "_API_SETTINGS_KEY", "api_key")
                    return f"[Error: {key} が設定されていません。Settings ページで設定してください]"
                return await fn(self, *args, **kwargs)
            return wrapper
    return decorator


def _api_guard_tool_turn(package: str):
    """パッケージのインポートとAPIキーの確認を行うデコレータ（ToolTurnResult返却用）。

    エラー時は ToolTurnResult(text=エラーメッセージ, tool_calls=[]) を返す。
    クラス属性 `_API_SETTINGS_KEY` からAPIキーの設定名を動的に取得する。
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            from backend.character_actions.executor import ToolTurnResult

            try:
                __import__(package)
            except ImportError:
                msg = (
                    f"[Error: {package} パッケージがインストールされていません。"
                    f"pip install {package} を実行してください]"
                )
                return ToolTurnResult(text=msg, tool_calls=[])
            if hasattr(self, "api_key") and not self.api_key:
                key = getattr(self, "_API_SETTINGS_KEY", "api_key")
                msg = f"[Error: {key} が設定されていません。Settings ページで設定してください]"
                return ToolTurnResult(text=msg, tool_calls=[])
            return await fn(self, *args, **kwargs)
        return wrapper
    return decorator


class BaseLLMProvider:
    """すべてのLLMプロバイダーが継承する抽象基底クラス。"""

    PROVIDER_ID: str = ""
    DEFAULT_MODEL: str = ""
    REQUIRES_API_KEY: bool = True
    # True のプロバイダーは _tool_turn() と _extend_messages_with_results() を実装し、
    # タグ方式ではなくtool-use（function calling）で記憶・DRIFTを操作する。
    SUPPORTS_TOOLS: bool = False
    # _api_guard / _api_guard_tool_turn デコレータがエラーメッセージで参照するキー名。
    # サブクラスで上書きすることで、継承先でも正しいキー名のエラーが出る。
    _API_SETTINGS_KEY: str = "api_key"

    def _log_request(self, params: Any) -> None:
        """プロバイダーAPIへのリクエストパラメータをデバッグログに記録する。"""
        if not self.PROVIDER_ID:
            raise ValueError(f"{self.__class__.__name__} が PROVIDER_ID を設定していません")
        logger.log_provider_request(self.PROVIDER_ID, params)

    def _log_response(self, data: Any) -> None:
        """プロバイダーAPIからのレスポンスをデバッグログに記録する。"""
        if not self.PROVIDER_ID:
            raise ValueError(f"{self.__class__.__name__} が PROVIDER_ID を設定していません")
        logger.log_provider_response(self.PROVIDER_ID, data)

    @classmethod
    def from_config(cls, model: str, settings: dict, **kwargs) -> "BaseLLMProvider":
        """ファクトリメソッド。サブクラスが各自の設定キーを使って初期化する。"""
        raise NotImplementedError(f"{cls.__name__}.from_config() is not implemented")

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """LLMから応答テキストを生成する（一括返却）。

        Args:
            system_prompt: build_system_prompt() で構築済みのシステムプロンプト。
            messages: {"role": str, "content": str} 形式の辞書リスト（user/assistantのみ）。

        Returns:
            キャラクターの応答テキスト。
        """
        raise NotImplementedError

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """ストリーミング生成。デフォルト実装はgenerate()の結果を一括でyield。

        サブクラスでオーバーライドすることでトークン単位のストリーミングが可能になる。
        """
        yield await self.generate(system_prompt, messages)

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """型付きチャンクのストリーミング生成。

        思考ブロック（ThinkingBlock）と通常テキストを区別してyieldする。
        デフォルト実装はgenerate_stream()を ("text", chunk) 形式でラップする。
        サブクラスでオーバーライドすることで思考ブロックも個別にyieldできる。

        Yields:
            tuple[str, str]: (type, content) 形式。
                type == "text"    : 通常の応答テキスト。
                type == "thinking": 思考ブロック（ThinkingBlock）のテキスト。
        """
        async for chunk in self.generate_stream(system_prompt, messages):
            yield ("text", chunk)

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tool_executor: "ToolExecutor",
    ) -> str:
        """tool-use（function calling）を使って記憶・DRIFTを操作しながら生成する。

        SUPPORTS_TOOLS = True のプロバイダーは _tool_turn() と
        _extend_messages_with_results() を実装することで、このループを利用できる。

        ループ構造:
            1. _tool_turn() でLLMを1ターン呼び出す
            2. ToolExecutorで各ツールを実行し結果を収集する
            3. _extend_messages_with_results() でメッセージリストを拡張する
            4. tool_calls がなくなるまで繰り返す

        Args:
            system_prompt: 構築済みのシステムプロンプト。
            messages: {"role": str, "content": str} 形式の辞書リスト。
            tool_executor: ツール呼び出しを実行するexecutorインスタンス。

        Returns:
            キャラクターの最終応答テキスト（ツール呼び出し行は含まない）。
        """
        api_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
        full_text = ""

        while True:
            result = await self._tool_turn(system_prompt, api_messages)
            full_text += result.text

            if not result.tool_calls:
                break

            # ツールを実行して結果を収集する
            results = {
                tc.id: tool_executor.execute(tc.name, tc.input)
                for tc in result.tool_calls
            }

            # switch_angle が呼ばれた場合はループを即中断する。
            # 再ディスパッチは呼び出し元 (service.py) が担う。
            if tool_executor.switch_request:
                break

            # プロバイダー固有のフォーマットでメッセージリストを拡張する
            api_messages = self._extend_messages_with_results(api_messages, result, results)

        return full_text

    async def _tool_turn(
        self, system_prompt: str, messages: list[dict]
    ) -> "ToolTurnResult":
        """1ターンのLLM呼び出し（ツール付き）。SUPPORTS_TOOLS=Trueのプロバイダーが実装する。

        Args:
            system_prompt: 構築済みのシステムプロンプト。
            messages: 現在の会話メッセージリスト（プロバイダー固有フォーマット）。

        Returns:
            テキスト・正規化ツール呼び出し・生レスポンスを含む ToolTurnResult。
        """
        raise NotImplementedError(f"{self.__class__.__name__}._tool_turn() is not implemented")

    def _extend_messages_with_results(
        self,
        messages: list[dict],
        turn_result: "ToolTurnResult",
        results: dict[str, str],
    ) -> list[dict]:
        """ツール実行結果をメッセージリストに追加する。SUPPORTS_TOOLS=Trueのプロバイダーが実装する。

        Args:
            messages: 現在の会話メッセージリスト。
            turn_result: 直前の _tool_turn() の結果（生レスポンスを含む）。
            results: {tool_call_id: result_text} 形式のツール実行結果 dict。

        Returns:
            アシスタントメッセージとツール結果メッセージを追加した新しいメッセージリスト。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._extend_messages_with_results() is not implemented"
        )
