"""LLMプロバイダーの基底クラス。"""


class BaseLLMProvider:
    """すべてのLLMプロバイダーが継承する抽象基底クラス。"""

    PROVIDER_ID: str = ""
    DEFAULT_MODEL: str = ""
    REQUIRES_API_KEY: bool = True

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
