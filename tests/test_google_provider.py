"""Tests for GoogleProvider — Gemma限定処理廃止とモジュールレベルimport対応。

今回の変更点を重点的に検証する：
1. _build_contents がシステムプロンプトをメッセージに埋め込まなくなったこと
2. _build_generate_config が常に system_instruction をセットすること
3. Gemmaモデルに対しても Gemini と同じ扱い（system_instruction セット）になること
4. モジュールレベルimport 移行後も _GOOGLE_GENAI_AVAILABLE=False 時にエラーを返すこと

外部API通信はすべてモックで置き換える。
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# ヘルパー: モジュール名前空間を差し替えるコンテキストマネージャ
# ---------------------------------------------------------------------------


def _make_mock_types():
    """google.genai.types の最小限モックを生成する。

    Returns:
        types モジュール相当の MagicMock オブジェクト。
    """
    mock_types = MagicMock()
    mock_types.Part = MagicMock(side_effect=lambda **kw: {"_part": kw})
    mock_types.Content = MagicMock(side_effect=lambda **kw: {"_content": kw})
    mock_types.SafetySetting = MagicMock(return_value=MagicMock())
    mock_types.GenerateContentConfig = MagicMock(side_effect=lambda **kw: {"_config": kw})
    mock_types.ThinkingConfig = MagicMock(return_value=MagicMock())
    mock_types.Tool = MagicMock(return_value=MagicMock())
    mock_types.FunctionResponse = MagicMock(return_value=MagicMock())
    return mock_types


def _patch_provider(mock_client=None, available: bool = True):
    """GoogleProvider のモジュール名前空間をパッチするコンテキストマネージャを返す。

    Args:
        mock_client: genai.Client() の返り値として使う MagicMock。
            None の場合は空の MagicMock を使用する。
        available: _GOOGLE_GENAI_AVAILABLE に設定する値。

    Returns:
        複数パッチを適用する contextlib.ExitStack 相当のコンテキスト。
    """
    from contextlib import ExitStack

    if mock_client is None:
        mock_client = MagicMock()

    mock_types = _make_mock_types()
    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    stack = ExitStack()
    stack.enter_context(patch("backend.providers.google_provider._GOOGLE_GENAI_AVAILABLE", available))
    stack.enter_context(patch("backend.providers.google_provider.genai", mock_genai))
    stack.enter_context(patch("backend.providers.google_provider.types", mock_types))
    stack.enter_context(patch("backend.lib.debug_logger.ChotgorLogger.log_provider_request"))
    stack._mock_types = mock_types  # type: ignore[attr-defined]
    stack._mock_genai = mock_genai  # type: ignore[attr-defined]
    return stack


# ---------------------------------------------------------------------------
# _build_contents — システムプロンプト非埋め込みの確認
# ---------------------------------------------------------------------------


class TestBuildContents:
    """_build_contents の動作を検証するテストクラス。

    Gemma/Gemini によるモデル分岐がなくなり、
    システムプロンプトをメッセージ内に埋め込まなくなったことを確認する。
    """

    def _make_provider(self, model: str = "gemini-2.0-flash"):
        """テスト用 GoogleProvider を生成するヘルパー。

        Args:
            model: 使用するモデルID。

        Returns:
            GoogleProvider インスタンス。
        """
        from backend.providers.google_provider import GoogleProvider
        return GoogleProvider(api_key="dummy", model=model)

    def test_text_message_converted_to_part(self):
        """プレーンテキストメッセージが types.Part(text=...) に変換されること。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy")
        messages = [{"role": "user", "content": "こんにちは"}]

        with _patch_provider() as stack:
            contents = provider._build_contents(messages)

        # types.Content が1件生成されること
        assert len(contents) == 1

    def test_assistant_role_mapped_to_model(self):
        """role='assistant' のメッセージが types.Content(role='model', ...) に変換されること。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy")
        messages = [{"role": "assistant", "content": "はい"}]

        captured_calls = []

        with _patch_provider() as stack:
            # Content コンストラクタの呼び出し引数を捕捉する
            stack._mock_types.Content = MagicMock(side_effect=lambda **kw: captured_calls.append(kw))
            contents = provider._build_contents(messages)

        assert any(c.get("role") == "model" for c in captured_calls), \
            "assistant ロールは 'model' にマッピングされるべき"

    def test_user_role_not_remapped(self):
        """role='user' のメッセージが types.Content(role='user', ...) のままであること。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy")
        messages = [{"role": "user", "content": "テスト"}]

        captured_calls = []

        with _patch_provider() as stack:
            stack._mock_types.Content = MagicMock(side_effect=lambda **kw: captured_calls.append(kw))
            contents = provider._build_contents(messages)

        assert any(c.get("role") == "user" for c in captured_calls)

    def test_system_prompt_not_injected_into_first_user_message(self):
        """_build_contents はシステムプロンプトをユーザーメッセージに埋め込まないこと。

        旧Gemma対応コードではシステムプロンプトを最初のユーザーターン先頭に
        挿入していたが、Gemma4以降は不要のため削除された。
        """
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy", model="gemma-3-27b-it")
        messages = [{"role": "user", "content": "本文"}]

        captured_part_texts: list[str] = []

        with _patch_provider() as stack:
            stack._mock_types.Part = MagicMock(
                side_effect=lambda **kw: captured_part_texts.append(kw.get("text", ""))
            )
            provider._build_contents(messages)

        # システムプロンプトに相当するテキストが混入していないこと
        assert captured_part_texts == ["本文"], \
            f"システムプロンプトがメッセージに埋め込まれている: {captured_part_texts}"

    def test_gemma_model_no_special_handling(self):
        """Gemmaモデルでも Gemini と同じコードパスを通ること（分岐なし）。

        旧コードにあった `supports_system_instruction` フラグが廃止され、
        戻り値がタプルではなくリスト単体になっていることを確認する。
        """
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy", model="gemma-3-27b-it")
        messages = [{"role": "user", "content": "hi"}]

        with _patch_provider():
            result = provider._build_contents(messages)

        # 旧コードはタプル (contents, bool) を返していたが、
        # 現在はリストのみを返すことを確認する
        assert isinstance(result, list), \
            f"_build_contents はリストを返すべきだが {type(result)} が返った"

    def test_empty_content_skipped(self):
        """content が空（None や空文字）のメッセージは contents に追加されないこと。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy")
        messages = [
            {"role": "user", "content": None},
            {"role": "user", "content": "有効メッセージ"},
        ]

        with _patch_provider():
            contents = provider._build_contents(messages)

        # None コンテンツはスキップされ、有効なメッセージのみ残ること
        assert len(contents) == 1


# ---------------------------------------------------------------------------
# _build_generate_config — system_instruction が常に設定されること
# ---------------------------------------------------------------------------


class TestBuildGenerateConfig:
    """_build_generate_config の動作を検証するテストクラス。

    Gemma/Gemini の分岐廃止により system_instruction が常にセットされることを確認する。
    """

    def test_system_instruction_always_set(self):
        """system_instruction が常に GenerateContentConfig に渡されること。

        旧コードでは Gemma の場合に system_instruction を省略していたが、
        廃止後は全モデルで設定される。
        """
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy", model="gemma-3-27b-it")
        captured_kwargs: dict = {}

        with _patch_provider() as stack:
            stack._mock_types.GenerateContentConfig = MagicMock(
                side_effect=lambda **kw: captured_kwargs.update(kw)
            )
            provider._build_generate_config("システムプロンプト")

        assert captured_kwargs.get("system_instruction") == "システムプロンプト", \
            "system_instruction が GenerateContentConfig に渡されていない"

    def test_system_instruction_set_for_gemini_too(self):
        """Geminiモデルでも system_instruction が設定されること（従来通り）。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash")
        captured_kwargs: dict = {}

        with _patch_provider() as stack:
            stack._mock_types.GenerateContentConfig = MagicMock(
                side_effect=lambda **kw: captured_kwargs.update(kw)
            )
            provider._build_generate_config("Geminiプロンプト")

        assert captured_kwargs.get("system_instruction") == "Geminiプロンプト"

    def test_thinking_config_set_when_thinking_level_not_default(self):
        """thinking_level != 'default' のとき thinking_config が設定されること。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy", model="gemini-2.5-flash", thinking_level="medium")
        thinking_kwargs: dict = {}

        with _patch_provider() as stack:
            stack._mock_types.ThinkingConfig = MagicMock(
                side_effect=lambda **kw: thinking_kwargs.update(kw) or MagicMock()
            )
            stack._mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            provider._build_generate_config("sys")

        assert thinking_kwargs.get("thinking_budget") == 5000  # medium
        assert thinking_kwargs.get("include_thoughts") is True

    def test_thinking_config_not_set_when_default(self):
        """thinking_level == 'default' のとき thinking_config が設定されないこと。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash", thinking_level="default")
        captured_kwargs: dict = {}

        with _patch_provider() as stack:
            stack._mock_types.GenerateContentConfig = MagicMock(
                side_effect=lambda **kw: captured_kwargs.update(kw)
            )
            provider._build_generate_config("sys")

        assert "thinking_config" not in captured_kwargs


# ---------------------------------------------------------------------------
# generate / generate_stream / generate_stream_typed —
# _GOOGLE_GENAI_AVAILABLE=False 時のエラー返却
# ---------------------------------------------------------------------------


class TestPackageUnavailable:
    """google-genai 未インストール時のエラー処理を検証するテストクラス。

    モジュールロード時に ImportError が発生した場合（_GOOGLE_GENAI_AVAILABLE=False）、
    各メソッドが適切なエラーメッセージを返すことを確認する。
    """

    @pytest.mark.asyncio
    async def test_generate_returns_error_when_unavailable(self):
        """google-genai 未インストール時に generate がエラー文字列を返すこと。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy")

        with patch("backend.providers.google_provider._GOOGLE_GENAI_AVAILABLE", False):
            result = await provider.generate("sys", [{"role": "user", "content": "hi"}])

        assert "google-genai" in result

    @pytest.mark.asyncio
    async def test_generate_stream_yields_error_when_unavailable(self):
        """google-genai 未インストール時に generate_stream がエラーをyieldして終了すること。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy")

        with patch("backend.providers.google_provider._GOOGLE_GENAI_AVAILABLE", False):
            chunks = []
            async for chunk in provider.generate_stream("sys", []):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert "google-genai" in chunks[0]

    @pytest.mark.asyncio
    async def test_generate_stream_typed_yields_error_when_unavailable(self):
        """google-genai 未インストール時に generate_stream_typed がエラータプルをyieldすること。"""
        from backend.providers.google_provider import GoogleProvider

        provider = GoogleProvider(api_key="dummy")

        with patch("backend.providers.google_provider._GOOGLE_GENAI_AVAILABLE", False):
            result = []
            async for item in provider.generate_stream_typed("sys", []):
                result.append(item)

        assert len(result) == 1
        t, msg = result[0]
        assert t == "text"
        assert "google-genai" in msg


# ---------------------------------------------------------------------------
# generate / generate_stream — thought パート除外の検証
# ---------------------------------------------------------------------------


def _make_mock_part(text: str, thought):
    """指定した text と thought 属性を持つ part モックを生成する。

    Args:
        text: パートのテキスト内容。
        thought: True=思考ブロック、None=通常テキスト（実APIの挙動に合わせる）。
    """
    part = MagicMock()
    part.text = text
    part.thought = thought
    return part


def _make_mock_response(parts_spec: list[tuple]):
    """指定したパーツ仕様から generate 用レスポンスモックを生成する。

    Args:
        parts_spec: (text, thought) のタプルリスト。thought は True または None。

    Returns:
        response モック。candidates[0].content.parts に parts_spec が反映される。
    """
    parts = [_make_mock_part(text, thought) for text, thought in parts_spec]
    content = MagicMock()
    content.parts = parts
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


class TestThoughtFiltering:
    """thought=True のパートが回答テキストに混入しないことを検証するテストクラス。

    Gemma4/Gemini ともに thought=True が思考ブロック、thought=None が通常テキスト。
    generate / generate_stream いずれも思考テキストを除外することを確認する。
    """

    @pytest.mark.asyncio
    async def test_generate_excludes_thought_parts(self):
        """generate が thought=True のパートを除外して通常テキストのみ返すこと。"""
        from backend.providers.google_provider import GoogleProvider

        response = _make_mock_response([
            ("思考テキスト", True),
            ("回答テキスト", None),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it")

        with _patch_provider(mock_client):
            result = await provider.generate("sys", [{"role": "user", "content": "hi"}])

        assert result == "回答テキスト", f"思考テキストが混入している: {result!r}"
        assert "思考テキスト" not in result

    @pytest.mark.asyncio
    async def test_generate_returns_only_text_when_no_thought(self):
        """thought パートがない場合は全テキストをそのまま返すこと。"""
        from backend.providers.google_provider import GoogleProvider

        response = _make_mock_response([
            ("part1", None),
            ("part2", None),
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response

        provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash")

        with _patch_provider(mock_client):
            result = await provider.generate("sys", [{"role": "user", "content": "hi"}])

        assert result == "part1part2"

    @pytest.mark.asyncio
    async def test_generate_stream_excludes_thought_parts(self):
        """generate_stream が thought=True のパートをyieldしないこと。"""
        from backend.providers.google_provider import GoogleProvider

        # thought=True パートと通常テキストパートを1チャンクに混在させる
        thought_part = _make_mock_part("思考テキスト", True)
        text_part = _make_mock_part("回答テキスト", None)
        content = MagicMock()
        content.parts = [thought_part, text_part]
        candidate = MagicMock()
        candidate.content = content
        chunk = MagicMock()
        chunk.candidates = [candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([chunk])

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it")

        with _patch_provider(mock_client):
            chunks = []
            async for c in provider.generate_stream("sys", [{"role": "user", "content": "hi"}]):
                chunks.append(c)

        assert chunks == ["回答テキスト"], f"思考テキストが混入している: {chunks}"

    @pytest.mark.asyncio
    async def test_generate_stream_typed_thought_true_is_thinking(self):
        """generate_stream_typed が thought=True を ('thinking', ...) としてyieldすること。"""
        from backend.providers.google_provider import GoogleProvider

        thought_part = _make_mock_part("思考テキスト", True)
        text_part = _make_mock_part("回答テキスト", None)
        content = MagicMock()
        content.parts = [thought_part, text_part]
        candidate = MagicMock()
        candidate.content = content
        chunk = MagicMock()
        chunk.candidates = [candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([chunk])

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it", thinking_level="medium")

        with _patch_provider(mock_client):
            result = []
            async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
                result.append(item)

        assert ("thinking", "思考テキスト") in result
        assert ("text", "回答テキスト") in result

    @pytest.mark.asyncio
    async def test_generate_stream_typed_thought_none_is_text(self):
        """generate_stream_typed が thought=None を ('text', ...) としてyieldすること。

        実APIは通常テキストに thought=None を返す（False ではない）。
        """
        from backend.providers.google_provider import GoogleProvider

        text_part = _make_mock_part("通常テキスト", None)
        content = MagicMock()
        content.parts = [text_part]
        candidate = MagicMock()
        candidate.content = content
        chunk = MagicMock()
        chunk.candidates = [candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([chunk])

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it")

        with _patch_provider(mock_client):
            result = []
            async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
                result.append(item)

        assert result == [("text", "通常テキスト")]


    @pytest.mark.asyncio
    async def test_tool_turn_separates_thought_into_thinking_field(self):
        """_tool_turn が thought=True のパートを text ではなく thinking フィールドに格納すること。

        generate_with_tools() → _tool_turn() の経路が SUPPORTS_TOOLS=True の
        実際の呼び出しパスであり、思考テキストは text ではなく thinking に格納されることで
        service.py が ("thinking", ...) チャンクとして UI に転送できるようになる。
        """
        from backend.providers.google_provider import GoogleProvider

        thought_part = _make_mock_part("思考テキスト", True)
        text_part = _make_mock_part("回答テキスト", None)
        content = MagicMock()
        content.parts = [thought_part, text_part]
        candidate = MagicMock()
        candidate.content = content
        response = MagicMock()
        response.candidates = [candidate]
        response.model_dump = MagicMock(return_value={})

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it")

        with _patch_provider(mock_client):
            result = await provider._tool_turn("sys", [{"role": "user", "content": "hi"}])

        assert result.text == "回答テキスト", \
            f"_tool_turn.text に思考テキストが混入している: {result.text!r}"
        assert result.thinking == "思考テキスト", \
            f"_tool_turn.thinking に思考テキストが格納されていない: {result.thinking!r}"


# ---------------------------------------------------------------------------
# generate — Gemmaモデルで system_instruction が使われること
# ---------------------------------------------------------------------------


class TestGemmaSystemInstruction:
    """Gemmaモデルで system_instruction が設定されることを統合的に検証するテストクラス。

    旧コードでは Gemma の場合にシステムプロンプトをメッセージに埋め込み、
    system_instruction を省略していた。廃止後は常に system_instruction が設定される。
    """

    @pytest.mark.asyncio
    async def test_generate_uses_system_instruction_for_gemma(self):
        """Gemmaモデルで generate を呼んだとき system_instruction が設定されること。"""
        from backend.providers.google_provider import GoogleProvider

        response = _make_mock_response([("応答", None)])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response

        provider = GoogleProvider(api_key="dummy", model="gemma-3-27b-it")

        captured_config_kwargs: dict = {}

        with _patch_provider(mock_client) as stack:
            stack._mock_types.GenerateContentConfig = MagicMock(
                side_effect=lambda **kw: captured_config_kwargs.update(kw) or MagicMock()
            )
            await provider.generate("テストシステムプロンプト", [{"role": "user", "content": "hi"}])

        assert captured_config_kwargs.get("system_instruction") == "テストシステムプロンプト", \
            "Gemmaモデルでも system_instruction が GenerateContentConfig に渡されるべき"
