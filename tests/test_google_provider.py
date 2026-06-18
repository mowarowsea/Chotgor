"""GoogleProvider のユニットテスト。

外部 API 通信（google-genai SDK）はすべてモックに置き換え、
SDK 境界の変換・組み立て・エラー伝搬ロジックを検証する。

検証する観点:
    - _build_contents — メッセージ → types.Content 変換
      （role マッピング・空コンテンツ除外・システムプロンプトを本文へ埋め込まないこと）
    - _build_generate_config — system_instruction / thinking_config の組み立て
    - generate / generate_stream / generate_stream_typed — thought=True パートの除外と
      ('thinking', ...) 分離（実APIは通常テキストに thought=None を返す。False ではない）
    - _extract_block_reason / ブロック時のエラー伝搬 — PROHIBITED_CONTENT 等で
      応答が空のとき UI へエラーが届くこと（過去の実バグの回帰テスト）
    - google-genai 未インストール時のエラー返却
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.providers.google_provider import GoogleProvider


# ---------------------------------------------------------------------------
# ヘルパー: モジュール名前空間・SDKレスポンスのモック生成
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
    """指定したパーツ仕様からレスポンス／ストリームチャンクのモックを生成する。

    generate のレスポンスと generate_stream のチャンクは同じ構造
    （candidates[0].content.parts）なので、どちらのテストでも共用する。

    Args:
        parts_spec: (text, thought) のタプルリスト。thought は True または None。

    Returns:
        response/chunk モック。candidates[0].content.parts に parts_spec が反映される。
    """
    parts = [_make_mock_part(text, thought) for text, thought in parts_spec]
    content = MagicMock()
    content.parts = parts
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    # debug_logger がレスポンスを JSON 化する経路向け（dict を返せば安全）
    response.model_dump = MagicMock(return_value={})
    return response


def _make_blocked_chunk(block_reason: str | None = None, finish_reason: str | None = None, has_content: bool = False):
    """ブロック検出テスト用の chunk/response モックを生成するヘルパー。

    PROHIBITED_CONTENT のような状況を再現するため、prompt_feedback.block_reason や
    candidates[0].finish_reason をセットできる柔軟なモックを返す。

    Args:
        block_reason: prompt_feedback.block_reason の値（None なら未設定）。
        finish_reason: candidates[0].finish_reason の値（None なら candidates 自体を空にする）。
        has_content: True なら candidate.content をテキスト付きで生成、False なら content=None。
    """
    obj = MagicMock()
    obj.model_dump = MagicMock(return_value={})
    if block_reason is not None:
        pf = MagicMock()
        pf.block_reason = block_reason
        obj.prompt_feedback = pf
    else:
        obj.prompt_feedback = None

    if finish_reason is not None:
        candidate = MagicMock()
        candidate.finish_reason = finish_reason
        if has_content:
            part = _make_mock_part("回答", None)
            content = MagicMock()
            content.parts = [part]
            candidate.content = content
        else:
            candidate.content = None
        obj.candidates = [candidate]
    else:
        obj.candidates = []
    return obj


# ---------------------------------------------------------------------------
# _build_contents — メッセージ変換
# ---------------------------------------------------------------------------


class TestBuildContents:
    """_build_contents のメッセージ → types.Content 変換を検証するテストクラス。

    role マッピング・空コンテンツ除外と、システムプロンプトをメッセージ本文へ
    埋め込まないこと（system_instruction 経由に一本化されていること）を確認する。
    """

    def test_assistant_role_mapped_to_model(self):
        """role='assistant' のメッセージが types.Content(role='model', ...) に変換されること。"""
        provider = GoogleProvider(api_key="dummy")
        messages = [{"role": "assistant", "content": "はい"}]

        captured_calls = []

        with _patch_provider() as stack:
            # Content コンストラクタの呼び出し引数を捕捉する
            stack._mock_types.Content = MagicMock(side_effect=lambda **kw: captured_calls.append(kw))
            provider._build_contents(messages)

        assert any(c.get("role") == "model" for c in captured_calls), \
            "assistant ロールは 'model' にマッピングされるべき"

    def test_user_role_not_remapped(self):
        """role='user' のメッセージが types.Content(role='user', ...) のままであること。"""
        provider = GoogleProvider(api_key="dummy")
        messages = [{"role": "user", "content": "テスト"}]

        captured_calls = []

        with _patch_provider() as stack:
            stack._mock_types.Content = MagicMock(side_effect=lambda **kw: captured_calls.append(kw))
            provider._build_contents(messages)

        assert any(c.get("role") == "user" for c in captured_calls)

    def test_system_prompt_not_injected_into_first_user_message(self):
        """_build_contents はシステムプロンプトをユーザーメッセージに埋め込まないこと。

        システムプロンプトは _build_generate_config の system_instruction 経由で
        渡される設計であり、メッセージ本文には混入しない。
        """
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

    def test_empty_content_skipped(self):
        """content が空（None や空文字）のメッセージは contents に追加されないこと。"""
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
# _build_generate_config — system_instruction / thinking_config
# ---------------------------------------------------------------------------


class TestBuildGenerateConfig:
    """_build_generate_config の設定組み立てを検証するテストクラス。

    system_instruction が全モデルで常にセットされること、
    thinking_level に応じた thinking_config の有無を確認する。
    """

    def test_system_instruction_always_set(self):
        """system_instruction が常に GenerateContentConfig に渡されること（Gemma 含む全モデル）。"""
        provider = GoogleProvider(api_key="dummy", model="gemma-3-27b-it")
        captured_kwargs: dict = {}

        with _patch_provider() as stack:
            stack._mock_types.GenerateContentConfig = MagicMock(
                side_effect=lambda **kw: captured_kwargs.update(kw)
            )
            provider._build_generate_config("システムプロンプト")

        assert captured_kwargs.get("system_instruction") == "システムプロンプト", \
            "system_instruction が GenerateContentConfig に渡されていない"

    def test_thinking_config_set_when_thinking_level_not_default(self):
        """thinking_level != 'default' のとき thinking_config が設定されること。"""
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
        provider = GoogleProvider(api_key="dummy")

        with patch("backend.providers.google_provider._GOOGLE_GENAI_AVAILABLE", False):
            result = await provider.generate("sys", [{"role": "user", "content": "hi"}])

        assert "google-genai" in result

    @pytest.mark.asyncio
    async def test_generate_stream_yields_error_when_unavailable(self):
        """google-genai 未インストール時に generate_stream がエラーをyieldして終了すること。"""
        provider = GoogleProvider(api_key="dummy")

        with patch("backend.providers.google_provider._GOOGLE_GENAI_AVAILABLE", False):
            chunks = []
            async for chunk in provider.generate_stream("sys", []):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert "google-genai" in chunks[0]

    @pytest.mark.asyncio
    async def test_generate_stream_typed_yields_error_when_unavailable(self):
        """google-genai 未インストール時に generate_stream_typed が ("error", ...) をyieldすること。

        "error" 型は呼び出し側が「出力に積まない／上書きしない」分岐を行うシグナル。
        """
        provider = GoogleProvider(api_key="dummy")

        with patch("backend.providers.google_provider._GOOGLE_GENAI_AVAILABLE", False):
            result = []
            async for item in provider.generate_stream_typed("sys", []):
                result.append(item)

        assert len(result) == 1
        t, msg = result[0]
        assert t == "error"
        assert "google-genai" in msg


# ---------------------------------------------------------------------------
# generate / generate_stream — thought パート除外の検証
# ---------------------------------------------------------------------------


class TestThoughtFiltering:
    """thought=True のパートが回答テキストに混入しないことを検証するテストクラス。

    Gemma4/Gemini ともに thought=True が思考ブロック、thought=None が通常テキスト。
    generate / generate_stream いずれも思考テキストを除外することを確認する。
    """

    @pytest.mark.asyncio
    async def test_generate_excludes_thought_parts(self):
        """generate が thought=True のパートを除外して通常テキストのみ返すこと。"""
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
        # thought=True パートと通常テキストパートを1チャンクに混在させる
        chunk = _make_mock_response([
            ("思考テキスト", True),
            ("回答テキスト", None),
        ])

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
        chunk = _make_mock_response([
            ("思考テキスト", True),
            ("回答テキスト", None),
        ])

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
        chunk = _make_mock_response([("通常テキスト", None)])

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
        response = _make_mock_response([
            ("思考テキスト", True),
            ("回答テキスト", None),
        ])

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
# _extract_block_reason / PROHIBITED_CONTENT 等のブロック検出
# ---------------------------------------------------------------------------


class TestExtractBlockReason:
    """_extract_block_reason の動作を検証するテストクラス。

    PROHIBITED_CONTENT 等で API がコンテンツを返さなかった場合に、
    prompt_feedback.block_reason または candidates[0].finish_reason を
    正しく抽出することを確認する。通常応答に対しては None を返すべき。
    """

    def test_returns_block_reason_from_prompt_feedback(self):
        """prompt_feedback.block_reason がある場合、その値を文字列で返すこと。"""
        obj = _make_blocked_chunk(block_reason="PROHIBITED_CONTENT")
        assert GoogleProvider._extract_block_reason(obj) == "PROHIBITED_CONTENT"

    def test_returns_finish_reason_when_content_empty(self):
        """finish_reason が STOP/MAX_TOKENS 以外かつ content=None のとき finish_reason を返すこと。"""
        obj = _make_blocked_chunk(finish_reason="SAFETY", has_content=False)
        assert GoogleProvider._extract_block_reason(obj) == "SAFETY"

    def test_returns_none_for_normal_stop(self):
        """finish_reason=STOP の通常応答に対しては None を返すこと。"""
        obj = _make_blocked_chunk(finish_reason="STOP", has_content=True)
        assert GoogleProvider._extract_block_reason(obj) is None

    def test_returns_none_for_max_tokens(self):
        """finish_reason=MAX_TOKENS は正常終了扱いで None を返すこと（途中まで応答あり）。"""
        obj = _make_blocked_chunk(finish_reason="MAX_TOKENS", has_content=True)
        assert GoogleProvider._extract_block_reason(obj) is None

    def test_returns_none_when_no_block_info(self):
        """ブロック情報も candidates もない通常チャンクに対しては None を返すこと。"""
        obj = _make_blocked_chunk()  # 何も設定しない
        assert GoogleProvider._extract_block_reason(obj) is None

    def test_prompt_feedback_takes_priority_over_finish_reason(self):
        """prompt_feedback.block_reason と finish_reason が両方ある場合、block_reason を優先すること。"""
        obj = _make_blocked_chunk(block_reason="PROHIBITED_CONTENT", finish_reason="SAFETY")
        assert GoogleProvider._extract_block_reason(obj) == "PROHIBITED_CONTENT"


class TestBlockedResponseHandling:
    """PROHIBITED_CONTENT 等で応答が空になった場合のエラー伝搬を検証するテストクラス。

    _tool_turn と generate_stream_typed の両方で、ブロック理由が検出された場合に
    UI に届くエラーメッセージが生成されることを確認する。これがないと
    「Gemini が拒否しただけなのに UI に何も表示されない」現象が起きる。
    """

    @pytest.mark.asyncio
    async def test_tool_turn_returns_error_on_blocked(self):
        """_tool_turn が PROHIBITED_CONTENT のとき error=True とエラーテキストを返すこと。"""
        response = _make_blocked_chunk(block_reason="PROHIBITED_CONTENT")

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it")

        with _patch_provider(mock_client):
            result = await provider._tool_turn("sys", [{"role": "user", "content": "hi"}])

        assert result.error is True, "ブロック時は error=True を返すべき"
        assert "PROHIBITED_CONTENT" in result.text, \
            f"ブロック理由がエラーメッセージに含まれていない: {result.text!r}"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_tool_turn_no_error_on_normal_response(self):
        """_tool_turn が通常応答に対しては error=False のままであること（誤検知防止）。"""
        response = _make_mock_response([("通常応答", None)])
        # 通常応答にも prompt_feedback がついてくる場合がある（block_reason=None）
        pf = MagicMock()
        pf.block_reason = None
        response.prompt_feedback = pf

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it")

        with _patch_provider(mock_client):
            result = await provider._tool_turn("sys", [{"role": "user", "content": "hi"}])

        assert result.error is False
        assert result.text == "通常応答"

    @pytest.mark.asyncio
    async def test_stream_typed_yields_error_on_blocked_missing_candidates(self):
        """generate_stream_typed が missing_candidates ケースでブロック理由を ("error", ...) で yield すること。

        実ログ（gemma-4-31b-it の PROHIBITED_CONTENT）と同じ構造を再現する:
        chunk.candidates が空かつ prompt_feedback.block_reason がセットされている状態。
        safety filter ブロックは "text" ではなく "error" 型で通知される（呼び出し側が
        蒸留対象から外す等の判定を行うため）。
        """
        chunk = _make_blocked_chunk(block_reason="PROHIBITED_CONTENT")
        chunk.text = None  # chunk.text フォールバックも発火しないこと

        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([chunk])

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it")

        with _patch_provider(mock_client):
            items = []
            async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
                items.append(item)

        # ("error", "[Google API blocked: PROHIBITED_CONTENT]") が含まれること
        assert any(
            t == "error" and "PROHIBITED_CONTENT" in msg
            for t, msg in items
        ), f"ブロック理由のテキストが yield されていない: {items}"

    @pytest.mark.asyncio
    async def test_stream_typed_yields_error_on_blocked_missing_content(self):
        """generate_stream_typed が candidate.content=None ケースでもブロック理由を ("error", ...) で yield すること。

        candidates は存在するが finish_reason=SAFETY 等で content が None になるパターン。
        """
        chunk = _make_blocked_chunk(finish_reason="SAFETY", has_content=False)
        chunk.text = None

        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([chunk])

        provider = GoogleProvider(api_key="dummy", model="gemma-4-31b-it")

        with _patch_provider(mock_client):
            items = []
            async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
                items.append(item)

        assert any(
            t == "error" and "SAFETY" in msg
            for t, msg in items
        ), f"ブロック理由のテキストが yield されていない: {items}"
