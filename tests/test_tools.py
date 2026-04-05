"""ToolExecutor・ToolCall・ToolTurnResult・ツールスキーマのテスト。

ツール呼び出しの正規化・記憶の書き込み（inscribe_memory）・inner_narrative の彫り込み（carve_narrative）・
SELF_DRIFT 操作・switch_angle が MemoryManager / DriftManager を通じて正しく実行されるかを検証する。
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.character_actions.executor import (
    ANTHROPIC_TOOLS,
    OPENAI_TOOLS,
    ToolCall,
    ToolExecutor,
    ToolTurnResult,
)


# ---------------------------------------------------------------------------
# ツールスキーマの整合性テスト
# ---------------------------------------------------------------------------

class TestToolSchemas:
    """ツールスキーマが正しく定義されているかを検証する。"""

    def test_anthropic_tools_have_required_fields(self):
        """ANTHROPIC_TOOLS の各エントリが name / description / input_schema を持つ。"""
        for tool in ANTHROPIC_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_openai_tools_have_required_fields(self):
        """OPENAI_TOOLS の各エントリが type: function と function フィールドを持つ。"""
        for tool in OPENAI_TOOLS:
            assert tool.get("type") == "function"
            fn = tool.get("function", {})
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_tool_names_match_between_formats(self):
        """Anthropic と OpenAI 形式でツール名が一致する。"""
        anthropic_names = {t["name"] for t in ANTHROPIC_TOOLS}
        openai_names = {t["function"]["name"] for t in OPENAI_TOOLS}
        assert anthropic_names == openai_names

    def test_expected_tool_names_present(self):
        """inscribe_memory / carve_narrative / drift / drift_reset / switch_angle が存在する。"""
        names = {t["name"] for t in ANTHROPIC_TOOLS}
        assert "inscribe_memory" in names
        assert "carve_narrative" in names
        assert "drift" in names
        assert "drift_reset" in names
        assert "switch_angle" in names

    def test_old_tool_name_carve_memory_is_absent(self):
        """旧ツール名 carve_memory が存在しないこと（改名済み確認）。"""
        names = {t["name"] for t in ANTHROPIC_TOOLS}
        assert "carve_memory" not in names

    def test_old_tool_name_inscribe_narrative_is_absent(self):
        """旧ツール名 inscribe_narrative が存在しないこと（改名済み確認）。"""
        names = {t["name"] for t in ANTHROPIC_TOOLS}
        assert "inscribe_narrative" not in names

    def test_inscribe_memory_required_params(self):
        """inscribe_memory ツールが content / category / impact を必須パラメータとして持つ。"""
        tool = next(t for t in ANTHROPIC_TOOLS if t["name"] == "inscribe_memory")
        required = tool["input_schema"]["required"]
        assert "content" in required
        assert "category" in required
        assert "impact" in required

    def test_category_enum_values(self):
        """inscribe_memory の category フィールドが4種のカテゴリを列挙している。"""
        tool = next(t for t in ANTHROPIC_TOOLS if t["name"] == "inscribe_memory")
        enum = tool["input_schema"]["properties"]["category"]["enum"]
        assert set(enum) == {"identity", "user", "semantic", "contextual"}

    def test_carve_narrative_required_params(self):
        """carve_narrative ツールが mode / content を必須パラメータとして持つ。"""
        tool = next(t for t in ANTHROPIC_TOOLS if t["name"] == "carve_narrative")
        required = tool["input_schema"]["required"]
        assert "mode" in required
        assert "content" in required

    def test_carve_narrative_mode_enum_values(self):
        """carve_narrative の mode フィールドが append / overwrite を列挙している。"""
        tool = next(t for t in ANTHROPIC_TOOLS if t["name"] == "carve_narrative")
        enum = tool["input_schema"]["properties"]["mode"]["enum"]
        assert set(enum) == {"append", "overwrite"}


# ---------------------------------------------------------------------------
# ToolCall / ToolTurnResult データクラスのテスト
# ---------------------------------------------------------------------------

class TestToolCallDataClass:
    """ToolCall データクラスの基本動作を検証する。"""

    def test_fields_are_accessible(self):
        """id / name / input フィールドに正しくアクセスできる。"""
        tc = ToolCall(id="call-1", name="inscribe_memory", input={"content": "test", "category": "user", "impact": 1.0})
        assert tc.id == "call-1"
        assert tc.name == "inscribe_memory"
        assert tc.input["category"] == "user"

    def test_carve_narrative_tool_call(self):
        """carve_narrative ツール呼び出しの ToolCall が正しく生成されること。"""
        tc = ToolCall(id="call-2", name="carve_narrative", input={"mode": "append", "content": "新しい指針"})
        assert tc.name == "carve_narrative"
        assert tc.input["mode"] == "append"
        assert tc.input["content"] == "新しい指針"


class TestToolTurnResultDataClass:
    """ToolTurnResult データクラスの基本動作を検証する。"""

    def test_no_tool_calls_case(self):
        """ツール呼び出しなしの場合、tool_calls は空リストになる。"""
        result = ToolTurnResult(text="こんにちは", tool_calls=[])
        assert result.text == "こんにちは"
        assert result.tool_calls == []

    def test_with_tool_calls(self):
        """ToolCall リストが正しく格納される。"""
        tc = ToolCall(id="x", name="drift", input={"content": "穏やかに話す"})
        result = ToolTurnResult(text="", tool_calls=[tc], _raw="raw_response")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "drift"
        assert result._raw == "raw_response"

    def test_raw_field_not_in_repr(self):
        """_raw フィールドは repr に含まれない（repr=False）。"""
        result = ToolTurnResult(text="hi", tool_calls=[], _raw="big_object")
        assert "_raw" not in repr(result)


# ---------------------------------------------------------------------------
# ToolExecutor のテスト
# ---------------------------------------------------------------------------

class TestToolExecutorExecute:
    """ToolExecutor.execute() がルーティングを正しく行うかを検証する。"""

    def _make_executor(self, memory_manager=None, drift_manager=None, session_id="sess-1"):
        """テスト用の ToolExecutor を生成するヘルパー。

        Args:
            memory_manager: MemoryManager のモック。None の場合は MagicMock を使用。
            drift_manager: DriftManager のモック。None の場合は MagicMock を使用。
            session_id: セッション ID。

        Returns:
            ToolExecutor インスタンス。
        """
        mm = memory_manager or MagicMock()
        dm = drift_manager or MagicMock()
        return ToolExecutor(
            character_id="char-1",
            session_id=session_id,
            memory_manager=mm,
            drift_manager=dm,
        )

    def test_unknown_tool_returns_error(self):
        """未知のツール名に対してエラーメッセージを返す。"""
        executor = self._make_executor()
        result = executor.execute("nonexistent_tool", {})
        assert "[Unknown tool:" in result

    def test_inscribe_memory_calls_write_memory(self):
        """inscribe_memory ツールが MemoryManager.write_memory() を呼び出す。"""
        mm = MagicMock()
        executor = self._make_executor(memory_manager=mm)
        result = executor.execute(
            "inscribe_memory",
            {"content": "ユーザは猫が好き", "category": "user", "impact": 1.2},
        )
        assert result == "記憶に刻んだ。"
        mm.write_memory.assert_called_once()
        call_kwargs = mm.write_memory.call_args.kwargs
        assert call_kwargs["character_id"] == "char-1"
        assert call_kwargs["content"] == "ユーザは猫が好き"
        assert call_kwargs["category"] == "user"

    def test_carve_narrative_append_calls_sqlite_update(self):
        """carve_narrative（append）ツールが sqlite_store.update_character を呼び出す。"""
        mm = MagicMock()
        # sqlite.get_character が Character-like オブジェクトを返すよう設定
        mock_char = MagicMock()
        mock_char.inner_narrative = "既存指針"
        mm.sqlite.get_character.return_value = mock_char

        executor = self._make_executor(memory_manager=mm)
        result = executor.execute(
            "carve_narrative",
            {"mode": "append", "content": "新しい自己指針"},
        )
        assert result == "inner_narrative を更新した。"
        mm.sqlite.update_character.assert_called_once()
        call_kwargs = mm.sqlite.update_character.call_args.kwargs
        assert call_kwargs.get("inner_narrative") == "既存指針\n新しい自己指針"

    def test_carve_narrative_empty_content_returns_error(self):
        """carve_narrative に空 content を渡した場合、エラーメッセージを返す。"""
        executor = self._make_executor()
        result = executor.execute("carve_narrative", {"mode": "append", "content": ""})
        assert "content が空" in result

    def test_drift_calls_add_drift(self):
        """drift ツールが DriftManager.add_drift() を呼び出す。"""
        dm = MagicMock()
        executor = self._make_executor(drift_manager=dm)
        result = executor.execute("drift", {"content": "穏やかに話す"})
        assert result == "指針を設定した。"
        dm.add_drift.assert_called_once_with("sess-1", "char-1", "穏やかに話す")

    def test_drift_reset_calls_reset_drifts(self):
        """drift_reset ツールが DriftManager.reset_drifts() を呼び出す。"""
        dm = MagicMock()
        executor = self._make_executor(drift_manager=dm)
        result = executor.execute("drift_reset", {})
        assert result == "指針をリセットした。"
        dm.reset_drifts.assert_called_once_with("sess-1", "char-1")

    def test_switch_angle_stores_switch_request(self):
        """switch_angle ツールが executor.switch_request にリクエストを格納する。"""
        executor = self._make_executor()
        result = executor.execute(
            "switch_angle",
            {"preset_name": "gemini2FlashLite", "self_instruction": "軽やかに話す"},
        )
        assert executor.switch_request == ("gemini2FlashLite", "軽やかに話す")
        assert "gemini2FlashLite" in result


# ---------------------------------------------------------------------------
# ToolExecutor: inscribe_memory のインポータンス計算テスト
# ---------------------------------------------------------------------------

class TestToolExecutorImportanceCalculation:
    """inscribe_memory のインパクト係数がインポータンス計算に正しく反映されることを検証する。"""

    def _run_inscribe_memory(self, category: str, impact: float) -> dict:
        """inscribe_memory を実行して write_memory に渡された kwargs を返すヘルパー。

        Args:
            category: 記憶カテゴリ。
            impact: 重要度係数。

        Returns:
            write_memory に渡された kwargs dict。
        """
        mm = MagicMock()
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=mm, drift_manager=None
        )
        executor.execute("inscribe_memory", {"content": "test", "category": category, "impact": impact})
        return mm.write_memory.call_args.kwargs

    def test_identity_category_has_high_identity_importance(self):
        """identity カテゴリでは identity_importance が最大になる。"""
        kwargs = self._run_inscribe_memory("identity", 1.0)
        assert kwargs["identity_importance"] > kwargs["contextual_importance"]
        assert kwargs["identity_importance"] > kwargs["user_importance"]

    def test_user_category_has_high_user_importance(self):
        """user カテゴリでは user_importance が最大になる。"""
        kwargs = self._run_inscribe_memory("user", 1.0)
        assert kwargs["user_importance"] > kwargs["semantic_importance"]
        assert kwargs["user_importance"] > kwargs["contextual_importance"]

    def test_impact_multiplier_scales_importance(self):
        """impact 係数 2.0 は 1.0 の2倍の重要度になる。"""
        kwargs_1x = self._run_inscribe_memory("contextual", 1.0)
        kwargs_2x = self._run_inscribe_memory("contextual", 2.0)
        assert abs(kwargs_2x["contextual_importance"] - kwargs_1x["contextual_importance"] * 2) < 1e-9

    def test_unknown_category_uses_default_base(self):
        """未知のカテゴリでもエラーにならず、デフォルト基準値 0.5 を使う。"""
        kwargs = self._run_inscribe_memory("unknown_cat", 1.0)
        assert kwargs["contextual_importance"] == pytest.approx(0.5)
        assert kwargs["semantic_importance"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ToolExecutor の境界条件・エラー処理テスト
# ---------------------------------------------------------------------------

class TestToolExecutorEdgeCases:
    """ToolExecutor の境界条件・エラー処理を検証する。"""

    def test_drift_without_session_id_returns_unavailable(self):
        """session_id が None の場合は drift ツールが利用不可メッセージを返す。"""
        dm = MagicMock()
        executor = ToolExecutor(
            character_id="c", session_id=None, memory_manager=MagicMock(), drift_manager=dm
        )
        result = executor.execute("drift", {"content": "test"})
        assert "利用できない" in result
        dm.add_drift.assert_not_called()

    def test_drift_without_drift_manager_returns_unavailable(self):
        """drift_manager が None の場合は drift ツールが利用不可メッセージを返す。"""
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=MagicMock(), drift_manager=None
        )
        result = executor.execute("drift", {"content": "test"})
        assert "利用できない" in result

    def test_drift_reset_without_session_id_returns_unavailable(self):
        """session_id が None の場合は drift_reset ツールが利用不可メッセージを返す。"""
        executor = ToolExecutor(
            character_id="c", session_id=None, memory_manager=MagicMock(), drift_manager=MagicMock()
        )
        result = executor.execute("drift_reset", {})
        assert "利用できない" in result

    def test_inscribe_memory_exception_returns_error_message(self):
        """write_memory が例外を投げた場合、エラーメッセージを返す（クラッシュしない）。"""
        mm = MagicMock()
        mm.write_memory.side_effect = RuntimeError("DB connection failed")
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=mm, drift_manager=None
        )
        result = executor.execute("inscribe_memory", {"content": "test", "category": "user", "impact": 1.0})
        assert "[inscribe_memory error:" in result

    def test_drift_exception_returns_error_message(self):
        """add_drift が例外を投げた場合、エラーメッセージを返す（クラッシュしない）。"""
        dm = MagicMock()
        dm.add_drift.side_effect = RuntimeError("DB error")
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=MagicMock(), drift_manager=dm
        )
        result = executor.execute("drift", {"content": "test"})
        assert "[drift error:" in result

    def test_carve_narrative_exception_returns_error_message(self):
        """update_character が例外を投げた場合、carve_narrative はエラーメッセージを返す（クラッシュしない）。"""
        mm = MagicMock()
        mm.sqlite.get_character.return_value = MagicMock(inner_narrative="")
        mm.sqlite.update_character.side_effect = RuntimeError("DB error")
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=mm, drift_manager=None
        )
        result = executor.execute("carve_narrative", {"mode": "append", "content": "指針"})
        assert "[carve_narrative error:" in result


# ---------------------------------------------------------------------------
# _api_guard デコレータのテスト
# ---------------------------------------------------------------------------

class TestApiGuardDecorator:
    """_api_guard デコレータが ImportError と API key チェックを正しく行うかを検証する。"""

    def test_missing_package_returns_error_string(self):
        """存在しないパッケージ名でデコレートされた generate() はエラー文字列を返す。"""
        import asyncio
        from backend.providers.base import _api_guard
        from backend.providers.base import BaseLLMProvider

        class FakeProvider(BaseLLMProvider):
            """テスト用フェイクプロバイダー（パッケージ不在）。"""
            _API_SETTINGS_KEY = "fake_api_key"

            @_api_guard("__nonexistent_package_xyz__")
            async def generate(self, system_prompt, messages):
                """テスト用 generate。"""
                return "OK"

        async def run():
            p = FakeProvider()
            return await p.generate("sys", [])

        result = asyncio.run(run())
        assert "[Error:" in result
        assert "__nonexistent_package_xyz__" in result

    def test_empty_api_key_returns_error_with_settings_key_name(self):
        """api_key が空の場合、_API_SETTINGS_KEY の名前を含むエラーメッセージを返す。"""
        import asyncio
        from backend.providers.base import _api_guard
        from backend.providers.base import BaseLLMProvider

        class FakeProvider(BaseLLMProvider):
            """テスト用フェイクプロバイダー（APIキー空）。"""
            _API_SETTINGS_KEY = "my_special_api_key"

            @_api_guard("sys")
            async def generate(self, system_prompt, messages):
                """テスト用 generate。"""
                return "OK"

        async def run():
            p = FakeProvider()
            p.api_key = ""
            return await p.generate("sys", [])

        result = asyncio.run(run())
        assert "[Error:" in result
        assert "my_special_api_key" in result

    def test_valid_api_key_passes_through(self):
        """api_key が設定されていれば元のメソッドを呼び出す。"""
        import asyncio
        from backend.providers.base import _api_guard
        from backend.providers.base import BaseLLMProvider

        class FakeProvider(BaseLLMProvider):
            """テスト用フェイクプロバイダー（正常）。"""

            @_api_guard("sys")
            async def generate(self, system_prompt, messages):
                """テスト用 generate。"""
                return "response_ok"

        async def run():
            p = FakeProvider()
            p.api_key = "sk-test"
            return await p.generate("sys", [])

        result = asyncio.run(run())
        assert result == "response_ok"

    def test_async_generator_yields_error_on_missing_package(self):
        """async generator にデコレートされた generate_stream() はエラーを yield してリターンする。"""
        import asyncio
        from backend.providers.base import _api_guard
        from backend.providers.base import BaseLLMProvider

        class FakeProvider(BaseLLMProvider):
            """テスト用フェイクプロバイダー（ストリーミング・パッケージ不在）。"""
            _API_SETTINGS_KEY = "fake_key"

            @_api_guard("__nonexistent_package_xyz__")
            async def generate_stream(self, system_prompt, messages):
                """テスト用 generate_stream。"""
                yield "chunk"

        async def run():
            p = FakeProvider()
            chunks = []
            async for chunk in p.generate_stream("sys", []):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        chunk_type, chunk_msg = chunks[0]
        assert chunk_type == "text"
        assert "[Error:" in chunk_msg

    def test_async_generator_yields_error_on_empty_api_key(self):
        """async generator にデコレートされた generate_stream() は API key エラーを yield してリターンする。"""
        import asyncio
        from backend.providers.base import _api_guard
        from backend.providers.base import BaseLLMProvider

        class FakeProvider(BaseLLMProvider):
            """テスト用フェイクプロバイダー（ストリーミング・APIキー空）。"""
            _API_SETTINGS_KEY = "my_key"

            @_api_guard("sys")
            async def generate_stream(self, system_prompt, messages):
                """テスト用 generate_stream。"""
                yield "chunk"

        async def run():
            p = FakeProvider()
            p.api_key = ""
            chunks = []
            async for chunk in p.generate_stream("sys", []):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        chunk_type, chunk_msg = chunks[0]
        assert chunk_type == "text"
        assert "my_key" in chunk_msg

    def test_xai_inherits_correct_settings_key(self):
        """XAIProvider の generate() は xai_api_key を含むエラーメッセージを返す。"""
        import asyncio
        from backend.providers.xai_provider import XAIProvider

        async def run():
            p = XAIProvider(api_key="")
            return await p.generate("sys", [{"role": "user", "content": "hi"}])

        result = asyncio.run(run())
        assert "xai_api_key" in result

    def test_anthropic_settings_key_in_error(self):
        """AnthropicProvider の generate() は anthropic_api_key を含むエラーメッセージを返す。"""
        import asyncio
        from backend.providers.anthropic_provider import AnthropicProvider

        async def run():
            p = AnthropicProvider(api_key="")
            return await p.generate("sys", [{"role": "user", "content": "hi"}])

        result = asyncio.run(run())
        assert "anthropic_api_key" in result


class TestApiGuardToolTurnDecorator:
    """_api_guard_tool_turn デコレータが ToolTurnResult でエラーを返すかを検証する。"""

    def test_missing_package_returns_tool_turn_result_with_error(self):
        """存在しないパッケージ名でデコレートされた _tool_turn() はエラー ToolTurnResult を返す。"""
        import asyncio
        from backend.providers.base import _api_guard_tool_turn, BaseLLMProvider
        from backend.character_actions.executor import ToolTurnResult

        class FakeProvider(BaseLLMProvider):
            """テスト用フェイクプロバイダー（ツールターン・パッケージ不在）。"""

            @_api_guard_tool_turn("__nonexistent_package_xyz__")
            async def _tool_turn(self, system_prompt, messages):
                """テスト用 _tool_turn。"""
                return ToolTurnResult(text="OK", tool_calls=[])

        async def run():
            p = FakeProvider()
            return await p._tool_turn("sys", [])

        result = asyncio.run(run())
        assert isinstance(result, ToolTurnResult)
        assert "[Error:" in result.text
        assert result.tool_calls == []

    def test_empty_api_key_returns_tool_turn_result_with_error(self):
        """api_key が空の場合、_API_SETTINGS_KEY の名前を含むエラー ToolTurnResult を返す。"""
        import asyncio
        from backend.providers.base import _api_guard_tool_turn, BaseLLMProvider
        from backend.character_actions.executor import ToolTurnResult

        class FakeProvider(BaseLLMProvider):
            """テスト用フェイクプロバイダー（ツールターン・APIキー空）。"""
            _API_SETTINGS_KEY = "my_tool_key"

            @_api_guard_tool_turn("sys")
            async def _tool_turn(self, system_prompt, messages):
                """テスト用 _tool_turn。"""
                return ToolTurnResult(text="OK", tool_calls=[])

        async def run():
            p = FakeProvider()
            p.api_key = ""
            return await p._tool_turn("sys", [])

        result = asyncio.run(run())
        assert isinstance(result, ToolTurnResult)
        assert "my_tool_key" in result.text
        assert result.tool_calls == []

    def test_anthropic_tool_turn_error_on_no_key(self):
        """AnthropicProvider の _tool_turn() は api_key なしでエラー ToolTurnResult を返す。"""
        import asyncio
        from backend.providers.anthropic_provider import AnthropicProvider

        async def run():
            p = AnthropicProvider(api_key="")
            return await p._tool_turn("sys", [])

        result = asyncio.run(run())
        assert isinstance(result, ToolTurnResult)
        assert "anthropic_api_key" in result.text
        assert result.tool_calls == []


# ---------------------------------------------------------------------------
# BaseLLMProvider.generate_with_tools のループロジックのテスト
# ---------------------------------------------------------------------------

class TestGenerateWithToolsLoop:
    """generate_with_tools のツールループが正しく機能するかをモックで検証する。"""

    def _make_provider(self, turn_results: list):
        """指定した ToolTurnResult のリストを順に返すモックプロバイダーを生成するヘルパー。

        Args:
            turn_results: _tool_turn() が順に返す ToolTurnResult のリスト。

        Returns:
            BaseLLMProvider を継承したモックプロバイダーインスタンス。
        """
        from backend.providers.base import BaseLLMProvider

        class MockProvider(BaseLLMProvider):
            """テスト用ツールループプロバイダー。"""

            def __init__(self):
                """モックプロバイダーを初期化する。"""
                self._turn_results = iter(turn_results)
                self._extend_calls = []

            async def _tool_turn(self, system_prompt, messages):
                """事前設定した ToolTurnResult を順に返す。"""
                return next(self._turn_results)

            def _extend_messages_with_results(self, messages, turn_result, results):
                """ツール結果をダミーメッセージとして追加する。"""
                self._extend_calls.append((turn_result, results))
                return messages + [{"role": "tool_result_dummy", "content": str(results)}]

        return MockProvider()

    def test_no_tool_calls_returns_text_directly(self):
        """ツール呼び出しなしの場合、テキストをそのまま返す。"""
        import asyncio

        provider = self._make_provider([
            ToolTurnResult(text="こんにちは", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", MagicMock(), MagicMock())

        text, thinking = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "こんにちは"
        assert thinking == ""

    def test_single_inscribe_memory_call_executes_and_continues(self):
        """1回の inscribe_memory 呼び出し → 実行 → 継続が正しく行われる。"""
        import asyncio

        tc = ToolCall(id="tc-1", name="inscribe_memory", input={"content": "X", "category": "user", "impact": 1.0})
        mm = MagicMock()
        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[tc]),
            ToolTurnResult(text="完了した", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", mm, MagicMock())

        text, _ = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "完了した"
        mm.write_memory.assert_called_once()
        assert len(provider._extend_calls) == 1

    def test_single_carve_narrative_call_executes_and_continues(self):
        """1回の carve_narrative 呼び出し → 実行 → 継続が正しく行われる。"""
        import asyncio

        tc = ToolCall(id="tc-2", name="carve_narrative", input={"mode": "append", "content": "新指針"})
        mm = MagicMock()
        mock_char = MagicMock()
        mock_char.inner_narrative = ""
        mm.sqlite.get_character.return_value = mock_char

        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[tc]),
            ToolTurnResult(text="指針を彫り込んだ", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", mm, None)

        text, _ = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "指針を彫り込んだ"
        mm.sqlite.update_character.assert_called_once()

    def test_multiple_tool_calls_in_sequence(self):
        """複数ターンのツール呼び出しが正しくループする。"""
        import asyncio

        tc1 = ToolCall(id="t1", name="drift", input={"content": "穏やか"})
        tc2 = ToolCall(id="t2", name="inscribe_memory", input={"content": "Y", "category": "identity", "impact": 0.5})
        dm = MagicMock()
        mm = MagicMock()
        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[tc1]),
            ToolTurnResult(text="", tool_calls=[tc2]),
            ToolTurnResult(text="終了", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", mm, dm)

        text, _ = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "終了"
        dm.add_drift.assert_called_once_with("s", "c", "穏やか")
        mm.write_memory.assert_called_once()
        assert len(provider._extend_calls) == 2

    def test_text_accumulates_across_turns(self):
        """複数ターンにわたるテキストが結合されて返される。"""
        import asyncio

        tc = ToolCall(id="t1", name="drift", input={"content": "テスト"})
        provider = self._make_provider([
            ToolTurnResult(text="前半", tool_calls=[tc]),
            ToolTurnResult(text="後半", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", MagicMock(), MagicMock())

        text, _ = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "前半後半"
