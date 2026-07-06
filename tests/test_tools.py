"""ToolExecutor・ToolCall・ToolTurnResult・ツールスキーマのテスト。

ツール呼び出しの正規化・記憶の書き込み（inscribe_memory）・inner_narrative の彫り込み（carve_narrative）・
SELF_DRIFT 操作・switch_angle が InscribedMemoryManager / DriftManager を通じて正しく実行されるかを検証する。
"""

import asyncio

import pytest
from unittest.mock import MagicMock, patch

from backend.providers.base import BaseLLMProvider, _api_guard, _api_guard_tool_turn

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
        """inscribe_memory / carve_narrative / post_working_memory_thread / read_working_memory_thread / power_recall / switch_angle が存在する。"""
        names = {t["name"] for t in ANTHROPIC_TOOLS}
        assert "inscribe_memory" in names
        assert "carve_narrative" in names
        assert "post_working_memory_thread" in names
        assert "read_working_memory_thread" in names
        assert "power_recall" in names
        assert "switch_angle" in names

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
        tc = ToolCall(id="x", name="post_working_memory_thread", input={"type": "topic", "summary": "話題"})
        result = ToolTurnResult(text="", tool_calls=[tc], _raw="raw_response")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "post_working_memory_thread"
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

    def _make_executor(self, memory_manager=None, working_memory_manager=None, session_id="sess-1"):
        """テスト用の ToolExecutor を生成するヘルパー。

        Args:
            memory_manager: InscribedMemoryManager のモック。None の場合は MagicMock を使用。
            working_memory_manager: WorkingMemoryManager のモック。None の場合は MagicMock を使用。
            session_id: セッション ID。

        Returns:
            ToolExecutor インスタンス。
        """
        mm = memory_manager or MagicMock()
        wm = working_memory_manager or MagicMock()
        return ToolExecutor(
            character_id="char-1",
            session_id=session_id,
            memory_manager=mm,
            working_memory_manager=wm,
        )

    def test_unknown_tool_returns_error(self):
        """未知のツール名に対してエラーメッセージを返す。"""
        executor = self._make_executor()
        result = executor.execute("nonexistent_tool", {})
        assert "[Unknown tool:" in result

    def test_inscribe_memory_calls_write_memory(self):
        """inscribe_memory ツールが InscribedMemoryManager.write_inscribed_memory() を呼び出す。"""
        mm = MagicMock()
        executor = self._make_executor(memory_manager=mm)
        result = executor.execute(
            "inscribe_memory",
            {"content": "ユーザは猫が好き", "category": "user", "impact": 1.2},
        )
        assert result == "記憶に刻んだ。"
        mm.write_inscribed_memory.assert_called_once()
        call_kwargs = mm.write_inscribed_memory.call_args.kwargs
        assert call_kwargs["character_id"] == "char-1"
        assert call_kwargs["content"] == "ユーザは猫が好き"
        assert call_kwargs["category"] == "user"

    def test_inscribe_memory_default_passes_force_insert_false(self):
        """batch_context 未指定（通常チャット経路）では write_inscribed_memory が force_insert=False で呼ばれる。

        後方互換性の回帰テスト。バッチではない通常の inscribe_memory ツール呼び出しでは、
        類似既存記憶の上書き挙動（重複排除）を維持する必要がある。
        """
        mm = MagicMock()
        executor = self._make_executor(memory_manager=mm)
        executor.execute(
            "inscribe_memory",
            {"content": "ユーザは猫が好き", "category": "user", "impact": 1.0},
        )
        call_kwargs = mm.write_inscribed_memory.call_args.kwargs
        assert call_kwargs.get("force_insert") is False

    def test_inscribe_memory_with_batch_context_passes_force_insert_true(self):
        """batch_context={"force_insert_memory": True} で生成した ToolExecutor は force_insert=True を伝播する。

        forget 蒸留バッチで使われる経路。キャラが inscribe_memory ツールを呼んだとき、
        類似既存への上書きをスキップさせるためのフラグがバッチ → ToolExecutor → Inscriber
        → InscribedMemoryManager.write_inscribed_memory まで一気通貫で伝わることを検証する。

        この経路が壊れると、蒸留結果が削除候補のIDに上書きされて、続く全件削除で消滅する
        （結合バグの再発）。
        """
        mm = MagicMock()
        executor = ToolExecutor(
            character_id="char-1",
            session_id="sess-1",
            memory_manager=mm,
            working_memory_manager=MagicMock(),
            batch_context={"force_insert_memory": True},
        )
        executor.execute(
            "inscribe_memory",
            {"content": "ユーザはコーヒーが好き", "category": "user", "impact": 1.0},
        )
        call_kwargs = mm.write_inscribed_memory.call_args.kwargs
        assert call_kwargs.get("force_insert") is True

    def test_inscribe_memory_batch_context_none_treated_as_empty(self):
        """batch_context=None でも例外にならず、force_insert=False が伝播する。

        ask_character_with_tools のデフォルト引数（batch_context=None）が
        ToolExecutor 側でちゃんと空 dict 扱いされることの確認。
        """
        mm = MagicMock()
        executor = ToolExecutor(
            character_id="char-1",
            session_id="sess-1",
            memory_manager=mm,
            working_memory_manager=MagicMock(),
            batch_context=None,
        )
        executor.execute(
            "inscribe_memory",
            {"content": "x", "category": "user", "impact": 1.0},
        )
        call_kwargs = mm.write_inscribed_memory.call_args.kwargs
        assert call_kwargs.get("force_insert") is False

    def test_carve_narrative_append_calls_sqlite_update(self):
        """carve_narrative（append）ツールが sqlite_store.carve_inner_narrative を呼び出す。

        追記の結合ロジック（改行区切り）は store 側（carve_inner_narrative）に移設済みの
        ため、ここでは mode / content が正しく保存終点へ渡ることだけを検証する
        （結合とタイムライン封筒は tests/test_timeline_events.py で検証）。
        """
        mm = MagicMock()

        executor = self._make_executor(memory_manager=mm)
        result = executor.execute(
            "carve_narrative",
            {"mode": "append", "content": "新しい自己指針"},
        )
        assert result == "inner_narrative を更新した。"
        mm.sqlite.carve_inner_narrative.assert_called_once()
        call_args = mm.sqlite.carve_inner_narrative.call_args.args
        assert call_args[1] == "append"
        assert call_args[2] == "新しい自己指針"

    def test_carve_narrative_empty_content_returns_error(self):
        """carve_narrative に空 content を渡した場合、エラーメッセージを返す。"""
        executor = self._make_executor()
        result = executor.execute("carve_narrative", {"mode": "append", "content": ""})
        assert "content が空" in result

    def test_post_working_memory_thread_new_calls_create_thread(self):
        """post_working_memory_thread ツール（thread_id 省略）が WorkingMemoryManager.create_thread() を呼び出す。"""
        wm = MagicMock()
        wm.create_thread.return_value = {"id": "thread-1"}
        executor = self._make_executor(working_memory_manager=wm)
        result = executor.execute(
            "post_working_memory_thread",
            {"type": "topic", "summary": "気になっている話題", "importance": 0.7},
        )
        assert "thread-1" in result
        wm.create_thread.assert_called_once()
        call_kwargs = wm.create_thread.call_args.kwargs
        assert call_kwargs["character_id"] == "char-1"
        assert call_kwargs["type"] == "topic"
        assert call_kwargs["summary"] == "気になっている話題"

    def test_post_working_memory_thread_existing_calls_add_post(self):
        """post_working_memory_thread ツール（thread_id 指定 + content）が WorkingMemoryManager.add_post() を呼び出す。"""
        wm = MagicMock()
        executor = self._make_executor(working_memory_manager=wm)
        result = executor.execute(
            "post_working_memory_thread",
            {"thread_id": "thread-9", "content": "新しい書き込み"},
        )
        assert "thread-9" in result
        wm.add_post.assert_called_once_with("thread-9", "新しい書き込み")

    def test_read_working_memory_thread_calls_get_thread_detail(self):
        """read_working_memory_thread ツールが WorkingMemoryManager.get_thread_detail() を呼び出す。"""
        wm = MagicMock()
        wm.get_thread_detail.return_value = {"id": "thread-9", "posts": []}
        executor = self._make_executor(working_memory_manager=wm)
        result = executor.execute("read_working_memory_thread", {"thread_id": "thread-9"})
        assert "thread-9" in result
        wm.get_thread_detail.assert_called_once_with("thread-9")

    def test_close_working_memory_thread_calls_set_open_false(self):
        """close_working_memory_thread ツールが WorkingMemoryManager.set_open(id, False) を呼び出す。"""
        wm = MagicMock()
        wm.set_open.return_value = True
        executor = self._make_executor(working_memory_manager=wm)
        result = executor.execute("close_working_memory_thread", {"thread_id": "thread-9"})
        wm.set_open.assert_called_once_with("thread-9", False)
        assert "閉じた" in result

    def test_reopen_working_memory_thread_calls_set_open_true(self):
        """reopen_working_memory_thread ツールが WorkingMemoryManager.set_open(id, True) を呼び出す。"""
        wm = MagicMock()
        wm.set_open.return_value = True
        executor = self._make_executor(working_memory_manager=wm)
        result = executor.execute("reopen_working_memory_thread", {"thread_id": "thread-9"})
        wm.set_open.assert_called_once_with("thread-9", True)
        assert "再オープン" in result

    def test_close_working_memory_thread_missing_returns_error(self):
        """存在しないスレッドの close は set_open が False を返し、エラー文字列になる。"""
        wm = MagicMock()
        wm.set_open.return_value = False
        executor = self._make_executor(working_memory_manager=wm)
        result = executor.execute("close_working_memory_thread", {"thread_id": "missing"})
        assert "見つかりません" in result

    def test_merge_working_memory_threads_closes_from_ids_and_posts(self):
        """merge ツールが into_id に経緯を post し、from_ids を close することを確認する。"""
        wm = MagicMock()
        wm.get_thread_detail.return_value = {"id": "into-1", "posts": []}
        wm.set_open.return_value = True
        executor = self._make_executor(working_memory_manager=wm)
        result = executor.execute(
            "merge_working_memory_threads",
            {"from_ids": ["a", "b"], "into_id": "into-1", "post": "同じ問題だった"},
        )
        wm.add_post.assert_called_once_with("into-1", "同じ問題だった")
        # from_ids 2本が close される
        assert wm.set_open.call_count == 2
        wm.set_open.assert_any_call("a", False)
        wm.set_open.assert_any_call("b", False)
        assert "2 本" in result

    def test_merge_working_memory_threads_missing_into_id_does_not_close_from_ids(self):
        """統合先が存在しない場合、from_ids を一切 close せずエラーを返すことを確認する。

        post の有無に関わらず into_id 存在チェックが先に走るため、存在しない統合先を
        指定したときに統合元だけが宛先なく閉じられる（統合の喪失）ことを防ぐ。
        """
        wm = MagicMock()
        wm.get_thread_detail.return_value = None  # 統合先が存在しない
        executor = self._make_executor(working_memory_manager=wm)
        result = executor.execute(
            "merge_working_memory_threads",
            # post を空にして「存在チェックを post 経由に頼っていた」旧バグの再現条件を作る
            {"from_ids": ["a", "b"], "into_id": "missing", "post": ""},
        )
        assert "見つかりません" in result
        # from_ids は1本も閉じられていない
        wm.set_open.assert_not_called()
        wm.add_post.assert_not_called()

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
        """inscribe_memory を実行して write_inscribed_memory に渡された kwargs を返すヘルパー。

        Args:
            category: 記憶カテゴリ。
            impact: 重要度係数。

        Returns:
            write_inscribed_memory に渡された kwargs dict。
        """
        mm = MagicMock()
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=mm, working_memory_manager=None
        )
        executor.execute("inscribe_memory", {"content": "test", "category": category, "impact": impact})
        return mm.write_inscribed_memory.call_args.kwargs

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
        """未知のカテゴリはContextualと同等として扱う"""
        kwargs_undif = self._run_inscribe_memory("unknown_cat", 1.0)
        kwargs_contx = self._run_inscribe_memory("contextual", 1.0)
        assert kwargs_undif["contextual_importance"] == kwargs_contx["contextual_importance"]
        assert kwargs_undif["semantic_importance"] == kwargs_contx["semantic_importance"]
        assert kwargs_undif["user_importance"] == kwargs_contx["user_importance"]
        assert kwargs_undif["identity_importance"] == kwargs_contx["identity_importance"]


# ---------------------------------------------------------------------------
# ToolExecutor の境界条件・エラー処理テスト
# ---------------------------------------------------------------------------

class TestToolExecutorEdgeCases:
    """ToolExecutor の境界条件・エラー処理を検証する。"""

    def test_post_working_memory_thread_without_working_memory_manager_returns_unavailable(self):
        """working_memory_manager が None の場合は post_working_memory_thread が利用不可メッセージを返す。"""
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=MagicMock(), working_memory_manager=None
        )
        result = executor.execute("post_working_memory_thread", {"type": "topic", "summary": "x"})
        assert "利用できない" in result

    def test_read_working_memory_thread_without_working_memory_manager_returns_unavailable(self):
        """working_memory_manager が None の場合は read_working_memory_thread が利用不可メッセージを返す。"""
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=MagicMock(), working_memory_manager=None
        )
        result = executor.execute("read_working_memory_thread", {"thread_id": "t-1"})
        assert "利用できない" in result

    def test_post_working_memory_thread_new_without_type_returns_error(self):
        """新規作成（thread_id 省略）で type が無い場合、エラーメッセージを返す。"""
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=MagicMock(), working_memory_manager=MagicMock()
        )
        result = executor.execute("post_working_memory_thread", {"summary": "概要だけ"})
        assert "[post_working_memory_thread error:" in result

    def test_inscribe_memory_exception_returns_error_message(self):
        """write_inscribed_memory が例外を投げた場合、エラーメッセージを返す（クラッシュしない）。"""
        mm = MagicMock()
        mm.write_inscribed_memory.side_effect = RuntimeError("DB connection failed")
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=mm, working_memory_manager=None
        )
        result = executor.execute("inscribe_memory", {"content": "test", "category": "user", "impact": 1.0})
        assert "[inscribe_memory error:" in result

    def test_post_working_memory_thread_exception_returns_error_message(self):
        """create_thread が例外を投げた場合、エラーメッセージを返す（クラッシュしない）。"""
        wm = MagicMock()
        wm.create_thread.side_effect = RuntimeError("DB error")
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=MagicMock(), working_memory_manager=wm
        )
        result = executor.execute("post_working_memory_thread", {"type": "topic", "summary": "x"})
        assert "[post_working_memory_thread error:" in result

    def test_carve_narrative_exception_returns_error_message(self):
        """carve_inner_narrative が例外を投げた場合、carve_narrative はエラーメッセージを返す（クラッシュしない）。"""
        mm = MagicMock()
        mm.sqlite.carve_inner_narrative.side_effect = RuntimeError("DB error")
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=mm, working_memory_manager=None
        )
        result = executor.execute("carve_narrative", {"mode": "append", "content": "指針"})
        assert "[carve_narrative error:" in result


class TestApplyInscribeMemoryTags:
    """ToolExecutor.apply_inscribe_memory_tags() のタグ抽出・impact 堅牢性を検証する。"""

    def test_malformed_impact_does_not_silently_inscribe(self):
        """impact が非数値のタグは黙って 1.0 で保存せず、その1件をスキップすることを確認する。

        旧 apply は float 失敗を握って impact=1.0 + 成功扱いにしていたため、不正タグが
        正常な記憶として保存されていた。非数値 impact は inscribe をスキップ（write を呼ばない）。
        """
        mm = MagicMock()
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=mm, working_memory_manager=None
        )
        clean = executor.apply_inscribe_memory_tags(
            "本文[INSCRIBE_MEMORY:contextual|high|不正なimpact]"
        )
        # 不正タグは保存されない
        mm.write_inscribed_memory.assert_not_called()
        # マーカーは本文から除去される
        assert "[INSCRIBE_MEMORY:" not in clean
        assert "本文" in clean

    def test_valid_impact_inscribes(self):
        """impact が数値の正常タグは write_inscribed_memory を呼ぶことを確認する。"""
        mm = MagicMock()
        executor = ToolExecutor(
            character_id="c", session_id="s", memory_manager=mm, working_memory_manager=None
        )
        executor.apply_inscribe_memory_tags("[INSCRIBE_MEMORY:user|0.8|ユーザは猫好き]")
        mm.write_inscribed_memory.assert_called_once()


# ---------------------------------------------------------------------------
# _api_guard デコレータのテスト
# ---------------------------------------------------------------------------

class TestApiGuardDecorator:
    """_api_guard デコレータが ImportError と API key チェックを正しく行うかを検証する。"""

    def test_missing_package_returns_error_string(self):
        """存在しないパッケージ名でデコレートされた generate() はエラー文字列を返す。"""

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
        """async generator にデコレートされた generate_stream() は ("error", ...) を yield してリターンする。

        "error" 型は呼び出し側が「出力に積まない／上書きしない」分岐を行うシグナル。
        """

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
        assert chunk_type == "error"
        assert "[Error:" in chunk_msg

    def test_async_generator_yields_error_on_empty_api_key(self):
        """async generator にデコレートされた generate_stream() は API key エラーを ("error", ...) で yield してリターンする。"""

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
        assert chunk_type == "error"
        assert "my_key" in chunk_msg

    def test_xai_inherits_correct_settings_key(self):
        """XAIProvider の generate() は xai_api_key を含むエラーメッセージを返す。"""
        from backend.providers.xai_provider import XAIProvider

        async def run():
            p = XAIProvider(api_key="")
            return await p.generate("sys", [{"role": "user", "content": "hi"}])

        result = asyncio.run(run())
        assert "xai_api_key" in result

    def test_anthropic_settings_key_in_error(self):
        """AnthropicProvider の generate() は anthropic_api_key を含むエラーメッセージを返す。"""
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

        provider = self._make_provider([
            ToolTurnResult(text="こんにちは", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", MagicMock(), MagicMock())

        text, thinking = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "こんにちは"
        assert thinking == ""

    def test_single_inscribe_memory_call_executes_and_continues(self):
        """1回の inscribe_memory 呼び出し → 実行 → 継続が正しく行われる。"""

        tc = ToolCall(id="tc-1", name="inscribe_memory", input={"content": "X", "category": "user", "impact": 1.0})
        mm = MagicMock()
        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[tc]),
            ToolTurnResult(text="完了した", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", mm, MagicMock())

        text, _ = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "完了した"
        mm.write_inscribed_memory.assert_called_once()
        assert len(provider._extend_calls) == 1

    def test_single_carve_narrative_call_executes_and_continues(self):
        """1回の carve_narrative 呼び出し → 実行 → 継続が正しく行われる。"""

        tc = ToolCall(id="tc-2", name="carve_narrative", input={"mode": "append", "content": "新指針"})
        mm = MagicMock()

        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[tc]),
            ToolTurnResult(text="指針を彫り込んだ", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", mm, None)

        text, _ = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "指針を彫り込んだ"
        mm.sqlite.carve_inner_narrative.assert_called_once()

    def test_multiple_tool_calls_in_sequence(self):
        """複数ターンのツール呼び出しが正しくループする。"""

        tc1 = ToolCall(id="t1", name="post_working_memory_thread", input={"type": "topic", "summary": "話題"})
        tc2 = ToolCall(id="t2", name="inscribe_memory", input={"content": "Y", "category": "identity", "impact": 0.5})
        wm = MagicMock()
        wm.create_thread.return_value = {"id": "thread-1"}
        mm = MagicMock()
        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[tc1]),
            ToolTurnResult(text="", tool_calls=[tc2]),
            ToolTurnResult(text="終了", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", mm, wm)

        text, _ = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "終了"
        wm.create_thread.assert_called_once()
        mm.write_inscribed_memory.assert_called_once()
        assert len(provider._extend_calls) == 2

    def test_text_accumulates_across_turns(self):
        """複数ターンにわたるテキストが結合されて返される。"""

        tc = ToolCall(id="t1", name="post_working_memory_thread", input={"type": "topic", "summary": "テスト"})
        provider = self._make_provider([
            ToolTurnResult(text="前半", tool_calls=[tc]),
            ToolTurnResult(text="後半", tool_calls=[]),
        ])
        executor = ToolExecutor("c", "s", MagicMock(), MagicMock())

        text, _ = asyncio.run(provider.generate_with_tools("sys", [], executor))
        assert text == "前半後半"


class TestToolExecutorEmbeddingDegraded:
    """embedding サーバ停止時（EmbeddingError）の ToolExecutor のキャラクター向け応答を検証する。

    背景: infinity 等の embedding サーバが落ちると、inscribe_memory は類似検索の時点で、
    power_recall はクエリのベクトル化の時点で EmbeddingError を送出する。
    従来は汎用の `[xxx error: ...]` に生の例外文字列が入るだけで、キャラクターには
    「自分の記憶がどうなったのか」が伝わらなかった。本クラスは、embedding 起因の失敗が
    「記憶は消えていない／保存はされていない」と実行可能な受け皿（WM 退避・Chronicle）を
    キャラクター本人に伝える専用メッセージへ変換されることを保証する。

    なお「復旧後にもう一度試して」という案内は禁句である。ツールコールの意図は
    後続ターンの文脈に残らないため、復旧時点では本人も何を刻もうとしていたか
    分からず、実行不可能なアドバイスになるため。
    """

    def _make_executor(self, memory_manager):
        """テスト用の ToolExecutor を生成するヘルパー。

        Args:
            memory_manager: InscribedMemoryManager のモック。

        Returns:
            ToolExecutor インスタンス。
        """
        return ToolExecutor(
            character_id="char-1",
            session_id="sess-1",
            memory_manager=memory_manager,
            working_memory_manager=MagicMock(),
        )

    def test_inscribe_memory_embedding_error_returns_character_facing_message(self):
        """inscribe_memory が EmbeddingError で失敗したとき、
        「まだ保存されていない」「既存の記憶は消えていない」を伝え、実行可能な受け皿
        （WM への退避・今夜の Chronicle）を案内するメッセージを返すこと。

        embedding 失敗時は類似検索の段階で例外となり SQLite にも書き込まれないため、
        「保存されていない」事実を正確に伝えないと、キャラクターが刻んだつもりに
        なってしまい記憶が静かに欠落する。受け皿2つはいずれも障害中に実在する経路:
        post_working_memory_thread は SQLite 先行書き込み（embed は遅延）のため保存でき、
        Chronicle は当日会話を SQLite から読み返すため embedding 非依存で機能する。
        """
        from backend.repositories.lance.store import EmbeddingError

        mm = MagicMock()
        mm.write_inscribed_memory.side_effect = EmbeddingError("connection refused")
        executor = self._make_executor(mm)
        result = executor.execute(
            "inscribe_memory",
            {"content": "ユーザは猫が好き", "category": "user_info", "impact": 1.0},
        )
        assert "[inscribe_memory error:" in result
        assert "embedding接続エラー" in result
        assert "まだ保存されていません" in result
        assert "消えたわけではありません" in result
        # 実行可能な受け皿2つ（WM 退避・Chronicle）が案内されていること
        assert "post_working_memory_thread" in result
        assert "棚卸し（Chronicle）" in result
        # 実行不可能な「復旧後に再試行」を案内していないこと
        assert "もう一度試して" not in result

    def test_power_recall_embedding_error_returns_character_facing_message(self):
        """power_recall が EmbeddingError で失敗したとき、
        「記憶が消えたわけではない」「復旧すればまた思い出せる」を伝えるメッセージを返すこと。

        想起の失敗を「忘却」と誤解させないことが目的。生の例外文字列ではなく、
        キャラクター本人に状況が伝わる文面であることを保証する。再試行の指示はしない
        （想起は必要が再び生じたときに自然に行われ、自動想起も復旧すれば勝手に再開するため）。
        """
        from backend.repositories.lance.store import EmbeddingError

        mm = MagicMock()
        mm.power_recall.side_effect = EmbeddingError("connection refused")
        executor = self._make_executor(mm)
        result = executor.execute("power_recall", {"query": "猫の話", "top_k": 5})
        assert "[power_recall error:" in result
        assert "embedding接続エラー" in result
        # 文言は「消えたわけではなく…」と続くため、活用に依存しない語幹で判定する
        assert "消えたわけでは" in result
        assert "思い出せます" in result
        # 実行不可能な「復旧後に再試行」を案内していないこと
        assert "もう一度試して" not in result
