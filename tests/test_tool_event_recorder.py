"""tool_event_recorder / ToolEventStoreMixin / ToolExecutor 記録経路のユニットテスト。

ツール実行イベント記録方式（2026-06-11 導入）の中核を検証する。
Logs 画面のツール使用表示は、生ログの逆解析（tag_extract.py）から
「実行時に tool_call_events テーブルへ確定事実を記録する」方式へ移行した。

対象:
    ToolEventStoreMixin.add_tool_call_event()          — 1イベント = 1行の INSERT
    ToolEventStoreMixin.get_tool_call_events_by_dir_ids() — dir_id 単位の一括取得
    tool_event_recorder.record_tool_event()             — ContextVar 自動補完付き記録
    tool_event_recorder.result_looks_like_error()       — 結果文字列のエラー規約判定
    ToolExecutor.execute()                              — 共通関門での自動記録と record=False
    api.mcp_tools._restore_log_context()                — env→HTTP リレー値の ContextVar 復元

テスト方針:
    - SQLite は tmp_path 配下の実ファイルで各テスト独立に生成する
    - ContextVar は log_context の set() で直接設定し、テスト後の汚染を避けるため
      contextvars.copy_context() ではなく毎テストで明示的に再設定する
    - ToolExecutor の依存（memory_manager 等）は Mock で代替する
      （switch_angle / 未知ツールは DB アクセスが発生しないため Mock で完結する）
"""

import json
from unittest.mock import Mock

import pytest

from backend.lib import tool_event_recorder
from backend.lib.log_context import (
    current_log_dir_id,
    current_log_feature,
    current_log_target,
    current_message_id,
    new_message_id,
)
from backend.lib.tool_event_recorder import record_tool_event, result_looks_like_error
from backend.repositories.sqlite.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    """テスト用 SQLiteStore を返すフィクスチャ。

    tmp_path 配下にファイルを作成して各テストを独立させる。
    Base.metadata.create_all により tool_call_events テーブルも自動作成される。
    """
    return SQLiteStore(str(tmp_path / "test.db"))


@pytest.fixture
def recorder_store(store):
    """tool_event_recorder にテスト用 store を注入し、テスト後に解除するフィクスチャ。

    モジュールグローバルの _store を差し替えるため、teardown で必ず None に戻して
    他テストへの汚染を防ぐ。
    """
    tool_event_recorder.set_store(store)
    yield store
    tool_event_recorder.set_store(None)


def _reset_context() -> None:
    """ログ用 ContextVar をすべて未設定デフォルトへ戻すヘルパ。

    pytest はテスト間で contextvars を隔離しないため、
    各テストの冒頭で明示的にリセットして独立性を保つ。
    """
    current_message_id.set("--------")
    current_log_dir_id.set("--------")
    current_log_feature.set("chat")
    current_log_target.set(None)


# ─── ToolEventStoreMixin ──────────────────────────────────────────────────────


class TestToolEventStore:
    """ToolEventStoreMixin の CRUD を検証するテストクラス。

    add_tool_call_event() で挿入した行が get_tool_call_events_by_dir_ids() で
    dir_id 単位に正しくグルーピング・実行順ソート・JSON パースされて返ることを確認する。
    """

    def test_add_and_get_roundtrip(self, store):
        """挿入したイベントが dir_id で取得でき、全フィールドが往復すること。"""
        store.add_tool_call_event(
            tool_name="inscribe_memory",
            arguments_json=json.dumps({"content": "テスト記憶", "category": "contextual", "impact": 1.0}, ensure_ascii=False),
            status="ok",
            source="tool_use",
            request_id="abc12345",
            dir_id="abc12345",
            target="はる@ClaudeCode",
            feature="chat",
        )
        events = store.get_tool_call_events_by_dir_ids(["abc12345"])["abc12345"]
        assert len(events) == 1
        ev = events[0]
        assert ev["tool_name"] == "inscribe_memory"
        assert ev["arguments"] == {"content": "テスト記憶", "category": "contextual", "impact": 1.0}
        assert ev["status"] == "ok"
        assert ev["source"] == "tool_use"
        assert ev["request_id"] == "abc12345"
        assert ev["target"] == "はる@ClaudeCode"
        assert ev["feature"] == "chat"

    def test_events_ordered_by_execution(self, store):
        """同一 dir_id の複数イベントが挿入順（= 実行順）で返ること。"""
        for i in range(3):
            store.add_tool_call_event(
                tool_name=f"tool_{i}", dir_id="dddd0000",
            )
        events = store.get_tool_call_events_by_dir_ids(["dddd0000"])["dddd0000"]
        assert [e["tool_name"] for e in events] == ["tool_0", "tool_1", "tool_2"]

    def test_missing_dir_id_returns_empty_list(self, store):
        """イベントが存在しない dir_id は空リストで返ること（KeyError にならない）。"""
        result = store.get_tool_call_events_by_dir_ids(["ffffffff"])
        assert result == {"ffffffff": []}

    def test_empty_dir_ids_returns_empty_dict(self, store):
        """dir_ids が空リストなら DB を引かず空辞書を返すこと。"""
        assert store.get_tool_call_events_by_dir_ids([]) == {}

    def test_multiple_dir_ids_grouped(self, store):
        """複数の dir_id を一括取得した場合、それぞれ独立にグルーピングされること。"""
        store.add_tool_call_event(tool_name="a", dir_id="11111111")
        store.add_tool_call_event(tool_name="b", dir_id="22222222")
        result = store.get_tool_call_events_by_dir_ids(["11111111", "22222222"])
        assert [e["tool_name"] for e in result["11111111"]] == ["a"]
        assert [e["tool_name"] for e in result["22222222"]] == ["b"]

    def test_broken_arguments_json_failsafe(self, store):
        """arguments_json が壊れていても例外にならず空 dict にフェイルセーフすること。"""
        store.add_tool_call_event(
            tool_name="x", arguments_json="{broken json", dir_id="33333333",
        )
        events = store.get_tool_call_events_by_dir_ids(["33333333"])["33333333"]
        assert events[0]["arguments"] == {}

    def test_error_event_fields(self, store):
        """status=error と error_message が往復すること。"""
        store.add_tool_call_event(
            tool_name="power_recall", status="error",
            error_message="[power_recall error: embedding接続エラー]",
            dir_id="44444444",
        )
        ev = store.get_tool_call_events_by_dir_ids(["44444444"])["44444444"][0]
        assert ev["status"] == "error"
        assert "embedding接続エラー" in ev["error_message"]


# ─── record_tool_event ────────────────────────────────────────────────────────


class TestRecordToolEvent:
    """record_tool_event() の ContextVar 自動補完と耐障害性を検証するテストクラス。

    usage_recorder と同型の「呼び出し側はツール名と引数だけ渡せばよい」設計が
    成立していること、および記録失敗が本流を妨げないことを確認する。
    """

    def test_context_vars_autocompleted(self, recorder_store):
        """ContextVar 設定済みなら request_id / dir_id / feature / target が自動補完されること。"""
        msg_id = new_message_id()
        current_log_feature.set("chronicle")
        current_log_target.set("はる@GhostModel")
        record_tool_event("carve_narrative", {"mode": "append", "content": "夜の棚卸し"})
        ev = recorder_store.get_tool_call_events_by_dir_ids([msg_id])[msg_id][0]
        assert ev["request_id"] == msg_id
        assert ev["dir_id"] == msg_id
        assert ev["feature"] == "chronicle"
        assert ev["target"] == "はる@GhostModel"
        assert ev["arguments"] == {"mode": "append", "content": "夜の棚卸し"}

    def test_unset_context_normalized_to_null(self, recorder_store):
        """ContextVar が未設定デフォルト（"--------"）なら NULL で記録されること。

        dir_id が NULL の行は dir_id 検索に引っかからないため、
        全件クエリで直接確認する。
        """
        _reset_context()
        current_log_target.set(None)
        record_tool_event("inscribe_memory", {"content": "x"})
        from backend.repositories.sqlite.models import ToolCallEvent
        with recorder_store.get_session() as sess:
            row = sess.query(ToolCallEvent).one()
            assert row.request_id is None
            assert row.dir_id is None
            assert row.target is None

    def test_no_store_is_noop(self):
        """store 未注入（テスト・起動直後）でも例外を出さず何もしないこと。"""
        tool_event_recorder.set_store(None)
        record_tool_event("inscribe_memory", {"content": "x"})  # 例外が出なければOK

    def test_store_failure_swallowed(self):
        """DB 書き込みが失敗してもチャット本流へ例外が漏れないこと。"""
        broken = Mock()
        broken.add_tool_call_event.side_effect = RuntimeError("DB死亡")
        tool_event_recorder.set_store(broken)
        try:
            record_tool_event("inscribe_memory", {"content": "x"})  # 例外が出なければOK
        finally:
            tool_event_recorder.set_store(None)

    def test_non_serializable_arguments_fallback(self, recorder_store):
        """JSON 化できない引数値も default=str で文字列化されて記録されること。"""
        msg_id = new_message_id()
        record_tool_event("inscribe_memory", {"obj": object()})
        ev = recorder_store.get_tool_call_events_by_dir_ids([msg_id])[msg_id][0]
        assert "obj" in ev["arguments"]


# ─── result_looks_like_error ──────────────────────────────────────────────────


class TestResultLooksLikeError:
    """result_looks_like_error() のエラー規約判定を検証するテストクラス。

    ToolExecutor の各ツールは例外を握り潰して "[<tool> error: ...]" 形式の
    文字列を返すため、この規約への一致/不一致を網羅的に確認する。
    """

    @pytest.mark.parametrize("text", [
        "[inscribe_memory error: 記憶システムが一時的に不調のため...]",
        "[power_recall error: embedding接続エラー]",
        "[carve_narrative error: DB死亡]",
        "[Error: ValueError: bad input]",
        "[Unknown tool: drift]",
        "  [web_search error: APIキー未設定]",  # 先頭空白も許容
    ])
    def test_error_results_detected(self, text):
        """エラー規約に一致する結果文字列が error と判定されること。"""
        assert result_looks_like_error(text) is True

    @pytest.mark.parametrize("text", [
        "記憶に刻んだ。",
        "inner_narrative を更新した。",
        "アングルを GhostModel に切り替えます。",
        "「過去」に関する記憶・会話は見つからなかった。",
        "[contextual] 昨日の散歩の記憶",  # 角括弧開始でも error を含まなければ OK
        "",
    ])
    def test_normal_results_not_detected(self, text):
        """正常結果の文字列が error と誤判定されないこと。"""
        assert result_looks_like_error(text) is False


# ─── ToolExecutor.execute の自動記録 ──────────────────────────────────────────


class TestToolExecutorRecording:
    """ToolExecutor.execute() がツール実行イベントを自動記録することを検証するテストクラス。

    execute() は tool-use 方式・MCP プロキシ・バッチの全経路が通る唯一の関門であり、
    ここでの記録がプロバイダー形式に依存しないツール使用表示を成立させている。
    DB アクセスが発生しない switch_angle / 未知ツールを使って記録動作だけを切り出す。
    """

    def _make_executor(self):
        """依存を Mock 化した ToolExecutor を返すヘルパ。"""
        from backend.character_actions.executor import ToolExecutor
        return ToolExecutor(
            character_id="char-1",
            session_id=None,
            memory_manager=Mock(),
            working_memory_manager=None,
        )

    def test_success_recorded_as_ok(self, recorder_store):
        """正常実行が status=ok で記録され、引数も保存されること。"""
        msg_id = new_message_id()
        executor = self._make_executor()
        result = executor.execute(
            "switch_angle", {"preset_name": "GhostModel", "self_instruction": "静かに"},
        )
        assert "GhostModel" in result
        ev = recorder_store.get_tool_call_events_by_dir_ids([msg_id])[msg_id][0]
        assert ev["tool_name"] == "switch_angle"
        assert ev["status"] == "ok"
        assert ev["arguments"]["preset_name"] == "GhostModel"

    def test_unknown_tool_recorded_as_error(self, recorder_store):
        """未知ツール（"[Unknown tool: ...]" 返却）が status=error で記録されること。"""
        msg_id = new_message_id()
        executor = self._make_executor()
        executor.execute("no_such_tool", {})
        ev = recorder_store.get_tool_call_events_by_dir_ids([msg_id])[msg_id][0]
        assert ev["status"] == "error"
        assert "Unknown tool" in ev["error_message"]

    def test_record_false_skips_recording(self, recorder_store):
        """record=False（claude_cli の switch_angle 転写経路）では記録されないこと。

        MCP プロキシ側で実行・記録済みの switch_angle を in-process executor へ
        転写する際の二重記録防止を担保する。
        """
        msg_id = new_message_id()
        executor = self._make_executor()
        executor.execute(
            "switch_angle", {"preset_name": "A", "self_instruction": ""}, record=False,
        )
        assert recorder_store.get_tool_call_events_by_dir_ids([msg_id])[msg_id] == []

    def test_dispatch_exception_recorded_then_reraised(self, recorder_store):
        """ディスパッチ段の例外（引数型変換等）が status=error で記録され、再送出されること。"""
        msg_id = new_message_id()
        executor = self._make_executor()
        # inscribe_memory の impact は float() 変換されるため、不正値で TypeError/ValueError になる
        with pytest.raises(Exception):
            executor.execute("inscribe_memory", {"impact": ["not-a-number"]})
        ev = recorder_store.get_tool_call_events_by_dir_ids([msg_id])[msg_id][0]
        assert ev["status"] == "error"


# ─── mcp_tools._restore_log_context ───────────────────────────────────────────


class TestRestoreLogContext:
    """_restore_log_context() の env→HTTP リレー値復元を検証するテストクラス。

    Claude CLI の MCP プロセスは backend と別プロセスのため ContextVar が届かない。
    CHOTGOR_LOG_CONTEXT env → mcp_server.py → HTTP ペイロード → 本関数、のリレーで
    元リクエストの request_id 等が ContextVar に復元されることを確認する。
    """

    def test_full_context_restored(self):
        """4キー全部が揃った log_context が ContextVar へ復元されること。"""
        from backend.api.mcp_tools import _restore_log_context
        _reset_context()
        _restore_log_context({
            "request_id": "deadbeef",
            "dir_id": "cafebabe",
            "feature": "scenario",
            "target": "はる@ClaudeCode",
        })
        assert current_message_id.get() == "deadbeef"
        assert current_log_dir_id.get() == "cafebabe"
        assert current_log_feature.get() == "scenario"
        assert current_log_target.get() == "はる@ClaudeCode"

    def test_none_is_noop(self):
        """log_context が None（旧バージョンの mcp_server からの呼び出し）なら何も変えないこと。"""
        from backend.api.mcp_tools import _restore_log_context
        _reset_context()
        _restore_log_context(None)
        assert current_message_id.get() == "--------"

    def test_partial_context_keeps_defaults(self):
        """欠けたキーはデフォルトのまま、存在するキーだけ復元されること。"""
        from backend.api.mcp_tools import _restore_log_context
        _reset_context()
        _restore_log_context({"request_id": "deadbeef"})
        assert current_message_id.get() == "deadbeef"
        assert current_log_dir_id.get() == "--------"

    def test_unset_relay_values_ignored(self):
        """リレー値が未設定デフォルト（"--------" 等の無意味値でなく空）なら上書きしないこと。"""
        from backend.api.mcp_tools import _restore_log_context
        _reset_context()
        _restore_log_context({"request_id": "", "dir_id": None, "feature": "", "target": ""})
        assert current_message_id.get() == "--------"
        assert current_log_dir_id.get() == "--------"
        assert current_log_feature.get() == "chat"
