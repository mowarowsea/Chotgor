"""debug_log_store モジュールのユニットテスト。

DebugLogStoreMixin（SQLiteStore 経由）の CRUD 操作を検証する。

対象メソッド:
    insert_debug_log_entry()              — 新規行の INSERT と返却 id
    update_debug_log_entry()              — 部分更新
    get_debug_log_entries_by_request_id() — request_id によるフィルタ・昇順
    get_debug_log_request_ids_paged()     — ユニーク request_id のページネーション

テスト方針:
    - インメモリ SQLite（:memory:）を使って外部依存なしに動作させる
    - 各テストで独立した SQLiteStore インスタンスを使用する
"""

import pytest

from backend.repositories.sqlite.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    """テスト用インメモリ SQLiteStore を返すフィクスチャ。

    tmp_path 配下にファイルを作成して各テストを独立させる。
    """
    db_path = str(tmp_path / "test.db")
    return SQLiteStore(db_path)


# ─── insert_debug_log_entry ───────────────────────────────────────────────────


class TestInsertDebugLogEntry:
    """insert_debug_log_entry() の動作を検証するテストクラス。"""

    def test_returns_positive_id(self, store):
        """INSERT が成功し、正の整数 id が返ること。"""
        entry_id = store.insert_debug_log_entry(
            request_id="abc12345",
            source_type="chat",
        )
        assert isinstance(entry_id, int)
        assert entry_id > 0

    def test_all_fields_stored(self, store):
        """全フィールドが正しく格納されること。"""
        entry_id = store.insert_debug_log_entry(
            request_id="abc12345",
            source_type="chat",
            session_id="sess_001",
            turn_sequence=3,
            target="はる",
            preset="ClaudeCode",
            user_message="こんにちは",
            response="こんにちは！",
            reasoning="思考内容",
            mcp_calls_json='[{"tag_name": "INSCRIBE_MEMORY"}]',
            has_error=False,
            warn_reason=None,
            raw_dir="debug/abc12345",
        )
        rows = store.get_debug_log_entries_by_request_id("abc12345")
        assert len(rows) == 1
        r = rows[0]
        assert r["id"] == entry_id
        assert r["request_id"] == "abc12345"
        assert r["source_type"] == "chat"
        assert r["session_id"] == "sess_001"
        assert r["turn_sequence"] == 3
        assert r["target"] == "はる"
        assert r["preset"] == "ClaudeCode"
        assert r["user_message"] == "こんにちは"
        assert r["response"] == "こんにちは！"
        assert r["reasoning"] == "思考内容"
        assert r["mcp_calls"] == [{"tag_name": "INSCRIBE_MEMORY"}]
        assert r["has_error"] is False
        assert r["raw_dir"] == "debug/abc12345"

    def test_has_error_true_stored_correctly(self, store):
        """has_error=True が正しく格納され True として返ること。"""
        store.insert_debug_log_entry(
            request_id="err00001",
            source_type="chat",
            has_error=True,
            warn_reason="タイムアウトが発生しました",
        )
        rows = store.get_debug_log_entries_by_request_id("err00001")
        assert rows[0]["has_error"] is True
        assert rows[0]["warn_reason"] == "タイムアウトが発生しました"

    def test_multiple_rows_same_request_id(self, store):
        """同一 request_id に複数行を INSERT できること（シナリオ再生成・farewell 等）。"""
        store.insert_debug_log_entry(request_id="multi001", source_type="chat")
        store.insert_debug_log_entry(request_id="multi001", source_type="farewell")
        store.insert_debug_log_entry(request_id="multi001", source_type="trigger")
        rows = store.get_debug_log_entries_by_request_id("multi001")
        assert len(rows) == 3
        source_types = {r["source_type"] for r in rows}
        assert source_types == {"chat", "farewell", "trigger"}

    def test_null_optional_fields(self, store):
        """オプションフィールドが None の場合、空文字または空リストで返ること。"""
        store.insert_debug_log_entry(request_id="null001", source_type="batch")
        rows = store.get_debug_log_entries_by_request_id("null001")
        r = rows[0]
        assert r["user_message"] == ""
        assert r["response"] == ""
        assert r["reasoning"] == ""
        assert r["mcp_calls"] == []
        assert r["warn_reason"] == ""
        assert r["raw_dir"] is None


# ─── update_debug_log_entry ───────────────────────────────────────────────────


class TestUpdateDebugLogEntry:
    """update_debug_log_entry() の動作を検証するテストクラス。"""

    def test_update_response(self, store):
        """response フィールドが更新されること。"""
        entry_id = store.insert_debug_log_entry(request_id="upd001", source_type="chat")
        store.update_debug_log_entry(entry_id, response="更新後の応答")
        rows = store.get_debug_log_entries_by_request_id("upd001")
        assert rows[0]["response"] == "更新後の応答"

    def test_update_reasoning(self, store):
        """reasoning フィールドが更新されること。"""
        entry_id = store.insert_debug_log_entry(request_id="upd002", source_type="chat")
        store.update_debug_log_entry(entry_id, reasoning="新しい思考内容")
        rows = store.get_debug_log_entries_by_request_id("upd002")
        assert rows[0]["reasoning"] == "新しい思考内容"

    def test_update_has_error(self, store):
        """has_error フィールドが更新されること。"""
        entry_id = store.insert_debug_log_entry(request_id="upd003", source_type="chat")
        store.update_debug_log_entry(entry_id, has_error=True, warn_reason="エラー発生")
        rows = store.get_debug_log_entries_by_request_id("upd003")
        assert rows[0]["has_error"] is True
        assert rows[0]["warn_reason"] == "エラー発生"

    def test_none_fields_not_overwritten(self, store):
        """None を渡したフィールドは既存値が上書きされないこと。"""
        entry_id = store.insert_debug_log_entry(
            request_id="upd004",
            source_type="chat",
            response="元の応答",
            reasoning="元の推論",
        )
        # response のみ None → 既存値維持、reasoning を更新
        store.update_debug_log_entry(entry_id, response=None, reasoning="新しい推論")
        rows = store.get_debug_log_entries_by_request_id("upd004")
        assert rows[0]["response"] == "元の応答"
        assert rows[0]["reasoning"] == "新しい推論"

    def test_nonexistent_id_is_noop(self, store):
        """存在しない id への更新は例外を送出せず何もしないこと。"""
        store.update_debug_log_entry(99999, response="あ")  # 例外が起きなければOK


# ─── get_debug_log_entries_by_request_id ──────────────────────────────────────


class TestGetEntriesByRequestId:
    """get_debug_log_entries_by_request_id() の動作を検証するテストクラス。"""

    def test_empty_when_not_found(self, store):
        """該当行がない場合、空リストを返すこと。"""
        rows = store.get_debug_log_entries_by_request_id("nonexistent")
        assert rows == []

    def test_returns_only_matching_rows(self, store):
        """指定 request_id の行のみ返し、他の request_id は含まないこと。"""
        store.insert_debug_log_entry(request_id="req_a", source_type="chat")
        store.insert_debug_log_entry(request_id="req_b", source_type="chat")
        rows = store.get_debug_log_entries_by_request_id("req_a")
        assert len(rows) == 1
        assert rows[0]["request_id"] == "req_a"

    def test_ascending_order_by_created_at(self, store):
        """複数行が created_at 昇順で返ること。

        シナリオ再生成では同一 request_id に複数行あり、古い順に並ぶことが期待される。
        """
        store.insert_debug_log_entry(request_id="ord001", source_type="chat", response="1回目")
        store.insert_debug_log_entry(request_id="ord001", source_type="chat", response="2回目")
        rows = store.get_debug_log_entries_by_request_id("ord001")
        assert rows[0]["response"] == "1回目"
        assert rows[1]["response"] == "2回目"

    def test_invalid_mcp_calls_json_returns_empty_list(self, store):
        """mcp_calls_json が不正 JSON の場合も例外なく空リストを返すこと。"""
        entry_id = store.insert_debug_log_entry(request_id="mcpbad1", source_type="chat")
        # 直接 DB を触って不正 JSON を注入する
        from backend.repositories.sqlite.store import DebugLogEntry
        with store.get_session() as sess:
            row = sess.get(DebugLogEntry, entry_id)
            row.mcp_calls_json = "not-json"
            sess.commit()
        rows = store.get_debug_log_entries_by_request_id("mcpbad1")
        assert rows[0]["mcp_calls"] == []


# ─── get_debug_log_request_ids_paged ─────────────────────────────────────────


class TestGetRequestIdsPaged:
    """get_debug_log_request_ids_paged() のページネーション・ソートを検証するテストクラス。"""

    def test_empty_when_no_entries(self, store):
        """エントリが1件もない場合、([], 0) を返すこと。"""
        ids, total = store.get_debug_log_request_ids_paged()
        assert ids == []
        assert total == 0

    def test_unique_request_ids_counted(self, store):
        """同一 request_id の複数行が1件としてカウントされること。"""
        store.insert_debug_log_entry(request_id="req_x", source_type="chat")
        store.insert_debug_log_entry(request_id="req_x", source_type="farewell")
        store.insert_debug_log_entry(request_id="req_y", source_type="chat")
        ids, total = store.get_debug_log_request_ids_paged()
        assert total == 2
        assert set(ids) == {"req_x", "req_y"}

    def test_newest_first_ordering(self, store):
        """最も新しい created_at を持つ request_id が先頭に来ること。

        シナリオ再生成で同一 request_id に複数行ある場合、最新行の時刻でソートされる。
        """
        import time
        store.insert_debug_log_entry(request_id="oldest1", source_type="chat")
        time.sleep(0.01)
        store.insert_debug_log_entry(request_id="newest2", source_type="chat")
        ids, _ = store.get_debug_log_request_ids_paged()
        assert ids[0] == "newest2"
        assert ids[1] == "oldest1"

    def test_pagination_page1(self, store):
        """ページ1が先頭 per_page 件を返すこと。"""
        for i in range(5):
            store.insert_debug_log_entry(request_id=f"page{i:06d}", source_type="chat")
        ids, total = store.get_debug_log_request_ids_paged(page=1, per_page=3)
        assert total == 5
        assert len(ids) == 3

    def test_pagination_last_page(self, store):
        """最終ページが余り件数を返すこと。"""
        for i in range(5):
            store.insert_debug_log_entry(request_id=f"last{i:06d}", source_type="chat")
        ids, total = store.get_debug_log_request_ids_paged(page=2, per_page=3)
        assert total == 5
        assert len(ids) == 2

    def test_scenario_rerun_uses_latest_time_for_ordering(self, store):
        """シナリオ再生成（同一 request_id）では最新行の時刻でソートされること。

        req_old が先に作られたが、その後 req_new より新しい再生成行が追加された場合、
        req_old が先頭になること。
        """
        import time
        store.insert_debug_log_entry(request_id="req_old", source_type="scenario")
        time.sleep(0.01)
        store.insert_debug_log_entry(request_id="req_new", source_type="scenario")
        time.sleep(0.01)
        # req_old の再生成（最新行が req_new より新しくなる）
        store.insert_debug_log_entry(request_id="req_old", source_type="scenario")
        ids, _ = store.get_debug_log_request_ids_paged()
        assert ids[0] == "req_old"
