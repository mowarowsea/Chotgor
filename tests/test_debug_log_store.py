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

    def test_filter_chat_type_includes_farewell_and_trigger(self, store):
        """request_type='chat': chat/farewell/trigger を含む request_id のみ返ること。

        farewell・trigger は Chat のサブタイプとして同じ request_id に混在するため、
        それらを含む request_id は Chat タブに表示される。
        """
        store.insert_debug_log_entry(request_id="chat001", source_type="chat")
        store.insert_debug_log_entry(request_id="chat001", source_type="farewell")
        store.insert_debug_log_entry(request_id="trig001", source_type="trigger")
        store.insert_debug_log_entry(request_id="scen001", source_type="scenario")
        store.insert_debug_log_entry(request_id="batch001", source_type="chronicle")
        ids, total = store.get_debug_log_request_ids_paged(request_type="chat")
        assert set(ids) == {"chat001", "trig001"}
        assert total == 2

    def test_filter_scenario_type(self, store):
        """request_type='scenario': scenario/scenario_chat を含む request_id のみ返ること。"""
        store.insert_debug_log_entry(request_id="scen001", source_type="scenario")
        store.insert_debug_log_entry(request_id="scen_chat001", source_type="scenario_chat")
        store.insert_debug_log_entry(request_id="chat001", source_type="chat")
        store.insert_debug_log_entry(request_id="batch001", source_type="chronicle")
        ids, total = store.get_debug_log_request_ids_paged(request_type="scenario")
        assert set(ids) == {"scen001", "scen_chat001"}
        assert total == 2

    def test_filter_batch_type_excludes_chat_and_scenario(self, store):
        """request_type='batch': chat 系・scenario 系のどちらにも属さない request_id のみ返ること。"""
        store.insert_debug_log_entry(request_id="batch001", source_type="chronicle")
        store.insert_debug_log_entry(request_id="batch002", source_type="forget")
        store.insert_debug_log_entry(request_id="chat001", source_type="chat")
        store.insert_debug_log_entry(request_id="scen001", source_type="scenario")
        ids, total = store.get_debug_log_request_ids_paged(request_type="batch")
        assert set(ids) == {"batch001", "batch002"}
        assert total == 2

    def test_filter_none_returns_all(self, store):
        """request_type=None（デフォルト）: 全件返ること。"""
        store.insert_debug_log_entry(request_id="chat001", source_type="chat")
        store.insert_debug_log_entry(request_id="scen001", source_type="scenario")
        store.insert_debug_log_entry(request_id="batch001", source_type="chronicle")
        ids, total = store.get_debug_log_request_ids_paged(request_type=None)
        assert total == 3
        assert set(ids) == {"chat001", "scen001", "batch001"}


# ─── get_debug_log_entries_by_request_ids ────────────────────────────────────


class TestGetEntriesByRequestIds:
    """get_debug_log_entries_by_request_ids() の動作を検証するテストクラス。

    N+1 クエリ解消のために追加された IN 句一括取得メソッドを検証する。
    単体クエリ版（get_debug_log_entries_by_request_id）と同じ結果を
    1回の SQL 呼び出しで返すことが求められる。
    """

    def test_empty_list_returns_empty_dict(self, store):
        """空リストを渡した場合、空辞書を返すこと。"""
        result = store.get_debug_log_entries_by_request_ids([])
        assert result == {}

    def test_single_id_returns_matching_rows(self, store):
        """1件の request_id を渡した場合、その行が返ること。"""
        store.insert_debug_log_entry(request_id="req_a", source_type="chat", response="回答A")
        result = store.get_debug_log_entries_by_request_ids(["req_a"])
        assert "req_a" in result
        assert len(result["req_a"]) == 1
        assert result["req_a"][0]["response"] == "回答A"

    def test_multiple_ids_returns_all(self, store):
        """複数の request_id を渡した場合、すべての行が返ること。"""
        store.insert_debug_log_entry(request_id="req_a", source_type="chat")
        store.insert_debug_log_entry(request_id="req_b", source_type="scenario")
        store.insert_debug_log_entry(request_id="req_c", source_type="chronicle")
        result = store.get_debug_log_entries_by_request_ids(["req_a", "req_b", "req_c"])
        assert set(result.keys()) == {"req_a", "req_b", "req_c"}
        assert len(result["req_a"]) == 1
        assert len(result["req_b"]) == 1
        assert len(result["req_c"]) == 1

    def test_nonexistent_id_returns_empty_list(self, store):
        """存在しない request_id のキーは空リストになること。"""
        store.insert_debug_log_entry(request_id="req_a", source_type="chat")
        result = store.get_debug_log_entries_by_request_ids(["req_a", "nonexistent"])
        assert len(result["req_a"]) == 1
        assert result["nonexistent"] == []

    def test_multiple_rows_same_id_returned_ascending(self, store):
        """同一 request_id の複数行が作成日時昇順で返ること。

        chat → farewell の順で INSERT されたとき、その順序で返ること（単体クエリ版と同じ仕様）。
        """
        store.insert_debug_log_entry(request_id="multi", source_type="chat", response="1回目")
        store.insert_debug_log_entry(request_id="multi", source_type="farewell", response="2回目")
        result = store.get_debug_log_entries_by_request_ids(["multi"])
        rows = result["multi"]
        assert len(rows) == 2
        assert rows[0]["response"] == "1回目"
        assert rows[1]["response"] == "2回目"

    def test_excludes_unrequested_ids(self, store):
        """渡していない request_id の行は含まれないこと。"""
        store.insert_debug_log_entry(request_id="req_a", source_type="chat")
        store.insert_debug_log_entry(request_id="req_b", source_type="chat")
        result = store.get_debug_log_entries_by_request_ids(["req_a"])
        assert "req_b" not in result

    def test_matches_single_query_method_result(self, store):
        """IN 句一括取得の結果が単体クエリ版と一致すること。

        get_debug_log_entries_by_request_id（1件ずつ）と
        get_debug_log_entries_by_request_ids（一括）が同じ結果を返すことを確認する。
        """
        store.insert_debug_log_entry(
            request_id="cmp001", source_type="chat",
            target="はる", preset="CC", response="応答X"
        )
        store.insert_debug_log_entry(request_id="cmp001", source_type="farewell")

        single = store.get_debug_log_entries_by_request_id("cmp001")
        bulk = store.get_debug_log_entries_by_request_ids(["cmp001"])["cmp001"]

        assert len(single) == len(bulk)
        for s, b in zip(single, bulk):
            assert s["source_type"] == b["source_type"]
            assert s["response"] == b["response"]


# ─── synopsis タブ振り分け ──────────────────────────────────────────────────────


class TestSynopsisTabFilter:
    """synopsis source_type が Scenario タブに振り分けられることを検証するテストクラス。

    シナリオあらすじ蒸留ログ（source_type='synopsis'）は Scenario タブに属すること、
    Batch タブには含まれないことを確認する。
    """

    def test_synopsis_appears_in_scenario_tab(self, store):
        """synopsis エントリが request_type='scenario' で返ること。"""
        store.insert_debug_log_entry(request_id="syn001", source_type="synopsis")
        ids, total = store.get_debug_log_request_ids_paged(request_type="scenario")
        assert "syn001" in ids
        assert total == 1

    def test_synopsis_excluded_from_batch_tab(self, store):
        """synopsis エントリが request_type='batch' に含まれないこと。"""
        store.insert_debug_log_entry(request_id="syn002", source_type="synopsis")
        store.insert_debug_log_entry(request_id="batch001", source_type="chronicle")
        ids, _ = store.get_debug_log_request_ids_paged(request_type="batch")
        assert "syn002" not in ids
        assert "batch001" in ids

    def test_synopsis_excluded_from_chat_tab(self, store):
        """synopsis エントリが request_type='chat' に含まれないこと。"""
        store.insert_debug_log_entry(request_id="syn003", source_type="synopsis")
        store.insert_debug_log_entry(request_id="chat001", source_type="chat")
        ids, _ = store.get_debug_log_request_ids_paged(request_type="chat")
        assert "syn003" not in ids
        assert "chat001" in ids

    def test_synopsis_in_scenario_tab_with_target(self, store):
        """synopsis エントリに target（シナリオ名）を付与しても Scenario タブで返ること。"""
        store.insert_debug_log_entry(
            request_id="syn004", source_type="synopsis",
            target="テストシナリオ", preset="default"
        )
        ids, total = store.get_debug_log_request_ids_paged(request_type="scenario")
        assert "syn004" in ids
