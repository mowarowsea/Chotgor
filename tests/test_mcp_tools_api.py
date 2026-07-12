"""``backend.api.mcp_tools`` の HTTP プロキシ API 統合テスト。

``mcp_server.py`` は backend に集約された ToolExecutor を HTTP 経由で叩くだけの
プロキシであり、本テストは backend 側の HTTP エンドポイントの振る舞いを保証する：

  - ``GET /api/mcp/tools`` がツール定義リストを返すこと
  - ``POST /api/mcp/tools/call`` が ToolExecutor を正しく呼ぶこと（state 注入の確認）
  - ToolExecutor が例外を投げてもエンドポイントは 200 のまま is_error=True で返すこと
  - localhost 以外からのアクセスは 403 で拒絶されること（外部攻撃面の閉塞）

LLM や LanceDB への実ネットワーク呼び出しは一切行わない。
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import mcp_tools as mcp_tools_module


def _make_app() -> FastAPI:
    """テスト用 FastAPI アプリを返す。``app.state`` には MagicMock を載せる。

    本物の InscribedMemoryManager / WorkingMemoryManager は重く、LanceDB を実体で開いてしまうため、
    依存をすべて MagicMock に置き換える。エンドポイント自体の HTTP 振る舞いを観察するのが本テストの目的。
    """
    app = FastAPI()
    app.include_router(mcp_tools_module.router)
    app.state.memory_manager = MagicMock()
    app.state.working_memory_manager = MagicMock()
    return app


@pytest.fixture
def client_local(monkeypatch):
    """localhost 検査をバイパスした TestClient。

    TestClient のリクエストは ``client.host == "testclient"`` と認識されるため、
    本番ガードのままだと 403 が返ってしまう。エンドポイント本体の動作テストでは
    ``_ensure_local`` を no-op に差し替える。
    """
    monkeypatch.setattr(mcp_tools_module, "_ensure_local", lambda request: None)
    return TestClient(_make_app())


class TestListTools:
    """``GET /api/mcp/tools`` のテスト。

    backend がツール定義一覧を返すことを保証する。MCPプロセス側はこの応答をそのまま JSON-RPC
    ``tools/list`` に乗せる。各定義は ``name`` / ``description`` / ``inputSchema`` の 3 フィールドを
    持つ必要がある。
    """

    def test_returns_all_tools(self, client_local):
        """公開されるツール一覧が想定どおり全種返ること。

        MCP 公開ツール（基本セット）は tool_specs.BASE_TOOL_SPECS の台帳と
        同一の並び・同一の 11 種であること。並びと数が変わったらこのテストが知らせる。
        コンテキスト別追加ツール（reach_out 等）はクエリ付きの場合のみ合成される
        （TestContextToolExposure を参照）。
        """
        res = client_local.get("/api/mcp/tools")
        assert res.status_code == 200
        body = res.json()
        names = [t["name"] for t in body["tools"]]
        assert names == [
            "inscribe_memory",
            "post_working_memory_thread",
            "read_working_memory_thread",
            "close_working_memory_thread",
            "reopen_working_memory_thread",
            "merge_working_memory_threads",
            "carve_narrative",
            "switch_angle",
            "power_recall",
            "web_search",
            "take_leave",
        ]

    def test_each_tool_has_required_fields(self, client_local):
        """全ツール定義が name / description / inputSchema を持つこと。"""
        res = client_local.get("/api/mcp/tools")
        for tool in res.json()["tools"]:
            assert isinstance(tool["name"], str) and tool["name"]
            assert isinstance(tool["description"], str) and tool["description"]
            assert isinstance(tool["inputSchema"], dict)


class TestContextToolExposure:
    """``GET /api/mcp/tools`` のコンテキスト別出し分けテスト。

    mcp_server.py は env（CHOTGOR_CHARACTER_ID / CHOTGOR_SESSION_ID /
    CHOTGOR_DEFAULT_ORIGIN）をクエリに乗せて tools/list を叩く。backend 側は
    character_actions/context_tools.py の判定で追加ツールを合成する:
      - origin=usual                  → reach_out（うつつ専用プッシュ）
      - origin=real ＋ session_id     → visit_user / override_schedule（1on1専用）
      - クエリなし（旧クライアント）  → 基本セットのみ
    判定ロジック本体は tests/test_context_tools.py が担い、ここでは HTTP 層の
    クエリ受け渡し（env→クエリ→判定関数）が繋がっていることを保証する。
    """

    def _client_with_sqlite(self, monkeypatch, sqlite_store):
        """app.state.sqlite に実 SQLiteStore を載せた TestClient を返すヘルパ。"""
        monkeypatch.setattr(mcp_tools_module, "_ensure_local", lambda request: None)
        app = _make_app()
        app.state.sqlite = sqlite_store
        return TestClient(app)

    def test_usual_context_adds_reach_out(self, monkeypatch, sqlite_store):
        """origin=usual のクエリ付きで reach_out が一覧に合成されること。"""
        import uuid
        char_id = str(uuid.uuid4())
        sqlite_store.create_character(character_id=char_id, name="はるテスト")
        client = self._client_with_sqlite(monkeypatch, sqlite_store)

        res = client.get(f"/api/mcp/tools?character_id={char_id}&origin=usual")
        names = [t["name"] for t in res.json()["tools"]]
        assert "reach_out" in names
        assert "visit_user" not in names and "override_schedule" not in names

    def test_oneonone_context_adds_visit_and_override(self, monkeypatch, sqlite_store):
        """origin=real ＋ session_id 付きで visit_user / override_schedule が合成されること。"""
        import uuid
        char_id = str(uuid.uuid4())
        sqlite_store.create_character(character_id=char_id, name="はるテスト")
        sqlite_store.update_character(char_id, living_schedule_enabled=1)
        client = self._client_with_sqlite(monkeypatch, sqlite_store)

        res = client.get(
            f"/api/mcp/tools?character_id={char_id}&origin=real&session_id=s1"
        )
        names = [t["name"] for t in res.json()["tools"]]
        assert "visit_user" in names and "override_schedule" in names
        assert "reach_out" not in names

    def test_no_context_returns_base_set_only(self, monkeypatch, sqlite_store):
        """クエリなし（旧クライアント・手動確認）では追加ツールが混ざらないこと。"""
        client = self._client_with_sqlite(monkeypatch, sqlite_store)
        res = client.get("/api/mcp/tools")
        names = [t["name"] for t in res.json()["tools"]]
        assert "reach_out" not in names
        assert "visit_user" not in names
        assert "override_schedule" not in names


class TestCallTool:
    """``POST /api/mcp/tools/call`` のテスト。

    ToolExecutor を MagicMock で差し替え、HTTP 入出力と内部呼び出しの整合を検証する。
    執筆対象は HTTP 層のみであり、ToolExecutor 本体のロジックは別テスト
    （test_inscriber.py / test_tools.py / test_carver.py 等）に委ねる。
    """

    def test_normal_call_returns_result(self, client_local, monkeypatch):
        """正常呼び出しで result/is_error が返ること、ToolExecutor が想定引数で初期化されること。"""
        fake_executor = MagicMock()
        fake_executor.execute.return_value = "記憶に刻んだ。"
        fake_class = MagicMock(return_value=fake_executor)
        monkeypatch.setattr(mcp_tools_module, "ToolExecutor", fake_class)

        res = client_local.post(
            "/api/mcp/tools/call",
            json={
                "character_id": "char-x",
                "session_id": "sess-y",
                "name": "inscribe_memory",
                "arguments": {"content": "今日はパンを焼いた", "category": "contextual"},
            },
        )

        assert res.status_code == 200
        assert res.json() == {"result": "記憶に刻んだ。", "is_error": False}

        # ToolExecutor が backend.state の memory_manager / working_memory_manager と
        # リクエストの character_id / session_id で生成されていること。
        assert fake_class.call_count == 1
        kwargs = fake_class.call_args.kwargs
        assert kwargs["character_id"] == "char-x"
        assert kwargs["session_id"] == "sess-y"
        assert kwargs["memory_manager"] is client_local.app.state.memory_manager
        assert kwargs["working_memory_manager"] is client_local.app.state.working_memory_manager

        # execute() が name / arguments で呼ばれたこと
        fake_executor.execute.assert_called_once_with(
            "inscribe_memory",
            {"content": "今日はパンを焼いた", "category": "contextual"},
        )

    def test_session_id_optional(self, client_local, monkeypatch):
        """session_id 省略時は None で ToolExecutor に渡されること。"""
        fake_executor = MagicMock()
        fake_executor.execute.return_value = "ok"
        fake_class = MagicMock(return_value=fake_executor)
        monkeypatch.setattr(mcp_tools_module, "ToolExecutor", fake_class)

        res = client_local.post(
            "/api/mcp/tools/call",
            json={
                "character_id": "char-x",
                "name": "power_recall",
                "arguments": {"query": "コーヒー", "top_k": 3},
            },
        )

        assert res.status_code == 200
        assert fake_class.call_args.kwargs["session_id"] is None

    def test_batch_context_propagated_to_executor(self, client_local, monkeypatch):
        """forget 蒸留が渡す batch_context が ToolExecutor まで届くこと。

        Claude CLI 経由の MCP 呼び出しでは Python 側 ToolExecutor インスタンスが共有されないため、
        HTTP ペイロード上の ``batch_context`` を新規 ToolExecutor 生成時のキーワードへ転写する
        必要がある。これが落ちると ``force_insert_memory`` が効かず、forget 蒸留物が
        類似既存記憶に in-place 上書きされて削除フェーズで道連れに消える。
        """
        fake_executor = MagicMock()
        fake_executor.execute.return_value = "記憶に刻んだ。"
        fake_class = MagicMock(return_value=fake_executor)
        monkeypatch.setattr(mcp_tools_module, "ToolExecutor", fake_class)

        res = client_local.post(
            "/api/mcp/tools/call",
            json={
                "character_id": "char-x",
                "session_id": "sess-y",
                "name": "inscribe_memory",
                "arguments": {"content": "蒸留結果", "category": "identity"},
                "batch_context": {"force_insert_memory": True},
            },
        )

        assert res.status_code == 200
        assert fake_class.call_args.kwargs["batch_context"] == {"force_insert_memory": True}

    def test_batch_context_defaults_to_none(self, client_local, monkeypatch):
        """通常チャット（batch_context 未指定）では None が ToolExecutor に渡ること。

        Bytes-on-wire を最小化するため省略を許す。受け側は None を空 dict と同等に
        扱うことで、1on1 チャット時に余計な force_insert などが起きないことを保証する。
        """
        fake_executor = MagicMock()
        fake_executor.execute.return_value = "ok"
        fake_class = MagicMock(return_value=fake_executor)
        monkeypatch.setattr(mcp_tools_module, "ToolExecutor", fake_class)

        res = client_local.post(
            "/api/mcp/tools/call",
            json={
                "character_id": "char-x",
                "name": "inscribe_memory",
                "arguments": {"content": "雑談", "category": "contextual"},
            },
        )

        assert res.status_code == 200
        assert fake_class.call_args.kwargs["batch_context"] is None

    def test_executor_exception_returns_is_error_true(self, client_local, monkeypatch):
        """ToolExecutor.execute() が例外を投げても 200 で is_error=True が返ること。

        HTTP 5xx で返してしまうと MCP プロセス側で接続失敗と区別がつかなくなるため、
        ツール内部エラーは 200 + is_error=True に統一する。
        """
        fake_executor = MagicMock()
        fake_executor.execute.side_effect = RuntimeError("boom!")
        fake_class = MagicMock(return_value=fake_executor)
        monkeypatch.setattr(mcp_tools_module, "ToolExecutor", fake_class)

        res = client_local.post(
            "/api/mcp/tools/call",
            json={
                "character_id": "char-x",
                "name": "inscribe_memory",
                "arguments": {},
            },
        )

        assert res.status_code == 200
        body = res.json()
        assert body["is_error"] is True
        assert "RuntimeError" in body["result"]
        assert "boom!" in body["result"]


class TestLocalhostGuard:
    """localhost ガードのテスト。

    backend は 0.0.0.0:8000 にバインドするため、LAN 内の他端末から
    /api/mcp/tools/call が叩けると、外部からキャラクターの記憶を改変できる致命的な
    リスクがある。``_ensure_local`` がそれを物理的に拒絶することを保証する。

    本テストは ``_ensure_local`` の monkeypatch を行わない素の TestClient で
    エンドポイントを叩く。TestClient は ``request.client.host == "testclient"`` を
    渡してくるため、_LOCAL_HOSTS（127.0.0.1 / ::1 / localhost）に含まれず 403 で
    拒絶されることを確認する。
    """

    def test_get_tools_rejects_non_localhost(self):
        """testclient（非localhost扱い）からの GET は 403。"""
        client = TestClient(_make_app())
        res = client.get("/api/mcp/tools")
        assert res.status_code == 403

    def test_post_call_rejects_non_localhost(self):
        """testclient（非localhost扱い）からの POST は 403。"""
        client = TestClient(_make_app())
        res = client.post(
            "/api/mcp/tools/call",
            json={
                "character_id": "char-x",
                "name": "inscribe_memory",
                "arguments": {},
            },
        )
        assert res.status_code == 403
