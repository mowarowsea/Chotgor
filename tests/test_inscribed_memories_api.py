"""backend.api.inscribed_memories の Chronicle / Forget エンドポイントのユニットテスト。

対象エンドポイント:
    POST /api/inscribed_memories/{character_id}/chronicle  — 個別キャラ Chronicle 手動実行
    POST /api/inscribed_memories/{character_id}/forget     — 個別キャラ Forget 手動実行
    POST /api/inscribed_memories/batch/chronicle           — 全キャラ Chronicle 一括実行
    POST /api/inscribed_memories/batch/forget              — 全キャラ Forget 一括実行

テスト方針:
    - FastAPI TestClient を使って HTTP レイヤーから検証する。
    - LLM を呼び出す run_chronicle / run_forget_process / run_pending_* は AsyncMock でパッチし、
      外部サービスへの依存を排除する。
    - SQLite は実 DB（tmp_path）を使い、キャラクター存在チェックを正確に再現する。
    - キャラクターが存在しない場合の 404 応答も確認する。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.inscribed_memories import router
from backend.repositories.sqlite.store import SQLiteStore


@pytest.fixture
def sqlite(tmp_path):
    """テスト用インメモリ相当の SQLiteStore を返すフィクスチャ。"""
    return SQLiteStore(str(tmp_path / "test.db"))


@pytest.fixture
def char_id(sqlite):
    """テスト用キャラクターを1件作成してその ID を返すフィクスチャ。"""
    cid = "test-char-id-0001"
    sqlite.create_character(
        character_id=cid,
        name="テストキャラ",
    )
    return cid


@pytest.fixture
def client(sqlite):
    """inscribed_memories ルーターを組み込んだ最小 FastAPI アプリのテストクライアントを返す。

    app.state に必要なサービスインスタンスを設定する。
    """
    app = FastAPI()
    app.include_router(router)

    app.state.sqlite = sqlite
    app.state.memory_manager = MagicMock()
    app.state.working_memory_manager = MagicMock()
    app.state.vector_store = MagicMock()

    return TestClient(app, raise_server_exceptions=False)


# ─── /chronicle エンドポイント ─────────────────────────────────────────────────


class TestTriggerChronicle:
    """POST /api/inscribed_memories/{character_id}/chronicle の動作を検証するテストクラス。"""

    def test_returns_200_on_success(self, client, char_id):
        """Chronicle が正常完了したとき 200 と結果 JSON が返ること。"""
        mock_result = {"status": "success", "inscribed": 2}
        with patch(
            "backend.api.inscribed_memories.run_chronicle",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(f"/api/inscribed_memories/{char_id}/chronicle")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_returns_404_when_character_not_found(self, client):
        """存在しないキャラクター ID を指定した場合 404 が返ること。"""
        resp = client.post("/api/inscribed_memories/nonexistent-id/chronicle")
        assert resp.status_code == 404

    def test_skipped_result_returned_as_is(self, client, char_id):
        """Chronicle が対象なしでスキップしたとき status='skipped' が返ること。"""
        mock_result = {"status": "skipped", "reason": "No messages"}
        with patch(
            "backend.api.inscribed_memories.run_chronicle",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(f"/api/inscribed_memories/{char_id}/chronicle")
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"


# ─── /forget エンドポイント ────────────────────────────────────────────────────


class TestTriggerForget:
    """POST /api/inscribed_memories/{character_id}/forget の動作を検証するテストクラス。"""

    def test_returns_200_on_success(self, client, char_id):
        """Forget が正常完了したとき 200 と結果 JSON が返ること。"""
        mock_result = {"status": "success", "candidates_count": 3, "deleted_count": 2}
        with patch(
            "backend.api.inscribed_memories.run_forget_process",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(f"/api/inscribed_memories/{char_id}/forget")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["deleted_count"] == 2

    def test_returns_404_when_character_not_found(self, client):
        """存在しないキャラクター ID を指定した場合 404 が返ること。"""
        resp = client.post("/api/inscribed_memories/nonexistent-id/forget")
        assert resp.status_code == 404

    def test_skipped_when_no_candidates(self, client, char_id):
        """対象記憶がない場合、status='skipped' が返ること。"""
        mock_result = {"status": "skipped", "reason": "No forgotten candidates found"}
        with patch(
            "backend.api.inscribed_memories.run_forget_process",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(f"/api/inscribed_memories/{char_id}/forget")
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

    def test_error_when_ghost_model_not_set(self, client, char_id):
        """ghost_model 未設定の場合、エラーステータスが返ること。"""
        mock_result = {"status": "error", "error": "ghost_model が設定されていません。"}
        with patch(
            "backend.api.inscribed_memories.run_forget_process",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(f"/api/inscribed_memories/{char_id}/forget")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_custom_threshold_passed_to_run_forget_process(self, client, char_id):
        """クエリパラメータの threshold が run_forget_process に正しく渡ること。"""
        mock_result = {"status": "skipped"}
        with patch(
            "backend.api.inscribed_memories.run_forget_process",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            client.post(f"/api/inscribed_memories/{char_id}/forget?threshold=0.5")
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["threshold"] == pytest.approx(0.5)


# ─── /batch/chronicle エンドポイント ──────────────────────────────────────────


class TestBatchChronicle:
    """POST /api/inscribed_memories/batch/chronicle の動作を検証するテストクラス。

    全キャラクター一括 Chronicle 処理を実行するエンドポイント。
    run_pending_chronicles が呼び出され、200 {"status": "ok"} が返ることを確認する。
    """

    def test_returns_200_ok(self, client):
        """一括 Chronicle が正常完了したとき 200 {"status": "ok"} が返ること。"""
        with patch(
            "backend.api.inscribed_memories.run_pending_chronicles",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = client.post("/api/inscribed_memories/batch/chronicle")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_run_pending_chronicles_called_once(self, client):
        """run_pending_chronicles が1回だけ呼び出されること。"""
        with patch(
            "backend.api.inscribed_memories.run_pending_chronicles",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fn:
            client.post("/api/inscribed_memories/batch/chronicle")
        mock_fn.assert_called_once()


# ─── /batch/forget エンドポイント ─────────────────────────────────────────────


class TestBatchForget:
    """POST /api/inscribed_memories/batch/forget の動作を検証するテストクラス。

    全キャラクター一括 Forget 処理を実行するエンドポイント。
    run_pending_forget が呼び出され、200 {"status": "ok"} が返ることを確認する。
    """

    def test_returns_200_ok(self, client):
        """一括 Forget が正常完了したとき 200 {"status": "ok"} が返ること。"""
        with patch(
            "backend.api.inscribed_memories.run_pending_forget",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = client.post("/api/inscribed_memories/batch/forget")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_run_pending_forget_called_once(self, client):
        """run_pending_forget が1回だけ呼び出されること。"""
        with patch(
            "backend.api.inscribed_memories.run_pending_forget",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fn:
            client.post("/api/inscribed_memories/batch/forget")
        mock_fn.assert_called_once()
