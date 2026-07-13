"""Tests for backend.api.ui.instruments — 計器パネルページ（めぐり Phase 2）。

実テンプレート（backend/templates/instruments.html）を使った描画スモークテストと、
アラーム確認済み化エンドポイントの動作を検証する。Jinja の構文崩れ・
未定義変数参照をテストで検出できるよう、テンプレートは本物を使う。
"""

import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from backend.api.ui import common as ui_common
from backend.api.ui.instruments import router

_TEMPLATES_DIR = str(Path(__file__).parent.parent / "backend" / "templates")


@pytest.fixture
def instruments_client(sqlite_store, monkeypatch):
    """instruments ルーターと実テンプレートを組み込んだテストクライアントを返すフィクスチャ。"""
    monkeypatch.setattr(ui_common, "templates", Jinja2Templates(directory=_TEMPLATES_DIR))
    app = FastAPI()
    app.include_router(router)
    app.state.sqlite = sqlite_store
    return TestClient(app)


class TestInstrumentsPanel:
    """GET /ui/instruments の描画スモークテスト。

    計器未稼働（データなし）・アラーム/スメル/メーターあり、の両状態で
    200 が返り、主要な表示要素（静音期間・検知器ラベル）が出ることを確認する。
    """

    def test_renders_without_data(self, instruments_client):
        """計器未稼働（アラームゼロ）でも 200 で描画されること。"""
        resp = instruments_client.get("/ui/instruments")
        assert resp.status_code == 200
        assert "計器パネル" in resp.text
        assert "計器未稼働" in resp.text

    def test_renders_with_alarms_and_meters(self, instruments_client, sqlite_store):
        """アラーム・スメル・メーターがある状態で描画され、内容が反映されること。"""
        char_id = str(uuid.uuid4())
        sqlite_store.create_character(character_id=char_id, name="はるテスト")
        sqlite_store.fire_alarm(
            "fabrication_backstop",
            details={"session_id": "s1", "note": "夜の営みの停止"},
        )
        sqlite_store.fire_alarm("smell_format_debris", severity="smell")
        sqlite_store.record_meter("wm_thread_count", 3, character_id=char_id)
        resp = instruments_client.get("/ui/instruments")
        assert resp.status_code == 200
        assert "GMのユーザ捏造" in resp.text       # invariant ラベル
        assert "WMスレッド数" in resp.text          # メーターラベル
        assert "はるテスト" in resp.text
        assert "夜の営みの停止" in resp.text
        assert "\\u591c\\u306e" not in resp.text

    def test_acknowledge_alarm(self, instruments_client, sqlite_store):
        """確認済み化 POST がアラームを ack してパネルへリダイレクトする。"""
        sqlite_store.fire_alarm("usual_scene_error")
        alarm = sqlite_store.list_alarms()[0]
        resp = instruments_client.post(
            f"/ui/instruments/alarms/{alarm.id}/ack", follow_redirects=False,
        )
        assert resp.status_code == 303
        assert sqlite_store.list_alarms(unacknowledged_only=True) == []
