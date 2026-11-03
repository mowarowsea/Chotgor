"""Tests for backend.api.ui.dashboard — ダッシュボード（index）ページ。"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from backend.api.ui import common as ui_common
from backend.api.ui.dashboard import _merge_period_rows, cache_hit_ratio, router

_TEMPLATES_DIR = str(Path(__file__).parent.parent / "backend" / "templates")


@pytest.fixture
def dashboard_client(sqlite_store, monkeypatch):
    """dashboard ルーターと実テンプレートを組み込んだテストクライアントを返すフィクスチャ。

    テンプレートは本物（backend/templates）を使い、Jinja の構文崩れ・
    未定義変数参照をテストで検出できるようにする。
    """
    monkeypatch.setattr(ui_common, "templates", Jinja2Templates(directory=_TEMPLATES_DIR))
    app = FastAPI()
    app.include_router(router)
    app.state.sqlite = sqlite_store
    return TestClient(app)


class TestCacheHitRatio:
    """cache_hit_ratio（キャッシュ読出 ÷ 入力全長）のテスト。prompt_cache_plan.md C案。"""

    def test_ratio_is_cache_over_total_input(self):
        """cache / (input + cache) を返すこと。"""
        assert cache_hit_ratio(input_tokens=2500, cache_tokens=7500) == 0.75

    def test_zero_input_is_none(self):
        """入力が 0（記録なし・出力のみ）なら None を返し、0 除算しないこと。"""
        assert cache_hit_ratio(0, 0) is None

    def test_no_cache_is_zero(self):
        """キャッシュ読出が無ければ 0.0（None ではない＝「効いていない」と表示できる）。"""
        assert cache_hit_ratio(1000, 0) == 0.0

    def test_none_tokens_are_treated_as_zero(self):
        """None（旧行・未記録）は 0 として扱うこと。"""
        assert cache_hit_ratio(None, None) is None
        assert cache_hit_ratio(None, 10) == 1.0


class TestMergePeriodRows:
    """_merge_period_rows（期間×プロバイダー別行 → 期間単位の合算）のテスト。"""

    def test_merges_providers_within_same_period(self):
        """同一期間の複数プロバイダー行が1期間に合算され、内訳として抱き合わされること。"""
        rows = [
            {"day": "2026-06-11", "provider": "claude_cli", "requests": 2,
             "input_tokens": 100, "cache_tokens": 800, "output_tokens": 20, "cost_usd": 0.01},
            {"day": "2026-06-11", "provider": "google", "requests": 1,
             "input_tokens": 50, "cache_tokens": 0, "output_tokens": 5, "cost_usd": 0.0},
        ]

        merged = _merge_period_rows(rows, "day")

        assert len(merged) == 1
        m = merged[0]
        assert m["label"] == "2026-06-11"
        assert m["requests"] == 3
        assert m["input_tokens"] == 150
        assert m["cache_tokens"] == 800
        assert m["output_tokens"] == 25
        assert m["cost_usd"] == 0.01
        assert m["providers"] == rows
        # キャッシュ率は合算後の値（800 / (150 + 800)）と内訳行それぞれに付く
        assert m["cache_ratio"] == pytest.approx(800 / 950)
        assert rows[0]["cache_ratio"] == pytest.approx(800 / 900)
        assert rows[1]["cache_ratio"] == 0.0

    def test_keeps_period_order(self):
        """入力の期間順（新しい順）がそのまま保たれること。"""
        rows = [
            {"day": "2026-06-11", "provider": "claude_cli", "requests": 1,
             "input_tokens": 1, "cache_tokens": 0, "output_tokens": 1, "cost_usd": 0.0},
            {"day": "2026-06-10", "provider": "claude_cli", "requests": 1,
             "input_tokens": 1, "cache_tokens": 0, "output_tokens": 1, "cost_usd": 0.0},
        ]

        merged = _merge_period_rows(rows, "day")

        assert [m["label"] for m in merged] == ["2026-06-11", "2026-06-10"]

    def test_empty_rows(self):
        """空入力からは空リストが返ること。"""
        assert _merge_period_rows([], "day") == []


class TestDashboardPage:
    """GET /ui/ の描画スモークテスト。

    集計クエリ → _merge_period_rows → dashboard.html の縦の経路を実テンプレートで
    通し、データ無し（初回起動）とデータ有りの両方で 200 が返ることを確認する。
    """

    def test_renders_without_data(self, dashboard_client):
        """使用量・キャラ・シナリオが空でも 200 で描画されること（初回起動状態）。"""
        resp = dashboard_client.get("/ui/")

        assert resp.status_code == 200
        assert "dashboard" in resp.text
        assert "まだ記録がありません" in resp.text

    def test_renders_usage_rows(self, dashboard_client, sqlite_store):
        """記録済みの使用量がサマリー・テーブルに描画されること。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli",
            model="claude-sonnet-4-6",
            target="織羽",
            feature="chat",
            input_tokens=1200,
            output_tokens=340,
            total_cost_usd=0.0123,
        )

        resp = dashboard_client.get("/ui/")

        assert resp.status_code == 200
        assert "1,200" in resp.text  # input_tokens の桁区切り表示
        assert "claude_cli" in resp.text
        assert "織羽" in resp.text

    def test_renders_cache_ratio(self, dashboard_client, sqlite_store):
        """キャッシュ率がサマリーカードと直近イベントに整数パーセントで出ること。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli",
            model="claude-sonnet-4-6",
            target="織羽",
            feature="chat",
            input_tokens=200,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=1500,
            output_tokens=10,
        )

        resp = dashboard_client.get("/ui/")

        assert resp.status_code == 200
        # 1500 / (200 + 300 + 1500) = 75%
        assert "(75%)" in resp.text  # サマリーカード
        assert "Cache率" in resp.text  # テーブル列
        assert "1,500" in resp.text
