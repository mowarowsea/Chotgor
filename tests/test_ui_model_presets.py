"""Tests for モデルプリセット一覧の並び順と /ui/model-presets ページ描画。

このモジュールは「プリセット一覧はプロバイダー表示順（PROVIDER_ORDER）を第一キー、
モデルIDを第二キー、プリセット名を第三キーに並べる」という規約を、
以下の3経路で守れているか検証する:

1. `SQLiteStore.list_model_presets()` — backend UI 各ページ（models / settings /
   character edit / memories / scenarios）が共通で使う一覧取得。
2. `GET /v1/models` — フロントエンド（Chotgor UI）のモデル切替メニューが表示順を
   そのまま引き継ぐ API。キャラクターごとに並べ替えられていることを見る。
3. `GET /ui/model-presets` — 一覧テーブルが「表示名・プロバイダー・モデルID」だけを持ち、
   残りの情報（思考レベル・タイムアウト・登録日時）が詳細モーダル側に出ること。

テンプレートは本物（backend/templates）を読み込むため、Jinja の構文崩れや
未定義変数参照もここで落ちる。
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from backend.adapters.openai.router import router as openai_router
from backend.api.ui import common as ui_common
from backend.api.ui.presets import router as presets_router
from backend.providers.registry import PROVIDER_ORDER, provider_sort_key

_TEMPLATES_DIR = str(Path(__file__).parent.parent / "backend" / "templates")


def _seed_presets(store) -> None:
    """並び順が created_at 順とは一致しないよう、わざと崩した順で登録する。

    anthropic に「モデルID が名前順と逆転する組（Bravo=claude-x / Alpha=claude-y）」と
    「同一モデルIDで名前だけ異なる組（Aardvark / Bravo = claude-x）」を含めて、
    第二キー=モデルID・第三キー=名前 の優先順位を判別できるようにしている。
    """
    store.create_model_preset("p-google", "Gemini", "google", "gemini-2.0-flash")
    store.create_model_preset("p-ant-b", "Bravo", "anthropic", "claude-x")
    store.create_model_preset("p-ollama", "Qwen", "ollama", "qwen2.5:latest")
    store.create_model_preset("p-ant-a", "Alpha", "anthropic", "claude-y")
    store.create_model_preset("p-cli", "CLI", "claude_cli", "")
    store.create_model_preset("p-ant-c", "Aardvark", "anthropic", "claude-x")


@pytest.fixture
def presets_client(sqlite_store, monkeypatch):
    """/ui/model-presets ルーターと実テンプレートを組み込んだテストクライアント。"""
    monkeypatch.setattr(ui_common, "templates", Jinja2Templates(directory=_TEMPLATES_DIR))
    app = FastAPI()
    app.include_router(presets_router)
    app.state.sqlite = sqlite_store
    return TestClient(app)


class TestProviderSortKey:
    """provider_sort_key — プロバイダー表示順のソートキー生成。"""

    def test_follows_provider_order(self):
        """PROVIDER_ORDER の並びどおりのキーが返ること。"""
        keys = [provider_sort_key(p) for p in PROVIDER_ORDER]
        assert keys == sorted(keys)

    def test_unknown_provider_goes_last(self):
        """PROVIDER_ORDER 未登録のプロバイダーは末尾へ回ること。"""
        assert provider_sort_key("unknown_provider") > provider_sort_key(PROVIDER_ORDER[-1])


class TestListModelPresetsOrder:
    """SQLiteStore.list_model_presets — 一覧の並び順。"""

    def test_sorted_by_provider_then_model_then_name(self, sqlite_store):
        """プロバイダー表示順 → モデルID → プリセット名 の優先順位で並ぶこと。"""
        _seed_presets(sqlite_store)

        names = [p.name for p in sqlite_store.list_model_presets()]

        # claude_cli → anthropic(claude-x: Aardvark, Bravo → claude-y: Alpha) → google → ollama
        assert names == ["CLI", "Aardvark", "Bravo", "Alpha", "Gemini", "Qwen"]

    def test_empty_model_id_comes_first_within_provider(self, sqlite_store):
        """モデルID 空欄（プロバイダー既定）が同一プロバイダー内の先頭に来ること。"""
        sqlite_store.create_model_preset("p1", "Zulu", "anthropic", "")
        sqlite_store.create_model_preset("p2", "Alpha", "anthropic", "claude-x")

        names = [p.name for p in sqlite_store.list_model_presets()]

        assert names == ["Zulu", "Alpha"]

    def test_name_order_is_case_insensitive(self, sqlite_store):
        """同一プロバイダー・同一モデルIDの名前順が大文字小文字を無視して並ぶこと。"""
        sqlite_store.create_model_preset("p1", "beta", "anthropic", "m")
        sqlite_store.create_model_preset("p2", "Alpha", "anthropic", "m")

        names = [p.name for p in sqlite_store.list_model_presets()]

        assert names == ["Alpha", "beta"]


class TestModelsApiOrder:
    """GET /v1/models — フロントエンドが引き継ぐ表示順。"""

    @pytest.fixture
    def models_client(self, sqlite_store):
        """openai 互換ルーターだけを組み込んだテストクライアント。"""
        app = FastAPI()
        app.include_router(openai_router)
        app.state.sqlite = sqlite_store
        return TestClient(app)

    def test_presets_sorted_within_character(self, sqlite_store, models_client):
        """キャラクターごとにプロバイダー順→モデルID順→プリセット名順で返ること。

        APIキー必須のプロバイダー（anthropic / google）は settings に鍵がないと
        除外されるため、ここで設定してから検証する。
        """
        _seed_presets(sqlite_store)
        sqlite_store.set_setting("anthropic_api_key", "dummy")
        sqlite_store.set_setting("google_api_key", "dummy")
        sqlite_store.create_character(
            "char-1",
            "はる",
            enabled_providers={
                "p-google": {},
                "p-ant-b": {},
                "p-ollama": {},
                "p-ant-a": {},
                "p-cli": {},
                "p-ant-c": {},
            },
        )

        data = models_client.get("/v1/models").json()["data"]

        assert [m["id"] for m in data] == [
            "はる@CLI",
            "はる@Aardvark",
            "はる@Bravo",
            "はる@Alpha",
            "はる@Gemini",
            "はる@Qwen",
        ]
        # 並び順の根拠（プロバイダー）をフロントへ渡していること
        assert [m["provider"] for m in data] == [
            "claude_cli", "anthropic", "anthropic", "anthropic", "google", "ollama",
        ]


class TestModelPresetsPage:
    """GET /ui/model-presets — 一覧テーブルと詳細モーダルの描画。"""

    def test_rows_sorted_by_provider_then_model_then_name(self, sqlite_store, presets_client):
        """テーブル行がプロバイダー順→モデルID順→名前順で並ぶこと（HTML 上の出現順で判定）。"""
        _seed_presets(sqlite_store)

        html = presets_client.get("/ui/model-presets").text

        expected = ["p-cli", "p-ant-c", "p-ant-b", "p-ant-a", "p-google", "p-ollama"]
        positions = [html.index(f'data-modal-open="preset-{pid}"') for pid in expected]
        assert positions == sorted(positions)

    def test_list_columns_are_slim(self, sqlite_store, presets_client):
        """一覧の列見出しが表示名・プロバイダー・モデルIDの3つに絞られていること。"""
        _seed_presets(sqlite_store)

        html = presets_client.get("/ui/model-presets").text

        head = html[html.index("<thead>"):html.index("</thead>")]
        assert "表示名" in head and "プロバイダー" in head and "モデルID" in head
        # 詳細情報は一覧の列から外れている
        assert "思考レベル" not in head and "タイムアウト秒" not in head

    def test_modal_shows_created_at_and_details(self, sqlite_store, presets_client):
        """詳細モーダルに登録日時と、一覧から外した項目が出ること。"""
        _seed_presets(sqlite_store)
        created = sqlite_store.get_model_preset("p-cli").created_at

        html = presets_client.get("/ui/model-presets").text

        modal = html[html.index('id="modal-preset-p-cli"'):]
        modal = modal[:modal.index("</div>\n    </div>")]
        assert created.strftime("%Y-%m-%d %H:%M") in modal
        assert "思考レベル" in modal
        assert "タイムアウト秒" in modal

    def test_empty_state(self, presets_client):
        """プリセットが1件もないときは空状態メッセージが出ること。"""
        html = presets_client.get("/ui/model-presets").text

        assert "登録されているモデルがありません" in html
