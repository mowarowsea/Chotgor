"""backend.api.translation エンドポイントのユニットテスト。

FastAPI TestClient を使い、モック化した app.state (sqlite) と
create_provider() でHTTP振る舞いを検証する。
LLM への実際のネットワーク呼び出しは一切行わない。

テスト対象:
    POST /api/translate — テキストを日本語に翻訳して返すエンドポイント

テスト方針:
    - SQLite 呼び出しは MagicMock で差し替える
    - create_provider() は patch() で差し替え、LLMコールをモック化する
    - 正常翻訳ケース・各種エラーケースを網羅する
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import translation as translation_module


# ---------------------------------------------------------------------------
# テスト用 FastAPI アプリのセットアップ
# ---------------------------------------------------------------------------


def _make_preset(preset_id: str = "preset-uuid", name: str = "TestModel",
                 provider: str = "anthropic", model_id: str = "claude-haiku",
                 thinking_level: str = "default"):
    """LLMModelPreset 風の MagicMock を返すヘルパー。"""
    p = MagicMock()
    p.id = preset_id
    p.name = name
    p.provider = provider
    p.model_id = model_id
    p.thinking_level = thinking_level
    return p


def _make_app(sqlite_mock) -> FastAPI:
    """モック state を持つ最小 FastAPI アプリを生成するファクトリ。"""
    app = FastAPI()
    app.include_router(translation_module.router)
    app.state.sqlite = sqlite_mock
    return app


# ---------------------------------------------------------------------------
# 正常系テスト
# ---------------------------------------------------------------------------


class TestTranslateTextSuccess:
    """POST /api/translate の正常系テスト。"""

    def test_returns_translated_text(self):
        """翻訳結果が {"translation": "..."} 形式で返ること。"""
        preset_id = str(uuid.uuid4())
        preset = _make_preset(preset_id=preset_id)

        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": preset_id}
        sqlite.get_model_preset.return_value = preset

        provider = MagicMock()
        provider.generate = AsyncMock(return_value="翻訳されたテキスト")

        app = _make_app(sqlite)
        with TestClient(app) as client:
            with patch("backend.api.translation.create_provider", return_value=provider):
                res = client.post("/api/translate", json={"text": "Hello, world!"})

        assert res.status_code == 200
        assert res.json()["translation"] == "翻訳されたテキスト"

    def test_uses_preset_id_from_request_when_provided(self):
        """リクエストの preset_id が設定より優先されること。"""
        request_preset_id = str(uuid.uuid4())
        settings_preset_id = str(uuid.uuid4())
        preset = _make_preset(preset_id=request_preset_id)

        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": settings_preset_id}
        sqlite.get_model_preset.return_value = preset

        provider = MagicMock()
        provider.generate = AsyncMock(return_value="応答")

        app = _make_app(sqlite)
        with TestClient(app) as client:
            with patch("backend.api.translation.create_provider", return_value=provider):
                client.post(
                    "/api/translate",
                    json={"text": "test", "preset_id": request_preset_id},
                )

        sqlite.get_model_preset.assert_called_with(request_preset_id)

    def test_uses_translation_preset_id_from_settings_when_not_in_request(self):
        """リクエストに preset_id がない場合、設定の translation_preset_id を使うこと。"""
        preset_id = str(uuid.uuid4())
        preset = _make_preset(preset_id=preset_id)

        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": preset_id}
        sqlite.get_model_preset.return_value = preset

        provider = MagicMock()
        provider.generate = AsyncMock(return_value="応答")

        app = _make_app(sqlite)
        with TestClient(app) as client:
            with patch("backend.api.translation.create_provider", return_value=provider):
                client.post("/api/translate", json={"text": "test"})

        sqlite.get_model_preset.assert_called_with(preset_id)

    def test_passes_correct_system_prompt_to_provider(self):
        """翻訳指示のシステムプロンプトが provider.generate() に渡されること。"""
        preset_id = str(uuid.uuid4())
        preset = _make_preset(preset_id=preset_id)

        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": preset_id}
        sqlite.get_model_preset.return_value = preset

        provider = MagicMock()
        provider.generate = AsyncMock(return_value="結果")

        app = _make_app(sqlite)
        with TestClient(app) as client:
            with patch("backend.api.translation.create_provider", return_value=provider):
                client.post("/api/translate", json={"text": "Translate me"})

        call_args = provider.generate.call_args[0]
        system_prompt = call_args[0]
        assert "translator" in system_prompt.lower() or "translate" in system_prompt.lower()

    def test_passes_text_as_user_message(self):
        """翻訳対象テキストが user ロールのメッセージとして渡されること。"""
        preset_id = str(uuid.uuid4())
        preset = _make_preset(preset_id=preset_id)

        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": preset_id}
        sqlite.get_model_preset.return_value = preset

        provider = MagicMock()
        provider.generate = AsyncMock(return_value="結果")

        app = _make_app(sqlite)
        with TestClient(app) as client:
            with patch("backend.api.translation.create_provider", return_value=provider):
                client.post("/api/translate", json={"text": "test input text"})

        call_args = provider.generate.call_args[0]
        messages = call_args[1]
        assert messages == [{"role": "user", "content": "test input text"}]


# ---------------------------------------------------------------------------
# エラー系テスト
# ---------------------------------------------------------------------------


class TestTranslateTextErrors:
    """POST /api/translate のエラー系テスト。"""

    def test_400_when_no_preset_configured(self):
        """translation_preset_id が未設定かつリクエストにも preset_id がない場合 400 を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {}  # translation_preset_id なし

        app = _make_app(sqlite)
        with TestClient(app) as client:
            res = client.post("/api/translate", json={"text": "hello"})

        assert res.status_code == 400
        assert "翻訳モデル" in res.json()["detail"]

    def test_400_when_preset_id_is_empty_string(self):
        """translation_preset_id が空文字の場合 400 を返すこと。"""
        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": ""}

        app = _make_app(sqlite)
        with TestClient(app) as client:
            res = client.post("/api/translate", json={"text": "hello"})

        assert res.status_code == 400

    def test_404_when_preset_not_found(self):
        """指定した preset_id が存在しない場合 404 を返すこと。"""
        preset_id = str(uuid.uuid4())

        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": preset_id}
        sqlite.get_model_preset.return_value = None  # プリセット未発見

        app = _make_app(sqlite)
        with TestClient(app) as client:
            res = client.post("/api/translate", json={"text": "hello"})

        assert res.status_code == 404

    def test_503_when_provider_creation_fails(self):
        """create_provider() が例外を送出した場合 503 を返すこと。"""
        preset_id = str(uuid.uuid4())
        preset = _make_preset(preset_id=preset_id)

        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": preset_id}
        sqlite.get_model_preset.return_value = preset

        app = _make_app(sqlite)
        with TestClient(app) as client:
            with patch(
                "backend.api.translation.create_provider",
                side_effect=RuntimeError("provider init failed"),
            ):
                res = client.post("/api/translate", json={"text": "hello"})

        assert res.status_code == 503
        assert "初期化" in res.json()["detail"]

    def test_503_when_llm_call_fails(self):
        """provider.generate() が例外を送出した場合 503 を返すこと。"""
        preset_id = str(uuid.uuid4())
        preset = _make_preset(preset_id=preset_id)

        sqlite = MagicMock()
        sqlite.get_all_settings.return_value = {"translation_preset_id": preset_id}
        sqlite.get_model_preset.return_value = preset

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=ConnectionError("LLM unreachable"))

        app = _make_app(sqlite)
        with TestClient(app) as client:
            with patch("backend.api.translation.create_provider", return_value=provider):
                res = client.post("/api/translate", json={"text": "hello"})

        assert res.status_code == 503
        assert "リクエスト" in res.json()["detail"]
