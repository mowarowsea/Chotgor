import pytest
import os
import tempfile
from unittest.mock import AsyncMock

from backend.repositories.sqlite.store import SQLiteStore

@pytest.fixture
def sqlite_store():
    """Provides a fresh temporary SQLite storage for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(path)
    yield store
    # Cleanup
    store.engine.dispose()
    try:
        os.remove(path)
    except PermissionError:
        pass


@pytest.fixture(autouse=True)
def _disable_intent_pickup(monkeypatch):
    """意図の拾い上げ（めぐり Phase 4）をテスト全体で既定無効化する autouse フィクスチャ。

    run_chronicle / run_usual_days_scene は末尾で run_intent_pickup を「同乗」させる
    （追加の LLM 呼び出しが1回発生する）。Chronicle・うつつの既存単体テストは
    「LLM 呼び出しは1回」を前提にプロンプトや呼び出し回数を検証しているため、
    同乗呼び出しをここで丸ごと no-op に差し替える。

    拾い上げ自体の検証は tests/test_intents.py が pickup モジュールを直接 import
    して行うので、このパッチ（パッケージ属性の差し替え）の影響を受けない。
    """
    import backend.services.intents as intents_pkg
    monkeypatch.setattr(
        intents_pkg,
        "run_intent_pickup",
        AsyncMock(return_value={"status": "skipped", "reason": "テストでは既定無効"}),
    )
