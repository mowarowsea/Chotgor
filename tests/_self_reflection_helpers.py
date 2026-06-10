"""自己参照ループ（SelfReflector）テスト群の共有フィクスチャ・ヘルパー。

キャラクター・各種プリセット・マネージャ・SelfReflector 本体のフィクスチャと、
モックプロバイダー生成ヘルパーを提供する。test_self_reflection_*.py から
import して使用する（フィクスチャは import するだけで pytest が認識する）。
sqlite_store は conftest.py で定義済み。
ファイル名先頭がアンダースコアのため pytest のテスト収集対象にはならない。
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.character_actions.reflector import SelfReflector


@pytest.fixture
def char_id(sqlite_store):
    """テスト用キャラクターをSQLiteに作成し、そのIDを返すフィクスチャ。"""
    cid = "reflector-test-char"
    sqlite_store.create_character(
        character_id=cid,
        name="自己参照テストキャラ",
        system_prompt_block1="テスト用設定",
    )
    return cid


@pytest.fixture
def trigger_preset_id(sqlite_store):
    """テスト用モデルプリセット（Ollama）をSQLiteに作成し、そのIDを返すフィクスチャ。"""
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Ollama",
        provider="ollama",
        model_id="qwen2.5:3b",
    )
    return pid


@pytest.fixture
def reflection_preset_id(sqlite_store):
    """テスト用モデルプリセット（Ollama: SUPPORTS_TOOLS=False）をSQLiteに作成し、そのIDを返すフィクスチャ。

    _run_reflection() のタグ方式フォールバックパスを検証するために使用する。
    Ollama は SUPPORTS_TOOLS=False のため ask_character() + タグパースを経由する。
    """
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Ollama-Reflection",
        provider="ollama",
        model_id="qwen2.5:7b",
    )
    return pid


@pytest.fixture
def tools_reflection_preset_id(sqlite_store):
    """テスト用モデルプリセット（claude_cli: SUPPORTS_TOOLS=True）をSQLiteに作成し、そのIDを返すフィクスチャ。

    _run_reflection() のMCPツール方式パスを検証するために使用する。
    claude_cli は SUPPORTS_TOOLS=True のため ask_character_with_tools() を経由する。
    """
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-ClaudeCLI-Reflection",
        provider="claude_cli",
        model_id="",
    )
    return pid


@pytest.fixture
def google_trigger_preset_id(sqlite_store):
    """テスト用モデルプリセット（Google）をSQLiteに作成し、そのIDを返すフィクスチャ。

    Ollamaに限らず任意プロバイダーを契機判断に使えることの検証用。
    """
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Google",
        provider="google",
        model_id="gemini-2.0-flash",
    )
    return pid


@pytest.fixture
def memory_manager(sqlite_store):
    """MagicMock の InscribedMemoryManager（sqlite は実DBに差し替え）を返すフィクスチャ。"""
    mm = MagicMock()
    mm.sqlite = sqlite_store
    return mm


@pytest.fixture
def working_memory_manager():
    """MagicMock の WorkingMemoryManager を返すフィクスチャ。"""
    return MagicMock()


@pytest.fixture
def reflector(memory_manager, working_memory_manager):
    """テスト用 SelfReflector インスタンスを返すフィクスチャ。"""
    return SelfReflector(
        memory_manager=memory_manager, working_memory_manager=working_memory_manager
    )


def _make_provider(response_text: str = "") -> MagicMock:
    """指定テキストを返す非同期 generate() を持つモックプロバイダーを生成する。

    Args:
        response_text: generate() が返すテキスト。

    Returns:
        AsyncMock の generate() を持つ MagicMock プロバイダー。
    """
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=response_text)
    return provider

