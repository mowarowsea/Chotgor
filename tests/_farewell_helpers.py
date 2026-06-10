"""退席判定（_run_farewell_detection）テスト群の共有ヘルパー・フィクスチャ。

FarewellResult / FarewellDetector / farewell_config の組み立てと、
累積カウント制御用のネガティブ退席セッション作成を提供する。
test_farewell_service_*.py から import して使用する
（フィクスチャは import するだけで pytest が認識する）。
sqlite_store は conftest.py で定義済み。
ファイル名先頭がアンダースコアのため pytest のテスト収集対象にはならない。
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.character_actions.farewell_detector import FarewellDetector, FarewellResult


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


def _make_farewell_result(
    should_exit: bool,
    farewell_type: str = "negative",
    reason: str = "もう話したくない。",
    emotions: dict | None = None,
) -> FarewellResult:
    """テスト用 FarewellResult を生成するヘルパー。

    Args:
        should_exit: 退席すべきかどうか。
        farewell_type: 退席タイプ（"negative" / "positive" / "neutral"）。
        reason: 退席メッセージ。
        emotions: 感情スコア dict。

    Returns:
        FarewellResult インスタンス。
    """
    return FarewellResult(
        should_exit=should_exit,
        farewell_type=farewell_type,
        reason=reason,
        emotions=emotions or {"anger": 0.9, "disgust": 0.5, "boredom": 0.1, "despair": 0.2},
    )


def _make_detector(sqlite_store, result: FarewellResult | None) -> FarewellDetector:
    """detect() の返値を固定したモック FarewellDetector を返すヘルパー。

    Args:
        sqlite_store: 実際の SQLiteStore インスタンス。
        result: detect() が返す FarewellResult（None も可）。

    Returns:
        モック済み FarewellDetector。
    """
    detector = MagicMock(spec=FarewellDetector)
    detector.sqlite = sqlite_store
    detector.detect = AsyncMock(return_value=result)
    return detector


def _make_farewell_config(
    lookback_days: int = 7,
    threshold: int = 3,
    include_estranged_msg: bool = False,
) -> dict:
    """テスト用 farewell_config を生成するヘルパー。

    Args:
        lookback_days: 集計対象の日数。
        threshold: 疎遠化の閾値（negative_exit_threshold）。
        include_estranged_msg: "estranged" キーのメッセージを含むかどうか。

    Returns:
        farewell_config 辞書。
    """
    farewell_messages = {
        "negative": "もう話したくない。",
        "positive": "ありがとう。",
        "neutral": "また今度。",
    }
    if include_estranged_msg:
        farewell_messages["estranged"] = "あなたとの関係を終わりにします。"
    return {
        "thresholds": {"anger": 0.8, "disgust": 0.7},
        "farewell_message": farewell_messages,
        "estrangement": {
            "lookback_days": lookback_days,
            "negative_exit_threshold": threshold,
        },
    }


def _create_negative_exit_sessions(sqlite_store, char_name: str, count: int) -> None:
    """指定数のネガティブ退席セッションを DB に作成するヘルパー。

    累積カウントをテストデータとして注入するために使用する。

    Args:
        sqlite_store: SQLiteStore インスタンス。
        char_name: 退席対象キャラクター名。
        count: 作成するネガティブ退席セッション数。
    """
    for _ in range(count):
        sess_id = str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=sess_id,
            model_id=f"{char_name}@test-preset",
        )
        sqlite_store.update_chat_session(
            sess_id,
            exited_chars=[{
                "char_name": char_name,
                "reason": "テスト退席",
                "farewell_type": "negative",
            }],
        )


def _run(coro):
    """asyncio.run のラッパー。テスト内でコルーチンを同期実行する。"""
    return asyncio.run(coro)


# ─── フィクスチャ ──────────────────────────────────────────────────────────────


@pytest.fixture
def char_id(sqlite_store):
    """テスト用キャラクターを SQLite に作成し、そのIDを返すフィクスチャ。"""
    cid = str(uuid.uuid4())
    sqlite_store.create_character(
        character_id=cid,
        name="別れサービステストキャラ",
        system_prompt_block1="テスト用設定",
    )
    return cid


@pytest.fixture
def session_id(sqlite_store, char_id):
    """テスト用セッションを SQLite に作成し、そのIDを返すフィクスチャ。"""
    sid = str(uuid.uuid4())
    sqlite_store.create_chat_session(
        session_id=sid,
        model_id="別れサービステストキャラ@test-preset",
    )
    return sid


@pytest.fixture
def farewell_config():
    """デフォルトの farewell_config（閾値3、lookback7日）を返すフィクスチャ。"""
    return _make_farewell_config(lookback_days=7, threshold=3)

