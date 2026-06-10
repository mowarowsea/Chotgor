"""退席判定 — 無動作ケース＆閾値未満のユニットテスト。

テスト方針:
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
    - FarewellDetector.detect() は AsyncMock でモックして実際のLLM呼び出しを回避する
    - FarewellResult を直接注入し、DB の状態変化（relationship_status / exited_chars）を確認する
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.chat.service import _run_farewell_detection

from tests._farewell_helpers import (  # noqa: F401
    _create_negative_exit_sessions,
    _make_detector,
    _make_farewell_config,
    _make_farewell_result,
    _run,
    char_id,
    farewell_config,
    session_id,
)

# ─── should_exit=False / None — 何も起きないケース ──────────────────────────────


class TestFarewellDetectionNoAction:
    """detect() が退席不要と判定した場合、DBが変化しないことを検証する。"""

    def test_should_exit_false_does_not_update_session(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """should_exit=False の場合、exited_chars が更新されないこと。"""
        result = _make_farewell_result(should_exit=False, farewell_type="neutral")
        detector = _make_detector(sqlite_store, result)

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=farewell_config,
            messages=[{"role": "user", "content": "テスト"}],
            settings={},
        ))

        session = sqlite_store.get_chat_session(session_id)
        exited = getattr(session, "exited_chars", None) or []
        assert len(exited) == 0

    def test_should_exit_false_does_not_set_estranged(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """should_exit=False の場合、relationship_status が変更されないこと。"""
        result = _make_farewell_result(should_exit=False, farewell_type="neutral")
        detector = _make_detector(sqlite_store, result)

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=farewell_config,
            messages=[],
            settings={},
        ))

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_detector_returns_none_does_not_update_session(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """detect() が None を返した場合、exited_chars が更新されないこと。"""
        detector = _make_detector(sqlite_store, None)

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=farewell_config,
            messages=[],
            settings={},
        ))

        session = sqlite_store.get_chat_session(session_id)
        exited = getattr(session, "exited_chars", None) or []
        assert len(exited) == 0


# ─── ネガティブ退席 — 閾値未満 ────────────────────────────────────────────────


class TestFarewellDetectionBelowThreshold:
    """ネガティブ退席の累積数が閾値未満の場合の動作を検証する。"""

    def test_below_threshold_does_not_set_estranged(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """累積数が閾値未満（2回、閾値3）の場合、relationship_status が "active" のままであること。"""
        # 既存のネガティブ退席を2件作成（prev_count=2、total=3、threshold=3 → ちょうど閾値）
        # ただしここでは threshold=3 に対して prev_count=1 のケースでテスト
        config = _make_farewell_config(threshold=3)
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 1)

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
        ))

        # total_count = prev_count(1) + 1 = 2 < threshold(3) → active のまま
        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_below_threshold_appends_warning_to_reason(
        self, sqlite_store, char_id, session_id
    ):
        """累積数が閾値未満の場合、警告テキストが退席メッセージに付加されること。"""
        config = _make_farewell_config(lookback_days=7, threshold=3)
        # prev_count=1 → total=2 < threshold=3
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 1)

        result = _make_farewell_result(should_exit=True, farewell_type="negative", reason="不機嫌。")
        detector = _make_detector(sqlite_store, result)

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
        ))

        session = sqlite_store.get_chat_session(session_id)
        exited = getattr(session, "exited_chars", None) or []
        assert len(exited) == 1
        reason_text = exited[0]["reason"]
        # 警告テキストの各構成要素が含まれること
        assert "7日間" in reason_text
        assert "3回" in reason_text
        assert "別れを決断" in reason_text

    def test_warning_contains_total_count(self, sqlite_store, char_id, session_id):
        """警告テキストに現在の累積退席数（prev + 1）が含まれること。"""
        config = _make_farewell_config(lookback_days=14, threshold=5)
        # prev_count=2 → total=3
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 2)

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
        ))

        session = sqlite_store.get_chat_session(session_id)
        exited = getattr(session, "exited_chars", None) or []
        assert len(exited) == 1
        # total_count=3 が警告に含まれること
        assert "3回" in exited[0]["reason"]

    def test_zero_previous_exits_shows_count_one_in_warning(
        self, sqlite_store, char_id, session_id
    ):
        """既存退席がゼロの場合、警告テキストに「1回」が含まれること。"""
        config = _make_farewell_config(threshold=5)
        # prev_count=0 → total=1

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
        ))

        session = sqlite_store.get_chat_session(session_id)
        exited = getattr(session, "exited_chars", None) or []
        assert "1回" in exited[0]["reason"]


