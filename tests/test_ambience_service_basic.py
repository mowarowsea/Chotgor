"""退席判定 — 無動作ケース＆閾値未満のユニットテスト。

テスト方針:
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
    - AmbienceJudge.detect() は AsyncMock でモックして実際のLLM呼び出しを回避する
    - AmbienceReading を直接注入し、DB の状態変化（relationship_status / exited_chars）を確認する
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.chat.service import run_ambience_detection

from tests._ambience_helpers import (  # noqa: F401
    _create_negative_exit_sessions,
    _make_judge,
    _make_farewell_config,
    _make_ambience_reading,
    _run,
    char_id,
    farewell_config,
    session_id,
)

# ─── should_exit=False / None — 何も起きないケース ──────────────────────────────


class TestAmbienceDetectionNoAction:
    """detect() が退席不要と判定した場合、DBが変化しないことを検証する。"""

    def test_should_exit_false_does_not_update_session(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """should_exit=False の場合、exited_chars が更新されないこと。"""
        result = _make_ambience_reading(should_exit=False, farewell_type="neutral")
        judge = _make_judge(sqlite_store, result)

        _run(run_ambience_detection(
            judge=judge,
            character_id=char_id,
            character_name="なりゆきサービステストキャラ",
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
        result = _make_ambience_reading(should_exit=False, farewell_type="neutral")
        judge = _make_judge(sqlite_store, result)

        _run(run_ambience_detection(
            judge=judge,
            character_id=char_id,
            character_name="なりゆきサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=farewell_config,
            messages=[],
            settings={},
        ))

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_judge_returns_none_does_not_update_session(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """detect() が None を返した場合、exited_chars が更新されないこと。"""
        judge = _make_judge(sqlite_store, None)

        _run(run_ambience_detection(
            judge=judge,
            character_id=char_id,
            character_name="なりゆきサービステストキャラ",
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


class TestAmbienceDetectionBelowThreshold:
    """ネガティブ退席の累積数が閾値未満の場合の動作を検証する。"""

    def test_below_threshold_does_not_set_estranged(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """累積数が閾値未満（2回、閾値3）の場合、relationship_status が "active" のままであること。"""
        # 既存のネガティブ退席を2件作成（prev_count=2、total=3、threshold=3 → ちょうど閾値）
        # ただしここでは threshold=3 に対して prev_count=1 のケースでテスト
        config = _make_farewell_config(threshold=3)
        _create_negative_exit_sessions(sqlite_store, "なりゆきサービステストキャラ", 1)

        result = _make_ambience_reading(should_exit=True, farewell_type="negative")
        judge = _make_judge(sqlite_store, result)

        _run(run_ambience_detection(
            judge=judge,
            character_id=char_id,
            character_name="なりゆきサービステストキャラ",
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
        _create_negative_exit_sessions(sqlite_store, "なりゆきサービステストキャラ", 1)

        result = _make_ambience_reading(should_exit=True, farewell_type="negative", reason="不機嫌。")
        judge = _make_judge(sqlite_store, result)

        _run(run_ambience_detection(
            judge=judge,
            character_id=char_id,
            character_name="なりゆきサービステストキャラ",
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
        _create_negative_exit_sessions(sqlite_store, "なりゆきサービステストキャラ", 2)

        result = _make_ambience_reading(should_exit=True, farewell_type="negative")
        judge = _make_judge(sqlite_store, result)

        _run(run_ambience_detection(
            judge=judge,
            character_id=char_id,
            character_name="なりゆきサービステストキャラ",
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

        result = _make_ambience_reading(should_exit=True, farewell_type="negative")
        judge = _make_judge(sqlite_store, result)

        _run(run_ambience_detection(
            judge=judge,
            character_id=char_id,
            character_name="なりゆきサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
        ))

        session = sqlite_store.get_chat_session(session_id)
        exited = getattr(session, "exited_chars", None) or []
        assert "1回" in exited[0]["reason"]


# ─── 場所判定の反映（current_bg_label / ambience Step 4） ────────────────────


class TestAmbienceBgLabelPersistence:
    """run_ambience_detection が location_label を chat_sessions.current_bg_label に
    反映する動作を検証するテストクラス。

    検証する観点:
        - 対面モード中: AmbienceReading.location_label が current_bg_label へ保存される
        - 対面モード OFF: judge が何を返そうと current_bg_label は変わらない
        - detect() が None（スキップ/失敗）: current_bg_label は前回値を維持する
        - ラベル変化なし: 値は同じまま（実害のない不変性確認）
        - detect() へ対面フラグ・候補ラベル・前回ラベルが引数リレーされる
    """

    def _set_face_to_face(self, sqlite_store, char_id, labels=("はるの部屋", "もわの部屋")):
        """キャラを対面モード ON + ラベル付き背景画像あり、に設定するヘルパー。"""
        sqlite_store.update_character(
            char_id,
            face_to_face_mode=1,
            face_to_face_bg_images=[
                {"label": lb, "image": f"data:image/png;base64,{i}"}
                for i, lb in enumerate(labels)
            ],
        )

    def _run_detection(self, sqlite_store, judge, char_id, session_id, farewell_config):
        """run_ambience_detection を既定引数で同期実行するヘルパー。"""
        _run(run_ambience_detection(
            judge=judge,
            character_id=char_id,
            character_name="なりゆきサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=farewell_config,
            messages=[{"role": "user", "content": "部屋に移動した"}],
            settings={},
        ))

    def test_label_saved_when_face_to_face(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """対面中に judge がラベルを返すと current_bg_label に保存されること。"""
        self._set_face_to_face(sqlite_store, char_id)
        result = _make_ambience_reading(should_exit=False, farewell_type="neutral")
        result.location_label = "はるの部屋"
        judge = _make_judge(sqlite_store, result)

        self._run_detection(sqlite_store, judge, char_id, session_id, farewell_config)

        session = sqlite_store.get_chat_session(session_id)
        assert session.current_bg_label == "はるの部屋"

    def test_label_not_saved_when_not_face_to_face(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """対面 OFF では judge がラベルを返しても保存されないこと。"""
        # face_to_face_mode は既定 0 のまま
        result = _make_ambience_reading(should_exit=False, farewell_type="neutral")
        result.location_label = "はるの部屋"
        judge = _make_judge(sqlite_store, result)

        self._run_detection(sqlite_store, judge, char_id, session_id, farewell_config)

        session = sqlite_store.get_chat_session(session_id)
        assert getattr(session, "current_bg_label", None) is None

    def test_label_kept_when_judge_returns_none(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """judge が None（失敗/スキップ）のとき前回ラベルが維持されること。"""
        self._set_face_to_face(sqlite_store, char_id)
        sqlite_store.update_chat_session(session_id, current_bg_label="もわの部屋")
        judge = _make_judge(sqlite_store, None)

        self._run_detection(sqlite_store, judge, char_id, session_id, farewell_config)

        session = sqlite_store.get_chat_session(session_id)
        assert session.current_bg_label == "もわの部屋"

    def test_label_change_overwrites_previous(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """前回と異なるラベルが返ると上書きされること。"""
        self._set_face_to_face(sqlite_store, char_id)
        sqlite_store.update_chat_session(session_id, current_bg_label="もわの部屋")
        result = _make_ambience_reading(should_exit=False, farewell_type="neutral")
        result.location_label = "はるの部屋"
        judge = _make_judge(sqlite_store, result)

        self._run_detection(sqlite_store, judge, char_id, session_id, farewell_config)

        session = sqlite_store.get_chat_session(session_id)
        assert session.current_bg_label == "はるの部屋"

    def test_detect_receives_location_context(
        self, sqlite_store, char_id, session_id, farewell_config
    ):
        """detect() に対面フラグ・候補ラベル・前回ラベルが渡されること。"""
        self._set_face_to_face(sqlite_store, char_id)
        sqlite_store.update_chat_session(session_id, current_bg_label="もわの部屋")
        result = _make_ambience_reading(should_exit=False, farewell_type="neutral")
        judge = _make_judge(sqlite_store, result)

        self._run_detection(sqlite_store, judge, char_id, session_id, farewell_config)

        kwargs = judge.detect.call_args.kwargs
        assert kwargs["face_to_face"] is True
        assert kwargs["bg_label_candidates"] == ["はるの部屋", "もわの部屋"]
        assert kwargs["prev_bg_label"] == "もわの部屋"


