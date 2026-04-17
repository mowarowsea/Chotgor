"""backend.services.chat.service._run_farewell_detection のユニットテスト。

今回の修正の核心部分: ネガティブ退席の累積数が閾値に達した際に、
relationship_status を "estranged" に即座に更新するロジックを検証する。
Chronicle バッチを待たず、チャット終了時点でリアルタイムに疎遠化を確定させる動作を保証する。

対象関数:
    _run_farewell_detection() — 退席判定結果を受けて DB を更新するコルーチン

テスト方針:
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
    - FarewellDetector.detect() は AsyncMock でモックして実際のLLM呼び出しを回避する
    - FarewellResult を直接注入し、DB の状態変化（relationship_status / exited_chars）を確認する
    - 既存ネガティブ退席セッションを DB に作成することで累積カウントを制御する
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.character_actions.farewell_detector import FarewellDetector, FarewellResult
from backend.services.chat.service import _run_farewell_detection


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


# ─── ネガティブ退席 — 閾値以上（疎遠化確定）────────────────────────────────────


class TestFarewellDetectionEstrangement:
    """ネガティブ退席の累積数が閾値に達した際の疎遠化ロジックを検証する。

    これが今回のバグ修正で追加された核心的な動作である。
    chronicle バッチを待たず、チャット終了時点で即座に estranged が確定することを保証する。
    """

    def test_exactly_at_threshold_sets_estranged(self, sqlite_store, char_id, session_id):
        """累積数がちょうど閾値（prev=2, total=3, threshold=3）で relationship_status が "estranged" になること。"""
        config = _make_farewell_config(threshold=3)
        # prev_count=2 → total=3 = threshold
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

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_above_threshold_sets_estranged(self, sqlite_store, char_id, session_id):
        """累積数が閾値を超えた場合（prev=5, total=6, threshold=3）も "estranged" になること。"""
        config = _make_farewell_config(threshold=3)
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 5)

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

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_estranged_still_updates_exited_chars(self, sqlite_store, char_id, session_id):
        """疎遠化確定時もセッションの exited_chars が更新されること。"""
        config = _make_farewell_config(threshold=3)
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 2)

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
        assert exited[0]["char_name"] == "別れサービステストキャラ"
        assert exited[0]["farewell_type"] == "negative"

    def test_estranged_reason_uses_negative_fallback_when_no_estranged_key(
        self, sqlite_store, char_id, session_id
    ):
        """farewell_config に "estranged" キーがない場合、ネガティブ退席メッセージにフォールバックすること。"""
        config = _make_farewell_config(threshold=3, include_estranged_msg=False)
        config["farewell_message"]["negative"] = "嫌になった。"
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 2)

        result = _make_farewell_result(should_exit=True, farewell_type="negative", reason="嫌になった。")
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
        # 警告テキストではなく、元のネガティブメッセージが使われること
        assert "嫌になった。" in exited[0]["reason"]
        assert "別れを決断" not in exited[0]["reason"]

    def test_estranged_reason_uses_estranged_key_when_present(
        self, sqlite_store, char_id, session_id
    ):
        """farewell_config に "estranged" キーがある場合、そのメッセージが使われること。"""
        config = _make_farewell_config(threshold=3, include_estranged_msg=True)
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 2)

        result = _make_farewell_result(should_exit=True, farewell_type="negative", reason="もう話したくない。")
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
        # "estranged" キーのメッセージが使われること
        assert "あなたとの関係を終わりにします。" in exited[0]["reason"]

    def test_one_exit_at_threshold_one_sets_estranged(self, sqlite_store, char_id, session_id):
        """閾値が1の場合、最初のネガティブ退席で即 "estranged" になること。"""
        config = _make_farewell_config(threshold=1)
        # prev_count=0 → total=1 = threshold

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

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_chroma_mark_definition_estranged_called_on_estrangement(
        self, sqlite_store, char_id, session_id
    ):
        """疎遠化確定時に chroma.mark_definition_estranged() が呼ばれること。

        SQLite の更新だけでは ChromaDB の status が "active" のままとなり、
        類似キャラクター登録ブロックが機能しない。ChromaDB の更新も必須。
        """
        config = _make_farewell_config(threshold=3)
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 2)

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)
        mock_chroma = MagicMock()

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
            chroma=mock_chroma,
        ))

        mock_chroma.mark_definition_estranged.assert_called_once_with(char_id)

    def test_chroma_none_does_not_raise_on_estrangement(self, sqlite_store, char_id, session_id):
        """chroma=None でも疎遠化確定時に例外が発生しないこと。"""
        config = _make_farewell_config(threshold=1)

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
            chroma=None,
        ))

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_chroma_error_does_not_prevent_sqlite_update(self, sqlite_store, char_id, session_id):
        """chroma.mark_definition_estranged() が例外を投げても SQLite の更新は完了すること。"""
        config = _make_farewell_config(threshold=1)

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)
        mock_chroma = MagicMock()
        mock_chroma.mark_definition_estranged.side_effect = RuntimeError("ChromaDB接続失敗")

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
            chroma=mock_chroma,
        ))

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_chroma_not_called_below_threshold(self, sqlite_store, char_id, session_id):
        """閾値未満の場合、chroma.mark_definition_estranged() が呼ばれないこと。"""
        config = _make_farewell_config(threshold=5)
        # prev=0 → total=1 < threshold=5

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)
        mock_chroma = MagicMock()

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
            chroma=mock_chroma,
        ))

        mock_chroma.mark_definition_estranged.assert_not_called()


# ─── 非ネガティブ退席 — 疎遠化が起きないこと ─────────────────────────────────


class TestFarewellDetectionNonNegative:
    """positive / neutral 退席タイプでは疎遠化カウントが進まないことを検証する。"""

    def test_positive_exit_does_not_set_estranged(self, sqlite_store, char_id, session_id):
        """positive タイプの退席では、閾値に関係なく "estranged" にならないこと。"""
        config = _make_farewell_config(threshold=1)
        # threshold=1 でも positive なら疎遠化しない

        result = _make_farewell_result(should_exit=True, farewell_type="positive", reason="ありがとう。")
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

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_neutral_exit_does_not_set_estranged(self, sqlite_store, char_id, session_id):
        """neutral タイプの退席では "estranged" にならないこと。"""
        config = _make_farewell_config(threshold=1)

        result = _make_farewell_result(should_exit=True, farewell_type="neutral", reason="また今度。")
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

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_positive_exit_updates_exited_chars(self, sqlite_store, char_id, session_id):
        """positive タイプの退席でも exited_chars は更新されること。"""
        config = _make_farewell_config()

        result = _make_farewell_result(should_exit=True, farewell_type="positive", reason="楽しかった。")
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
        assert exited[0]["farewell_type"] == "positive"


# ─── 重複退席チェック ─────────────────────────────────────────────────────────


class TestFarewellDetectionDuplicateCheck:
    """同一セッションで同一キャラクターが二重退席しないことを検証する。"""

    def test_already_exited_session_is_skipped(self, sqlite_store, char_id, session_id):
        """既に exited_chars にエントリが存在するセッションは更新されないこと。"""
        # セッションにすでに退席エントリを追加する
        sqlite_store.update_chat_session(
            session_id,
            exited_chars=[{
                "char_name": "別れサービステストキャラ",
                "reason": "初回退席",
                "farewell_type": "negative",
            }],
        )

        config = _make_farewell_config(threshold=10)
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
        # エントリが増えていないこと（1件のまま）
        assert len(exited) == 1
        assert exited[0]["reason"] == "初回退席"

    def test_already_exited_does_not_set_estranged_even_at_threshold(
        self, sqlite_store, char_id, session_id
    ):
        """重複退席スキップ時は、閾値超過であっても relationship_status が変更されないこと。

        重複チェックで早期リターンするため、疎遠化判定コードに到達しない。
        """
        sqlite_store.update_chat_session(
            session_id,
            exited_chars=[{
                "char_name": "別れサービステストキャラ",
                "reason": "初回退席",
                "farewell_type": "negative",
            }],
        )

        # 既存のネガティブ退席を大量に作成（閾値を超える件数）
        config = _make_farewell_config(threshold=1)
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 5)

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

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "active"


# ─── 境界値 — lookback_days ───────────────────────────────────────────────────


class TestFarewellDetectionLookbackDays:
    """lookback_days 境界値の動作を検証する。"""

    def test_old_exits_outside_lookback_are_not_counted(self, sqlite_store, char_id, session_id):
        """lookback_days より前の退席セッションがカウントされないこと。

        updated_at が 8日以上前のセッションは lookback_days=7 の集計から除外される。
        get_negative_exit_count に since 引数として渡す日時より前のセッションはカウント対象外。
        ここでは get_negative_exit_count をモックして直接テストする。
        """
        config = _make_farewell_config(lookback_days=7, threshold=3)
        # lookback 外の退席は 0件扱いにする（モックで制御）
        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)
        # get_negative_exit_count を 0 を返すようにモック
        with patch.object(sqlite_store, "get_negative_exit_count", return_value=0):
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

        # prev=0, total=1, threshold=3 → active のまま
        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_get_negative_exit_count_called_with_correct_since(
        self, sqlite_store, char_id, session_id
    ):
        """get_negative_exit_count に lookback_days に基づく since 日時が渡されること。"""
        config = _make_farewell_config(lookback_days=14, threshold=999)
        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)

        captured_args = []
        original_fn = sqlite_store.get_negative_exit_count

        def capture_fn(character_name, since):
            captured_args.append((character_name, since))
            return 0

        with patch.object(sqlite_store, "get_negative_exit_count", side_effect=capture_fn):
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

        assert len(captured_args) == 1
        _, since = captured_args[0]
        # since は約14日前であること（1分の誤差を許容）
        expected_since = datetime.now() - timedelta(days=14)
        assert abs((since - expected_since).total_seconds()) < 60
