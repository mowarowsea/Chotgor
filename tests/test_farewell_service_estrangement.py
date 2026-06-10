"""退席判定 — 疎遠化確定・非ネガティブ・重複チェック・lookback 境界のユニットテスト。

ネガティブ退席の累積数が閾値に達した際に relationship_status を "estranged" に
即座に更新するロジック（Chronicle バッチを待たないリアルタイム疎遠化）を検証する。
"""

import uuid
from datetime import datetime, timedelta
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

    def test_vector_store_mark_definition_estranged_called_on_estrangement(
        self, sqlite_store, char_id, session_id
    ):
        """疎遠化確定時に vector_store.mark_definition_estranged() が呼ばれること。

        SQLite の更新だけでは LanceDB の status が "active" のままとなり、
        類似キャラクター登録ブロックが機能しない。LanceDB の更新も必須。
        """
        config = _make_farewell_config(threshold=3)
        _create_negative_exit_sessions(sqlite_store, "別れサービステストキャラ", 2)

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)
        mock_vector_store = MagicMock()

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
            vector_store=mock_vector_store,
        ))

        mock_vector_store.mark_definition_estranged.assert_called_once_with(char_id)

    def test_vector_store_none_does_not_raise_on_estrangement(self, sqlite_store, char_id, session_id):
        """vector_store=None でも疎遠化確定時に例外が発生しないこと。"""
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
            vector_store=None,
        ))

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_vector_store_error_does_not_prevent_sqlite_update(self, sqlite_store, char_id, session_id):
        """vector_store.mark_definition_estranged() が例外を投げても SQLite の更新は完了すること。"""
        config = _make_farewell_config(threshold=1)

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)
        mock_vector_store = MagicMock()
        mock_vector_store.mark_definition_estranged.side_effect = RuntimeError("LanceDB接続失敗")

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
            vector_store=mock_vector_store,
        ))

        char = sqlite_store.get_character(char_id)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_vector_store_not_called_below_threshold(self, sqlite_store, char_id, session_id):
        """閾値未満の場合、vector_store.mark_definition_estranged() が呼ばれないこと。"""
        config = _make_farewell_config(threshold=5)
        # prev=0 → total=1 < threshold=5

        result = _make_farewell_result(should_exit=True, farewell_type="negative")
        detector = _make_detector(sqlite_store, result)
        mock_vector_store = MagicMock()

        _run(_run_farewell_detection(
            detector=detector,
            character_id=char_id,
            character_name="別れサービステストキャラ",
            session_id=session_id,
            preset_id="dummy-preset",
            farewell_config=config,
            messages=[],
            settings={},
            vector_store=mock_vector_store,
        ))

        mock_vector_store.mark_definition_estranged.assert_not_called()


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
