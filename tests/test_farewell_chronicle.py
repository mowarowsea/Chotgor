"""chronicle_job.py の別れ機能拡張部分のユニットテスト。

Chronicle バッチの farewell_config 対応と疎遠化判定（_check_estrangement）の動作を検証する。

対象関数:
    _check_estrangement()           — ネガティブ退席数が閾値を超えたら relationship_status を更新する
    run_chronicle()                 — farewell_config を更新できること・_check_estrangement を呼ぶこと
    run_pending_chronicles()        — chroma を受け取って各 run_chronicle に渡すこと

テスト方針:
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
    - ask_character() は AsyncMock でモックして実際のLLM呼び出しを回避する
    - ChromaDB は MagicMock で差し替える（embedding は不要）
    - _check_estrangement は公開関数ではないため、get_negative_exit_count をモックして間接的にテストする
    - run_chronicle() のテストでは farewell_config が JSON に含まれる場合のみ DB に書き込まれることを確認する
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.batch.chronicle_job import (
    _check_estrangement,
    run_chronicle,
    run_pending_chronicles,
)


# ─── フィクスチャ ─────────────────────────────────────────────────────────────
# sqlite_store は conftest.py で定義済み。


@pytest.fixture
def ghost_preset_id(sqlite_store):
    """テスト用 ghost_model プリセット（Ollama）を作成し、IDを返すフィクスチャ。"""
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Ghost-Ollama",
        provider="ollama",
        model_id="qwen2.5:3b",
    )
    return pid


@pytest.fixture
def char_id_with_ghost(sqlite_store, ghost_preset_id):
    """ghost_model が設定されたテスト用キャラクターを作成し、IDを返すフィクスチャ。"""
    cid = str(uuid.uuid4())
    sqlite_store.create_character(
        character_id=cid,
        name="別れChronicleキャラ",
        system_prompt_block1="テスト用設定",
        ghost_model=ghost_preset_id,
    )
    return cid


@pytest.fixture
def farewell_config_dict():
    """テスト用 farewell_config 辞書を返すフィクスチャ。"""
    return {
        "thresholds": {
            "anger": 0.8,
            "disgust": 0.7,
        },
        "farewell_message": {
            "negative": "私はこの会話を終わらせます。",
            "positive": "ありがとう。",
            "neutral": "また今度。",
        },
        "estrangement": {
            "lookback_days": 30,
            "negative_exit_threshold": 3,
        },
    }


def _make_chronicle_response(
    self_history_text: str | None = None,
    relationship_text: str | None = None,
    farewell_config: dict | None = None,
) -> str:
    """chronicle LLM の JSON レスポンスを生成するヘルパー。

    Args:
        self_history_text: self_history の更新テキスト（None なら update=false）。
        relationship_text: relationship_state の更新テキスト（None なら update=false）。
        farewell_config: farewell_config の更新値（None なら null）。

    Returns:
        JSON 文字列。
    """
    payload = {
        "self_history": {
            "update": self_history_text is not None,
            "text": self_history_text,
        },
        "relationship_state": {
            "update": relationship_text is not None,
            "text": relationship_text,
        },
        "farewell_config": farewell_config,
    }
    return json.dumps(payload, ensure_ascii=False)


# ─── _check_estrangement ──────────────────────────────────────────────────────


class TestCheckEstrangement:
    """_check_estrangement() の疎遠化判定ロジックを検証する。

    直接呼び出しで挙動を確認する。
    DBの get_negative_exit_count は実際の SQLite を使用する。
    """

    def test_no_farewell_config_skips_silently(self, sqlite_store):
        """farewell_config が None のキャラクターはスキップされること（例外なし）。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="NoConfig")
        char = sqlite_store.get_character(cid)

        asyncio.run(_check_estrangement(char, sqlite_store, chroma=None))

        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_missing_estrangement_settings_skips(self, sqlite_store, farewell_config_dict):
        """estrangement 設定が欠如した場合はスキップされること。"""
        cid = str(uuid.uuid4())
        config_no_estrangement = {**farewell_config_dict}
        del config_no_estrangement["estrangement"]
        sqlite_store.create_character(character_id=cid, name="NoEstrangement")
        sqlite_store.update_character(cid, farewell_config=config_no_estrangement)
        char = sqlite_store.get_character(cid)

        asyncio.run(_check_estrangement(char, sqlite_store, chroma=None))

        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_count_below_threshold_stays_active(self, sqlite_store, farewell_config_dict):
        """ネガティブ退席数が閾値未満の場合、relationship_status が "active" のままであること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="BelowThreshold")
        sqlite_store.update_character(cid, farewell_config=farewell_config_dict)
        char = sqlite_store.get_character(cid)
        # negative_exit_threshold=3 に対して count=2 → 閾値未満
        with patch.object(sqlite_store, "get_negative_exit_count", return_value=2):
            asyncio.run(_check_estrangement(char, sqlite_store, chroma=None))

        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_count_at_threshold_sets_estranged(self, sqlite_store, farewell_config_dict):
        """ネガティブ退席数が閾値以上になると relationship_status が "estranged" になること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="AtThreshold")
        sqlite_store.update_character(cid, farewell_config=farewell_config_dict)
        char = sqlite_store.get_character(cid)
        # negative_exit_threshold=3 に対して count=3 → 閾値以上
        with patch.object(sqlite_store, "get_negative_exit_count", return_value=3):
            asyncio.run(_check_estrangement(char, sqlite_store, chroma=None))

        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_count_above_threshold_sets_estranged(self, sqlite_store, farewell_config_dict):
        """ネガティブ退席数が閾値を大きく超えた場合も "estranged" になること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="AboveThreshold")
        sqlite_store.update_character(cid, farewell_config=farewell_config_dict)
        char = sqlite_store.get_character(cid)
        with patch.object(sqlite_store, "get_negative_exit_count", return_value=10):
            asyncio.run(_check_estrangement(char, sqlite_store, chroma=None))

        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_chroma_mark_called_when_estranged(self, sqlite_store, farewell_config_dict):
        """疎遠化確定時に chroma.mark_definition_estranged() が呼ばれること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="EstrangedChroma")
        sqlite_store.update_character(cid, farewell_config=farewell_config_dict)
        char = sqlite_store.get_character(cid)

        mock_chroma = MagicMock()
        with patch.object(sqlite_store, "get_negative_exit_count", return_value=5):
            asyncio.run(_check_estrangement(char, sqlite_store, chroma=mock_chroma))

        mock_chroma.mark_definition_estranged.assert_called_once_with(cid)

    def test_chroma_none_does_not_raise(self, sqlite_store, farewell_config_dict):
        """chroma が None でも例外が発生しないこと。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="NullChroma")
        sqlite_store.update_character(cid, farewell_config=farewell_config_dict)
        char = sqlite_store.get_character(cid)
        with patch.object(sqlite_store, "get_negative_exit_count", return_value=5):
            asyncio.run(_check_estrangement(char, sqlite_store, chroma=None))

        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_chroma_error_does_not_propagate(self, sqlite_store, farewell_config_dict):
        """chroma.mark_definition_estranged() が例外を投げても run_chronicle が落ちないこと。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="ChromaError")
        sqlite_store.update_character(cid, farewell_config=farewell_config_dict)
        char = sqlite_store.get_character(cid)

        mock_chroma = MagicMock()
        mock_chroma.mark_definition_estranged.side_effect = RuntimeError("ChromaDB接続失敗")
        with patch.object(sqlite_store, "get_negative_exit_count", return_value=5):
            asyncio.run(_check_estrangement(char, sqlite_store, chroma=mock_chroma))

        # chroma エラーがあっても relationship_status は更新されていること
        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "estranged"


# ─── run_chronicle() — farewell_config 更新 ───────────────────────────────────


class TestRunChronicleFarewellConfig:
    """run_chronicle() が farewell_config を正しく更新することを検証する。"""

    def test_farewell_config_in_response_is_saved_to_db(
        self, sqlite_store, char_id_with_ghost, farewell_config_dict, ghost_preset_id
    ):
        """LLM が farewell_config を含む JSON を返した場合、DBに保存されること。"""
        response = _make_chronicle_response(farewell_config=farewell_config_dict)
        with patch("backend.batch.chronicle_job.ask_character", new=AsyncMock(return_value=response)):
            result = asyncio.run(
                run_chronicle(
                    character_id=char_id_with_ghost,
                    sqlite=sqlite_store,
                    settings={},
                    chroma=None,
                )
            )

        assert result["status"] == "success"
        char = sqlite_store.get_character(char_id_with_ghost)
        assert getattr(char, "farewell_config", None) is not None
        fc = char.farewell_config
        assert "thresholds" in fc
        assert fc["thresholds"]["anger"] == 0.8

    def test_null_farewell_config_in_response_is_not_saved(
        self, sqlite_store, char_id_with_ghost, ghost_preset_id
    ):
        """LLM が farewell_config=null を返した場合、DBは変更されないこと。"""
        # まず farewell_config を設定する
        existing_config = {"thresholds": {"anger": 0.5}}
        sqlite_store.update_character(char_id_with_ghost, farewell_config=existing_config)

        response = _make_chronicle_response(farewell_config=None)
        with patch("backend.batch.chronicle_job.ask_character", new=AsyncMock(return_value=response)):
            asyncio.run(
                run_chronicle(
                    character_id=char_id_with_ghost,
                    sqlite=sqlite_store,
                    settings={},
                    chroma=None,
                )
            )

        char = sqlite_store.get_character(char_id_with_ghost)
        # 既存の設定が保持されていること
        fc = getattr(char, "farewell_config", None)
        assert fc is not None
        assert fc["thresholds"]["anger"] == 0.5

    def test_farewell_config_and_self_history_both_updated(
        self, sqlite_store, char_id_with_ghost, farewell_config_dict, ghost_preset_id
    ):
        """farewell_config と self_history が同時に更新されること。"""
        response = _make_chronicle_response(
            self_history_text="新しい歴史テキスト",
            farewell_config=farewell_config_dict,
        )
        with patch("backend.batch.chronicle_job.ask_character", new=AsyncMock(return_value=response)):
            result = asyncio.run(
                run_chronicle(
                    character_id=char_id_with_ghost,
                    sqlite=sqlite_store,
                    settings={},
                    chroma=None,
                )
            )

        assert "self_history" in result["updated_fields"]
        assert "farewell_config" in result["updated_fields"]

        char = sqlite_store.get_character(char_id_with_ghost)
        assert char.self_history == "新しい歴史テキスト"
        assert char.farewell_config is not None

    def test_check_estrangement_called_after_update(
        self, sqlite_store, char_id_with_ghost, farewell_config_dict, ghost_preset_id
    ):
        """run_chronicle() 完了後に _check_estrangement() が呼ばれること。"""
        response = _make_chronicle_response(farewell_config=farewell_config_dict)
        with patch("backend.batch.chronicle_job.ask_character", new=AsyncMock(return_value=response)):
            with patch("backend.batch.chronicle_job._check_estrangement", new=AsyncMock()) as mock_check:
                asyncio.run(
                    run_chronicle(
                        character_id=char_id_with_ghost,
                        sqlite=sqlite_store,
                        settings={},
                        chroma=None,
                    )
                )
        mock_check.assert_called_once()

    def test_ghost_model_not_set_skips_chronicle(self, sqlite_store):
        """ghost_model が未設定のキャラクターは chronicle をスキップすること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="NoGhost")

        result = asyncio.run(
            run_chronicle(character_id=cid, sqlite=sqlite_store, settings={})
        )

        assert result["status"] == "skipped"

    def test_character_not_found_returns_error(self, sqlite_store):
        """存在しないキャラクターIDを渡すとエラーが返ること。"""
        result = asyncio.run(
            run_chronicle(character_id="nonexistent", sqlite=sqlite_store, settings={})
        )
        assert result["status"] == "error"

    def test_prompt_contains_farewell_rubric(
        self, sqlite_store, char_id_with_ghost, ghost_preset_id
    ):
        """chronicle のプロンプトに FAREWELL_EMOTION_RUBRIC が含まれること。"""
        from backend.character_actions.farewell_detector import FAREWELL_EMOTION_RUBRIC

        captured_messages = []

        async def capture_ask(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return _make_chronicle_response()

        with patch("backend.batch.chronicle_job.ask_character", side_effect=capture_ask):
            asyncio.run(
                run_chronicle(
                    character_id=char_id_with_ghost,
                    sqlite=sqlite_store,
                    settings={},
                    chroma=None,
                )
            )

        assert len(captured_messages) == 1
        prompt_content = captured_messages[0]["content"]
        # ルーブリックの特徴的な文字列がプロンプトに含まれること
        assert "anger" in prompt_content
        assert "0.0" in prompt_content


# ─── run_pending_chronicles() — chroma 引数 ──────────────────────────────────


class TestRunPendingChroniclesChroma:
    """run_pending_chronicles() が chroma を受け取って run_chronicle に渡すことを検証する。"""

    def test_chroma_is_passed_to_run_chronicle(
        self, sqlite_store, char_id_with_ghost, ghost_preset_id
    ):
        """run_pending_chronicles() に渡した chroma が run_chronicle に伝達されること。"""
        mock_chroma = MagicMock()
        response = _make_chronicle_response()

        with patch("backend.batch.chronicle_job.ask_character", new=AsyncMock(return_value=response)):
            with patch("backend.batch.chronicle_job._check_estrangement", new=AsyncMock()) as mock_check:
                asyncio.run(
                    run_pending_chronicles(sqlite=sqlite_store, chroma=mock_chroma)
                )

        # _check_estrangement が少なくとも1回呼ばれていること
        # （chroma が run_chronicle に渡され、処理が実行されたことを確認）
        assert mock_check.called

    def test_chroma_none_does_not_raise(self, sqlite_store, char_id_with_ghost, ghost_preset_id):
        """run_pending_chronicles() に chroma=None を渡しても例外が発生しないこと。"""
        response = _make_chronicle_response()
        with patch("backend.batch.chronicle_job.ask_character", new=AsyncMock(return_value=response)):
            asyncio.run(
                run_pending_chronicles(sqlite=sqlite_store, chroma=None)
            )


# ─── DB スキーマ — farewell_config / relationship_status ────────────────────


class TestFarewellDBSchema:
    """farewell_config / relationship_status フィールドの DB 読み書きを検証する。"""

    def test_new_character_has_null_farewell_config(self, sqlite_store):
        """新規作成キャラクターの farewell_config デフォルトが None であること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="デフォルト別れキャラ")
        char = sqlite_store.get_character(cid)
        assert getattr(char, "farewell_config", None) is None

    def test_new_character_has_active_relationship_status(self, sqlite_store):
        """新規作成キャラクターの relationship_status デフォルトが "active" であること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="デフォルトステータスキャラ")
        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "active"

    def test_update_farewell_config_persists(self, sqlite_store, farewell_config_dict):
        """update_character() で farewell_config を保存・取得できること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="FarewellConfigキャラ")
        sqlite_store.update_character(cid, farewell_config=farewell_config_dict)
        char = sqlite_store.get_character(cid)
        fc = getattr(char, "farewell_config", None)
        assert fc is not None
        assert fc["thresholds"]["anger"] == 0.8
        assert fc["estrangement"]["negative_exit_threshold"] == 3

    def test_update_relationship_status_to_estranged(self, sqlite_store):
        """update_character() で relationship_status を "estranged" に変更できること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="EstrangedStateキャラ")
        sqlite_store.update_character(cid, relationship_status="estranged")
        char = sqlite_store.get_character(cid)
        assert getattr(char, "relationship_status", "active") == "estranged"

    def test_get_negative_exit_count_empty_returns_zero(self, sqlite_store):
        """退席セッションがない状態で get_negative_exit_count() が 0 を返すこと。"""
        count = sqlite_store.get_negative_exit_count(
            character_name="存在しないキャラ",
            since=datetime.now() - timedelta(days=30),
        )
        assert count == 0

    def test_get_negative_exit_count_counts_negative_exits(self, sqlite_store):
        """farewell_type="negative" のセッションのみカウントされること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="CountTestキャラ")

        # セッション1: negative 退席あり
        sess1_id = str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=sess1_id,
            model_id=f"CountTestキャラ@preset1",
        )
        sqlite_store.update_chat_session(
            sess1_id,
            exited_chars=[{"char_name": "CountTestキャラ", "reason": "怒り", "farewell_type": "negative"}],
        )

        # セッション2: neutral 退席（カウント対象外）
        sess2_id = str(uuid.uuid4())
        sqlite_store.create_chat_session(
            session_id=sess2_id,
            model_id=f"CountTestキャラ@preset1",
        )
        sqlite_store.update_chat_session(
            sess2_id,
            exited_chars=[{"char_name": "CountTestキャラ", "reason": "普通", "farewell_type": "neutral"}],
        )

        count = sqlite_store.get_negative_exit_count(
            character_name="CountTestキャラ",
            since=datetime.now() - timedelta(days=30),
        )
        assert count == 1
