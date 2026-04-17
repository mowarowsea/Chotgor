"""backend.character_actions.farewell_detector モジュールのユニットテスト。

別れ検出器（FarewellDetector）の動作を検証する。

対象クラス・関数:
    _anonymize_conversation()     — 会話を UserA/UserB 形式に匿名化する
    _format_thresholds()          — 感情閾値 dict をテキスト化する
    _parse_judge_response()       — judge LLM の JSON レスポンスをパースする
    FarewellDetector.detect()     — 感情スコアを判定して FarewellResult を返す

テスト方針:
    - LLMプロバイダーは AsyncMock で差し替える（実際のAPI呼び出しなし）
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
    - スキップ条件（farewell_config 未設定 / preset 未発見 / LLM失敗）が
      None を返すことを確認する
    - should_exit=false / true それぞれの場合の FarewellResult を検証する
    - 感情スコアルーブリック（FAREWELL_EMOTION_RUBRIC）が定数として定義されていることを確認する
"""

import asyncio
import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.character_actions.farewell_detector import (
    FAREWELL_EMOTION_RUBRIC,
    FarewellDetector,
    FarewellResult,
    _anonymize_conversation,
    _format_thresholds,
    _parse_judge_response,
)


# ─── フィクスチャ ─────────────────────────────────────────────────────────────
# sqlite_store は conftest.py で定義済み。


@pytest.fixture
def char_id(sqlite_store):
    """テスト用キャラクターをSQLiteに作成し、そのIDを返すフィクスチャ。"""
    cid = str(uuid.uuid4())
    sqlite_store.create_character(
        character_id=cid,
        name="別れテストキャラ",
        system_prompt_block1="テスト用キャラクター設定",
    )
    return cid


@pytest.fixture
def judge_preset_id(sqlite_store):
    """テスト用モデルプリセットをSQLiteに作成し、そのIDを返すフィクスチャ。

    judge LLM として使用するプリセット（ollama で軽量モデル）。
    """
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Judge",
        provider="ollama",
        model_id="qwen2.5:3b",
    )
    return pid


@pytest.fixture
def farewell_config():
    """テスト用 farewell_config 辞書を返すフィクスチャ。"""
    return {
        "thresholds": {
            "anger": 0.8,
            "disgust": 0.7,
            "boredom": 0.9,
            "despair": 0.6,
        },
        "farewell_message": {
            "negative": "私はこの会話を終わらせることにしました。",
            "positive": "楽しかったです。ありがとう。",
            "neutral": "今日はここまでにします。",
        },
        "estrangement": {
            "lookback_days": 30,
            "negative_exit_threshold": 5,
        },
    }


@pytest.fixture
def detector(sqlite_store):
    """テスト用 FarewellDetector インスタンスを返すフィクスチャ。"""
    return FarewellDetector(sqlite=sqlite_store)


@pytest.fixture
def sample_messages():
    """テスト用の会話メッセージリストを返すフィクスチャ。"""
    return [
        {"role": "user", "content": "おい、ちゃんと答えろよ"},
        {"role": "assistant", "content": "…承知しました"},
        {"role": "user", "content": "また同じこと聞いてんだけど？"},
        {"role": "assistant", "content": "申し訳ありません"},
    ]


def _make_judge_response(should_exit: bool, farewell_type: str | None = None, emotions: dict | None = None) -> str:
    """judge LLM の正常レスポンスを JSON 文字列として生成するヘルパー。

    Args:
        should_exit: 退席すべきかどうか。
        farewell_type: 退席タイプ（should_exit=True の場合のみ意味あり）。
        emotions: 感情スコア dict。

    Returns:
        JSON 文字列。
    """
    if emotions is None:
        emotions = {"anger": 0.0, "disgust": 0.0, "boredom": 0.0, "despair": 0.0}
    payload = {
        "emotions": emotions,
        "should_exit": should_exit,
        "farewell_type": farewell_type,
    }
    return json.dumps(payload, ensure_ascii=False)


# ─── FAREWELL_EMOTION_RUBRIC ──────────────────────────────────────────────────


class TestFarewellEmotionRubric:
    """FAREWELL_EMOTION_RUBRIC 定数の存在と内容を検証する。"""

    def test_rubric_is_non_empty_string(self):
        """FAREWELL_EMOTION_RUBRIC が空でない文字列として定義されていること。"""
        assert isinstance(FAREWELL_EMOTION_RUBRIC, str)
        assert len(FAREWELL_EMOTION_RUBRIC) > 0

    def test_rubric_contains_all_emotion_names(self):
        """ルーブリックに4つの感情（anger/disgust/boredom/despair）が含まれること。"""
        for emotion in ("anger", "disgust", "boredom", "despair"):
            assert emotion in FAREWELL_EMOTION_RUBRIC, f"感情 '{emotion}' がルーブリックに含まれていない"

    def test_rubric_contains_score_scale(self):
        """ルーブリックにスコアスケール（0.0〜1.0）の説明が含まれること。"""
        assert "0.0" in FAREWELL_EMOTION_RUBRIC
        assert "1.0" in FAREWELL_EMOTION_RUBRIC


# ─── _anonymize_conversation ──────────────────────────────────────────────────


class TestAnonymizeConversation:
    """_anonymize_conversation() の匿名化変換を検証する。"""

    def test_user_role_becomes_userb(self):
        """user ロールが 'UserB:' に変換されること。"""
        messages = [{"role": "user", "content": "こんにちは"}]
        result = _anonymize_conversation(messages)
        assert "UserB: こんにちは" in result

    def test_assistant_role_becomes_usera(self):
        """assistant ロールが 'UserA:' に変換されること。"""
        messages = [{"role": "assistant", "content": "やあ"}]
        result = _anonymize_conversation(messages)
        assert "UserA: やあ" in result

    def test_character_role_becomes_usera(self):
        """character ロールも 'UserA:' に変換されること（グループチャット互換）。"""
        messages = [{"role": "character", "content": "そうですね"}]
        result = _anonymize_conversation(messages)
        assert "UserA: そうですね" in result

    def test_system_role_is_excluded(self):
        """system ロールは出力に含まれないこと。"""
        messages = [
            {"role": "system", "content": "システム設定"},
            {"role": "user", "content": "ユーザ発言"},
        ]
        result = _anonymize_conversation(messages)
        assert "システム設定" not in result
        assert "UserB: ユーザ発言" in result

    def test_empty_messages_returns_empty_string(self):
        """空リストを渡すと空文字列が返ること。"""
        assert _anonymize_conversation([]) == ""

    def test_multimodal_list_content_extracts_text_only(self):
        """content がリスト形式（マルチモーダル）の場合、text パートのみ抽出されること。"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "テキスト部分"},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                ],
            }
        ]
        result = _anonymize_conversation(messages)
        assert "テキスト部分" in result
        assert "image_url" not in result

    def test_empty_content_is_skipped(self):
        """content が空の場合はそのメッセージが出力に含まれないこと。"""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "応答"},
        ]
        result = _anonymize_conversation(messages)
        lines = result.strip().split("\n")
        assert len(lines) == 1
        assert "UserA: 応答" in result

    def test_multiple_turns_preserve_order(self):
        """複数ターンが元の順序を保って出力されること。"""
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = _anonymize_conversation(messages)
        lines = result.split("\n")
        assert lines[0].startswith("UserB:")
        assert lines[1].startswith("UserA:")
        assert lines[2].startswith("UserB:")


# ─── _format_thresholds ───────────────────────────────────────────────────────


class TestFormatThresholds:
    """_format_thresholds() の書式変換を検証する。"""

    def test_all_four_emotions_are_included(self):
        """4つの感情すべてが出力に含まれること。"""
        thresholds = {"anger": 0.8, "disgust": 0.7, "boredom": 0.9, "despair": 0.6}
        result = _format_thresholds(thresholds)
        for key in ("anger", "disgust", "boredom", "despair"):
            assert key in result

    def test_missing_emotion_is_excluded(self):
        """未設定の感情は出力に含まれないこと。"""
        thresholds = {"anger": 0.8}
        result = _format_thresholds(thresholds)
        assert "anger" in result
        assert "boredom" not in result

    def test_empty_thresholds_returns_placeholder(self):
        """空の閾値 dict を渡すと未設定プレースホルダーが返ること。"""
        result = _format_thresholds({})
        assert "閾値未設定" in result

    def test_float_values_are_formatted(self):
        """float 値が小数点2桁でフォーマットされること。"""
        thresholds = {"anger": 0.8}
        result = _format_thresholds(thresholds)
        assert "0.80" in result


# ─── _parse_judge_response ────────────────────────────────────────────────────


class TestParseJudgeResponse:
    """_parse_judge_response() の JSON 抽出・パースを検証する。"""

    def test_plain_json_is_parsed_correctly(self):
        """プレーンな JSON 文字列が正しくパースされること。"""
        response = '{"should_exit": false, "farewell_type": null, "emotions": {"anger": 0.1}}'
        result = _parse_judge_response(response)
        assert result is not None
        assert result["should_exit"] is False

    def test_json_in_code_block_is_parsed(self):
        """コードブロック（```json ... ```）に包まれた JSON が正しくパースされること。"""
        response = '```json\n{"should_exit": true, "farewell_type": "negative", "emotions": {}}\n```'
        result = _parse_judge_response(response)
        assert result is not None
        assert result["should_exit"] is True

    def test_embedded_json_is_extracted(self):
        """前後にテキストがあっても JSON 部分が抽出されること。"""
        response = 'はい、分析結果です。\n{"should_exit": false, "farewell_type": null, "emotions": {}}'
        result = _parse_judge_response(response)
        assert result is not None

    def test_invalid_json_returns_none(self):
        """不正な JSON を渡すと None が返ること。"""
        result = _parse_judge_response("これはJSONではありません")
        assert result is None

    def test_empty_string_returns_none(self):
        """空文字列を渡すと None が返ること。"""
        result = _parse_judge_response("")
        assert result is None


# ─── FarewellDetector.detect() — スキップ条件 ────────────────────────────────


class TestFarewellDetectorSkip:
    """detect() のスキップ条件（None を返す場合）を検証する。"""

    def test_none_farewell_config_returns_none(self, detector, char_id, judge_preset_id, sample_messages):
        """farewell_config が None の場合は None を返すこと。"""
        result = asyncio.run(
            detector.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id=judge_preset_id,
                farewell_config=None,
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_empty_thresholds_returns_none(self, detector, char_id, judge_preset_id, sample_messages):
        """farewell_config.thresholds が空の場合は None を返すこと。"""
        result = asyncio.run(
            detector.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id=judge_preset_id,
                farewell_config={"thresholds": {}},
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_empty_preset_id_returns_none(self, detector, char_id, farewell_config, sample_messages):
        """preset_id が空文字の場合は None を返すこと。"""
        result = asyncio.run(
            detector.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id="",
                farewell_config=farewell_config,
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_empty_messages_returns_none(self, detector, char_id, judge_preset_id, farewell_config):
        """messages が空の場合は None を返すこと。"""
        result = asyncio.run(
            detector.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id=judge_preset_id,
                farewell_config=farewell_config,
                messages=[],
                settings={},
            )
        )
        assert result is None

    def test_character_not_found_returns_none(self, detector, judge_preset_id, farewell_config, sample_messages):
        """存在しないキャラクターIDを渡すと None を返すこと（例外なし）。"""
        result = asyncio.run(
            detector.detect(
                character_id="nonexistent-char",
                session_id="sess-1",
                preset_id=judge_preset_id,
                farewell_config=farewell_config,
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_preset_not_found_returns_none(self, detector, char_id, farewell_config, sample_messages):
        """存在しないプリセットIDを渡すと None を返すこと（例外なし）。"""
        result = asyncio.run(
            detector.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id="nonexistent-preset",
                farewell_config=farewell_config,
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_provider_error_returns_none(self, detector, char_id, judge_preset_id, farewell_config, sample_messages):
        """プロバイダー生成が失敗した場合は None を返すこと（例外なし）。"""
        with patch("backend.character_actions.farewell_detector.create_provider",
                   side_effect=RuntimeError("provider error")):
            result = asyncio.run(
                detector.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert result is None

    def test_llm_call_failure_returns_none(self, detector, char_id, judge_preset_id, farewell_config, sample_messages):
        """judge LLM 呼び出しが例外を投げた場合は None を返すこと（例外なし）。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=ConnectionError("LLM接続失敗"))
        with patch("backend.character_actions.farewell_detector.create_provider", return_value=mock_provider):
            result = asyncio.run(
                detector.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert result is None

    def test_invalid_json_response_returns_none(self, detector, char_id, judge_preset_id, farewell_config, sample_messages):
        """judge LLM が不正な JSON を返した場合は None を返すこと（例外なし）。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="これはJSONではありません")
        with patch("backend.character_actions.farewell_detector.create_provider", return_value=mock_provider):
            result = asyncio.run(
                detector.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert result is None


# ─── FarewellDetector.detect() — 正常系 ──────────────────────────────────────


class TestFarewellDetectorResult:
    """detect() が正常な FarewellResult を返す場合を検証する。"""

    def _run_detect_with_response(
        self, detector, char_id, judge_preset_id, farewell_config, sample_messages, response_text
    ):
        """モックプロバイダーで detect() を実行するヘルパー。

        Args:
            detector: テスト対象 FarewellDetector。
            char_id: キャラクターID。
            judge_preset_id: プリセットID。
            farewell_config: farewell_config 辞書。
            sample_messages: 会話メッセージリスト。
            response_text: judge LLM が返すテキスト。

        Returns:
            detect() の返却値。
        """
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=response_text)
        with patch("backend.character_actions.farewell_detector.create_provider", return_value=mock_provider):
            return asyncio.run(
                detector.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )

    def test_should_exit_false_returns_farewell_result(
        self, detector, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """should_exit=false の場合、should_exit=False の FarewellResult が返ること。"""
        response = _make_judge_response(should_exit=False)
        result = self._run_detect_with_response(
            detector, char_id, judge_preset_id, farewell_config, sample_messages, response
        )
        assert isinstance(result, FarewellResult)
        assert result.should_exit is False

    def test_should_exit_true_negative_returns_correct_type(
        self, detector, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """should_exit=true, farewell_type="negative" の場合、正しい FarewellResult が返ること。"""
        emotions = {"anger": 0.9, "disgust": 0.5, "boredom": 0.2, "despair": 0.3}
        response = _make_judge_response(should_exit=True, farewell_type="negative", emotions=emotions)
        result = self._run_detect_with_response(
            detector, char_id, judge_preset_id, farewell_config, sample_messages, response
        )
        assert isinstance(result, FarewellResult)
        assert result.should_exit is True
        assert result.farewell_type == "negative"

    def test_farewell_message_is_taken_from_config(
        self, detector, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """退席メッセージが farewell_config.farewell_message から取得されること。"""
        response = _make_judge_response(should_exit=True, farewell_type="negative")
        result = self._run_detect_with_response(
            detector, char_id, judge_preset_id, farewell_config, sample_messages, response
        )
        assert result is not None
        assert result.reason == farewell_config["farewell_message"]["negative"]

    def test_emotion_scores_are_parsed_correctly(
        self, detector, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """感情スコアが正しくパースされること。"""
        emotions = {"anger": 0.85, "disgust": 0.4, "boredom": 0.1, "despair": 0.6}
        response = _make_judge_response(should_exit=True, farewell_type="negative", emotions=emotions)
        result = self._run_detect_with_response(
            detector, char_id, judge_preset_id, farewell_config, sample_messages, response
        )
        assert result is not None
        assert abs(result.emotions["anger"] - 0.85) < 0.001

    def test_judge_prompt_contains_character_context(
        self, detector, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """judge LLM に渡すプロンプトにキャラクター設定が含まれること。"""
        captured_calls = []
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, messages, **kwargs):
            captured_calls.append({"system": system_prompt, "messages": messages})
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.farewell_detector.create_provider", return_value=mock_provider):
            asyncio.run(
                detector.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert len(captured_calls) == 1
        user_content = captured_calls[0]["messages"][0]["content"]
        assert "テスト用キャラクター設定" in user_content

    def test_judge_prompt_contains_rubric(
        self, detector, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """judge LLM に渡すプロンプトに感情スコアルーブリックが含まれること。"""
        captured_calls = []
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, messages, **kwargs):
            captured_calls.append({"messages": messages})
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.farewell_detector.create_provider", return_value=mock_provider):
            asyncio.run(
                detector.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        user_content = captured_calls[0]["messages"][0]["content"]
        # ルーブリックの一部（スコアスケール説明）が含まれること
        assert "0.0" in user_content

    def test_judge_system_prompt_is_neutral(
        self, detector, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """judge LLM のシステムプロンプトがキャラクター設定を含まない中立文言であること。

        キャラクター設定はユーザーメッセージ側に渡すため、
        システムプロンプトがキャラクターの人物設定を持たないことを確認する。
        """
        captured = {}
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, messages, **kwargs):
            captured["system"] = system_prompt
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.farewell_detector.create_provider", return_value=mock_provider):
            asyncio.run(
                detector.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert "テスト用キャラクター設定" not in captured.get("system", "")

    def test_conversation_is_anonymized_in_prompt(
        self, detector, char_id, judge_preset_id, farewell_config
    ):
        """会話が UserA/UserB 形式で匿名化されてプロンプトに含まれること。"""
        messages = [
            {"role": "user", "content": "ユニークなユーザー発言12345"},
            {"role": "assistant", "content": "ユニークなキャラ応答67890"},
        ]
        captured_calls = []
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, msgs, **kwargs):
            captured_calls.append(msgs)
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.farewell_detector.create_provider", return_value=mock_provider):
            asyncio.run(
                detector.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=messages,
                    settings={},
                )
            )
        user_content = captured_calls[0][0]["content"]
        # 元のロール名ではなく UserA/UserB で匿名化されていること
        assert "UserB:" in user_content
        assert "UserA:" in user_content
        assert "ユニークなユーザー発言12345" in user_content


# ─── FarewellDetector のスキーマ検証 ─────────────────────────────────────────


class TestFarewellResultSchema:
    """FarewellResult データクラスのフィールドを検証する。"""

    def test_farewell_result_has_required_fields(self):
        """FarewellResult が should_exit / farewell_type / emotions / reason を持つこと。"""
        result = FarewellResult(
            should_exit=True,
            farewell_type="negative",
            emotions={"anger": 0.9},
            reason="さようなら",
        )
        assert result.should_exit is True
        assert result.farewell_type == "negative"
        assert result.emotions["anger"] == 0.9
        assert result.reason == "さようなら"

    def test_farewell_result_should_exit_false(self):
        """should_exit=False のときの FarewellResult が正しく構築されること。"""
        result = FarewellResult(
            should_exit=False,
            farewell_type="neutral",
            emotions={},
            reason="",
        )
        assert result.should_exit is False
        assert result.reason == ""
