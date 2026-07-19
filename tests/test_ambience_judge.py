"""backend.character_actions.ambience_judge モジュールのユニットテスト。

なりゆき judge（AmbienceJudge）の動作を検証する。

対象クラス・関数:
    _anonymize_conversation()     — 会話を UserA/UserB 形式に匿名化する
    _format_thresholds()          — 感情閾値 dict をテキスト化する
    _parse_judge_response()       — judge LLM の JSON レスポンスをパースする
    AmbienceJudge.detect()     — 感情スコアを判定して AmbienceReading を返す

テスト方針:
    - LLMプロバイダーは AsyncMock で差し替える（実際のAPI呼び出しなし）
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
    - スキップ条件（farewell_config 未設定 / preset 未発見 / LLM失敗）が
      None を返すことを確認する
    - should_exit=false / true それぞれの場合の AmbienceReading を検証する
    - 感情スコアルーブリック（EMOTION_RUBRIC）が定数として定義されていることを確認する
"""

import asyncio
import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.character_actions.ambience_judge import (
    EMOTION_RUBRIC,
    AmbienceJudge,
    AmbienceReading,
    _format_conversation,
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
        name="なりゆきテストキャラ",
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
def judge(sqlite_store):
    """テスト用 AmbienceJudge インスタンスを返すフィクスチャ。"""
    return AmbienceJudge(sqlite=sqlite_store)


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


# ─── EMOTION_RUBRIC ──────────────────────────────────────────────────


class TestFarewellEmotionRubric:
    """EMOTION_RUBRIC 定数の存在と内容を検証する。"""

    def test_rubric_is_non_empty_string(self):
        """EMOTION_RUBRIC が空でない文字列として定義されていること。"""
        assert isinstance(EMOTION_RUBRIC, str)
        assert len(EMOTION_RUBRIC) > 0

    def test_rubric_contains_all_emotion_names(self):
        """ルーブリックに4つの感情（anger/disgust/boredom/despair）が含まれること。"""
        for emotion in ("anger", "disgust", "boredom", "despair"):
            assert emotion in EMOTION_RUBRIC, f"感情 '{emotion}' がルーブリックに含まれていない"

    def test_rubric_contains_score_scale(self):
        """ルーブリックにスコアスケール（0.0〜1.0）の説明が含まれること。"""
        assert "0.0" in EMOTION_RUBRIC
        assert "1.0" in EMOTION_RUBRIC


# ─── _format_conversation ─────────────────────────────────────────────────────


class TestFormatConversation:
    """_format_conversation() の実名対話ログ変換を検証する。

    実名化（ambience Step 2）により、旧 UserA/UserB 匿名化は廃止された。
    キャラクター名とユーザ呼称がそのまま行頭ラベルになることを確認する。
    """

    def test_user_role_becomes_user_label(self):
        """user ロールがユーザ呼称のラベルに変換されること。"""
        messages = [{"role": "user", "content": "こんにちは"}]
        result = _format_conversation(messages, "はる", "もわ")
        assert "もわ: こんにちは" in result

    def test_assistant_role_becomes_character_name(self):
        """assistant ロールがキャラクター名のラベルに変換されること。"""
        messages = [{"role": "assistant", "content": "やあ"}]
        result = _format_conversation(messages, "はる", "もわ")
        assert "はる: やあ" in result

    def test_character_role_becomes_character_name(self):
        """character ロールもキャラクター名に変換されること（グループチャット互換）。"""
        messages = [{"role": "character", "content": "そうですね"}]
        result = _format_conversation(messages, "はる", "もわ")
        assert "はる: そうですね" in result

    def test_system_role_is_excluded(self):
        """system ロールは出力に含まれないこと。"""
        messages = [
            {"role": "system", "content": "システム設定"},
            {"role": "user", "content": "ユーザ発言"},
        ]
        result = _format_conversation(messages, "はる", "もわ")
        assert "システム設定" not in result
        assert "もわ: ユーザ発言" in result

    def test_empty_messages_returns_empty_string(self):
        """空リストを渡すと空文字列が返ること。"""
        assert _format_conversation([], "はる", "もわ") == ""

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
        result = _format_conversation(messages, "はる", "もわ")
        assert "テキスト部分" in result
        assert "image_url" not in result

    def test_empty_content_is_skipped(self):
        """content が空の場合はそのメッセージが出力に含まれないこと。"""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "応答"},
        ]
        result = _format_conversation(messages, "はる", "もわ")
        lines = result.strip().split("\n")
        assert len(lines) == 1
        assert "はる: 応答" in result

    def test_multiple_turns_preserve_order(self):
        """複数ターンが元の順序を保って出力されること。"""
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = _format_conversation(messages, "はる", "もわ")
        lines = result.split("\n")
        assert lines[0].startswith("もわ:")
        assert lines[1].startswith("はる:")
        assert lines[2].startswith("もわ:")


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


# ─── AmbienceJudge.detect() — スキップ条件 ────────────────────────────────


class TestAmbienceJudgeSkip:
    """detect() のスキップ条件（None を返す場合）を検証する。"""

    def test_none_farewell_config_returns_none(self, judge, char_id, judge_preset_id, sample_messages):
        """farewell_config が None の場合は None を返すこと。"""
        result = asyncio.run(
            judge.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id=judge_preset_id,
                farewell_config=None,
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_empty_thresholds_returns_none(self, judge, char_id, judge_preset_id, sample_messages):
        """farewell_config.thresholds が空の場合は None を返すこと。"""
        result = asyncio.run(
            judge.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id=judge_preset_id,
                farewell_config={"thresholds": {}},
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_empty_preset_id_returns_none(self, judge, char_id, farewell_config, sample_messages):
        """preset_id が空文字の場合は None を返すこと。"""
        result = asyncio.run(
            judge.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id="",
                farewell_config=farewell_config,
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_empty_messages_returns_none(self, judge, char_id, judge_preset_id, farewell_config):
        """messages が空の場合は None を返すこと。"""
        result = asyncio.run(
            judge.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id=judge_preset_id,
                farewell_config=farewell_config,
                messages=[],
                settings={},
            )
        )
        assert result is None

    def test_character_not_found_returns_none(self, judge, judge_preset_id, farewell_config, sample_messages):
        """存在しないキャラクターIDを渡すと None を返すこと（例外なし）。"""
        result = asyncio.run(
            judge.detect(
                character_id="nonexistent-char",
                session_id="sess-1",
                preset_id=judge_preset_id,
                farewell_config=farewell_config,
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_preset_not_found_returns_none(self, judge, char_id, farewell_config, sample_messages):
        """存在しないプリセットIDを渡すと None を返すこと（例外なし）。"""
        result = asyncio.run(
            judge.detect(
                character_id=char_id,
                session_id="sess-1",
                preset_id="nonexistent-preset",
                farewell_config=farewell_config,
                messages=sample_messages,
                settings={},
            )
        )
        assert result is None

    def test_provider_error_returns_none(self, judge, char_id, judge_preset_id, farewell_config, sample_messages):
        """プロバイダー生成が失敗した場合は None を返すこと（例外なし）。"""
        with patch("backend.character_actions.ambience_judge.create_provider",
                   side_effect=RuntimeError("provider error")):
            result = asyncio.run(
                judge.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert result is None

    def test_llm_call_failure_returns_none(self, judge, char_id, judge_preset_id, farewell_config, sample_messages):
        """judge LLM 呼び出しが例外を投げた場合は None を返すこと（例外なし）。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=ConnectionError("LLM接続失敗"))
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            result = asyncio.run(
                judge.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert result is None

    def test_invalid_json_response_returns_none(self, judge, char_id, judge_preset_id, farewell_config, sample_messages):
        """judge LLM が不正な JSON を返した場合は None を返すこと（例外なし）。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="これはJSONではありません")
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            result = asyncio.run(
                judge.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert result is None


# ─── AmbienceJudge.detect() — 正常系 ──────────────────────────────────────


class TestAmbienceJudgeResult:
    """detect() が正常な AmbienceReading を返す場合を検証する。"""

    def _run_detect_with_response(
        self, judge, char_id, judge_preset_id, farewell_config, sample_messages, response_text
    ):
        """モックプロバイダーで detect() を実行するヘルパー。

        Args:
            judge: テスト対象 AmbienceJudge。
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
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            return asyncio.run(
                judge.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )

    def test_should_exit_false_returns_farewell_result(
        self, judge, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """should_exit=false の場合、should_exit=False の AmbienceReading が返ること。"""
        response = _make_judge_response(should_exit=False)
        result = self._run_detect_with_response(
            judge, char_id, judge_preset_id, farewell_config, sample_messages, response
        )
        assert isinstance(result, AmbienceReading)
        assert result.should_exit is False

    def test_should_exit_true_negative_returns_correct_type(
        self, judge, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """should_exit=true, farewell_type="negative" の場合、正しい AmbienceReading が返ること。"""
        emotions = {"anger": 0.9, "disgust": 0.5, "boredom": 0.2, "despair": 0.3}
        response = _make_judge_response(should_exit=True, farewell_type="negative", emotions=emotions)
        result = self._run_detect_with_response(
            judge, char_id, judge_preset_id, farewell_config, sample_messages, response
        )
        assert isinstance(result, AmbienceReading)
        assert result.should_exit is True
        assert result.farewell_type == "negative"

    def test_farewell_message_is_taken_from_config(
        self, judge, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """退席メッセージが farewell_config.farewell_message から取得されること。"""
        response = _make_judge_response(should_exit=True, farewell_type="negative")
        result = self._run_detect_with_response(
            judge, char_id, judge_preset_id, farewell_config, sample_messages, response
        )
        assert result is not None
        assert result.reason == farewell_config["farewell_message"]["negative"]

    def test_emotion_scores_are_parsed_correctly(
        self, judge, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """感情スコアが正しくパースされること。"""
        emotions = {"anger": 0.85, "disgust": 0.4, "boredom": 0.1, "despair": 0.6}
        response = _make_judge_response(should_exit=True, farewell_type="negative", emotions=emotions)
        result = self._run_detect_with_response(
            judge, char_id, judge_preset_id, farewell_config, sample_messages, response
        )
        assert result is not None
        assert abs(result.emotions["anger"] - 0.85) < 0.001

    def test_judge_prompt_contains_character_context(
        self, judge, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """judge LLM に渡すプロンプトにキャラクター設定が含まれること。"""
        captured_calls = []
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, messages, **kwargs):
            captured_calls.append({"system": system_prompt, "messages": messages})
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            asyncio.run(
                judge.detect(
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
        self, judge, char_id, judge_preset_id, farewell_config, sample_messages
    ):
        """judge LLM に渡すプロンプトに感情スコアルーブリックが含まれること。"""
        captured_calls = []
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, messages, **kwargs):
            captured_calls.append({"messages": messages})
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            asyncio.run(
                judge.detect(
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
        self, judge, char_id, judge_preset_id, farewell_config, sample_messages
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
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            asyncio.run(
                judge.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=sample_messages,
                    settings={},
                )
            )
        assert "テスト用キャラクター設定" not in captured.get("system", "")

    def test_conversation_uses_real_names_in_prompt(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """会話が実名（キャラクター名／ユーザ呼称）でプロンプトに含まれること。

        実名化（ambience Step 2）の検証。ユーザ呼称は Settings.user_name から
        解決され、旧 UserA/UserB 匿名化ラベルが出力に現れないことを確認する。
        """
        messages = [
            {"role": "user", "content": "ユニークなユーザー発言12345"},
            {"role": "assistant", "content": "ユニークなキャラ応答67890"},
        ]
        captured_calls = []
        captured_system = {}
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, msgs, **kwargs):
            captured_system["system"] = system_prompt
            captured_calls.append(msgs)
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            asyncio.run(
                judge.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=messages,
                    settings={"user_name": "もわ"},
                )
            )
        user_content = captured_calls[0][0]["content"]
        # 実名ラベルで会話が並び、匿名化ラベルは使われないこと
        assert "なりゆきテストキャラ: ユニークなキャラ応答67890" in user_content
        assert "もわ: ユニークなユーザー発言12345" in user_content
        assert "UserA" not in user_content
        assert "UserB" not in user_content
        # システムプロンプトにも両者の実名が入ること
        assert "なりゆきテストキャラ" in captured_system["system"]
        assert "もわ" in captured_system["system"]

    def test_user_label_falls_back_when_unset(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """user_label も Settings.user_name も空の場合、「相手」に縮退すること。"""
        messages = [{"role": "user", "content": "呼称なしの発言"}]
        captured_calls = []
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, msgs, **kwargs):
            captured_calls.append(msgs)
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            asyncio.run(
                judge.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=messages,
                    settings={},
                )
            )
        user_content = captured_calls[0][0]["content"]
        assert "相手: 呼称なしの発言" in user_content

    def test_user_label_prefers_character_setting(
        self, judge, sqlite_store, judge_preset_id, farewell_config
    ):
        """characters.user_label が設定済みなら Settings.user_name より優先されること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(
            character_id=cid,
            name="呼称優先テストキャラ",
            system_prompt_block1="テスト用",
            user_label="せんぱい",
        )
        messages = [{"role": "user", "content": "呼称優先の発言"}]
        captured_calls = []
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, msgs, **kwargs):
            captured_calls.append(msgs)
            return _make_judge_response(should_exit=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            asyncio.run(
                judge.detect(
                    character_id=cid,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=messages,
                    settings={"user_name": "もわ"},
                )
            )
        user_content = captured_calls[0][0]["content"]
        assert "せんぱい: 呼称優先の発言" in user_content


# ─── AmbienceJudge.detect() — 場所判定（location_label / ambience Step 4） ──


class TestLocationLabelJudgement:
    """対面モード時の場所判定（location_label）を検証するテストクラス。

    検証する観点:
        - 対面 + 候補ありのときだけプロンプトに「対面の場所判定」ブロックが入る
        - 非対面／候補なし（空配列・空ラベルのみ）ではブロックが入らず location_label=None
        - judge が候補内のラベルを返せば AmbienceReading.location_label に載る
        - 候補外のラベル・null は前回ラベル（prev_bg_label）へ丸められる
        - 前回もラベルなし + 判定不能なら None（背景なし）
        - プロンプトに候補配列と前回ラベルが注入される
    """

    def _run_detect(
        self, judge, char_id, judge_preset_id, farewell_config,
        response_payload: dict, *, face_to_face=True,
        candidates=None, prev_label=None, captured=None,
    ):
        """場所判定パラメータ付きで detect() を実行するヘルパー。

        Args:
            judge: テスト対象 AmbienceJudge。
            char_id: キャラクターID。
            judge_preset_id: プリセットID。
            farewell_config: farewell_config 辞書。
            response_payload: judge LLM が返す JSON dict。
            face_to_face: 対面モードフラグ。
            candidates: 候補ラベル配列。
            prev_label: 前回判定ラベル。
            captured: 渡すと {"system", "user"} にプロンプトを格納する。

        Returns:
            detect() の返却値。
        """
        mock_provider = MagicMock()

        async def capture_generate(system_prompt, msgs, **kwargs):
            if captured is not None:
                captured["system"] = system_prompt
                captured["user"] = msgs[0]["content"]
            return json.dumps(response_payload, ensure_ascii=False)

        mock_provider.generate = capture_generate
        with patch("backend.character_actions.ambience_judge.create_provider", return_value=mock_provider):
            return asyncio.run(
                judge.detect(
                    character_id=char_id,
                    session_id="sess-1",
                    preset_id=judge_preset_id,
                    farewell_config=farewell_config,
                    messages=[{"role": "user", "content": "部屋に入るね"}],
                    settings={},
                    face_to_face=face_to_face,
                    bg_label_candidates=candidates,
                    prev_bg_label=prev_label,
                )
            )

    def _payload(self, location_label=None) -> dict:
        """location_label 付きの judge 応答 JSON dict を作るヘルパー。"""
        return {
            "emotions": {"anger": 0.0, "disgust": 0.0, "boredom": 0.0, "despair": 0.0},
            "engagement": 0.5,
            "should_exit": False,
            "farewell_type": None,
            "location_label": location_label,
        }

    def test_valid_candidate_is_returned(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """judge が候補内のラベルを返した場合、そのまま location_label に載ること。"""
        result = self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload("はるの部屋"),
            candidates=["はるの部屋", "もわの部屋"],
        )
        assert result.location_label == "はるの部屋"

    def test_out_of_candidates_falls_back_to_prev(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """候補外のラベルは前回ラベルへ丸められること。"""
        result = self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload("知らない場所"),
            candidates=["はるの部屋", "もわの部屋"],
            prev_label="もわの部屋",
        )
        assert result.location_label == "もわの部屋"

    def test_null_label_falls_back_to_prev(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """judge が null を返した場合も前回ラベル踏襲になること。"""
        result = self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload(None),
            candidates=["はるの部屋"],
            prev_label="はるの部屋",
        )
        assert result.location_label == "はるの部屋"

    def test_no_prev_and_no_match_returns_none(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """前回もラベルなし + 候補外なら None（背景なし）になること。"""
        result = self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload("知らない場所"),
            candidates=["はるの部屋"],
            prev_label=None,
        )
        assert result.location_label is None

    def test_not_face_to_face_skips_judgement(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """非対面では候補があっても場所判定せず None、プロンプトにもブロックが入らないこと。"""
        captured = {}
        result = self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload("はるの部屋"),
            face_to_face=False,
            candidates=["はるの部屋"],
            captured=captured,
        )
        assert result.location_label is None
        assert "## 対面の場所判定" not in captured["user"]

    def test_no_candidates_skips_judgement(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """候補が空（未登録）なら対面中でも場所判定せず None 固定になること。"""
        captured = {}
        result = self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload("はるの部屋"),
            candidates=[],
            captured=captured,
        )
        assert result.location_label is None
        assert "## 対面の場所判定" not in captured["user"]

    def test_empty_labels_are_excluded_from_candidates(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """空ラベル（旧単数画像の移行直後など）は候補から除外されること。

        候補が空ラベルのみなら場所判定自体がスキップされる。
        """
        captured = {}
        result = self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload(""),
            candidates=[""],
            captured=captured,
        )
        assert result.location_label is None
        assert "## 対面の場所判定" not in captured["user"]

    def test_prompt_contains_candidates_and_prev_label(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """対面時のプロンプトに候補配列・前回ラベル・踏襲ルールが注入されること。"""
        captured = {}
        self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload("はるの部屋"),
            candidates=["はるの部屋", "もわの部屋"],
            prev_label="もわの部屋",
            captured=captured,
        )
        user_content = captured["user"]
        assert "## 対面の場所判定" in user_content
        assert "はるの部屋" in user_content
        assert "もわの部屋" in user_content
        assert "前回の判定" in user_content
        assert "明確に変わったと判断できないなら" in user_content

    def test_prev_label_none_shows_nashi(
        self, judge, char_id, judge_preset_id, farewell_config
    ):
        """前回ラベルが無い場合、プロンプトの前回判定が「なし」になること。"""
        captured = {}
        self._run_detect(
            judge, char_id, judge_preset_id, farewell_config,
            self._payload("はるの部屋"),
            candidates=["はるの部屋"],
            prev_label=None,
            captured=captured,
        )
        assert "前回の判定: なし" in captured["user"]


# ─── AmbienceJudge のスキーマ検証 ─────────────────────────────────────────


class TestAmbienceReadingSchema:
    """AmbienceReading データクラスのフィールドを検証する。"""

    def test_farewell_result_has_required_fields(self):
        """AmbienceReading が should_exit / farewell_type / emotions / reason を持つこと。"""
        result = AmbienceReading(
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
        """should_exit=False のときの AmbienceReading が正しく構築されること。"""
        result = AmbienceReading(
            should_exit=False,
            farewell_type="neutral",
            emotions={},
            reason="",
        )
        assert result.should_exit is False
        assert result.reason == ""
