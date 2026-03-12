"""
グループチャット司会AI (director.py) のユニットテスト。

テスト対象:
  - _parse_director_response: 応答テキストから次発言者リストを解析する
  - _build_director_messages: 司会AIへ渡すシステムプロンプトとユーザーメッセージを構築する
  - decide_next_speakers: 非同期でキャラクタープロバイダーを呼び出して発言者リストを返す

LLMへの実際のネットワーク呼び出しはすべてモック化する。
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.group_chat.director import (
    _build_director_messages,
    _parse_director_response,
    decide_next_speakers,
)


# ---------------------------------------------------------------------------
# _parse_director_response のテスト
# ---------------------------------------------------------------------------

class TestParseDirectorResponse:
    """_parse_director_response の解析ロジックを網羅するテストスイート。

    司会AIは [キャラ名] / [A, B] / [] の形式で返すことが仕様だが、
    前後に説明文が混入する場合もある。すべての形式を正しく解析できることを検証する。
    """

    PARTICIPANTS = ["はる", "Chotgor君", "サブキャラ"]

    def test_single_character(self):
        """単一キャラクター指定 [はる] を正しく解析できること。"""
        result = _parse_director_response("[はる]", self.PARTICIPANTS)
        assert result == ["はる"]

    def test_multiple_characters(self):
        """複数キャラクター指定 [はる, Chotgor君] を正しく解析できること。"""
        result = _parse_director_response("[はる, Chotgor君]", self.PARTICIPANTS)
        assert result == ["はる", "Chotgor君"]

    def test_empty_bracket_returns_user_turn(self):
        """[] はユーザーターンを意味するため、空リストを返すこと。"""
        result = _parse_director_response("[]", self.PARTICIPANTS)
        assert result == []

    def test_unknown_character_is_filtered(self):
        """参加者リストにいないキャラクター名はフィルタリングされること。"""
        result = _parse_director_response("[存在しないキャラ]", self.PARTICIPANTS)
        assert result == []

    def test_mixed_valid_and_invalid_characters(self):
        """有効なキャラと存在しないキャラが混在する場合、有効なキャラのみ返すこと。"""
        result = _parse_director_response("[はる, 幻のキャラ]", self.PARTICIPANTS)
        assert result == ["はる"]

    def test_extra_text_before_bracket(self):
        """[]の前に余計なテキストがあっても、最初の [] を正しく解析できること。"""
        result = _parse_director_response("次は[Chotgor君]が発言するべきです。", self.PARTICIPANTS)
        assert result == ["Chotgor君"]

    def test_extra_spaces_inside_bracket(self):
        """[] 内のスペースはトリムして解析できること。"""
        result = _parse_director_response("[ はる , Chotgor君 ]", self.PARTICIPANTS)
        assert result == ["はる", "Chotgor君"]

    def test_no_bracket_returns_empty(self):
        """[] がない応答はパース失敗として空リストを返すこと。"""
        result = _parse_director_response("はるが次に発言します", self.PARTICIPANTS)
        assert result == []

    def test_empty_string_returns_empty(self):
        """空文字列は空リストを返すこと。"""
        result = _parse_director_response("", self.PARTICIPANTS)
        assert result == []

    def test_all_three_participants(self):
        """全参加者が指定された場合、全員を返すこと。"""
        result = _parse_director_response("[はる, Chotgor君, サブキャラ]", self.PARTICIPANTS)
        assert result == ["はる", "Chotgor君", "サブキャラ"]


# ---------------------------------------------------------------------------
# _build_director_messages のテスト
# ---------------------------------------------------------------------------

class TestBuildDirectorMessages:
    """_build_director_messages のプロンプト構築を検証するテストスイート。

    キャラクター設定・会話履歴が正しくユーザーメッセージに含まれること、
    フォーマット指示がシステムプロンプトに含まれることを検証する。
    """

    def _make_sqlite(self, char_name, system_prompt):
        """指定したキャラクターを返すモックSQLiteStoreを生成するヘルパー。"""
        sqlite = MagicMock()
        char = MagicMock()
        char.name = char_name
        char.system_prompt_block1 = system_prompt
        sqlite.get_character_by_name.return_value = char
        sqlite.get_character.return_value = char
        return sqlite

    def test_contains_character_names(self):
        """ユーザーメッセージに参加者名のセクションヘッダーが含まれること。"""
        sqlite = self._make_sqlite("はる", "はるはAIキャラクターです。")
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        _, user_message = _build_director_messages("<user>こんにちは</user>", participants, sqlite)
        assert "## はる" in user_message

    def test_contains_system_prompt_content(self):
        """参加者のsystem_prompt_block1がユーザーメッセージに含まれること。"""
        sqlite = self._make_sqlite("はる", "はるはフレンドリーなキャラクターです。")
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        _, user_message = _build_director_messages("会話履歴", participants, sqlite)
        assert "はるはフレンドリーなキャラクターです。" in user_message

    def test_contains_history(self):
        """会話履歴テキストがユーザーメッセージに含まれること。"""
        sqlite = self._make_sqlite("はる", "")
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        history_text = "<user>テストメッセージ</user>"
        _, user_message = _build_director_messages(history_text, participants, sqlite)
        assert history_text in user_message

    def test_contains_format_instructions(self):
        """フォーマット指示 ([正式名称] / [] など) がシステムプロンプトに含まれること。"""
        sqlite = self._make_sqlite("はる", "")
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        system_prompt, _ = _build_director_messages("", participants, sqlite)
        assert "[正式名称]" in system_prompt
        assert "[]" in system_prompt

    def test_contains_participant_names_in_instructions(self):
        """ユーザーメッセージ内に参加者名リストが含まれること。"""
        sqlite = self._make_sqlite("はる", "")
        participants = [
            {"char_name": "はる", "preset_name": "Sonnet"},
            {"char_name": "Chotgor君", "preset_name": "Gemini"},
        ]
        sqlite.get_character_by_name.side_effect = lambda name: MagicMock(
            name=name, system_prompt_block1=f"{name}の設定"
        )
        _, user_message = _build_director_messages("", participants, sqlite)
        assert "はる" in user_message
        assert "Chotgor君" in user_message


# ---------------------------------------------------------------------------
# decide_next_speakers のテスト
# ---------------------------------------------------------------------------

class TestDecideNextSpeakers:
    """decide_next_speakers の非同期動作を検証するテストスイート。

    LLM呼び出しはすべてモック化し、ネットワークアクセスを発生させない。
    タイムアウト・接続エラー時のフォールバック動作、ハルシネーションフィルタを
    重点的に検証する。
    """

    def _make_history(self, messages: list[tuple[str, str]]):
        """(role, content) タプルのリストから ChatMessage 風モックリストを生成するヘルパー。

        Args:
            messages: (role, content) のタプルリスト。roleは "user" または "character"。
            character_name は "character" ロールの場合 "テストキャラ" を設定する。
        """
        result = []
        for role, content in messages:
            m = MagicMock()
            m.role = role
            m.content = content
            m.character_name = "テストキャラ" if role == "character" else None
            result.append(m)
        return result

    def _make_sqlite(self, participants: list[dict]):
        """参加者情報からモックSQLiteStoreを生成するヘルパー。

        キャラクター情報取得と、司会役プリセット取得をモック化する。
        """
        sqlite = MagicMock()

        def get_char_by_name(name):
            char = MagicMock()
            char.name = name
            char.system_prompt_block1 = f"{name}のキャラクター設定"
            return char

        sqlite.get_character_by_name.side_effect = get_char_by_name
        sqlite.get_character.side_effect = get_char_by_name

        # 司会役のプリセットモック
        preset = MagicMock()
        preset.provider = "google"
        preset.model_id = "gemma-3-27b-it"
        sqlite.get_model_preset_by_name.return_value = preset
        sqlite.get_model_preset.return_value = preset

        return sqlite

    def _make_provider_mock(self, return_value: str):
        """指定した応答を返すプロバイダーモックを生成するヘルパー。"""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value=return_value)
        return mock_provider

    @pytest.mark.asyncio
    async def test_returns_parsed_speakers_on_success(self):
        """プロバイダーが [はる] を返したとき、["はる"] を返すこと。"""
        participants = [
            {"char_name": "はる", "preset_name": "Sonnet"},
            {"char_name": "Chotgor君", "preset_name": "Gemini"},
        ]
        sqlite = self._make_sqlite(participants)
        history = self._make_history([("user", "こんにちは")])

        with patch(
            "backend.core.group_chat.director.create_provider",
            return_value=self._make_provider_mock("[はる]"),
        ):
            result = await decide_next_speakers(
                history=history,
                participants=participants,
                sqlite=sqlite,
                settings={},
                director_char_name="司会",
                director_preset_name="Gemini",
                timeout=5,
            )

        assert result == ["はる"]

    @pytest.mark.asyncio
    async def test_returns_multiple_speakers(self):
        """プロバイダーが [はる, Chotgor君] を返したとき、両方を返すこと。"""
        participants = [
            {"char_name": "はる", "preset_name": "Sonnet"},
            {"char_name": "Chotgor君", "preset_name": "Gemini"},
        ]
        sqlite = self._make_sqlite(participants)
        history = self._make_history([("user", "みんな聞いてる？")])

        with patch(
            "backend.core.group_chat.director.create_provider",
            return_value=self._make_provider_mock("[はる, Chotgor君]"),
        ):
            result = await decide_next_speakers(
                history=history,
                participants=participants,
                sqlite=sqlite,
                settings={},
                director_char_name="司会",
                director_preset_name="Gemini",
            )

        assert result == ["はる", "Chotgor君"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_user_turn(self):
        """プロバイダーが [] を返したとき、空リスト（ユーザーターン）を返すこと。"""
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        sqlite = self._make_sqlite(participants)
        history = self._make_history([("user", "ありがとう")])

        with patch(
            "backend.core.group_chat.director.create_provider",
            return_value=self._make_provider_mock("[]"),
        ):
            result = await decide_next_speakers(
                history=history,
                participants=participants,
                sqlite=sqlite,
                settings={},
                director_char_name="司会",
                director_preset_name="Gemini",
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self):
        """プロバイダーへの接続が失敗したとき、例外を吐かず None を返すこと。

        None はエラーによるフォールバックを意味し、service 側でユーザーターンへ戻す。
        """
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        sqlite = self._make_sqlite(participants)
        history = self._make_history([("user", "こんにちは")])

        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=Exception("Connection refused"))

        with patch(
            "backend.core.group_chat.director.create_provider",
            return_value=mock_provider,
        ):
            result = await decide_next_speakers(
                history=history,
                participants=participants,
                sqlite=sqlite,
                settings={},
                director_char_name="司会",
                director_preset_name="Gemini",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self):
        """プロバイダーがタイムアウトしたとき、例外を吐かず None を返すこと。"""
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        sqlite = self._make_sqlite(participants)
        history = self._make_history([("user", "こんにちは")])

        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch(
            "backend.core.group_chat.director.create_provider",
            return_value=mock_provider,
        ):
            result = await decide_next_speakers(
                history=history,
                participants=participants,
                sqlite=sqlite,
                settings={},
                director_char_name="司会",
                director_preset_name="Gemini",
                timeout=1,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_filters_hallucinated_character_names(self):
        """プロバイダーが参加者にいないキャラ名を返したとき、フィルタリングされること。

        LLMのハルシネーション（存在しないキャラを指名する）への対策を確認する。
        """
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        sqlite = self._make_sqlite(participants)
        history = self._make_history([("user", "こんにちは")])

        with patch(
            "backend.core.group_chat.director.create_provider",
            return_value=self._make_provider_mock("[幻のキャラクター]"),
        ):
            result = await decide_next_speakers(
                history=history,
                participants=participants,
                sqlite=sqlite,
                settings={},
                director_char_name="司会",
                director_preset_name="Gemini",
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_history_with_character_messages(self):
        """キャラクターメッセージが含まれる履歴でも正常に動作すること。"""
        participants = [
            {"char_name": "はる", "preset_name": "Sonnet"},
            {"char_name": "Chotgor君", "preset_name": "Gemini"},
        ]
        sqlite = self._make_sqlite(participants)
        history = self._make_history([
            ("user", "こんにちは"),
            ("character", "やあ！"),
            ("user", "どうしてる？"),
        ])
        history[1].character_name = "はる"

        with patch(
            "backend.core.group_chat.director.create_provider",
            return_value=self._make_provider_mock("[Chotgor君]"),
        ):
            result = await decide_next_speakers(
                history=history,
                participants=participants,
                sqlite=sqlite,
                settings={},
                director_char_name="司会",
                director_preset_name="Gemini",
            )

        assert result == ["Chotgor君"]

    @pytest.mark.asyncio
    async def test_returns_none_when_preset_not_found(self):
        """司会役のプリセットが見つからないとき、None を返すこと。

        None はエラーによるフォールバックを意味し、service 側でユーザーターンへ戻す。
        """
        participants = [{"char_name": "はる", "preset_name": "Sonnet"}]
        sqlite = self._make_sqlite(participants)
        # プリセットが見つからない状態にする
        sqlite.get_model_preset_by_name.return_value = None
        sqlite.get_model_preset.return_value = None
        history = self._make_history([("user", "こんにちは")])

        result = await decide_next_speakers(
            history=history,
            participants=participants,
            sqlite=sqlite,
            settings={},
            director_char_name="司会",
            director_preset_name="存在しないプリセット",
        )

        assert result is None
