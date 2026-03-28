"""
グループチャット コンテキスト整形 (context.py) のユニットテスト。

テスト対象:
  - format_group_history_for_character: グループ履歴を指定キャラ視点のOpenAI messages形式に変換する

検証ポイント:
  - ユーザーメッセージは <user> タグ付きで role="user" になること
  - 自分自身の発言は role="assistant" になること
  - 他キャラクターの発言はタグ付きで role="user" になること
  - 連続する同ロールのメッセージは1つにマージされること（OpenAI仕様への適合）
"""

from unittest.mock import MagicMock

import pytest

from backend.services.group_chat.context import format_group_history_for_character


def _msg(role: str, content: str, character_name: str | None = None):
    """テスト用ChatMessage風モックを生成するヘルパー。

    Args:
        role: "user" または "character"。
        content: メッセージ本文。
        character_name: グループチャット時の発言キャラクター名（role="character"のみ）。
    """
    m = MagicMock()
    m.role = role
    m.content = content
    m.character_name = character_name
    return m


class TestFormatGroupHistoryForCharacter:
    """format_group_history_for_character の変換ロジックを網羅するテストスイート。

    OpenAI messages 形式の制約（role は "user" / "assistant" のみ、
    連続する同ロールは不可）に適合した出力になることを重点的に検証する。
    """

    SELF = "はる"
    OTHER = "Chotgor君"

    def test_empty_history_returns_empty_list(self):
        """空の履歴は空リストを返すこと。"""
        result = format_group_history_for_character([], self.SELF)
        assert result == []

    def test_user_message_becomes_user_role_with_tag(self):
        """ユーザーメッセージは role='user' で <user> タグ付きになること。"""
        history = [_msg("user", "こんにちは")]
        result = format_group_history_for_character(history, self.SELF)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "<user>こんにちは</user>"

    def test_self_message_becomes_assistant_role(self):
        """自分自身の発言は role='assistant' で本文そのままになること。"""
        history = [_msg("character", "やあ！", character_name=self.SELF)]
        result = format_group_history_for_character(history, self.SELF)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "やあ！"

    def test_other_character_becomes_user_role_with_tag(self):
        """他キャラクターの発言は role='user' でキャラ名タグ付きになること。"""
        history = [_msg("character", "俺の番だ！", character_name=self.OTHER)]
        result = format_group_history_for_character(history, self.SELF)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == f"<{self.OTHER}>俺の番だ！</{self.OTHER}>"

    def test_consecutive_user_messages_are_merged(self):
        """連続するユーザーメッセージは1エントリにマージされること。

        ユーザーメッセージの後に他キャラのメッセージが来ると、
        APIの仕様上 user → user になるため、マージが必要。
        """
        history = [
            _msg("user", "最初のユーザーメッセージ"),
            _msg("character", "他キャラの発言", character_name=self.OTHER),
        ]
        result = format_group_history_for_character(history, self.SELF)
        # 両方とも user ロール → マージされて1エントリになること
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "<user>最初のユーザーメッセージ</user>" in result[0]["content"]
        assert f"<{self.OTHER}>他キャラの発言</{self.OTHER}>" in result[0]["content"]

    def test_alternating_user_and_self_is_not_merged(self):
        """ユーザーとキャラクターが交互に発言する正常な会話はマージされないこと。"""
        history = [
            _msg("user", "ユーザー1"),
            _msg("character", "はるの返答", character_name=self.SELF),
            _msg("user", "ユーザー2"),
            _msg("character", "はるの返答2", character_name=self.SELF),
        ]
        result = format_group_history_for_character(history, self.SELF)
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"
        assert result[3]["role"] == "assistant"

    def test_consecutive_assistant_messages_are_merged(self):
        """自分の発言が連続する場合、1エントリにマージされること（edge case）。"""
        history = [
            _msg("character", "1回目", character_name=self.SELF),
            _msg("character", "2回目", character_name=self.SELF),
        ]
        result = format_group_history_for_character(history, self.SELF)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "1回目" in result[0]["content"]
        assert "2回目" in result[0]["content"]

    def test_typical_group_conversation(self):
        """典型的なグループ会話（user → キャラA → キャラB → user）を正しく変換すること。

        キャラA視点:
          user → user (そのまま)
          キャラA → assistant
          キャラB → user (タグ付き)
          user2 → user (マージ: タグ付きキャラB発言 + <user>user2)
        """
        history = [
            _msg("user", "みんなに質問です"),
            _msg("character", "はいはい！", character_name=self.SELF),
            _msg("character", "俺もいるぞ", character_name=self.OTHER),
            _msg("user", "ありがとう"),
        ]
        result = format_group_history_for_character(history, self.SELF)

        # user / assistant / user(マージ) の3エントリになること
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "<user>みんなに質問です</user>"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "はいはい！"
        assert result[2]["role"] == "user"
        assert f"<{self.OTHER}>俺もいるぞ</{self.OTHER}>" in result[2]["content"]
        assert "<user>ありがとう</user>" in result[2]["content"]

    def test_character_message_without_name_defaults_gracefully(self):
        """character_name が None の場合、自分以外として処理されること。

        1on1チャットの履歴を誤って渡されても、最低限動作することを確認する。
        """
        history = [
            _msg("user", "ユーザーメッセージ"),
            _msg("character", "応答", character_name=None),
        ]
        result = format_group_history_for_character(history, self.SELF)
        # character_name=None は自分 ("はる") ではないので user 側にマージされる
        assert all(r["role"] in ("user", "assistant") for r in result)
