"""Tests for claude_cli_provider._format_conversation — Issue #10 coverage."""

import pytest

from backend.core.providers.claude_cli_provider import _format_conversation


class TestFormatConversation:
    def test_empty_messages(self):
        assert _format_conversation([]) == ""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "こんにちは"}]
        result = _format_conversation(messages)
        assert result == "こんにちは"

    def test_single_user_message_with_char_name(self):
        messages = [{"role": "user", "content": "こんにちは"}]
        result = _format_conversation(messages, character_name="織羽")
        # 1件のみの場合は history なし、そのまま返す
        assert result == "こんにちは"

    def test_multi_turn_uses_character_name_tag(self):
        """Issue #10: assistant ロールはキャラクター名タグで囲まれる。"""
        messages = [
            {"role": "user", "content": "調子はどう？"},
            {"role": "assistant", "content": "まあまあかな。"},
            {"role": "user", "content": "そっか"},
        ]
        result = _format_conversation(messages, character_name="織羽")
        assert "<織羽>まあまあかな。</織羽>" in result
        assert "<ai>" not in result
        assert "assistant" not in result

    def test_multi_turn_fallback_when_no_character_name(self):
        """キャラクター名なし → <character> タグにフォールバック。"""
        messages = [
            {"role": "user", "content": "ねえ"},
            {"role": "assistant", "content": "なに？"},
            {"role": "user", "content": "なんでもない"},
        ]
        result = _format_conversation(messages)
        assert "<character>なに？</character>" in result
        assert "<ai>" not in result

    def test_system_role_is_skipped(self):
        messages = [
            {"role": "system", "content": "このメッセージは無視される"},
            {"role": "user", "content": "最初のユーザー発言"},
            {"role": "assistant", "content": "最初の返答"},
            {"role": "user", "content": "2番目の質問"},
        ]
        result = _format_conversation(messages, character_name="テスト")
        assert "このメッセージは無視される" not in result

    def test_history_wrapped_in_history_tag(self):
        messages = [
            {"role": "user", "content": "過去の質問"},
            {"role": "assistant", "content": "過去の返答"},
            {"role": "user", "content": "最新の質問"},
        ]
        result = _format_conversation(messages, character_name="キャラ")
        assert result.startswith("<history>")
        assert "</history>" in result
        # 最後のユーザーメッセージは history の外
        assert result.endswith("最新の質問")

    def test_human_tag_for_user_in_history(self):
        messages = [
            {"role": "user", "content": "ユーザー発言"},
            {"role": "assistant", "content": "キャラ返答"},
            {"role": "user", "content": "最新"},
        ]
        result = _format_conversation(messages, character_name="キャラ")
        assert "<human>ユーザー発言</human>" in result

    def test_character_name_with_spaces_is_stripped(self):
        messages = [
            {"role": "user", "content": "過去"},
            {"role": "assistant", "content": "返答"},
            {"role": "user", "content": "今"},
        ]
        result = _format_conversation(messages, character_name="  織羽  ")
        assert "<織羽>返答</織羽>" in result
