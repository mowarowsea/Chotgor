"""Tests for claude_cli_provider._format_conversation — Issue #10 coverage."""

import pytest

from backend.providers.claude_cli_provider import _format_conversation


class TestFormatConversation:
    def test_empty_messages(self):
        assert _format_conversation([]) == ""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "こんにちは"}]
        assert _format_conversation(messages) == "こんにちは"

    def test_single_user_message_with_char_name(self):
        messages = [{"role": "user", "content": "こんにちは"}]
        # 1件のみの場合は history なし、そのまま返す
        assert _format_conversation(messages, character_name="織羽") == "こんにちは"

    def test_multi_turn_uses_character_name_tag(self):
        """Issue #10: assistant ロールはキャラクター名タグで囲まれる。"""
        messages = [
            {"role": "user", "content": "調子はどう？"},
            {"role": "assistant", "content": "まあまあかな。"},
            {"role": "user", "content": "そっか"},
        ]
        result = _format_conversation(messages, character_name="織羽")
        expected = (
            "<history>\n"
            "<human>調子はどう？</human>\n"
            "<織羽>まあまあかな。</織羽>\n"
            "</history>\n\n"
            "そっか"
        )
        assert result == expected

    def test_multi_turn_fallback_when_no_character_name(self):
        """キャラクター名なし → <character> タグにフォールバック。"""
        messages = [
            {"role": "user", "content": "ねえ"},
            {"role": "assistant", "content": "なに？"},
            {"role": "user", "content": "なんでもない"},
        ]
        result = _format_conversation(messages)
        expected = (
            "<history>\n"
            "<human>ねえ</human>\n"
            "<character>なに？</character>\n"
            "</history>\n\n"
            "なんでもない"
        )
        assert result == expected

    def test_system_role_is_skipped(self):
        """system ロールはプロンプトフォーマットに含めない。"""
        messages = [
            {"role": "system", "content": "このメッセージは無視される"},
            {"role": "user", "content": "最初のユーザー発言"},
            {"role": "assistant", "content": "最初の返答"},
            {"role": "user", "content": "2番目の質問"},
        ]
        result = _format_conversation(messages, character_name="テスト")
        expected = (
            "<history>\n"
            "<human>最初のユーザー発言</human>\n"
            "<テスト>最初の返答</テスト>\n"
            "</history>\n\n"
            "2番目の質問"
        )
        assert result == expected

    def test_history_formatting(self):
        """historyタグの付与、humanタグの付与、キャラ名の余白除去をまとめてテスト。"""
        messages = [
            {"role": "user", "content": "過去の質問"},
            {"role": "assistant", "content": "過去の返答"},
            {"role": "user", "content": "最新の質問"},
        ]
        result = _format_conversation(messages, character_name="  織羽  ")
        expected = (
            "<history>\n"
            "<human>過去の質問</human>\n"
            "<織羽>過去の返答</織羽>\n"
            "</history>\n\n"
            "最新の質問"
        )
        assert result == expected
