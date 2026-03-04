"""Tests for backend.core.system_prompt — system prompt builder."""

import pytest

from backend.core.system_prompt import build_system_prompt, CHOTGOR_BLOCK3_TEMPLATE


class TestBuildSystemPrompt:
    def test_character_prompt_always_included(self):
        result = build_system_prompt(character_system_prompt="私はAIキャラクターです。")
        assert "私はAIキャラクターです。" in result

    def test_block3_always_included(self):
        """Chotgorメモリ記法の説明（Block3）は必ず末尾に含まれる。"""
        result = build_system_prompt(character_system_prompt="テスト")
        assert "MEMORY:" in result
        # Block3 が最後のブロックであること
        assert result.endswith(CHOTGOR_BLOCK3_TEMPLATE.strip())

    def test_recalled_memories_appear_in_prompt(self):
        memories = [
            {"content": "ユーザーは猫が好き。", "metadata": {"category": "user"}},
            {"content": "Chotgorは記憶管理システム。", "metadata": {"category": "semantic"}},
        ]
        result = build_system_prompt(
            character_system_prompt="テスト",
            recalled_memories=memories,
        )
        assert "ユーザーは猫が好き" in result
        assert "Chotgorは記憶管理システム" in result
        assert "Relevant Memories" in result

    def test_no_memories_no_memory_block(self):
        result = build_system_prompt(
            character_system_prompt="テスト",
            recalled_memories=[],
        )
        assert "Relevant Memories" not in result

    def test_fetched_web_content_included(self):
        fetched = [{"url": "https://example.com", "content": "サンプルページの内容", "truncated": False}]
        result = build_system_prompt(
            character_system_prompt="テスト",
            fetched_contents=fetched,
        )
        assert "Fetched Web Content" in result
        assert "サンプルページの内容" in result

    def test_fetched_web_content_with_error(self):
        fetched = [{"url": "https://bad.example.com", "error": "接続できませんでした"}]
        result = build_system_prompt(
            character_system_prompt="テスト",
            fetched_contents=fetched,
        )
        assert "接続できませんでした" in result

    def test_meta_instructions_included(self):
        result = build_system_prompt(
            character_system_prompt="テスト",
            meta_instructions="絶対に英語で返答すること。",
        )
        assert "絶対に英語で返答すること" in result
        assert "Character-specific Instructions" in result

    def test_meta_instructions_empty_skipped(self):
        result = build_system_prompt(
            character_system_prompt="テスト",
            meta_instructions="",
        )
        assert "Character-specific Instructions" not in result

    def test_blocks_separated_by_divider(self):
        """ブロック間は `---` で区切られる。"""
        result = build_system_prompt(
            character_system_prompt="テスト",
            meta_instructions="追加指示",
        )
        assert "---" in result

    def test_truncated_content_marker(self):
        fetched = [{"url": "https://example.com", "content": "長いコンテンツ", "truncated": True}]
        result = build_system_prompt(
            character_system_prompt="テスト",
            fetched_contents=fetched,
        )
        assert "省略" in result
