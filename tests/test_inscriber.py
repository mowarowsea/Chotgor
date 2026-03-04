"""Tests for backend.core.memory.inscriber — [MEMORY:...] marker parsing."""

from unittest.mock import MagicMock, call

import pytest

from backend.core.memory.inscriber import carve, _extract


# ---------------------------------------------------------------------------
# _extract() — テキストからマーカーを取り出す
# ---------------------------------------------------------------------------

class TestExtract:
    def test_single_memory_marker(self):
        text = "返答本文です。\n[MEMORY:identity|1.5|私はお茶が好き。]"
        clean, memories = _extract(text)
        assert "[MEMORY:" not in clean
        assert len(memories) == 1
        cat, impact, content = memories[0]
        assert cat == "identity"
        assert impact == "1.5"
        assert "お茶が好き" in content

    def test_multiple_memory_markers(self):
        text = (
            "本文\n"
            "[MEMORY:user|1.0|ユーザーは猫派。]\n"
            "[MEMORY:contextual|0.5|明日は晴れ。]"
        )
        clean, memories = _extract(text)
        assert len(memories) == 2
        assert memories[0][0] == "user"
        assert memories[1][0] == "contextual"

    def test_clean_text_has_no_markers(self):
        text = "普通の返答。\n[MEMORY:semantic|1.0|ChromaDBはベクトルDB。]"
        clean, _ = _extract(text)
        assert "MEMORY" not in clean
        assert "普通の返答。" in clean

    def test_no_markers_returns_empty_list(self):
        text = "何も覚えることはなかった。"
        clean, memories = _extract(text)
        assert memories == []
        assert clean == text

    def test_fallback_format_without_impact(self):
        """[MEMORY:category] content 形式（パイプなし）のフォールバック解析。"""
        text = "[MEMORY:identity] 私はコーヒーより紅茶が好き。"
        clean, memories = _extract(text)
        assert len(memories) == 1
        cat, impact, content = memories[0]
        assert cat == "identity"
        assert impact == "1.0"
        assert "紅茶が好き" in content


# ---------------------------------------------------------------------------
# carve() — 解析 + write_memory 呼び出し
# ---------------------------------------------------------------------------

class TestCarve:
    def test_carve_calls_write_memory(self, mock_memory_manager):
        text = "返答です。\n[MEMORY:user|1.2|ユーザーは音楽好き。]"
        clean = carve(text, "char-001", mock_memory_manager)
        mock_memory_manager.write_memory.assert_called_once()
        call_kwargs = mock_memory_manager.write_memory.call_args.kwargs
        assert call_kwargs["character_id"] == "char-001"
        assert "音楽好き" in call_kwargs["content"]
        assert call_kwargs["category"] == "user"

    def test_carve_returns_clean_text(self, mock_memory_manager):
        text = "本文です。\n[MEMORY:semantic|1.0|Chotgorは記憶管理システム。]"
        clean = carve(text, "char-001", mock_memory_manager)
        assert "[MEMORY:" not in clean
        assert "本文です。" in clean

    def test_carve_no_marker_does_not_write(self, mock_memory_manager):
        text = "今日は特に覚えることはなかった。"
        carve(text, "char-001", mock_memory_manager)
        mock_memory_manager.write_memory.assert_not_called()

    def test_carve_impact_exceeds_1(self, mock_memory_manager):
        """impact=2.0 の場合、importance が 1.0 を超えることを許容する。"""
        text = "[MEMORY:identity|2.0|非常に重要な記憶。]"
        carve(text, "char-001", mock_memory_manager)
        call_kwargs = mock_memory_manager.write_memory.call_args.kwargs
        
        # identityカテゴリに対するベース値 (identity: 0.9) x 2.0 = 1.8 になるはず
        assert call_kwargs["identity_importance"] == 1.8
        # 他のスコアも同様に倍増していること (例: semantic: 0.4 x 2.0 = 0.8)
        assert call_kwargs["semantic_importance"] == 0.8

    def test_carve_multiple_memories(self, mock_memory_manager):
        text = (
            "返答。\n"
            "[MEMORY:user|1.0|ユーザーはゲーマー。]\n"
            "[MEMORY:contextual|0.5|今日は雨。]"
        )
        carve(text, "char-001", mock_memory_manager)
        assert mock_memory_manager.write_memory.call_count == 2
