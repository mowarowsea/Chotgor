"""Tests for backend.core.memory.manager — MemoryManager coordination."""

from unittest.mock import MagicMock, call, patch

import pytest

from backend.core.memory.manager import MemoryManager


@pytest.fixture
def mock_sqlite():
    store = MagicMock()
    store.create_memory.return_value = MagicMock()
    store.touch_memory.return_value = None
    store.soft_delete_memory.return_value = True
    store.list_memories.return_value = []
    return store


@pytest.fixture
def mock_chroma():
    store = MagicMock()
    store.recall_memory.return_value = [
        {"id": "mem-1", "content": "猫が好き", "distance": 0.1, "metadata": {"category": "user"}}
    ]
    store.add_memory.return_value = None
    store.delete_memory.return_value = None
    return store


@pytest.fixture
def manager(mock_sqlite, mock_chroma):
    return MemoryManager(sqlite=mock_sqlite, chroma=mock_chroma)


class TestWriteMemory:
    def test_writes_to_both_stores(self, manager, mock_sqlite, mock_chroma):
        manager.write_memory(
            character_id="char-001",
            content="ユーザーは猫派。",
            category="user",
        )
        mock_sqlite.create_memory.assert_called_once()
        mock_chroma.add_memory.assert_called_once()

    def test_returns_memory_id_string(self, manager):
        with patch("backend.core.memory.manager.uuid.uuid4", return_value="fixed-uuid"):
            memory_id = manager.write_memory(
                character_id="char-001",
                content="テスト記憶",
            )
        assert memory_id == "fixed-uuid"

    def test_sqlite_receives_correct_args(self, manager, mock_sqlite):
        manager.write_memory(
            character_id="char-001",
            content="テスト内容",
            category="identity",
            identity_importance=0.9,
        )
        call_kwargs = mock_sqlite.create_memory.call_args.kwargs
        assert call_kwargs["character_id"] == "char-001"
        assert call_kwargs["content"] == "テスト内容"
        assert call_kwargs["memory_category"] == "identity"
        assert call_kwargs["identity_importance"] == 0.9

    def test_chroma_receives_correct_metadata(self, manager, mock_chroma):
        manager.write_memory(
            character_id="char-001",
            content="テスト",
            category="semantic",
            semantic_importance=0.8,
        )
        call_kwargs = mock_chroma.add_memory.call_args.kwargs
        assert call_kwargs["character_id"] == "char-001"
        assert call_kwargs["metadata"]["category"] == "semantic"
        assert call_kwargs["metadata"]["semantic_importance"] == 0.8


class TestRecallMemory:
    def test_returns_chroma_results(self, manager, mock_chroma):
        results = manager.recall_memory("char-001", "猫")
        assert len(results) == 1
        assert results[0]["content"] == "猫が好き"

    def test_touches_each_recalled_memory_in_sqlite(self, manager, mock_sqlite, mock_chroma):
        manager.recall_memory("char-001", "猫")
        mock_sqlite.touch_memory.assert_called_once_with("mem-1")

    def test_empty_recall_does_not_touch(self, manager, mock_sqlite, mock_chroma):
        mock_chroma.recall_memory.return_value = []
        manager.recall_memory("char-001", "存在しない記憶")
        mock_sqlite.touch_memory.assert_not_called()


class TestDeleteMemory:
    def test_soft_deletes_in_sqlite(self, manager, mock_sqlite):
        manager.delete_memory("mem-1", "char-001")
        mock_sqlite.soft_delete_memory.assert_called_once_with("mem-1")

    def test_hard_deletes_in_chroma_when_sqlite_succeeds(self, manager, mock_chroma):
        manager.delete_memory("mem-1", "char-001")
        mock_chroma.delete_memory.assert_called_once_with("mem-1", "char-001")

    def test_chroma_not_deleted_when_sqlite_fails(self, manager, mock_sqlite, mock_chroma):
        mock_sqlite.soft_delete_memory.return_value = False
        manager.delete_memory("nonexistent", "char-001")
        mock_chroma.delete_memory.assert_not_called()

    def test_returns_true_on_success(self, manager):
        result = manager.delete_memory("mem-1", "char-001")
        assert result is True

    def test_returns_false_when_not_found(self, manager, mock_sqlite):
        mock_sqlite.soft_delete_memory.return_value = False
        result = manager.delete_memory("nonexistent", "char-001")
        assert result is False
