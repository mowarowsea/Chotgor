"""Shared pytest fixtures for Chotgor tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_memory_manager():
    """MemoryManager の mock。SQLite / ChromaDB への実アクセスを避ける。"""
    manager = MagicMock()
    manager.recall_memory.return_value = []
    manager.write_memory.return_value = "mock-memory-id"
    manager.delete_memory.return_value = True
    manager.list_memories.return_value = []
    return manager
