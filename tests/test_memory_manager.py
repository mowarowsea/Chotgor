from unittest.mock import MagicMock, patch
import pytest
from backend.core.memory.manager import MemoryManager

@pytest.fixture
def mock_chroma():
    store = MagicMock()
    store.recall_memory.return_value = [
        {"id": "mem-1", "content": "test content", "distance": 0.1, "metadata": {"category": "general"}}
    ]
    return store

@pytest.fixture
def manager(sqlite_store, mock_chroma):
    import backend.core.memory.manager
    return MemoryManager(sqlite=sqlite_store, chroma=mock_chroma)

def test_write_memory(manager, sqlite_store, mock_chroma):
    char_id = "char-001"
    manager.write_memory(char_id, "Hello world", category="user")
    
    # Check SQLite
    mems = sqlite_store.list_memories(char_id)
    assert len(mems) == 1
    assert mems[0].content == "Hello world"
    assert mems[0].memory_category == "user"
    
    # Check Chroma
    mock_chroma.add_memory.assert_called_once()

def test_recall_memory_hybrid(manager, sqlite_store, mock_chroma):
    char_id = "char-001"
    # Seed SQLite with a memory that matches mock_chroma's return ID
    sqlite_store.create_memory("mem-1", char_id, "real content")
    
    results = manager.recall_memory(char_id, "test")
    assert len(results) == 1
    # Manager now adds hybrid_score and other metadata
    assert "hybrid_score" in results[0]
    assert "decayed_score" in results[0]
    
def test_delete_memory(manager, sqlite_store, mock_chroma):
    char_id = "char-001"
    sqlite_store.create_memory("mem-1", char_id, "to be deleted")
    
    ok = manager.delete_memory("mem-1", char_id)
    assert ok is True
    assert sqlite_store.get_memory("mem-1").deleted_at is not None
    mock_chroma.delete_memory.assert_called_once_with("mem-1", char_id)
