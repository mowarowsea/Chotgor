import pytest
from datetime import datetime, timedelta
import tempfile
import os
from backend.core.memory.sqlite_store import SQLiteStore, Memory
from backend.core.memory.manager import MemoryManager
from unittest.mock import MagicMock

@pytest.fixture
def sqlite_store():
    # Use an in-memory or temporary file db for tests
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(path)
    yield store
    # Cleanup DB connection before removing file
    store.engine.dispose()
    try:
        os.remove(path)
    except PermissionError:
        pass

@pytest.fixture
def memory_manager(sqlite_store):
    chroma = MagicMock()
    return MemoryManager(sqlite_store, chroma)

def test_calculate_decayed_score(memory_manager):
    manager = memory_manager
    
    # Create a dummy memory
    now = datetime.now()
    old_time = now - timedelta(days=14)
    
    m = Memory(
        id="test-mem",
        character_id="char1",
        content="test",
        contextual_importance=1.0,  # fast decay (7 days)
        user_importance=1.0,       # medium decay (30 days)
        semantic_importance=1.0,   # slow decay (90 days)
        identity_importance=1.0,   # no decay
        created_at=old_time,
        last_accessed_at=old_time
    )
    
    score = manager.calculate_decayed_score(m, now)
    
    # Contextual after 14 days (2 half-lives) = 1.0 * 0.25 = 0.25 * 1.0 weight = 0.25
    # User after 14 days (~0.46 half-lives) = 1.0 * 0.72 = 0.72 * 0.8 weight = 0.576
    # Semantic after 14 days (~0.15 half-lives) = 1.0 * 0.89 = 0.89 * 0.6 weight = 0.534
    # Identity after 14 days (0 half-lives) = 1.0 * 1.0 = 1.0 * 0.3 weight = 0.3
    # Total ~ 1.66
    
    assert score > 1.5
    assert score < 1.8
    
    # Immediate recall should have near max score
    m2 = Memory(
        id="test-mem2",
        character_id="char1",
        content="test",
        contextual_importance=1.0,
        user_importance=1.0,
        semantic_importance=1.0,
        identity_importance=1.0,
        created_at=now,
        last_accessed_at=now
    )
    score2 = manager.calculate_decayed_score(m2, now)
    # Weights sum: 1.0 + 0.8 + 0.6 + 0.3 = 2.7
    assert score2 > 2.6
    
def test_get_forgotten_candidates(memory_manager):
    manager = memory_manager
    char_id = "test-char"
    manager.sqlite.create_character(char_id, "Test")
    
    # Memory 1: Old and unimportant
    manager.sqlite.create_memory(
        memory_id="mem1",
        character_id=char_id,
        content="old stuff",
        contextual_importance=0.1,
        user_importance=0.1,
        semantic_importance=0.1,
        identity_importance=0.1,
    )
    
    # Memory 2: New and important
    manager.sqlite.create_memory(
        memory_id="mem2",
        character_id=char_id,
        content="new stuff",
        contextual_importance=1.0,
        user_importance=1.0,
        semantic_importance=1.0,
        identity_importance=1.0,
    )
    
    # Fake dates: hard update SQLite
    now = datetime.now()
    old_time = now - timedelta(days=60)
    
    with manager.sqlite.get_session() as session:
        m1 = session.query(Memory).filter_by(id="mem1").first()
        m1.created_at = old_time
        m1.last_accessed_at = old_time
        session.commit()
    
    # Threshold 1.0 -> should catch old/unimportant but not new
    candidates = manager.get_forgotten_candidates(char_id, threshold=1.0)
    ids = [c.id for c in candidates]
    
    assert "mem1" in ids
    assert "mem2" not in ids
