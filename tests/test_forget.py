import pytest
from datetime import datetime, timedelta
import tempfile
import os
from unittest.mock import MagicMock, AsyncMock, patch

from backend.core.memory.sqlite_store import SQLiteStore, Memory
from backend.core.memory.manager import MemoryManager
from backend.core.memory.forget import run_forget_process

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

    # 現在の実装パラメータ（manager.py より）:
    #   contextual: 半減期 4日, weight 1.0
    #   user:       半減期 10日, weight 0.8
    #   semantic:   半減期 20日, weight 0.6
    #   identity:   半減期 90日, weight 0.3
    #
    # 14日後の各スコア（importance=1.0）:
    #   contextual: exp(-ln2/4 * 14) ≈ 0.0884 * 1.0 = 0.088
    #   user:       exp(-ln2/10 * 14) ≈ 0.3789 * 0.8 = 0.303
    #   semantic:   exp(-ln2/20 * 14) ≈ 0.6160 * 0.6 = 0.370
    #   identity:   exp(-ln2/90 * 14) ≈ 0.8978 * 0.3 = 0.269
    #   合計 ≈ 1.030

    assert score > 1.0
    assert score < 1.1
    
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


# --- run_forget_process tests ---

def _make_manager_with_candidates(sqlite_store, char_id, memory_ids):
    """Helper: create a character + memories and return a MemoryManager whose
    get_forgotten_candidates() returns those memories."""
    chroma = MagicMock()
    manager = MemoryManager(sqlite_store, chroma)
    sqlite_store.create_character(char_id, "TestChar")

    now = datetime.now()
    old_time = now - timedelta(days=60)

    for mid in memory_ids:
        sqlite_store.create_memory(
            memory_id=mid,
            character_id=char_id,
            content=f"content of {mid}",
            contextual_importance=0.1,
            user_importance=0.1,
            semantic_importance=0.1,
            identity_importance=0.1,
        )
        with sqlite_store.get_session() as session:
            m = session.query(Memory).filter_by(id=mid).first()
            m.created_at = old_time
            m.last_accessed_at = old_time
            session.commit()

    return manager


@pytest.mark.asyncio
async def test_forget_default_keeps_all_when_no_delete(sqlite_store):
    """If LLM returns no [DELETE: ...], all candidates must be kept (not deleted)."""
    char_id = "char-keep-all"
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2", "m3"])

    with patch("backend.core.memory.forget._call_llm_for_forget", new=AsyncMock(return_value="何も手放しません。")):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            character_system_prompt="You are TestChar.",
            memory_manager=manager,
            sqlite=sqlite_store,
            threshold=1.0,
        )

    assert result["status"] == "success"
    assert result["deleted_count"] == 0
    assert result["kept_count"] == 3

    # Verify nothing was actually deleted in DB
    active = sqlite_store.get_all_active_memories(char_id)
    ids = [m.id for m in active]
    assert "m1" in ids
    assert "m2" in ids
    assert "m3" in ids


@pytest.mark.asyncio
async def test_forget_deletes_only_explicitly_listed(sqlite_store):
    """Only IDs in [DELETE: ...] are removed; others survive."""
    char_id = "char-partial-delete"
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2", "m3"])

    with patch("backend.core.memory.forget._call_llm_for_forget", new=AsyncMock(return_value="[DELETE: m2]")):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            character_system_prompt="You are TestChar.",
            memory_manager=manager,
            sqlite=sqlite_store,
            threshold=1.0,
        )

    assert result["status"] == "success"
    assert result["deleted_count"] == 1
    assert result["kept_count"] == 2

    active = sqlite_store.get_all_active_memories(char_id)
    ids = [m.id for m in active]
    assert "m1" in ids
    assert "m2" not in ids
    assert "m3" in ids


@pytest.mark.asyncio
async def test_forget_delete_none_keeps_all(sqlite_store):
    """[DELETE: NONE] explicitly keeps everything."""
    char_id = "char-delete-none"
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2"])

    with patch("backend.core.memory.forget._call_llm_for_forget", new=AsyncMock(return_value="[DELETE: NONE]")):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            character_system_prompt="You are TestChar.",
            memory_manager=manager,
            sqlite=sqlite_store,
            threshold=1.0,
        )

    assert result["deleted_count"] == 0
    assert result["kept_count"] == 2


@pytest.mark.asyncio
async def test_forget_parse_failure_keeps_all(sqlite_store):
    """Garbled LLM output (no parseable [DELETE:]) must not delete anything."""
    char_id = "char-garbled"
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2"])

    garbled = "うーん、どれも大切かな... KEEP m1 DELETE maybe m2??"
    with patch("backend.core.memory.forget._call_llm_for_forget", new=AsyncMock(return_value=garbled)):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            character_system_prompt="You are TestChar.",
            memory_manager=manager,
            sqlite=sqlite_store,
            threshold=1.0,
        )

    assert result["deleted_count"] == 0
    assert result["kept_count"] == 2
