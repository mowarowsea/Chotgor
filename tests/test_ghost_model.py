"""Tests for ghost_model: DB persistence, digest/forget routing, and logging."""

import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.memory.sqlite_store import SQLiteStore
from backend.core.memory.manager import MemoryManager
from backend.core.memory.digest import run_daily_digest, _call_llm_for_digest
from backend.core.memory.forget import run_forget_process, _call_llm_for_forget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(path)
    yield store
    store.engine.dispose()
    try:
        os.remove(path)
    except PermissionError:
        pass


@pytest.fixture
def memory_manager(sqlite_store):
    chroma = MagicMock()
    chroma.add_memory = MagicMock()
    chroma.delete_memory = MagicMock()
    chroma.search = MagicMock(return_value=[])
    return MemoryManager(sqlite_store, chroma)


# ---------------------------------------------------------------------------
# ghost_model DB persistence
# ---------------------------------------------------------------------------

def test_ghost_model_saved_and_retrieved(sqlite_store):
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    char = sqlite_store.get_character(char_id)
    assert char.ghost_model == preset_id


def test_ghost_model_default_is_none(sqlite_store):
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")

    char = sqlite_store.get_character(char_id)
    assert char.ghost_model is None


def test_ghost_model_updated(sqlite_store):
    char_id = str(uuid.uuid4())
    preset_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")
    sqlite_store.update_character(char_id, ghost_model=preset_id)

    char = sqlite_store.get_character(char_id)
    assert char.ghost_model == preset_id


def test_ghost_model_cleared(sqlite_store):
    char_id = str(uuid.uuid4())
    preset_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)
    sqlite_store.update_character(char_id, ghost_model=None)

    char = sqlite_store.get_character(char_id)
    assert char.ghost_model is None


# ---------------------------------------------------------------------------
# _call_llm_for_digest: error cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_llm_for_digest_no_ghost_model(sqlite_store):
    with pytest.raises(RuntimeError, match="ghost_model"):
        await _call_llm_for_digest("sys", "mem_text", None, sqlite_store)


@pytest.mark.asyncio
async def test_call_llm_for_digest_unknown_preset(sqlite_store):
    with pytest.raises(RuntimeError, match="プリセット"):
        await _call_llm_for_digest("sys", "mem_text", "nonexistent-id", sqlite_store)


# ---------------------------------------------------------------------------
# _call_llm_for_forget: error cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_llm_for_forget_no_ghost_model(sqlite_store):
    with pytest.raises(RuntimeError, match="ghost_model"):
        await _call_llm_for_forget("sys", "mem_text", None, sqlite_store)


@pytest.mark.asyncio
async def test_call_llm_for_forget_unknown_preset(sqlite_store):
    with pytest.raises(RuntimeError, match="プリセット"):
        await _call_llm_for_forget("sys", "mem_text", "nonexistent-id", sqlite_store)


# ---------------------------------------------------------------------------
# _call_llm_for_digest: calls create_provider with correct args + logs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_llm_for_digest_uses_ghost_model_preset(sqlite_store):
    preset_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value="digest summary")

    with patch("backend.core.memory.digest.create_provider", return_value=mock_provider) as mock_cp, \
         patch("backend.core.memory.digest.log_llm_request") as mock_req, \
         patch("backend.core.memory.digest.log_llm_response") as mock_res:

        result = await _call_llm_for_digest("sys_prompt", "memory content", preset_id, sqlite_store)

    assert result == "digest summary"
    mock_cp.assert_called_once_with("google", "gemini-2.0-flash", sqlite_store.get_all_settings())
    mock_provider.generate.assert_called_once_with(
        "sys_prompt", [{"role": "user", "content": "memory content"}]
    )
    mock_req.assert_called_once()
    mock_res.assert_called_once_with("digest summary")


@pytest.mark.asyncio
async def test_call_llm_for_forget_uses_ghost_model_preset(sqlite_store):
    preset_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "anthropic", "claude-3-5-haiku-latest")

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value="[KEEP: NONE]")

    with patch("backend.core.memory.forget.create_provider", return_value=mock_provider) as mock_cp, \
         patch("backend.core.memory.forget.log_llm_request") as mock_req, \
         patch("backend.core.memory.forget.log_llm_response") as mock_res:

        result = await _call_llm_for_forget("sys_prompt", "candidates text", preset_id, sqlite_store)

    assert result == "[KEEP: NONE]"
    mock_cp.assert_called_once_with("anthropic", "claude-3-5-haiku-latest", sqlite_store.get_all_settings())
    mock_req.assert_called_once()
    mock_res.assert_called_once_with("[KEEP: NONE]")


# ---------------------------------------------------------------------------
# run_daily_digest: ghost_model propagated, error recorded in digest log
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_daily_digest_error_when_no_ghost_model(sqlite_store, memory_manager):
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")
    sqlite_store.create_memory(
        memory_id=str(uuid.uuid4()),
        character_id=char_id,
        content="something happened",
    )
    # Backdate the memory to the target date
    from datetime import datetime, timedelta
    target = (datetime.now() - timedelta(days=1)).date().isoformat()
    with sqlite_store.get_session() as session:
        from backend.core.memory.sqlite_store import Memory
        m = session.query(Memory).filter_by(character_id=char_id).first()
        m.created_at = datetime.fromisoformat(target)
        session.commit()

    result = await run_daily_digest(
        character_id=char_id,
        character_name="TestChar",
        character_system_prompt="You are TestChar.",
        target_date=target,
        memory_manager=memory_manager,
        sqlite=sqlite_store,
        ghost_model=None,
    )

    assert result["status"] == "error"
    assert "ghost_model" in result["error"]


@pytest.mark.asyncio
async def test_run_daily_digest_skipped_when_no_memories(sqlite_store, memory_manager):
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")

    result = await run_daily_digest(
        character_id=char_id,
        character_name="TestChar",
        character_system_prompt="You are TestChar.",
        target_date="2026-01-01",
        memory_manager=memory_manager,
        sqlite=sqlite_store,
        ghost_model="any-preset-id",
    )

    assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# run_forget_process: ghost_model propagated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_forget_process_error_when_no_ghost_model(sqlite_store, memory_manager):
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")

    # Create an old low-importance memory so candidates list is non-empty
    from datetime import datetime, timedelta
    sqlite_store.create_memory(
        memory_id=str(uuid.uuid4()),
        character_id=char_id,
        content="faint memory",
        contextual_importance=0.01,
        user_importance=0.01,
        semantic_importance=0.01,
        identity_importance=0.01,
    )
    with sqlite_store.get_session() as session:
        from backend.core.memory.sqlite_store import Memory
        m = session.query(Memory).filter_by(character_id=char_id).first()
        m.created_at = datetime.now() - timedelta(days=180)
        m.last_accessed_at = datetime.now() - timedelta(days=180)
        session.commit()

    result = await run_forget_process(
        character_id=char_id,
        character_name="TestChar",
        character_system_prompt="You are TestChar.",
        memory_manager=memory_manager,
        sqlite=sqlite_store,
        threshold=10.0,  # high threshold to ensure candidates exist
        ghost_model=None,
    )

    assert result["status"] == "error"
    assert "ghost_model" in result["error"]
