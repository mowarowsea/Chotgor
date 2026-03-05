import pytest
import os
import tempfile
from backend.core.memory.sqlite_store import SQLiteStore

@pytest.fixture
def sqlite_store():
    """Provides a fresh temporary SQLite storage for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(path)
    yield store
    # Cleanup
    store.engine.dispose()
    try:
        os.remove(path)
    except PermissionError:
        pass
