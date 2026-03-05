from backend.core.memory.inscriber import carve, _extract
from unittest.mock import MagicMock

def test_extract_memories():
    text = "Hello! [MEMORY:user|0.8|User is happy]\nHow are you?"
    clean, mems = _extract(text)
    
    assert len(mems) == 1
    assert mems[0][0] == "user"
    assert mems[0][1] == "0.8"
    assert mems[0][2] == "User is happy"

def test_carve_memories():
    text = "Hello! [MEMORY:user|0.8|User is happy]\nHow are you?"
    memory_manager = MagicMock()
    
    clean_text = carve(text, "char-1", memory_manager)
    
    assert "[MEMORY:" not in clean_text
    assert "Hello!" in clean_text
    assert "How are you?" in clean_text
    memory_manager.write_memory.assert_called_once()
