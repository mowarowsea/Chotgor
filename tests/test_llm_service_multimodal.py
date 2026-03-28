import pytest
from backend.services.chat.service import extract_text_content

def test_extract_text_content_string():
    assert extract_text_content("hello") == "hello"

def test_extract_text_content_none():
    assert extract_text_content(None) == ""

def test_extract_text_content_list():
    content = [
        {"type": "text", "text": "Hello "},
        {"type": "image_url", "image_url": {"url": "..."}},
        "world"
    ]
    # "Hello " (from dict) + "world" (from str)
    assert extract_text_content(content) == "Hello world"

def test_extract_text_content_empty_list():
    assert extract_text_content([]) == ""
