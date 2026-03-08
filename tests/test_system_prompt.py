from backend.core.system_prompt import build_system_prompt

def test_build_system_prompt_basic():
    char_prompt = "You are a cat."
    prompt = build_system_prompt(char_prompt)

    assert char_prompt in prompt
    assert "## あなたの記憶について" in prompt


def test_build_system_prompt_includes_chotgor_context():
    """【前提】ブロックは常に含まれること。"""
    prompt = build_system_prompt("You are a cat.")
    assert "【前提】" in prompt
    assert "Chotgor" in prompt


def test_build_system_prompt_chotgor_context_after_block1():
    """【前提】ブロックは Block 1 の後に来ること。"""
    char_prompt = "You are a cat."
    prompt = build_system_prompt(char_prompt)
    assert prompt.index(char_prompt) < prompt.index("【前提】")


def test_build_system_prompt_time_disabled_by_default():
    """enable_time_awareness=False のとき時刻情報は含まれないこと。"""
    prompt = build_system_prompt("You are a cat.", current_time_str="2026-03-08T12:00:00")
    assert "【現在時刻" not in prompt


def test_build_system_prompt_time_requires_current_time_str():
    """enable_time_awareness=True でも current_time_str が空なら時刻ブロックは入らないこと。"""
    prompt = build_system_prompt(
        "You are a cat.",
        enable_time_awareness=True,
        current_time_str="",
    )
    assert "【現在時刻" not in prompt


def test_build_system_prompt_time_no_last_interaction():
    """前回の経過時間がない場合は現在時刻のみ含まれること。"""
    prompt = build_system_prompt(
        "You are a cat.",
        enable_time_awareness=True,
        current_time_str="2026-03-08T12:00:00",
    )
    assert "【現在時刻：2026-03-08T12:00:00】" in prompt
    assert "【前回の交流から" not in prompt

def test_build_system_prompt_with_memories():
    char_prompt = "You are a cat."
    memories = [
        {"content": "User likes fish", "metadata": {"category": "user"}}
    ]
    prompt = build_system_prompt(char_prompt, recalled_memories=memories)
    
    assert "User likes fish" in prompt
    assert "[MEMORY:カテゴリ|インパクト係数|内容テキスト]" in prompt

def test_build_system_prompt_with_time():
    char_prompt = "You are a cat."
    prompt = build_system_prompt(
        char_prompt, 
        enable_time_awareness=True, 
        current_time_str="2026-03-05 12:00",
        time_since_last_interaction="1 hour"
    )
    
    assert "【現在時刻：2026-03-05 12:00】" in prompt
    assert "【前回の交流から：1 hour】" in prompt

def test_build_system_prompt_with_web():
    char_prompt = "You are a cat."
    web = [{"url": "http://example.com", "content": "Example page"}]
    prompt = build_system_prompt(char_prompt, fetched_contents=web)
    
    assert "## Fetched Web Content" in prompt
    assert "Example page" in prompt
