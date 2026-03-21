"""backend.core.system_prompt モジュールのテスト。

build_system_prompt() が各ブロックを正しく構築・配置することを検証する。
"""

from backend.core.system_prompt import build_system_prompt


def test_build_system_prompt_basic():
    """キャラクター設定が含まれ、記憶ブロックのヘッダーが存在すること。"""
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
    """【前提】ブロックは Block 1（キャラクター設定）の後に来ること。"""
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
    """recalled_memories が注入されたとき記憶内容と [INSCRIBE_MEMORY:...] ガイドが含まれること。"""
    char_prompt = "You are a cat."
    memories = [
        {"content": "User likes fish", "metadata": {"category": "user"}}
    ]
    prompt = build_system_prompt(char_prompt, recalled_memories=memories)

    assert "User likes fish" in prompt
    assert "[INSCRIBE_MEMORY:" in prompt


def test_build_system_prompt_memories_does_not_use_old_memory_tag():
    """古い [MEMORY:...] タグガイドがシステムプロンプトに含まれないこと（改名済み確認）。"""
    import re
    prompt = build_system_prompt(
        "You are a cat.",
        recalled_memories=[{"content": "内容", "metadata": {"category": "user"}}],
    )
    # [MEMORY:カテゴリ|... という古い形式が残っていないこと
    old_tag = re.compile(r'\[MEMORY:')
    assert not old_tag.search(prompt), (
        "システムプロンプトに古いタグ形式 [MEMORY:...] が残っています"
    )


def test_build_system_prompt_with_time():
    """時刻情報が正しく含まれること。"""
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
    """fetched_contents が注入されたとき Fetched Web Content ブロックが含まれること。"""
    char_prompt = "You are a cat."
    web = [{"url": "http://example.com", "content": "Example page"}]
    prompt = build_system_prompt(char_prompt, fetched_contents=web)

    assert "## Fetched Web Content" in prompt
    assert "Example page" in prompt


def test_build_system_prompt_with_inner_narrative():
    """inner_narrative が設定されているとき、専用ブロックとして含まれること。"""
    prompt = build_system_prompt(
        "You are a cat.",
        inner_narrative="知的好奇心を大切にしたい。",
    )
    assert "inner_narrative" in prompt
    assert "知的好奇心を大切にしたい。" in prompt


def test_build_system_prompt_with_carve_narrative_guide():
    """use_tools=False のとき CARVE_NARRATIVE タグガイドが含まれること。"""
    prompt = build_system_prompt("You are a cat.", use_tools=False)
    assert "[CARVE_NARRATIVE:" in prompt


def test_build_system_prompt_tools_mode_uses_tool_hint():
    """use_tools=True のとき carve_narrative ツール案内（ツール形式）が含まれること。"""
    prompt = build_system_prompt("You are a cat.", use_tools=True)
    assert "carve_narrative" in prompt
