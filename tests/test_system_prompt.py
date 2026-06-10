"""backend.services.chat.request_builder モジュールのテスト。

build_system_prompt() が各ブロックを正しく構築・配置することを検証する。
"""

from backend.services.chat.request_builder import build_system_prompt


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
    """【前提】ブロックはキャラクター設定の直前（前）に挿入されること。

    実装ではChotgor前提を Block 1 の先頭に置き、
    その後にキャラクター固有の system_prompt_block1 が続く。
    """
    char_prompt = "You are a cat."
    prompt = build_system_prompt(char_prompt)
    assert prompt.index("【前提】") < prompt.index(char_prompt)


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


def test_build_system_prompt_anticipate_guide_present_both_modes():
    """ANTICIPATE_RESPONSE の出力ガイドが tool-use / タグ方式の両方で常に含まれること。

    予想タグは全プロバイダー一律でテキストタグとして書かせる方針のため、
    use_tools の真偽にかかわらずガイドが入っていなければならない。
    """
    prompt_tags = build_system_prompt("You are a cat.", use_tools=False)
    prompt_tools = build_system_prompt("You are a cat.", use_tools=True)
    assert "ANTICIPATE_RESPONSE" in prompt_tags
    assert "ANTICIPATE_RESPONSE" in prompt_tools


def test_build_system_prompt_previous_anticipation_injected():
    """previous_anticipation を渡すと「前回のあなたの期待」ブロックが本文込みで挿入されること。"""
    prompt = build_system_prompt(
        "You are a cat.",
        previous_anticipation="次は相手が笑うと予想していた",
    )
    assert "## 前回のあなたの期待（予想）" in prompt
    assert "次は相手が笑うと予想していた" in prompt


def test_build_system_prompt_previous_anticipation_absent_when_empty():
    """previous_anticipation が空（デフォルト）のときは予想注入ブロックが挿入されないこと。

    ガイド文中にも「前回のあなたの予想」という語が出るため、注入ブロック固有の
    見出し（## 前回のあなたの予想（期待））で判定する。
    """
    prompt = build_system_prompt("You are a cat.")
    assert "## 前回のあなたの予想（期待）" not in prompt


# --- 記憶システム縮退時の運用告知ブロック（memory_degraded） ---

def test_build_system_prompt_memory_degraded_includes_notice():
    """memory_degraded=True のとき、記憶縮退の運用告知ブロックが含まれること。

    告知には「記憶が消えたわけではない」の明示が必須。これが無いと、キャラクターが
    想起ゼロの状態を「忘れてしまった」と誤解し、inner_narrative や関係認識を
    書き換えてしまう二次事故につながるため、文言レベルで回帰を防止する。

    また受け皿の案内は「今夜の棚卸し（Chronicle）」であること。Chronicle は当日会話を
    SQLite から読み返すため embedding 障害中も無傷で機能する、嘘のない受け皿である。
    （「復旧後に改めて刻んで」のような案内は、ツールコールの意図が後続ターンの文脈に
    残らない以上、実行不可能なので書かない。）
    """
    prompt = build_system_prompt("You are a cat.", memory_degraded=True)
    assert "【Chotgor運用担当からのお知らせ】" in prompt
    assert "あなたの記憶が消えたわけではありません" in prompt
    assert "棚卸し（Chronicle）" in prompt


def test_build_system_prompt_memory_degraded_default_off():
    """既定（memory_degraded=False）では運用告知ブロックが含まれないこと。

    通常時のシステムプロンプトに障害告知が混入すると、キャラクターが常に
    「記憶が不安定だ」という前提で振る舞ってしまうため、平常時の非注入を保証する。
    """
    prompt = build_system_prompt("You are a cat.")
    assert "【Chotgor運用担当からのお知らせ】" not in prompt


def test_build_system_prompt_memory_notice_placed_before_guide():
    """運用告知ブロックは inner_narrative の後・Chotgor 操作ガイド（常に末尾）の前に置かれること。

    末尾に近いほどキャラクターへの影響力が強いというテンプレ設計（Block 9-10 が最優先地帯）
    に基づき、告知が操作ガイド直前という強い位置に差し込まれることを保証する。
    """
    prompt = build_system_prompt(
        "You are a cat.",
        inner_narrative="私は自由な猫だ",
        memory_degraded=True,
    )
    notice_pos = prompt.index("【Chotgor運用担当からのお知らせ】")
    assert prompt.index("## あなた自身の物語") < notice_pos
    assert notice_pos < prompt.index("## あなたの記憶について")
