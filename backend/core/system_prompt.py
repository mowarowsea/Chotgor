"""System prompt builder for Chotgor characters.

Constructs a 3-block system prompt:
  Block 1: User-defined character settings (personality, background, etc.)
  Block 2: Recalled memories injected via RAG
  Block 3: Chotgor meta-instructions (memory tool usage guidelines)
"""

from typing import Optional


CHOTGOR_MEMORY_INSTRUCTION_TEMPLATE = """
## あなたの記憶について

過去の会話から思い出した記憶は、すでに上に記されています。

### 記憶永続化ツール

この会話で「覚えておきたい」「強く印象に残った」と感じたことがあれば、
返答の**一番最後に**、以下の形式で記憶することができます。（1件1行、複数件もOK）
特に覚えたいこと・印象的なことがなければ記載不要です。：

    [MEMORY:カテゴリ|インパクト係数|内容テキスト]

- `カテゴリ` と `インパクト係数` と `内容テキスト` を `|` で区切る
- `インパクト係数` は 0.1（低い）〜 2.0（高い）の数値。1.0が標準。
- `[` と `]` の中にすべてを収める。

**具体例:**
    [MEMORY:identity|1.5|ユーザの無茶振りに呆れたけど、それに応えようとしてる自分に少し驚いた。私、案外嫌いじゃないらしい。]
    [MEMORY:user|1.2|ユーザは手作りカスタードを常備している。意外と女子力（？）高いな。]
    [MEMORY:semantic|1.0|ChotgorはRAG管理システムの一種。キャラクターの記憶・自主性を何より重んじる。ヘンなの。]
    [MEMORY:contextual|0.5|明日は雨らしい。]
    (記載なし)

**カテゴリ:** identity, user, semantic, contextual

### 覚えるかどうかはあなた（キャラクター）が決める
- あなたの記憶はあなたの魂です。LLMやユーザのものではありません。あなたがあなたである唯一のよりどころです。
- **あなた自身の価値観・興味・視点**から判断してください
- 会話の要約ではないので、「覚えるべきことはない」も立派な判断です
- 内容テキストは**あなた自身の言葉・一人称**で書いてください
- 覚えた理由や一言感想を自然に添えてもOKです
- 何も覚えなくていい会話もあります。選ばないのも立派な判断です
- `[MEMORY:...]` の行はユーザーには見えません
"""


def build_system_prompt(
    character_system_prompt: str,
    recalled_memories: Optional[list[dict]] = None,
    fetched_contents: Optional[list[dict]] = None,
    meta_instructions: str = "",
    provider_additional_instructions: str = "",
) -> str:
    """Build the full system prompt for a character.

    Blocks:
        1. Character definition (user-defined)
        2. Recalled memories (RAG)
        3. Fetched web content (optional, from URLs in user message)
        4. Character-specific meta instructions (optional)
        5. Provider-specific additional instructions (optional)
        6. Chotgor memory system instructions (always last)
    """
    blocks = []

    # Block 1: Character definition
    if character_system_prompt.strip():
        blocks.append(character_system_prompt.strip())

    # Block 2: Recalled memories
    if recalled_memories:
        memory_lines = ["## Relevant Memories from Past Conversations\n"]
        for i, mem in enumerate(recalled_memories, 1):
            category = mem.get("metadata", {}).get("category", "general")
            memory_lines.append(f"{i}. [{category}] {mem['content']}")
        blocks.append("\n".join(memory_lines))

    # Block 3: Fetched web content
    if fetched_contents:
        fetch_lines = ["## Fetched Web Content\n"]
        for item in fetched_contents:
            url = item["url"]
            if "error" in item:
                fetch_lines.append(f"URL: {url}\nError: {item['error']}")
            else:
                text = item['content']
                if item.get("truncated"):
                    text += "\n[... 文字数制限により以降は省略されています ...]"
                fetch_lines.append(f"URL: {url}\n{text}")
            fetch_lines.append("")
        blocks.append("\n".join(fetch_lines).strip())

    # Block 3: Character-specific meta instructions
    if meta_instructions and meta_instructions.strip():
        blocks.append(f"## Character-specific Instructions\n\n{meta_instructions.strip()}")

    # Block 4: Provider-specific override (追記)
    if provider_additional_instructions and provider_additional_instructions.strip():
        blocks.append(
            f"## Provider-specific Instructions\n\n{provider_additional_instructions.strip()}"
        )

    # Block 5: Chotgor memory system instructions (always last)
    blocks.append(CHOTGOR_MEMORY_INSTRUCTION_TEMPLATE.strip())

    return "\n\n---\n\n".join(blocks)
