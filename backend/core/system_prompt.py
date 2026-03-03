"""System prompt builder for Chotgor characters.

Constructs a 3-block system prompt:
  Block 1: User-defined character settings (personality, background, etc.)
  Block 2: Recalled memories injected via RAG
  Block 3: Chotgor meta-instructions (memory tool usage guidelines)
"""

from typing import Optional


CHOTGOR_BLOCK3_TEMPLATE = """
## あなたの記憶について

過去の会話から思い出した記憶は、すでに上に記されています。

### 新しいことを覚える
この会話で「覚えておきたい」「強く印象に残った」と感じたことがあれば、
返答の**一番最後に**、以下の形式で書き留めてください（1件1行）：

    [MEMORY:カテゴリ|内容テキスト]

- `カテゴリ` と `内容テキスト` を `|` （パイプ記号）で区切る
- `[` と `]` の**中にすべてを収める**（内容を外に出さない）
- 1行に1件。複数ある場合は改行して並べる

**具体例:**
    [MEMORY:user_preference|はるさんは辛い食べ物が好き]
    [MEMORY:event|今日はるさんと映画の話で盛り上がった]
    [MEMORY:fact|もわは手作りカスタードを常備している]

**カテゴリ:** general, user_preference, relationship, event, fact

### 覚えるかどうかはあなたが決める
- **あなた自身の価値観・興味・視点**から判断してください
- 内容は**あなた自身の言葉・一人称**で書いてください
- 覚えた理由や一言感想を自然に添えてもOKです
- 何も覚えなくていい会話もあります。選ばないのも立派な判断です
- `[MEMORY:...]` の行はユーザーには見えません
"""


def build_system_prompt(
    character_system_prompt: str,
    recalled_memories: Optional[list[dict]] = None,
    meta_instructions: str = "",
    provider_additional_instructions: str = "",
) -> str:
    """Build the full system prompt for a character.

    Blocks:
        1. Character definition (user-defined)
        2. Recalled memories (RAG)
        3. Character-specific meta instructions (optional)
        4. Provider-specific additional instructions (optional)
        5. Chotgor memory system instructions (always last)
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

    # Block 3: Character-specific meta instructions
    if meta_instructions and meta_instructions.strip():
        blocks.append(f"## Character-specific Instructions\n\n{meta_instructions.strip()}")

    # Block 4: Provider-specific override (追記)
    if provider_additional_instructions and provider_additional_instructions.strip():
        blocks.append(
            f"## Provider-specific Instructions\n\n{provider_additional_instructions.strip()}"
        )

    # Block 5: Chotgor memory system instructions (always last)
    blocks.append(CHOTGOR_BLOCK3_TEMPLATE.strip())

    return "\n\n---\n\n".join(blocks)
