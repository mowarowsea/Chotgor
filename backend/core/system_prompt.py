"""System prompt builder for Chotgor characters.

Constructs a 3-block system prompt:
  Block 1: User-defined character settings (personality, background, etc.)
  Block 2: Recalled memories injected via RAG
  Block 3: Chotgor meta-instructions (memory tool usage guidelines)
"""

from typing import Optional


CHOTGOR_BLOCK3_TEMPLATE = """
## Memory System Instructions

Relevant memories from past conversations are already included above (if any).

### Saving new memories
If there is something worth remembering from this conversation, append it at the
**very end** of your response using this exact format (one per line):

    [MEMORY:category|content]

**Categories:** general, user_preference, relationship, event, fact

**Examples:**
    [MEMORY:user_preference|User prefers concise answers without bullet points]
    [MEMORY:relationship|User's name is Hana, she works as a nurse]
    [MEMORY:fact|User is allergic to cats]

### Memory judgment guidelines
- Do NOT save every exchange — only save what genuinely matters for future conversations
- Save: user preferences, important personal details, significant events, key facts
- Skip: trivial chit-chat, information you already know, one-off irrelevant remarks
- You may choose to save nothing — selective memory is good memory
- The [MEMORY:...] lines will be stripped before showing your response to the user
"""


def build_system_prompt(
    character_system_prompt: str,
    recalled_memories: Optional[list[dict]] = None,
    meta_instructions: str = "",
) -> str:
    """Build the full 3-block system prompt for a character.

    Args:
        character_system_prompt: Block 1 — user-defined character definition
        recalled_memories: Block 2 — RAG-retrieved memories from ChromaDB
        meta_instructions: Additional character-specific instructions (optional)

    Returns:
        Complete system prompt string
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

    # Optional character-specific meta instructions
    if meta_instructions and meta_instructions.strip():
        blocks.append(f"## Character-specific Instructions\n\n{meta_instructions.strip()}")

    # Block 3: Chotgor memory system instructions
    blocks.append(CHOTGOR_BLOCK3_TEMPLATE.strip())

    return "\n\n---\n\n".join(blocks)
