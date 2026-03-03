"""LLM dispatcher and SSE streaming layer.

Receives a chat request, recalls memories, builds the system prompt,
dispatches to the appropriate LLM provider, extracts memory markers,
saves new memories, and returns an SSE stream.

Architecture:
  OpenWebUI → Chotgor(host) → LLM provider

Memory flow:
  - Pre-chat: ChromaDB similarity search → injected into system prompt (Block 2)
  - Post-chat: [MEMORY:category|content] markers in response → saved to ChromaDB + SQLite
"""

import json
import re
from typing import AsyncIterator, Optional

from .memory.manager import MemoryManager
from .providers.anthropic_provider import AnthropicProvider
from .providers.claude_cli_provider import ClaudeCliProvider
from .providers.google_provider import GoogleProvider
from .providers.openai_provider import OpenAIProvider
from .system_prompt import build_system_prompt

# [MEMORY:category|content] canonical form
MEMORY_PATTERN = re.compile(r"\[MEMORY:(\w+)\|([^\]]+)\]", re.DOTALL)
# Fallback: [MEMORY:category] content (no pipe, content outside bracket)
MEMORY_PATTERN_FALLBACK = re.compile(
    r"\[MEMORY:(\w+)\]\s*(.+?)(?=\[MEMORY:|\Z)", re.DOTALL
)


def _get_provider(provider: str, model: str, settings: dict):
    """Return the appropriate LLM provider instance."""
    if provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.get("anthropic_api_key", ""),
            model=model,
        )
    elif provider == "openai":
        return OpenAIProvider(
            api_key=settings.get("openai_api_key", ""),
            model=model,
        )
    elif provider == "xai":
        return OpenAIProvider(
            api_key=settings.get("xai_api_key", ""),
            model=model,
            base_url="https://api.x.ai/v1",
        )
    elif provider == "google":
        return GoogleProvider(
            api_key=settings.get("google_api_key", ""),
            model=model,
        )
    else:
        # Default: claude_cli
        return ClaudeCliProvider(model=model)


async def stream_chat(
    messages: list[dict],
    character_id: str,
    character_system_prompt: str,
    meta_instructions: str,
    memory_manager: MemoryManager,
    provider: str = "claude_cli",
    model: str = "",
    provider_additional_instructions: str = "",
    settings: Optional[dict] = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Dispatch to LLM provider and return SSE chunks."""
    if settings is None:
        settings = {}

    # --- 1. Pre-recall relevant memories ---
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    recalled = []
    if last_user_msg:
        try:
            recalled = memory_manager.recall_memory(character_id, last_user_msg)
        except Exception:
            pass

    # --- 2. Build system prompt (3-block structure) ---
    system_prompt = build_system_prompt(
        character_system_prompt=character_system_prompt,
        recalled_memories=recalled,
        meta_instructions=meta_instructions,
        provider_additional_instructions=provider_additional_instructions,
    )

    # --- 3. Dispatch to provider ---
    provider_impl = _get_provider(provider, model, settings)
    try:
        response_text = await provider_impl.generate(system_prompt, messages)
    except Exception as e:
        import traceback
        yield _sse_chunk(f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]")
        yield "data: [DONE]\n\n"
        return

    # --- 4. Extract [MEMORY:...] markers and save ---
    clean_text, memories = _extract_memories(response_text)

    for category, content in memories:
        try:
            memory_manager.write_memory(
                character_id=character_id,
                content=content.strip(),
                category=category.strip(),
            )
        except Exception:
            pass

    # --- 5. Debug log ---
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"[CHAT] character={character_id} provider={provider} model={model or '(default)'}")
    for m in messages:
        role = m.get("role", "?").upper()
        content = m.get("content", "")
        print(f"  [{role}] {content[:300]}{'...' if len(content) > 300 else ''}")
    print(f"  [ASSISTANT] {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")
    print(sep, flush=True)

    # --- 6. Return SSE ---
    if clean_text:
        yield _sse_chunk(clean_text)
    yield "data: [DONE]\n\n"


def _extract_memories(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Extract [MEMORY:category|content] markers; return (cleaned_text, memories)."""
    memories = MEMORY_PATTERN.findall(text)
    clean = MEMORY_PATTERN.sub("", text).strip()

    # Fallback: no pipe-form but [MEMORY:xxx] remnants exist
    if not memories and "[MEMORY:" in clean:
        fb_memories = MEMORY_PATTERN_FALLBACK.findall(clean)
        if fb_memories:
            memories = [(cat, content.strip()) for cat, content in fb_memories]
            clean = MEMORY_PATTERN_FALLBACK.sub("", clean).strip()

    return clean, memories


def _sse_chunk(text: str) -> str:
    payload = {
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": text}, "finish_reason": None}
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
