"""LLM Service — リクエスト受付からSSE返却までのサービス層。

Architecture:
  OpenWebUI → Chotgor(host) → LLM provider

Flow:
  1. 記憶を想起 (ChromaDB RAG)
  2. URL自動fetch
  3. システムプロンプト構築
  4. プロバイダーへディスパッチ
  5. 応答から記憶を刻み込み (Inscriber.carve)
  6. SSEとして返却
"""

import json
from typing import AsyncIterator, Optional, Union

from .memory.inscriber import carve
from .memory.manager import MemoryManager
from .providers.anthropic_provider import AnthropicProvider
from .providers.claude_cli_provider import ClaudeCliProvider
from .providers.google_provider import GoogleProvider
from .providers.openai_provider import OpenAIProvider
from .system_prompt import build_system_prompt
from .web_fetch import fetch_urls, find_urls


def _get_provider(provider: str, model: str, settings: dict):
    """プロバイダー識別子から適切なプロバイダーインスタンスを返す。"""
    # ... (unchanged)

def extract_text_content(content: Union[str, list, None]) -> str:
    """メッセージの content (str or list) からプレーンテキストのみを抽出する。"""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


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
    """LLMにディスパッチしてSSEチャンクをyieldする。"""
    if settings is None:
        settings = {}

    # --- 1. 記憶の想起 ---
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = extract_text_content(m.get("content"))
            break

    recalled = []
    if last_user_msg:
        try:
            recalled = memory_manager.recall_memory(character_id, last_user_msg)
        except Exception:
            pass

    # --- 2. URLの自動fetch ---
    fetched_contents = []
    if last_user_msg:
        urls = find_urls(last_user_msg)
        if urls:
            try:
                fetched_contents = await fetch_urls(urls)
            except Exception:
                pass

    # --- 3. システムプロンプト構築 ---
    system_prompt = build_system_prompt(
        character_system_prompt=character_system_prompt,
        recalled_memories=recalled,
        fetched_contents=fetched_contents,
        meta_instructions=meta_instructions,
        provider_additional_instructions=provider_additional_instructions,
    )

    # --- 4. プロバイダーへディスパッチ ---
    provider_impl = _get_provider(provider, model, settings)
    try:
        response_text = await provider_impl.generate(system_prompt, messages)
    except Exception as e:
        import traceback
        yield _sse_chunk(f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]")
        yield "data: [DONE]\n\n"
        return

    # --- 5. 記憶を刻み込む ---
    clean_text = carve(response_text, character_id, memory_manager)

    # --- 6. デバッグログ ---
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"[CHAT] character={character_id} provider={provider} model={model or '(default)'}")
    for m in messages:
        role = m.get("role", "?").upper()
        content = extract_text_content(m.get("content"))
        print(f"  [{role}] {content[:300]}{'...' if len(content) > 300 else ''}")
    print(f"  [ASSISTANT] {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")
    print(sep, flush=True)

    # --- 7. SSEとして返却 ---
    if clean_text:
        yield _sse_chunk(clean_text)
    yield "data: [DONE]\n\n"


def _sse_chunk(text: str) -> str:
    payload = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
