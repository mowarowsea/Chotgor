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
from typing import AsyncIterator, Optional

from .debug_logger import log_llm_request, log_llm_response
from .memory.inscriber import carve
from .memory.manager import MemoryManager
from .providers.anthropic_provider import AnthropicProvider
from .providers.claude_cli_provider import ClaudeCliProvider
from .providers.google_provider import GoogleProvider
from .providers.openai_provider import OpenAIProvider
from .system_prompt import build_system_prompt
from .web_fetch import fetch_urls, find_urls


def _get_provider(provider: str, model: str, settings: dict, character_name: str = ""):
    """プロバイダー識別子から適切なプロバイダーインスタンスを返す。"""
    if provider == "anthropic":
        return AnthropicProvider(api_key=settings.get("anthropic_api_key", ""), model=model)
    elif provider == "openai":
        return OpenAIProvider(api_key=settings.get("openai_api_key", ""), model=model)
    elif provider == "xai":
        return OpenAIProvider(
            api_key=settings.get("xai_api_key", ""),
            model=model,
            base_url="https://api.x.ai/v1",
        )
    elif provider == "google":
        return GoogleProvider(api_key=settings.get("google_api_key", ""), model=model)
    else:
        return ClaudeCliProvider(model=model, character_name=character_name)


async def stream_chat(
    messages: list[dict],
    character_id: str,
    character_name: str,
    character_system_prompt: str,
    meta_instructions: str,
    memory_manager: MemoryManager,
    provider: str = "claude_cli",
    model: str = "",
    provider_additional_instructions: str = "",
    settings: Optional[dict] = None,
    enable_time_awareness: bool = False,
    current_time_str: str = "",
    time_since_last_interaction: str = "",
    **kwargs,
) -> AsyncIterator[str]:
    """LLMにディスパッチしてSSEチャンクをyieldする。"""
    if settings is None:
        settings = {}

    # --- 1. 記憶の想起 ---
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
        enable_time_awareness=enable_time_awareness,
        current_time_str=current_time_str,
        time_since_last_interaction=time_since_last_interaction,
    )

    # --- 4. プロバイダーへディスパッチ ---
    log_llm_request(system_prompt, messages)
    
    provider_impl = _get_provider(provider, model, settings, character_name)
    try:
        response_text = await provider_impl.generate(system_prompt, messages)
        log_llm_response(response_text)
    except Exception as e:
        import traceback
        yield _sse_chunk(f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]")
        yield "data: [DONE]\n\n"
        return

    # --- 5. 記憶を刻み込む ---
    clean_text = carve(response_text, character_id, memory_manager)

    # --- 6. デバッグログ ---
    char_label = character_name.strip() if character_name.strip() else "CHARACTER"
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"[CHAT] character={character_id}({char_label}) provider={provider} model={model or '(default)'}")
    for m in messages:
        role = m.get("role", "")
        display_role = char_label if role == "assistant" else role.upper()
        content = m.get("content", "")
        print(f"  [{display_role}] {content[:300]}{'...' if len(content) > 300 else ''}")
    print(f"  [{char_label}] {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")
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
