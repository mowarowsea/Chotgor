"""OpenAI-compatible API endpoints.

GET  /v1/models          - List available models (character@provider combinations)
POST /v1/chat/completions - Chat with streaming SSE
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.debug_logger import log_front_input, log_front_output
from ..core.llm_service import stream_chat
from .schemas import ChatCompletionRequest, ChatMessage

router = APIRouter()

PROVIDER_LABELS = {
    "claude_cli": "Claude CLI",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "xai": "xAI",
    "google": "Google",
}


def _available_providers(settings: dict) -> set[str]:
    """Return providers that have API keys configured (claude_cli always included)."""
    result = {"claude_cli"}
    for p in ("anthropic", "openai", "xai", "google"):
        if settings.get(f"{p}_api_key"):
            result.add(p)
    return result


# remove models from here


@router.get("/v1/models")
async def list_models(request: Request):
    """Return all enabled character@provider combinations as model entries."""
    state = request.app.state
    settings = state.sqlite.get_all_settings()
    available = _available_providers(settings)
    characters = state.sqlite.list_characters()

    data = []
    for char in characters:
        for provider, config in (char.enabled_providers or {}).items():
            if provider not in available:
                continue
            label = PROVIDER_LABELS.get(provider, provider)
            data.append({
                "id": f"{char.id}@{provider}",
                "object": "model",
                "created": int(char.created_at.timestamp()) if char.created_at else 0,
                "owned_by": "chotgor",
                "name": f"{char.name} ({label})",
            })
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    log_front_input(body.model_dump())
    state = request.app.state

    if "@" not in body.model:
        raise HTTPException(
            status_code=400,
            detail="Model must be in format {character_id}@{provider}",
        )

    char_id, provider = body.model.rsplit("@", 1)
    character = state.sqlite.get_character(char_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{char_id}' not found")

    provider_config = (character.enabled_providers or {}).get(provider)
    if provider_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider}' is not enabled for this character",
        )

    settings = state.sqlite.get_all_settings()
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    from datetime import datetime
    now = datetime.now()

    enable_time_awareness = settings.get("enable_time_awareness", "true") == "true"
    current_time_str = ""
    time_since_last_interaction = ""

    if enable_time_awareness:
        current_time_str = now.isoformat(timespec="seconds")
        last_str = settings.get(f"last_interaction_{character.id}")
        if last_str:
            try:
                last_dt = datetime.fromisoformat(last_str)
                diff = now - last_dt
                hours = diff.total_seconds() / 3600
                if hours < 1:
                    m = int(hours * 60)
                    time_since_last_interaction = f"約 {m} 分" if m > 0 else "数分以内"
                elif hours < 24:
                    time_since_last_interaction = f"約 {hours:.1f} 時間"
                else:
                    days = int(hours / 24)
                    time_since_last_interaction = f"約 {days} 日"
            except Exception:
                pass

    if messages and messages[-1]["role"] == "user":
        state.sqlite.set_setting(f"last_interaction_{character.id}", now.isoformat())

    chat_kwargs = dict(
        messages=messages,
        character_id=character.id,
        character_name=character.name,
        character_system_prompt=character.system_prompt_block1,
        meta_instructions=character.meta_instructions,
        memory_manager=state.memory_manager,
        provider=provider,
        model=provider_config.get("model", ""),
        provider_additional_instructions=provider_config.get("additional_instructions", ""),
        settings=settings,
        enable_time_awareness=enable_time_awareness,
        current_time_str=current_time_str,
        time_since_last_interaction=time_since_last_interaction,
    )

    if body.stream:
        async def generate():
            full_text = ""
            async for chunk in stream_chat(**chat_kwargs):
                if chunk.startswith("data: ") and not chunk.endswith("[DONE]\n\n"):
                    try:
                        import json
                        data = json.loads(chunk[6:])
                        delta = data["choices"][0]["delta"].get("content", "")
                        full_text += delta
                    except Exception:
                        pass
                yield chunk
            log_front_output(full_text)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        full_text = ""
        async for chunk in stream_chat(**chat_kwargs):
            if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
                import json
                try:
                    data = json.loads(chunk[6:])
                    delta = data["choices"][0]["delta"].get("content", "")
                    full_text += delta
                except Exception:
                    pass

        log_front_output(full_text)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }
