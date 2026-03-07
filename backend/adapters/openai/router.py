"""OpenAI互換 API エンドポイント。

GET  /v1/models          - 利用可能なモデル一覧 (character@preset_id)
POST /v1/chat/completions - チャット (streaming SSE / non-streaming)
"""

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...core.chat.models import ChatRequest, Message
from ...core.debug_logger import log_front_input
from ...core.providers.registry import PROVIDER_LABELS
from .schemas import OAIChatRequest

router = APIRouter()


def _available_providers(settings: dict) -> set[str]:
    """APIキーが設定済みのプロバイダーを返す (claude_cli は常に含む)。"""
    result = {"claude_cli"}
    for p in ("anthropic", "openai", "xai", "google"):
        if settings.get(f"{p}_api_key"):
            result.add(p)
    return result


def _sse_chunk(text: str) -> str:
    payload = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _format_completion(model: str, text: str) -> dict:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


@router.get("/v1/models")
async def list_models(request: Request):
    """利用可能な character@preset_id の組み合わせをモデル一覧として返す。"""
    state = request.app.state
    settings = state.sqlite.get_all_settings()
    available = _available_providers(settings)
    characters = state.sqlite.list_characters()

    data = []
    for char in characters:
        for preset_id in (char.enabled_providers or {}):
            preset = state.sqlite.get_model_preset(preset_id)
            if preset is None or preset.provider not in available:
                continue
            label = PROVIDER_LABELS.get(preset.provider, preset.provider)
            data.append({
                "id": f"{char.id}@{preset_id}",
                "object": "model",
                "created": int(char.created_at.timestamp()) if char.created_at else 0,
                "owned_by": "chotgor",
                "name": f"{char.name} ({preset.name})",
            })
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: OAIChatRequest):
    """OpenAI互換チャット補完エンドポイント。"""
    state = request.app.state

    log_front_input(body.model_dump())

    if "@" not in body.model:
        raise HTTPException(
            status_code=400,
            detail="Model must be in format {character_id}@{preset_id}",
        )

    char_id, preset_id = body.model.rsplit("@", 1)
    character = state.sqlite.get_character(char_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{char_id}' not found")

    model_config = (character.enabled_providers or {}).get(preset_id)
    if model_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model preset '{preset_id}' is not enabled for this character",
        )

    preset = state.sqlite.get_model_preset(preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Model preset '{preset_id}' not found")

    settings = state.sqlite.get_all_settings()

    chat_request = ChatRequest(
        character_id=character.id,
        character_name=character.name,
        provider=preset.provider,
        model=preset.model_id,
        messages=[Message(role=m.role, content=m.content) for m in body.messages],
        character_system_prompt=character.system_prompt_block1,
        meta_instructions=character.meta_instructions,
        provider_additional_instructions=model_config.get("additional_instructions", ""),
        settings=settings,
    )

    chat_service = state.chat_service

    if body.stream:
        async def generate():
            text = await chat_service.execute(chat_request)
            if text:
                yield _sse_chunk(text)
            yield "data: [DONE]\n\n"

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
        text = await chat_service.execute(chat_request)
        return _format_completion(body.model, text)
