"""OpenAI-compatible API endpoints.

GET  /v1/models          - List available models (characters)
POST /v1/chat/completions - Chat with streaming SSE
"""

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..core.claude_wrapper import stream_chat
from ..core.memory.manager import MemoryManager

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str  # Used as character_id
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = None  # Ignored — Claude handles this


def get_app_state(request: Request):
    return request.app.state


@router.get("/v1/models")
async def list_models(request: Request):
    """Return all characters as available models."""
    state = get_app_state(request)
    characters = state.sqlite.list_characters()
    return {
        "object": "list",
        "data": [
            {
                "id": char.id,
                "object": "model",
                "created": int(char.created_at.timestamp()) if char.created_at else 0,
                "owned_by": "chotgor",
                "name": char.name,
            }
            for char in characters
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    state = get_app_state(request)

    character = state.sqlite.get_character(body.model)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{body.model}' not found")

    settings = state.sqlite.get_all_settings()
    tavily_key = settings.get("tavily_api_key")

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        async def generate():
            async for chunk in stream_chat(
                messages=messages,
                character_id=character.id,
                character_system_prompt=character.system_prompt_block1,
                meta_instructions=character.meta_instructions,
                memory_manager=state.memory_manager,
                tavily_api_key=tavily_key,
            ):
                yield chunk

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
        # Non-streaming: collect all chunks
        full_text = ""
        async for chunk in stream_chat(
            messages=messages,
            character_id=character.id,
            character_system_prompt=character.system_prompt_block1,
            meta_instructions=character.meta_instructions,
            memory_manager=state.memory_manager,
            tavily_api_key=tavily_key,
        ):
            if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
                import json
                try:
                    data = json.loads(chunk[6:])
                    delta = data["choices"][0]["delta"].get("content", "")
                    full_text += delta
                except Exception:
                    pass

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
