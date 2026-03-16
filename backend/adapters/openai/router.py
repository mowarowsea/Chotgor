"""OpenAI互換 API エンドポイント。

GET  /v1/models          - 利用可能なモデル一覧 (character@preset_id)
POST /v1/chat/completions - チャット (streaming SSE / non-streaming)
"""

import hashlib
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...core.chat.models import ChatRequest, Message
from ...core.chat.service import extract_text_content
from ...core.debug_logger import log_front_input
from ...core.providers.registry import PROVIDER_LABELS
from ...core.utils import format_time_delta
from .schemas import OAIChatRequest

router = APIRouter()


def _derive_session_id(character_id: str, messages: list) -> str:
    """会話の最初のuserメッセージ + character_idからsession_idを導出する。

    OpenWebUIはsession_idを送ってこないため、同じ会話スレッドを識別するための
    擬似session_idを生成する。会話の先頭userメッセージが同じ場合は同じIDになる。
    """
    first_user = next((m for m in messages if m.role == "user"), None)
    if not first_user:
        return ""
    seed = f"{character_id}:{extract_text_content(first_user.content)}"
    return hashlib.md5(seed.encode()).hexdigest()


def _available_providers(settings: dict) -> set[str]:
    """APIキーが設定済みのプロバイダーを返す (claude_cli は常に含む)。"""
    result = {"claude_cli"}
    for p in ("anthropic", "openai", "xai", "google"):
        if settings.get(f"{p}_api_key"):
            result.add(p)
    return result


def _sse_chunk(text: str) -> str:
    """通常テキストのSSEチャンクを生成する。delta.content に格納する。"""
    payload = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_chunk_reasoning(text: str) -> str:
    """思考ブロック・想起記憶用のSSEチャンクを生成する。

    delta.reasoning_content に格納することで、OpenWebUI が
    「思考」として折りたたみ表示する（DeepSeek R1 と同仕様）。
    """
    payload = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _format_memories_display(recalled: list) -> str:
    """想起した記憶リストを思考ブロック内表示用テキストにフォーマットする。"""
    from ...core.memory.format import format_recalled_memories
    return format_recalled_memories(recalled)


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
                "id": f"{char.name}@{preset.name}",
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

    # UUID優先、なければ名前で検索
    character = state.sqlite.get_character(char_id) or state.sqlite.get_character_by_name(char_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{char_id}' not found")

    # preset_idはUUIDか名前かを判断してルックアップ
    preset = state.sqlite.get_model_preset(preset_id) or state.sqlite.get_model_preset_by_name(preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Model preset '{preset_id}' not found")

    model_config = (character.enabled_providers or {}).get(preset.id)
    if model_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model preset '{preset_id}' is not enabled for this character",
        )

    settings = state.sqlite.get_all_settings()

    # --- 時刻計算 ---
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
                time_since_last_interaction = format_time_delta(now - last_dt)
            except Exception:
                pass

    if body.messages and body.messages[-1].role == "user":
        state.sqlite.set_setting(f"last_interaction_{character.id}", now.isoformat())

    messages = [Message(role=m.role, content=m.content) for m in body.messages]
    session_id = _derive_session_id(character.id, messages)

    chat_request = ChatRequest(
        character_id=character.id,
        character_name=character.name,
        provider=preset.provider,
        model=preset.model_id,
        messages=messages,
        character_system_prompt=character.system_prompt_block1,
        meta_instructions=character.meta_instructions,
        provider_additional_instructions=model_config.get("additional_instructions", ""),
        thinking_level=preset.thinking_level or "default",
        settings=settings,
        enable_time_awareness=enable_time_awareness,
        current_time_str=current_time_str,
        time_since_last_interaction=time_since_last_interaction,
        session_id=session_id,
    )

    chat_service = state.chat_service

    if body.stream:
        async def generate():
            """型付きチャンクを OpenAI SSE 形式に変換して送信する。

            - ("memories", list) → reasoning_content として記憶一覧を送信
            - ("thinking", str)  → reasoning_content として思考ブロックを送信
            - ("text", str)      → content として応答テキストを送信
            """
            async for chunk_type, content in chat_service.execute_stream(chat_request):
                if chunk_type == "memories":
                    display = _format_memories_display(content)
                    if display:
                        yield _sse_chunk_reasoning(display)
                elif chunk_type == "thinking":
                    if content:
                        yield _sse_chunk_reasoning(content)
                elif chunk_type == "text":
                    if content:
                        yield _sse_chunk(content)
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
