"""OpenAI互換 API エンドポイント。

GET  /v1/models          - 利用可能なモデル一覧 (character@preset_id)
POST /v1/chat/completions - チャット (streaming SSE / non-streaming)
"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.api.resource_resolver import parse_model_id, resolve_character, resolve_preset, require_model_config
from backend.api.utils import build_available_presets
from backend.services.chat.models import ChatRequest, Message
from backend.services.chat.service import extract_text_content
from backend.lib.debug_logger import logger
from backend.lib.log_context import new_message_id
from backend.providers.registry import PROVIDER_LABELS
from backend.lib.time_awareness import compute_time_awareness
from backend.adapters.openai.schemas import OAIChatRequest
from backend.services.memory.format import format_recalled_memories
router = APIRouter()


def _derive_session_id(character_id: str, messages: list) -> str:
    """会話の最初のuserメッセージ + character_idからsession_idを導出する。"""
    import hashlib
    first_user = next((m for m in messages if m.role == "user"), None)
    if not first_user:
        return ""
    seed = f"{character_id}:{extract_text_content(first_user.content)}"
    return hashlib.md5(seed.encode()).hexdigest()


def _available_providers(settings: dict) -> set[str]:
    """APIキーが設定済みのプロバイダーを返す (claude_cli は常に含む)。
    settings の全キーを走査し `{provider}_api_key` 形式のものを自動検出するため、
    新プロバイダー追加時もこの関数の変更は不要。
    """
    result = {"claude_cli"}
    for key, value in settings.items():
        if key.endswith("_api_key") and value:
            result.add(key.removesuffix("_api_key"))
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
    new_message_id()

    state = request.app.state

    logger.log_front_input(body.model_dump())

    char_id, preset_id = parse_model_id(body.model)

    # UUID優先、なければ名前で検索
    character = resolve_character(state.sqlite, char_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{char_id}' not found")

    preset = resolve_preset(state.sqlite, preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Model preset '{preset_id}' not found")

    model_config = require_model_config(character, preset)

    settings = state.sqlite.get_all_settings()

    # 時刻認識
    now = datetime.now()
    ta = compute_time_awareness(settings, character.id, state.sqlite, now)
    if body.messages and body.messages[-1].role == "user":
        state.sqlite.set_setting(f"last_interaction_{character.id}", now.isoformat())

    messages = [Message(role=m.role, content=m.content) for m in body.messages]
    session_id = _derive_session_id(character.id, messages)

    available_presets = build_available_presets(character, preset, state.sqlite)

    # Afterglow（感情継続機構）: 新規会話かつ afterglow_default が ON の場合、
    # 同キャラの直近5ターンをメッセージ先頭に注入する。
    # OpenWebUI は全履歴を毎回送信するため、user メッセージが1件のみのときを新規会話と判定する。
    if len(messages) == 1 and messages[0].role == "user":
        afterglow_default = getattr(character, "afterglow_default", 0)
        if afterglow_default:
            prev_session_id = state.sqlite.find_latest_session_for_character(
                character_name=character.name,
                exclude_session_id=session_id,
            )
            if prev_session_id:
                from backend.services.chat.content import build_1on1_history
                afterglow_raw = state.sqlite.get_recent_turns(prev_session_id, n_turns=5)
                uploads_dir = getattr(state, "uploads_dir", "")
                afterglow_msgs = build_1on1_history(afterglow_raw, state.sqlite, uploads_dir)
                messages = afterglow_msgs + messages

    chat_request = ChatRequest(
        character_id=character.id,
        character_name=character.name,
        provider=preset.provider,
        model=preset.model_id,
        messages=messages,
        character_system_prompt=character.system_prompt_block1,
        self_history=character.self_history,
        relationship_state=character.relationship_state,
        inner_narrative=character.inner_narrative,
        provider_additional_instructions=model_config.get("additional_instructions", ""),
        thinking_level=preset.thinking_level or "default",
        settings=settings,
        enable_time_awareness=ta.enabled,
        current_time_str=ta.current_time_str,
        time_since_last_interaction=ta.time_since_last_interaction,
        session_id=session_id,
        current_preset_name=preset.name,
        current_preset_id=preset.id,
        available_presets=available_presets,
        self_reflection_mode=character.self_reflection_mode,
        self_reflection_preset_id=character.self_reflection_preset_id or "",
        self_reflection_n_turns=character.self_reflection_n_turns,
    )

    chat_service = state.chat_service

    if body.stream:
        async def generate():
            """型付きチャンクを OpenAI SSE 形式に変換して送信する。"""
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
