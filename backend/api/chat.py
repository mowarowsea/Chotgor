"""チャットセッション管理API。

フロントエンドからの直接アクセスに対応する。
セッション管理 + LLM呼び出しを担当する。

画像管理: chat_images.py
SELF_DRIFT: chat_drifts.py
"""

import json
import uuid
from datetime import datetime
from typing import List, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.chat.models import ChatRequest, Message
from ..core.debug_logger import log_front_output
from ..core.time_awareness import compute_time_awareness
from .resource_resolver import parse_model_id, require_character, require_preset, require_model_config
from .utils import build_1on1_history, build_message_content, format_memories_for_sse, message_to_dict, session_to_dict

router = APIRouter(prefix="/api/chat", tags=["chat"])


# --- Pydantic スキーマ ---

class SessionCreate(BaseModel):
    """セッション作成リクエスト。

    afterglow が True の場合、同キャラクターの最新1on1セッションを自動特定し、
    その直近5ターンを Afterglow（感情継続機構）として引き継ぐ。
    """

    model_id: str       # "{char_name}@{preset_name}"
    title: Optional[str] = None
    afterglow: bool = False


class SessionUpdate(BaseModel):
    """セッションタイトル更新リクエスト。"""

    title: str


class MessageCreate(BaseModel):
    """メッセージ送信リクエスト。"""

    content: str
    image_ids: Optional[List[str]] = None
    model_id: Optional[str] = None  # 送信時に使用するモデルを上書きする。省略時はセッションの model_id を使う。


def _prepend_afterglow(state, session, history_messages: list) -> list:
    """Afterglow（感情継続機構）の引き継ぎメッセージを history_messages の先頭に追加する。

    セッションに afterglow_session_id が設定されている場合、引き継ぎ元セッションの
    直近5ターン（最大10メッセージ）を取得してプリペンドする。
    これらのメッセージはDBには保存せず、LLMへのコンテキストとしてのみ使用される。

    Args:
        state: アプリケーションステート（sqlite / uploads_dir を含む）。
        session: 現在のチャットセッション。
        history_messages: 現在セッションの会話履歴。

    Returns:
        Afterglowメッセージがプリペンドされたメッセージリスト。
    """
    afterglow_id = getattr(session, "afterglow_session_id", None)
    if not afterglow_id:
        return history_messages
    afterglow_msgs = state.sqlite.get_recent_turns(afterglow_id, n_turns=5)
    return afterglow_msgs + history_messages


async def _build_chat_request(
    request: Request,
    session,
    history_messages: list,
    user_content: Union[str, list],
    model_id: Optional[str] = None,
) -> ChatRequest:
    """セッション情報からChatRequestを構築する内部ヘルパー。

    Args:
        model_id: 使用するモデルIDを明示的に指定する場合に渡す。省略時はセッションの model_id を使う。
    """
    state = request.app.state

    effective_id = model_id or session.model_id
    char_name, preset_name = parse_model_id(effective_id)

    character = require_character(state.sqlite, char_name)
    preset = require_preset(state.sqlite, preset_name)
    model_config = require_model_config(character, preset)

    settings = state.sqlite.get_all_settings()

    # 時刻認識
    now = datetime.now()
    ta = compute_time_awareness(settings, character.id, state.sqlite, now)
    state.sqlite.set_setting(f"last_interaction_{character.id}", now.isoformat())

    messages = build_1on1_history(history_messages, state.sqlite, state.uploads_dir)
    messages.append(Message(role="user", content=user_content))

    # switch_angle 用プリセット一覧を構築する
    enabled_providers = character.enabled_providers or {}
    available_presets = []
    if character.switch_angle_enabled and len(enabled_providers) > 1:
        all_presets = state.sqlite.list_model_presets()
        for p in all_presets:
            if p.id == preset.id:
                continue
            cfg = enabled_providers.get(p.id)
            if cfg is None:
                continue
            available_presets.append({
                "preset_id": p.id,
                "preset_name": p.name,
                "provider": p.provider,
                "model_id": p.model_id,
                "additional_instructions": cfg.get("additional_instructions", ""),
                "thinking_level": p.thinking_level or "default",
                "when_to_switch": cfg.get("when_to_switch", ""),
            })

    return ChatRequest(
        character_id=character.id,
        character_name=character.name,
        provider=preset.provider,
        model=preset.model_id,
        messages=messages,
        character_system_prompt=character.system_prompt_block1,
        inner_narrative=character.inner_narrative,
        provider_additional_instructions=model_config.get("additional_instructions", ""),
        thinking_level=preset.thinking_level or "default",
        settings=settings,
        enable_time_awareness=ta.enabled,
        current_time_str=ta.current_time_str,
        time_since_last_interaction=ta.time_since_last_interaction,
        session_id=session.id,
        available_presets=available_presets,
        current_preset_name=preset.name,
        current_preset_id=preset.id,
    )


async def _call_llm(request: Request, session, history_messages: list, user_content: str) -> str:
    """LLMを呼び出してキャラクターの応答テキストを返す。"""
    chat_request = await _build_chat_request(request, session, history_messages, user_content)
    return await request.app.state.chat_service.execute(chat_request)


# --- エンドポイント ---

@router.get("/settings/user-name")
async def get_user_name(request: Request):
    """設定からユーザ名を取得する。"""
    user_name = request.app.state.sqlite.get_setting("user_name", "ユーザ")
    return {"user_name": user_name}


@router.get("/sessions")
async def list_sessions(request: Request):
    """チャットセッション一覧を新しい順で返す。"""
    sessions = request.app.state.sqlite.list_chat_sessions()
    return [session_to_dict(s) for s in sessions]


@router.post("/sessions", status_code=201)
async def create_session(request: Request, body: SessionCreate):
    """新しいチャットセッションを作成する。

    afterglow=True の場合、同キャラクターの最新1on1セッションを引き継ぎ元として設定する。
    グループチャット（session_type="group"）では afterglow は無視される。
    """
    session_id = str(uuid.uuid4())
    title = body.title or "新しいチャット"

    # Afterglow: 同キャラの最新セッションを特定する
    afterglow_session_id: Optional[str] = None
    if body.afterglow:
        char_name = body.model_id.split("@")[0] if "@" in body.model_id else body.model_id
        afterglow_session_id = request.app.state.sqlite.find_latest_session_for_character(
            character_name=char_name,
            exclude_session_id=session_id,
        )

    session = request.app.state.sqlite.create_chat_session(
        session_id=session_id,
        model_id=body.model_id,
        title=title,
        afterglow_session_id=afterglow_session_id,
    )
    return session_to_dict(session)


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    """セッションとそのメッセージ一覧を返す。"""
    session = request.app.state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = request.app.state.sqlite.list_chat_messages(session_id)
    result = session_to_dict(session)
    result["messages"] = [message_to_dict(m) for m in messages]
    return result


@router.patch("/sessions/{session_id}")
async def update_session(request: Request, session_id: str, body: SessionUpdate):
    """セッションのタイトルを更新する。"""
    session = request.app.state.sqlite.update_chat_session(session_id, title=body.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_to_dict(session)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: str):
    """セッションとそのメッセージ・添付画像ファイルを削除する。"""
    import os
    images = request.app.state.sqlite.list_chat_images_by_session(session_id)
    for img in images:
        img_path = os.path.join(request.app.state.uploads_dir, img.id)
        try:
            os.remove(img_path)
        except FileNotFoundError:
            pass
    ok = request.app.state.sqlite.delete_chat_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/sessions/{session_id}/messages/from/{message_id}", status_code=204)
async def delete_messages_from(request: Request, session_id: str, message_id: str):
    """指定メッセージ以降（自身を含む）をすべて削除する。"""
    session = request.app.state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    ok = request.app.state.sqlite.delete_chat_messages_from(session_id, message_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Message not found")


@router.post("/sessions/{session_id}/messages")
async def send_message(request: Request, session_id: str, body: MessageCreate):
    """ユーザーメッセージを送信し、キャラクターの応答を返す。"""
    state = request.app.state

    session = state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg_id = str(uuid.uuid4())
    user_msg = state.sqlite.create_chat_message(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=body.content,
    )

    history_before = state.sqlite.list_chat_messages(session_id)
    history = [m for m in history_before if m.id != user_msg_id]

    if len(history) == 0 and session.title == "新しいチャット":
        auto_title = body.content[:30].replace("\n", " ")
        state.sqlite.update_chat_session(session_id, title=auto_title)

    # Afterglow: 引き継ぎ元セッションの直近ターンを先頭に追加する
    history_with_afterglow = _prepend_afterglow(state, session, history)

    try:
        response_text = await _call_llm(request, session, history_with_afterglow, body.content)
    except HTTPException:
        raise
    except Exception as e:
        response_text = f"[エラー: {e}]"

    char_msg_id = str(uuid.uuid4())
    char_msg = state.sqlite.create_chat_message(
        message_id=char_msg_id,
        session_id=session_id,
        role="character",
        content=response_text,
    )

    state.sqlite.update_chat_session(
        session_id,
        title=state.sqlite.get_chat_session(session_id).title,
    )

    return {
        "user_message": message_to_dict(user_msg),
        "character_message": message_to_dict(char_msg),
    }


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(request: Request, session_id: str, body: MessageCreate):
    """ユーザーメッセージを送信し、キャラクターの応答をSSEでストリーミング返却する。"""
    state = request.app.state

    session = state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    effective_model_id = body.model_id or session.model_id

    user_msg_id = str(uuid.uuid4())
    user_msg = state.sqlite.create_chat_message(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=body.content,
        images=body.image_ids or None,
    )

    history_before = state.sqlite.list_chat_messages(session_id)
    history = [m for m in history_before if m.id != user_msg_id]

    if len(history) == 0 and session.title == "新しいチャット":
        effective_title = body.content[:30].replace("\n", " ")
        state.sqlite.update_chat_session(session_id, title=effective_title)
    else:
        effective_title = session.title

    user_content: Union[str, list] = build_message_content(
        body.content, body.image_ids or [], state.sqlite, state.uploads_dir
    )

    # Afterglow: 引き継ぎ元セッションの直近ターンを先頭に追加する
    history_with_afterglow = _prepend_afterglow(state, session, history)

    try:
        chat_request = await _build_chat_request(request, session, history_with_afterglow, user_content, model_id=effective_model_id)
    except HTTPException:
        raise

    async def sse_generator():
        nonlocal effective_model_id
        full_text = ""
        accumulated_reasoning = ""
        try:
            async for chunk_type, content in state.chat_service.execute_stream(chat_request):
                if chunk_type == "memories":
                    display = format_memories_for_sse(content)
                    if display:
                        accumulated_reasoning += display
                        data = json.dumps({"type": "reasoning", "content": display}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                elif chunk_type == "thinking":
                    if content:
                        accumulated_reasoning += content
                        data = json.dumps({"type": "reasoning", "content": content}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                elif chunk_type == "text":
                    full_text = content
                    if content:
                        data = json.dumps({"type": "chunk", "content": content}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                elif chunk_type == "angle_switched":
                    effective_model_id = content
        except Exception as e:
            err_data = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {err_data}\n\n"
            return

        clean_text = full_text
        log_front_output(clean_text)

        used_char_name, used_preset_name = effective_model_id.rsplit("@", 1) if "@" in effective_model_id else (effective_model_id, None)

        char_msg_id = str(uuid.uuid4())
        char_msg = state.sqlite.create_chat_message(
            message_id=char_msg_id,
            session_id=session_id,
            role="character",
            content=clean_text,
            reasoning=accumulated_reasoning if accumulated_reasoning else None,
            character_name=used_char_name,
            preset_name=used_preset_name,
        )

        state.sqlite.update_chat_session(session_id, title=effective_title, model_id=effective_model_id)

        done_data = json.dumps({
            "type": "done",
            "user_message": message_to_dict(user_msg),
            "character_message": message_to_dict(char_msg),
        }, ensure_ascii=False)
        yield f"data: {done_data}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
