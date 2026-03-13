"""チャットセッション管理API。

フロントエンドからの直接アクセスに対応する。
セッション管理 + LLM呼び出しを担当する。
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Union

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..core.chat.models import ChatRequest, Message
from ..core.debug_logger import log_front_output
from ..core.utils import format_time_delta
from .utils import build_1on1_history, build_message_content, format_memories_for_sse, message_to_dict, session_to_dict

router = APIRouter(prefix="/api/chat", tags=["chat"])


# --- Pydantic スキーマ ---

class SessionCreate(BaseModel):
    """セッション作成リクエスト。"""

    model_id: str       # "{char_name}@{preset_name}"
    title: Optional[str] = None


class SessionUpdate(BaseModel):
    """セッションタイトル更新リクエスト。"""

    title: str


class MessageCreate(BaseModel):
    """メッセージ送信リクエスト。"""

    content: str
    image_ids: Optional[List[str]] = None


async def _build_chat_request(request: Request, session, history_messages: list, user_content: Union[str, list]) -> ChatRequest:
    """セッション情報からChatRequestを構築する内部ヘルパー。"""
    state = request.app.state

    # model_id を {char_name}@{preset_name} にパース
    if "@" not in session.model_id:
        raise HTTPException(status_code=400, detail="Invalid model_id format")
    char_name, preset_name = session.model_id.rsplit("@", 1)

    # キャラクターをDBから取得（名前優先）
    character = state.sqlite.get_character_by_name(char_name) or state.sqlite.get_character(char_name)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{char_name}' not found")

    # プリセットをDBから取得（名前優先）
    preset = state.sqlite.get_model_preset_by_name(preset_name) or state.sqlite.get_model_preset(preset_name)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Model preset '{preset_name}' not found")

    model_config = (character.enabled_providers or {}).get(preset.id)
    if model_config is None:
        raise HTTPException(
            status_code=400,
            detail=f"Preset '{preset_name}' is not enabled for character '{char_name}'",
        )

    settings = state.sqlite.get_all_settings()

    # 時刻計算
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
    state.sqlite.set_setting(f"last_interaction_{character.id}", now.isoformat())

    # OpenAI role形式に変換（character → assistant、画像付きメッセージは vision 形式に変換）
    messages = build_1on1_history(history_messages, state.sqlite, state.uploads_dir)
    messages.append(Message(role="user", content=user_content))

    return ChatRequest(
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
    """新しいチャットセッションを作成する。"""
    session_id = str(uuid.uuid4())
    title = body.title or "新しいチャット"
    session = request.app.state.sqlite.create_chat_session(
        session_id=session_id,
        model_id=body.model_id,
        title=title,
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
    # ディスク上の画像ファイルを先に削除する
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


@router.post("/sessions/{session_id}/images", status_code=201)
async def upload_images(
    request: Request,
    session_id: str,
    files: List[UploadFile] = File(...),
):
    """複数の画像ファイルをアップロードしてセッションに紐づける。

    受け付けるMIMEタイプ: image/*
    ファイルは uploads_dir/{image_id} として保存される。

    Returns:
        [{"id": image_id, "url": "/api/chat/images/{image_id}"}] の形式で返す。
    """
    state = request.app.state
    session = state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    results = []
    for file in files:
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' は画像ファイルではありません",
            )
        image_id = str(uuid.uuid4())
        data = await file.read()
        img_path = os.path.join(state.uploads_dir, image_id)
        with open(img_path, "wb") as f:
            f.write(data)
        state.sqlite.create_chat_image(
            image_id=image_id,
            session_id=session_id,
            mime_type=file.content_type or "image/jpeg",
        )
        results.append({"id": image_id, "url": f"/api/chat/images/{image_id}"})
    return results


@router.get("/images/{image_id}")
async def get_image(request: Request, image_id: str):
    """添付画像ファイルを配信する。"""
    img = request.app.state.sqlite.get_chat_image(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    img_path = os.path.join(request.app.state.uploads_dir, image_id)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(img_path, media_type=img.mime_type)


@router.delete("/sessions/{session_id}/messages/from/{message_id}", status_code=204)
async def delete_messages_from(request: Request, session_id: str, message_id: str):
    """指定メッセージ以降（自身を含む）をすべて削除する。

    ユーザメッセージ編集・キャラクター応答再生成の前処理として呼び出す。
    削除後、フロントは同セッションに対して streamMessage を再送する。
    """
    session = request.app.state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    ok = request.app.state.sqlite.delete_chat_messages_from(session_id, message_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Message not found")


@router.post("/sessions/{session_id}/messages")
async def send_message(request: Request, session_id: str, body: MessageCreate):
    """
    ユーザーメッセージを送信し、キャラクターの応答を返す。

    処理フロー:
    1. ユーザーメッセージを保存
    2. 会話履歴を取得
    3. LLMを呼び出し
    4. キャラクターの応答を保存
    5. 両メッセージを返す
    """
    state = request.app.state

    session = state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ユーザーメッセージ保存
    user_msg_id = str(uuid.uuid4())
    user_msg = state.sqlite.create_chat_message(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=body.content,
    )

    # 送信済みのユーザーメッセージを除いた履歴を取得（LLMには現在のメッセージを別途渡す）
    history_before = state.sqlite.list_chat_messages(session_id)
    history = [m for m in history_before if m.id != user_msg_id]

    # セッションタイトル自動設定（最初のメッセージから先頭30文字を使う）
    if len(history) == 0 and session.title == "新しいチャット":
        auto_title = body.content[:30].replace("\n", " ")
        state.sqlite.update_chat_session(session_id, title=auto_title)

    # LLM呼び出し
    try:
        response_text = await _call_llm(request, session, history, body.content)
    except HTTPException:
        raise
    except Exception as e:
        response_text = f"[エラー: {e}]"

    # キャラクター応答を保存
    char_msg_id = str(uuid.uuid4())
    char_msg = state.sqlite.create_chat_message(
        message_id=char_msg_id,
        session_id=session_id,
        role="character",
        content=response_text,
    )

    # セッションの updated_at を最新化
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
    """
    ユーザーメッセージを送信し、キャラクターの応答をSSEでストリーミング返却する。

    SSEイベント形式:
        data: {"type": "chunk", "content": "..."}   — テキストチャンク
        data: {"type": "done", "user_message": {...}, "character_message": {...}}  — 完了
        data: {"type": "error", "message": "..."}   — エラー
    """
    state = request.app.state

    session = state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ユーザーメッセージを保存
    user_msg_id = str(uuid.uuid4())
    user_msg = state.sqlite.create_chat_message(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=body.content,
        images=body.image_ids or None,
    )

    # 現在のユーザーメッセージを除いた履歴を取得
    history_before = state.sqlite.list_chat_messages(session_id)
    history = [m for m in history_before if m.id != user_msg_id]

    # セッションタイトル自動設定（最初のメッセージから先頭30文字）
    if len(history) == 0 and session.title == "新しいチャット":
        auto_title = body.content[:30].replace("\n", " ")
        state.sqlite.update_chat_session(session_id, title=auto_title)

    # 画像が添付されている場合は OpenAI vision 形式のコンテンツリストを構築する
    user_content: Union[str, list] = build_message_content(
        body.content, body.image_ids or [], state.sqlite, state.uploads_dir
    )

    # ChatRequestの構築（HTTPExceptionが発生した場合はここで止まる）
    try:
        chat_request = await _build_chat_request(request, session, history, user_content)
    except HTTPException:
        raise

    async def sse_generator():
        """SSEストリームのジェネレータ。型付きチャンクをyieldし、完了後にDBへ保存する。

        execute_stream() は (type, content) タプルをyieldする:
            ("memories", list[dict]) — 想起した記憶リスト
            ("thinking", str)        — 思考ブロック（リアルタイム）
            ("text", str)            — carve済みの応答テキスト（1回だけyield）

        SSEイベント形式:
            {"type": "reasoning", "content": "..."} — 思考・記憶（フロントで折りたたみ表示）
            {"type": "chunk",     "content": "..."} — 応答テキスト
        """
        full_text = ""
        accumulated_reasoning = ""
        try:
            async for chunk_type, content in state.chat_service.execute_stream(chat_request):
                if chunk_type == "memories":
                    # 記憶リストを表示用テキストにフォーマットして送信・蓄積する
                    display = format_memories_for_sse(content)
                    if display:
                        accumulated_reasoning += display
                        data = json.dumps({"type": "reasoning", "content": display}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                elif chunk_type == "thinking":
                    # 思考ブロックをリアルタイム送信・蓄積する
                    if content:
                        accumulated_reasoning += content
                        data = json.dumps({"type": "reasoning", "content": content}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                elif chunk_type == "text":
                    # carve済みの全テキストを1チャンクとして送信する
                    full_text = content
                    if content:
                        data = json.dumps({"type": "chunk", "content": content}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
        except Exception as e:
            err_data = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {err_data}\n\n"
            return

        # execute_stream() 内で carve 済みのため再実行不要
        clean_text = full_text
        log_front_output(clean_text)

        # キャラクターの応答をDBに保存（reasoning があれば一緒に保存する）
        char_msg_id = str(uuid.uuid4())
        char_msg = state.sqlite.create_chat_message(
            message_id=char_msg_id,
            session_id=session_id,
            role="character",
            content=clean_text,
            reasoning=accumulated_reasoning if accumulated_reasoning else None,
        )

        # セッションのupdated_atを最新化
        current_session = state.sqlite.get_chat_session(session_id)
        if current_session:
            state.sqlite.update_chat_session(session_id, title=current_session.title)

        # 完了イベントを送信
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
