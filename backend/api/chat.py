"""チャットセッション管理API。

フロントエンドからの直接アクセスに対応する。
セッション管理 + LLM呼び出しを担当する。
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..core.chat.models import ChatRequest, Message
from ..core.utils import format_time_delta

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


# --- ヘルパー ---

def _session_to_dict(s) -> dict:
    """ChatSession ORMオブジェクトを辞書に変換する。"""
    return {
        "id": s.id,
        "model_id": s.model_id,
        "title": s.title,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _message_to_dict(m) -> dict:
    """ChatMessage ORMオブジェクトを辞書に変換する。"""
    return {
        "id": m.id,
        "session_id": m.session_id,
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


async def _call_llm(request: Request, session, history_messages: list, user_content: str) -> str:
    """LLMを呼び出してキャラクターの応答テキストを返す。"""
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

    # OpenAI role形式に変換（character → assistant）
    messages = []
    for msg in history_messages:
        role = "assistant" if msg.role == "character" else "user"
        messages.append(Message(role=role, content=msg.content))
    messages.append(Message(role="user", content=user_content))

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
    )

    return await state.chat_service.execute(chat_request)


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
    return [_session_to_dict(s) for s in sessions]


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
    return _session_to_dict(session)


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    """セッションとそのメッセージ一覧を返す。"""
    session = request.app.state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = request.app.state.sqlite.list_chat_messages(session_id)
    result = _session_to_dict(session)
    result["messages"] = [_message_to_dict(m) for m in messages]
    return result


@router.patch("/sessions/{session_id}")
async def update_session(request: Request, session_id: str, body: SessionUpdate):
    """セッションのタイトルを更新する。"""
    session = request.app.state.sqlite.update_chat_session(session_id, title=body.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_dict(session)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: str):
    """セッションとそのメッセージを削除する。"""
    ok = request.app.state.sqlite.delete_chat_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")


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
        "user_message": _message_to_dict(user_msg),
        "character_message": _message_to_dict(char_msg),
    }
