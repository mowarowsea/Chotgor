"""グループチャット API。

複数キャラクターによるグループチャットセッションの作成・管理・メッセージ送受信を担当する。
エンドポイントはすべて /api/group プレフィックス以下に配置する。
"""

import json
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.group_chat.service import run_group_turn

router = APIRouter(prefix="/api/group", tags=["group_chat"])


# --- Pydantic スキーマ ---

class ParticipantModel(BaseModel):
    """グループチャット参加者の設定。

    Attributes:
        model_id: "{char_name}@{preset_name}" 形式の参加者モデル指定。
    """

    model_id: str  # "{char_name}@{preset_name}"


class GroupSessionCreate(BaseModel):
    """グループセッション作成リクエスト。

    Attributes:
        participants: 参加者モデルIDのリスト（最低2名）。
        director_model_id: 司会役キャラクターのモデルID（"{char_name}@{preset_name}" 形式）。
        max_auto_turns: キャラクター同士の最大連続発言回数（1〜10）。
        turn_timeout_sec: 司会AI・各キャラクターへのタイムアウト秒数。
        title: セッションタイトル（省略時は自動生成）。
    """

    participants: list[ParticipantModel] = Field(min_length=2)
    director_model_id: str  # "{char_name}@{preset_name}" 形式
    max_auto_turns: int = Field(default=3, ge=1, le=10)
    turn_timeout_sec: int = Field(default=30, ge=10, le=120)
    title: Optional[str] = None


class GroupMessageCreate(BaseModel):
    """グループチャットメッセージ送信リクエスト。"""

    content: str


# --- ヘルパー ---

def _parse_participant(model_id: str) -> dict:
    """"{char_name}@{preset_name}" 形式の文字列を参加者辞書に変換する。

    Args:
        model_id: "{char_name}@{preset_name}" 形式の文字列。

    Returns:
        {"char_name": str, "preset_name": str} の辞書。

    Raises:
        ValueError: フォーマットが不正な場合。
    """
    if "@" not in model_id:
        raise ValueError(f"Invalid participant model_id format: '{model_id}' (expected 'CharName@PresetName')")
    char_name, preset_name = model_id.rsplit("@", 1)
    return {"char_name": char_name, "preset_name": preset_name}


def _session_to_dict(s) -> dict:
    """グループセッション ORMオブジェクトを辞書に変換する。"""
    result = {
        "id": s.id,
        "model_id": s.model_id,
        "title": s.title,
        "session_type": getattr(s, "session_type", "group"),
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }
    group_config = getattr(s, "group_config", None)
    if group_config:
        result["group_config"] = group_config
    return result


def _message_to_dict(m) -> dict:
    """ChatMessage ORMオブジェクトを辞書に変換する。"""
    result = {
        "id": m.id,
        "session_id": m.session_id,
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }
    if getattr(m, "character_name", None):
        result["character_name"] = m.character_name
    if getattr(m, "reasoning", None):
        result["reasoning"] = m.reasoning
    return result


# --- エンドポイント ---

@router.post("/sessions", status_code=201)
async def create_group_session(request: Request, body: GroupSessionCreate):
    """グループチャットセッションを作成する。

    max_auto_turns >= 5 の場合は警告フラグを付与して返す。

    Returns:
        作成されたセッションの辞書と、必要に応じた警告メッセージ。
    """
    state = request.app.state

    # 司会キャラクターをパースして存在チェック
    if "@" not in body.director_model_id:
        raise HTTPException(
            status_code=400,
            detail=f"director_model_id のフォーマットが不正です: '{body.director_model_id}' (expected 'CharName@PresetName')",
        )
    director_char_name, director_preset_name = body.director_model_id.rsplit("@", 1)
    director_char = (
        state.sqlite.get_character_by_name(director_char_name)
        or state.sqlite.get_character(director_char_name)
    )
    if not director_char:
        raise HTTPException(status_code=404, detail=f"司会キャラクター '{director_char_name}' が見つかりません")
    director_preset = (
        state.sqlite.get_model_preset_by_name(director_preset_name)
        or state.sqlite.get_model_preset(director_preset_name)
    )
    if not director_preset:
        raise HTTPException(status_code=404, detail=f"プリセット '{director_preset_name}' が見つかりません")

    # 参加者をパースして存在チェック
    try:
        participants = [_parse_participant(p.model_id) for p in body.participants]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    for p in participants:
        char = state.sqlite.get_character_by_name(p["char_name"]) or state.sqlite.get_character(p["char_name"])
        if not char:
            raise HTTPException(status_code=404, detail=f"キャラクター '{p['char_name']}' が見つかりません")
        preset = state.sqlite.get_model_preset_by_name(p["preset_name"]) or state.sqlite.get_model_preset(p["preset_name"])
        if not preset:
            raise HTTPException(status_code=404, detail=f"プリセット '{p['preset_name']}' が見つかりません")

    # グループ設定を構築してDBに保存する
    group_config = {
        "participants": participants,
        "director_model_id": body.director_model_id,
        "max_auto_turns": body.max_auto_turns,
        "turn_timeout_sec": body.turn_timeout_sec,
    }
    char_names = [p["char_name"] for p in participants]
    title = body.title or "、".join(char_names) + " のグループチャット"

    session_id = str(uuid.uuid4())
    session = state.sqlite.create_chat_session(
        session_id=session_id,
        model_id="group",
        title=title,
        session_type="group",
        group_config=json.dumps(group_config, ensure_ascii=False),
    )

    result = _session_to_dict(session)
    if body.max_auto_turns >= 5:
        result["warning"] = f"max_auto_turns={body.max_auto_turns} はAPI消費量が増加します"
    return result


@router.get("/sessions/{session_id}")
async def get_group_session(request: Request, session_id: str):
    """グループセッションとメッセージ一覧を返す。"""
    state = request.app.state
    session = state.sqlite.get_chat_session(session_id)
    if not session or getattr(session, "session_type", "1on1") != "group":
        raise HTTPException(status_code=404, detail="グループセッションが見つかりません")
    messages = state.sqlite.list_chat_messages(session_id)
    result = _session_to_dict(session)
    result["messages"] = [_message_to_dict(m) for m in messages]
    return result


@router.post("/sessions/{session_id}/messages/stream")
async def stream_group_message(request: Request, session_id: str, body: GroupMessageCreate):
    """ユーザーメッセージを送信し、グループターンの経過をSSEでストリーミング返却する。

    SSEイベント形式:
        {"type": "user_saved",       "message": {...}}           — ユーザーメッセージ保存完了
        {"type": "speaker_decided",  "speakers": [...]}          — 司会AIが次発言者を決定
        {"type": "character_message","character": "...", "message": {...}} — キャラクター応答完了
        {"type": "user_turn",        "auto_turns_used": N}       — ユーザーターン開始
        {"type": "error",            "message": "...", "character": "..."} — エラー発生
        {"type": "done"}                                          — ストリーム終了
    """
    state = request.app.state

    session = state.sqlite.get_chat_session(session_id)
    if not session or getattr(session, "session_type", "1on1") != "group":
        raise HTTPException(status_code=404, detail="グループセッションが見つかりません")

    # グループ設定をデシリアライズする
    raw_config = getattr(session, "group_config", None)
    if not raw_config:
        raise HTTPException(status_code=400, detail="グループ設定が見つかりません")
    try:
        group_config = json.loads(raw_config)
    except Exception:
        raise HTTPException(status_code=400, detail="グループ設定が不正です")

    # ユーザーメッセージをDBに保存する
    user_msg_id = str(uuid.uuid4())
    user_msg = state.sqlite.create_chat_message(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=body.content,
    )

    # セッションタイトルを自動設定する（最初のメッセージから）
    existing = state.sqlite.list_chat_messages(session_id)
    if len(existing) == 1 and session.title.endswith("のグループチャット"):
        auto_title = body.content[:30].replace("\n", " ")
        state.sqlite.update_chat_session(session_id, title=auto_title)

    settings = state.sqlite.get_all_settings()

    async def sse_generator():
        """グループターンのSSEジェネレーター。"""
        # ユーザーメッセージ保存完了を通知する
        user_msg_dict = _message_to_dict(user_msg)
        yield f"data: {json.dumps({'type': 'user_saved', 'message': user_msg_dict}, ensure_ascii=False)}\n\n"

        try:
            async for event_type, payload in run_group_turn(
                session_id=session_id,
                group_config=group_config,
                sqlite=state.sqlite,
                memory_manager=state.memory_manager,
                settings=settings,
            ):
                data = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
                yield f"data: {data}\n\n"
        except Exception as e:
            err = json.dumps({"type": "error", "message": str(e), "character": ""}, ensure_ascii=False)
            yield f"data: {err}\n\n"

        # セッションの updated_at を最新化する
        current = state.sqlite.get_chat_session(session_id)
        if current:
            state.sqlite.update_chat_session(session_id, title=current.title)

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
