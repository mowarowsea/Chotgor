"""グループチャット API。

複数キャラクターによるグループチャットセッションの作成・管理・メッセージ送受信を担当する。
エンドポイントはすべて /api/group プレフィックス以下に配置する。
"""

import asyncio
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.chat.indexer import get_participant_char_ids, index_message_sync
from ..core.group_chat.service import run_group_turn
from .resource_resolver import parse_model_id, require_character, require_preset
from .utils import message_to_dict, session_to_dict

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
    image_ids: Optional[List[str]] = None


def _parse_participant(model_id: str) -> dict:
    """"{char_name}@{preset_name_or_id}" 形式の文字列をパースして (char_name, preset_key) を返す。

    Args:
        model_id: "{char_name}@{preset_name_or_id}" 形式の文字列。

    Returns:
        {"char_name": str, "preset_key": str} の辞書。preset_key は名前またはIDの文字列。

    Raises:
        ValueError: フォーマットが不正な場合。
    """
    if "@" not in model_id:
        raise ValueError(f"Invalid participant model_id format: '{model_id}' (expected 'CharName@PresetName')")
    char_name, preset_key = model_id.rsplit("@", 1)
    return {"char_name": char_name, "preset_key": preset_key}


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
    director_char_name, director_preset_key = parse_model_id(body.director_model_id)
    director_char = require_character(state.sqlite, director_char_name)
    director_preset = require_preset(state.sqlite, director_preset_key)

    # 参加者をパースして存在チェックし、preset_id を解決する
    try:
        parsed = [_parse_participant(p.model_id) for p in body.participants]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    participants = []
    for p in parsed:
        char = require_character(state.sqlite, p["char_name"])
        preset = require_preset(state.sqlite, p["preset_key"])
        # 名前ではなくIDで保存することで、プリセット名変更の影響を受けないようにする
        participants.append({"char_name": p["char_name"], "preset_id": preset.id})

    # グループ設定を構築してDBに保存する（IDで保存）
    group_config = {
        "participants": participants,
        "director_char_name": director_char_name,
        "director_preset_id": director_preset.id,
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

    result = session_to_dict(session)
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
    result = session_to_dict(session)
    result["messages"] = [message_to_dict(m) for m in messages]
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

    # チャット履歴インデックス登録用にセッション参加キャラIDとユーザ名を解決する
    _chat_char_ids = get_participant_char_ids(session, state.sqlite)
    _chat_user_name = state.sqlite.get_setting("user_name", "ユーザ")

    # ユーザーメッセージをDBに保存する（画像IDも含む）
    user_msg_id = str(uuid.uuid4())
    user_msg = state.sqlite.create_chat_message(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=body.content,
        images=body.image_ids or None,
    )
    asyncio.create_task(asyncio.to_thread(
        index_message_sync, user_msg, _chat_char_ids, state.chroma, _chat_user_name
    ))

    # セッションタイトルを自動設定する（最初のメッセージから）
    existing = state.sqlite.list_chat_messages(session_id)
    if len(existing) == 1 and session.title.endswith("のグループチャット"):
        auto_title = body.content[:30].replace("\n", " ")
        state.sqlite.update_chat_session(session_id, title=auto_title)

    settings = state.sqlite.get_all_settings()

    async def sse_generator():
        """グループターンのSSEジェネレーター。"""
        # ユーザーメッセージ保存完了を通知する
        user_msg_dict = message_to_dict(user_msg)
        yield f"data: {json.dumps({'type': 'user_saved', 'message': user_msg_dict}, ensure_ascii=False)}\n\n"

        try:
            async for event_type, payload in run_group_turn(
                session_id=session_id,
                group_config=group_config,
                sqlite=state.sqlite,
                settings=settings,
                chat_service=state.chat_service,
                message_to_dict=message_to_dict,
                uploads_dir=state.uploads_dir,
                chroma=state.chroma,
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
