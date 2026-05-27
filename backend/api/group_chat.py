"""グループチャット API。

複数キャラクターによるグループチャットセッションの作成・管理・メッセージ送受信を担当する。
エンドポイントはすべて /api/group プレフィックス以下に配置する。
"""

import asyncio
import json
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.chat.indexer import get_participant_char_ids, index_message_sync
from backend.services.group_chat.service import run_group_turn
from backend.api.resource_resolver import parse_model_id, require_character, require_preset
from backend.api.utils import message_to_dict, session_to_dict

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

    司会モデルはセッション単位ではなくシステム設定（`group_director_preset_id`）で
    一括管理されるため、作成リクエストには含めない。

    Attributes:
        participants: 参加者モデルIDのリスト（最低2名）。
        max_auto_turns: キャラクター同士の最大連続発言回数（1〜10）。
        turn_timeout_sec: 司会AI・各キャラクターへのタイムアウト秒数。
        title: セッションタイトル（省略時は自動生成）。
    """

    participants: list[ParticipantModel] = Field(min_length=2)
    max_auto_turns: int = Field(default=3, ge=1, le=10)
    turn_timeout_sec: int = Field(default=30, ge=10, le=120)
    title: str | None = None


class GroupMessageCreate(BaseModel):
    """グループチャットメッセージ送信リクエスト。

    Attributes:
        content: メッセージ本文。skip=True / target_character 指定時は無視される。
        image_ids: 添付画像IDリスト。
        skip: True の場合、ユーザメッセージを保存せずに司会へ直接ターンを委譲する（ユーザターンスキップ機能）。
        target_character: 指定された場合、司会を介さずこのキャラクターを強制的に発言させる
                          （ユーザによる手動指名）。司会エラー時の代替手段として使う。
    """

    content: str
    image_ids: list[str] | None = None
    skip: bool = False
    target_character: str | None = None


# --- エンドポイント ---

@router.post("/sessions", status_code=201)
async def create_group_session(request: Request, body: GroupSessionCreate):
    """グループチャットセッションを作成する。

    max_auto_turns >= 5 の場合は警告フラグを付与して返す。

    Returns:
        作成されたセッションの辞書と、必要に応じた警告メッセージ。
    """
    state = request.app.state

    # 参加者をパースして存在チェックし、preset_id を解決する
    # parse_model_id は "@" 不正時に HTTPException 400 を送出する
    participants = []
    for p_model in body.participants:
        char_name, preset_key = parse_model_id(p_model.model_id)
        char = require_character(state.sqlite, char_name)
        preset = require_preset(state.sqlite, preset_key)
        # 実体参照は ID（プリセット名変更の影響を受けない）。
        # preset_name はヘッダー表示用のスナップショット（多少古くても表示にしか使わない）。
        participants.append(
            {
                "char_name": char_name,
                "preset_id": preset.id,
                "preset_name": preset.name,
            }
        )

    # グループ設定を構築してDBに保存する（IDで保存）
    # 司会モデルはシステム設定で一括管理するため group_config には含めない。
    group_config = {
        "participants": participants,
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

    body.skip=True でユーザターンをスキップして司会へ委譲できる。
    body.target_character 指定時は司会を介さずそのキャラクターを手動指名する。

    SSEイベント形式:
        {"type": "user_saved",       "message": {...}}           — ユーザーメッセージ保存完了
        {"type": "speaker_decided",  "speakers": [...]}          — 司会AIが次発言者を決定
        {"type": "character_message","character": "...", "message": {...}} — キャラクター応答完了
        {"type": "director_error",   "message": "..."}            — 司会エラー（手動再試行・手動指名で復帰可能）
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

    settings = state.sqlite.get_all_settings()

    # 手動指名キャラクターは参加者に含まれているか検証する
    if body.target_character is not None:
        participant_names = {p["char_name"] for p in group_config.get("participants", [])}
        if body.target_character not in participant_names:
            raise HTTPException(
                status_code=400,
                detail=f"'{body.target_character}' はこのグループの参加者ではありません",
            )

    # スキップ・手動指名時はユーザメッセージを保存せずに直接ターンへ委譲する
    saved_user_msg = None
    if not body.skip and body.target_character is None:
        # チャット履歴インデックス登録用にセッション参加キャラIDとユーザ名を解決する
        _chat_char_ids = get_participant_char_ids(session, state.sqlite)
        _chat_user_name = state.sqlite.get_setting("user_name", "ユーザ")

        # ユーザーメッセージをDBに保存する（画像IDも含む）
        user_msg_id = str(uuid.uuid4())
        saved_user_msg = state.sqlite.create_chat_message(
            message_id=user_msg_id,
            session_id=session_id,
            role="user",
            content=body.content,
            images=body.image_ids or None,
        )
        asyncio.create_task(asyncio.to_thread(
            index_message_sync, saved_user_msg, _chat_char_ids, state.vector_store, _chat_user_name
        ))

        # セッションタイトルを自動設定する（最初のメッセージから）
        existing = state.sqlite.list_chat_messages(session_id)
        if len(existing) == 1 and session.title.endswith("のグループチャット"):
            auto_title = body.content[:30].replace("\n", " ")
            state.sqlite.update_chat_session(session_id, title=auto_title)

    async def sse_generator():
        """グループターンのSSEジェネレーター。"""
        # ユーザーメッセージ保存完了を通知する（スキップ時は送信しない）
        if saved_user_msg is not None:
            user_msg_dict = message_to_dict(saved_user_msg)
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
                vector_store=state.vector_store,
                forced_speaker=body.target_character,
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
