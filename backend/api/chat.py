"""チャットセッション管理API。

フロントエンドからの直接アクセスに対応する。
セッション管理 + LLM呼び出しを担当する。

画像管理: chat_images.py
SELF_DRIFT: chat_drifts.py
"""

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.chat.indexer import get_participant_char_ids, index_message_sync
from backend.services.chat.models import Message
from backend.lib.debug_logger import logger
from backend.lib.log_context import (
    new_message_id,
    current_log_session_id,
    current_log_target,
)
from backend.api.resource_resolver import parse_model_id, require_character, require_preset, require_model_config
from backend.api.utils import build_1on1_history, build_message_content, format_memories_for_sse, message_to_dict, session_to_dict
from backend.services.chat.request_factory import build_character_request, build_available_presets, latest_anticipation
from backend.services.chat.content import apply_context_window
from backend.services.chat_flow.scene_loop import LoopState, SceneLoop
from backend.services.chat_flow.strategies import OneOnOneExecutor, OneOnOneRouter
from backend.services.memory.format import format_recalled_threads

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _build_farewell_message(char_name: str, reason: str, farewell_type: str) -> str:
    """退席タイプに応じた退席メッセージを生成する。

    Args:
        char_name: 退席したキャラクター名。
        reason: 退席理由・補足テキスト（警告メッセージを含む場合あり）。
        farewell_type: "negative" / "positive" / "neutral" のいずれか。

    Returns:
        退席メッセージテキスト。
    """
    if farewell_type == "negative":
        base = f"{char_name}はこの会話を終わらせました。"
    elif farewell_type == "positive":
        base = f"{char_name}は満足して会話を終わらせました。"
    else:
        base = f"{char_name}は退席しました。"
    if reason:
        return f"{base}\n{reason}"
    return base


def _build_all_exited_message(exited_chars: list[dict]) -> str:
    """全員退席時の通知テキストを生成する。

    Args:
        exited_chars: [{"char_name": str, "reason": str, "farewell_type": str}] のリスト。

    Returns:
        各退席者の退席メッセージを改行で連結したテキスト。
    """
    lines = [
        _build_farewell_message(
            e["char_name"], e.get("reason", ""), e.get("farewell_type", "neutral")
        )
        for e in exited_chars
    ]
    lines.append("（チャットを再開するには新しいセッションを作成してください）")
    return "\n".join(lines)


# --- Pydantic スキーマ ---

class SessionCreate(BaseModel):
    """セッション作成リクエスト。"""

    model_id: str       # "{char_name}@{preset_name}"
    title: str | None = None


class SessionUpdate(BaseModel):
    """セッションタイトル更新リクエスト。"""

    title: str


class MessageCreate(BaseModel):
    """メッセージ送信リクエスト。"""

    content: str
    image_ids: list[str] | None = None
    model_id: str | None = None  # 送信時に使用するモデルを上書きする。省略時はセッションの model_id を使う。


async def build_1on1_chat_request(
    state,
    session,
    history_messages: list,
    user_content: str | list,
    model_id: str | None = None,
):
    """セッション情報からChatRequestを構築するヘルパー。

    1on1 固有の処理（コンテキストウィンドウ適用・available_presets 構築）を行った後、
    共通ファクトリ build_character_request に委譲する。
    SSE エンドポイントのほか、預かりメッセージの能動配達
    （services/gate/delivery.py — HTTP リクエスト文脈なし）からも呼ばれるため、
    Request ではなく app.state を直接受け取る。

    Args:
        state: FastAPI の app.state（sqlite / uploads_dir を持つ）。
        model_id: 使用するモデルIDを明示的に指定する場合に渡す。省略時はセッションの model_id を使う。
    """
    effective_id = model_id or session.model_id
    char_name, preset_name = parse_model_id(effective_id)

    character = require_character(state.sqlite, char_name)
    preset = require_preset(state.sqlite, preset_name)
    # プリセットがキャラクターで有効化されているか検証する（無効なら HTTPException 400）
    require_model_config(character, preset)

    settings = state.sqlite.get_all_settings()

    # chronicle済みメッセージの保持上限をグローバル設定から取得（デフォルト: 10件）
    max_chronicled = int(settings.get("context_window_max_chronicled", 10))
    windowed = apply_context_window(history_messages, max_chronicled=max_chronicled)
    if len(windowed) < len(history_messages):
        # コンテキスト圧縮は正常動作なので WARN ではなくお知らせ（Notice）として記録する
        logger.log_notice(
            "context_window",
            f"全{len(history_messages)}件 → {len(windowed)}件に圧縮 (chronicle済み上限: {max_chronicled})",
        )
    messages = build_1on1_history(windowed, state.sqlite, state.uploads_dir)
    messages.append(Message(role="user", content=user_content))

    available_presets = build_available_presets(character, preset, state.sqlite)
    # 直前のキャラクター応答に含まれていた予想（ANTICIPATE_RESPONSE）を次ターンに注入する
    previous_anticipation = latest_anticipation(history_messages)

    # 対面モードは 1on1 専用の概念。scenario / 通常PC からは流れないよう、
    # ここ（1on1 経路）でのみ override として明示的に渡す。
    face_to_face = bool(getattr(character, "face_to_face_mode", 0))

    return build_character_request(
        character, preset, messages, session.id, settings, state.sqlite,
        available_presets=available_presets,
        previous_anticipation=previous_anticipation,
        face_to_face=face_to_face,
    )


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


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(request: Request, session_id: str, body: MessageCreate):
    """ユーザーメッセージを送信し、キャラクターの応答をSSEでストリーミング返却する。

    退席済みセッションへのリクエストは LLM をスキップし、退席者一覧のシステムメッセージを返す。
    relationship_status が "estranged" のキャラクターへのリクエストは恒久的に拒否する。
    別れの検出は FarewellDetector がバックグラウンドで行い、次リクエスト時に反映される。
    """
    log_msg_id = new_message_id()
    current_log_session_id.set(session_id)

    state = request.app.state

    session = state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    effective_model_id = body.model_id or session.model_id

    # current_log_target を log_front_input より先にセットしないと
    # _insert_main_entry が target=None で DB に INSERT してしまう
    char_name_for_check = effective_model_id.split("@")[0] if "@" in effective_model_id else effective_model_id
    current_log_target.set(char_name_for_check)

    logger.log_front_input(body.model_dump())

    # estranged チェック: relationship_status="estranged" のキャラクターへのリクエストをSSEで拒否する
    char_for_estranged = state.sqlite.get_character_by_name(char_name_for_check)
    # 当該キャラの現在の対面モード（0/1）。送信時の値をメッセージへ焼き付ける。
    # キャラ未取得（None）なら 0（テキスト）扱い。
    current_face_to_face = int(getattr(char_for_estranged, "face_to_face_mode", 0) or 0) if char_for_estranged else 0
    if char_for_estranged and getattr(char_for_estranged, "relationship_status", "active") == "estranged":
        estranged_text = f"{char_name_for_check}はあなたとの別れを決断しました。この関係は修復できません。"
        user_msg_id_e = str(uuid.uuid4())
        user_msg_e = state.sqlite.create_chat_message(
            message_id=user_msg_id_e,
            session_id=session_id,
            role="user",
            content=body.content,
            images=body.image_ids or None,
            face_to_face=current_face_to_face,
        )
        sys_msg_id_e = str(uuid.uuid4())
        sys_msg_e = state.sqlite.create_chat_message(
            message_id=sys_msg_id_e,
            session_id=session_id,
            role="character",
            content=estranged_text,
            character_name=char_name_for_check,
            is_system_message=True,
            face_to_face=current_face_to_face,
        )

        async def estranged_generator():
            """estranged キャラクター向けSSEジェネレーター。"""
            done_data = json.dumps({
                "type": "done",
                "user_message": message_to_dict(user_msg_e),
                "character_message": message_to_dict(sys_msg_e),
            }, ensure_ascii=False)
            yield f"data: {done_data}\n\n"

        return StreamingResponse(
            estranged_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # 退席済みチェック: 1on1セッションでキャラクターが退席済みの場合はLLMをスキップする
    exited_chars: list[dict] = getattr(session, "exited_chars", None) or []
    already_exited = any(e["char_name"] == char_name_for_check for e in exited_chars)

    # チャット履歴インデックス登録用にセッション参加キャラIDとユーザ名を解決する
    _chat_char_ids = get_participant_char_ids(session, state.sqlite)
    _chat_user_name = state.sqlite.get_setting("user_name", "ユーザ")

    # --- 応答可能性ゲート（めぐり Phase 5 → 生活カレンダー §5 で一般化） ---
    # 同期 SSE で即時応答するのは OnTime（対面含む）だけ。それ以外（active / busy /
    # offline）のユーザ発言は預かり（delivered_at=NULL）で保存だけして LLM を呼ばない。
    # active / busy は能動配達スケジューラがチェック間隔×返信率で配達・返信し、
    # offline は従来どおり次の非 offline チェック点（または次のユーザリクエスト）で
    # まとめてキャラに渡る。従来キャラ（生活カレンダー無効）は available が
    # OnTime / offline の二値に写るため、挙動は従来と変わらない。
    from backend.services.gate import check_availability, is_usual_scene_running
    _availability = check_availability(
        char_for_estranged,
        usual_scene_running=(
            is_usual_scene_running(state.sqlite, char_for_estranged.id)
            if char_for_estranged else False
        ),
        sqlite=state.sqlite,
    )
    _sync_response = _availability.state == "OnTime"

    user_msg_id = str(uuid.uuid4())
    user_msg = state.sqlite.create_chat_message(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=body.content,
        images=body.image_ids or None,
        face_to_face=current_face_to_face,
        delivered=_sync_response,
    )
    asyncio.create_task(asyncio.to_thread(
        index_message_sync, user_msg, _chat_char_ids, state.vector_store, _chat_user_name
    ))

    if not _sync_response and not already_exited:
        # 預かり通知（システムメッセージ）: 既存の退席フローと同じ SSE 形で返す。
        # 通知自体は保存する（リロード後もユーザが状況を確認できるように）。
        # offline（席にいない）と active/busy（いるが手が離せない）で文言を分ける。
        _reason = _availability.reason or "予定あり"
        if _availability.state == "offline":
            escrow_text = (
                f"{char_name_for_check}はいま席を外しています（{_reason}）。"
                "メッセージは届いていて、戻ったら読みます。"
            )
        else:
            escrow_text = (
                f"{char_name_for_check}はいま手が離せないようです（{_reason}）。"
                "メッセージは届いていて、手が空いたときに読みます。"
            )
        escrow_sys_msg = state.sqlite.create_chat_message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="character",
            content=escrow_text,
            character_name=char_name_for_check,
            is_system_message=True,
            face_to_face=current_face_to_face,
        )

        async def escrow_generator():
            """預かり（escrow）向け SSE ジェネレーター。LLM は呼ばない。"""
            done_data = json.dumps({
                "type": "done",
                "user_message": message_to_dict(user_msg),
                "character_message": message_to_dict(escrow_sys_msg),
            }, ensure_ascii=False)
            yield f"data: {done_data}\n\n"

        return StreamingResponse(
            escrow_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    if already_exited:
        # 退席済み → 全退席者のシステムメッセージを1件返す
        sys_text = _build_all_exited_message(exited_chars)
        sys_msg_id = str(uuid.uuid4())
        sys_msg = state.sqlite.create_chat_message(
            message_id=sys_msg_id,
            session_id=session_id,
            role="character",
            content=sys_text,
            character_name=char_name_for_check,
            is_system_message=True,
            face_to_face=current_face_to_face,
        )

        async def already_exited_generator():
            """退席済みセッション向けSSEジェネレーター。"""
            done_data = json.dumps({
                "type": "done",
                "user_message": message_to_dict(user_msg),
                "character_message": message_to_dict(sys_msg),
            }, ensure_ascii=False)
            yield f"data: {done_data}\n\n"

        return StreamingResponse(
            already_exited_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    history_before = state.sqlite.list_chat_messages(session_id)
    history = [m for m in history_before if m.id != user_msg_id]

    # 預かり分の配達（めぐり Phase 5）: available でここまで来たら、過去の預かり
    # メッセージをまとめて配達する。LLM に渡すコピーにだけ時間差注釈を付け
    # （DB の本文は変更しない）、delivered_at と chat.message 封筒を確定させる。
    from backend.services.gate import format_escrow_annotation
    _pending = [
        m for m in history
        if getattr(m, "delivered_at", None) is None
        and not getattr(m, "is_system_message", None)
    ]
    if _pending:
        for m in _pending:
            m.content = format_escrow_annotation(m)
        state.sqlite.mark_messages_delivered([m.id for m in _pending])

    if len(history) == 0 and session.title == "新しいチャット":
        effective_title = body.content[:30].replace("\n", " ")
        state.sqlite.update_chat_session(session_id, title=effective_title)
    else:
        effective_title = session.title

    user_content: str | list = build_message_content(
        body.content, body.image_ids or [], state.sqlite, state.uploads_dir
    )

    try:
        chat_request = await build_1on1_chat_request(state, session, history, user_content, model_id=effective_model_id)
    except HTTPException:
        raise

    async def sse_generator():
        """通常チャット向けSSEジェネレーター。"""
        nonlocal effective_model_id
        full_text = ""
        accumulated_reasoning = ""
        anticipation_text = ""
        # 1on1 は SceneLoop の最小構成（max_iterations=1）で回す。Scenario と同じ
        # フロー骨に乗ることで、将来的な拡張（複数ターン連鎖、エラーリトライ等）も
        # Strategy 差し替えだけで対応できるようにしておく。
        scene_loop = SceneLoop(
            router=OneOnOneRouter(),
            executor=OneOnOneExecutor(state.chat_service),
            max_iterations=1,
        )
        loop_state = LoopState(context={"pending_request": chat_request})
        try:
            async for event in scene_loop.run(initial_state=loop_state):
                # SceneLoop 終端通知は API 層では使わない（テキスト末尾の確定処理は
                # 既存どおり ``done`` イベントで送る）。
                if event[0] == "loop_complete":
                    continue
                chunk_type, content = event
                if chunk_type == "inscribed_memories":
                    display = format_memories_for_sse(content)
                    if display:
                        accumulated_reasoning += display
                        data = json.dumps({"type": "reasoning", "content": display}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                elif chunk_type == "recall_error":
                    # 想起失敗メッセージを reasoning 行として流す。
                    # 記憶行/スレッド行のパターンに一致しないため、フロントではスケッチ欄に表示される。
                    display = content + "\n"
                    accumulated_reasoning += display
                    data = json.dumps({"type": "reasoning", "content": display}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                elif chunk_type == "working_memory_threads":
                    display = format_recalled_threads(content)
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
                    full_text += content
                    if content:
                        data = json.dumps({"type": "chunk", "content": content}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                elif chunk_type == "anticipation":
                    # キャラクターの予想（期待）。UIには本文として流さず、DB保存して次ターンに注入する。
                    anticipation_text = content
                elif chunk_type == "angle_switched":
                    effective_model_id = content["model_id"]
                    # Frontend が selectedModel を更新できるよう SSE で通知する。
                    # 通知しないと次ターン以降も古い model_id でリクエストされ、
                    # switch_angle の効果がそのターン限りで消えてしまう。
                    data = json.dumps({"type": "angle_switched", "model_id": content["model_id"]}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
        except Exception as e:
            err_data = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {err_data}\n\n"
            return

        clean_text = full_text
        # log_front_output は service.py 内で呼び出し済みのため、ここでは不要
        if accumulated_reasoning:
            logger.log_reasoning(accumulated_reasoning)
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
            log_message_id=log_msg_id,
            anticipation=anticipation_text or None,
            face_to_face=current_face_to_face,
        )
        asyncio.create_task(asyncio.to_thread(
            index_message_sync, char_msg, _chat_char_ids, state.vector_store, _chat_user_name
        ))

        # 計器 Tier 2: キャラ応答の外形スキャン（フォーマット残骸・エラー形状・
        # Assistant 混入・言語逸脱）。誤検知許容の smell 記録で、失敗しても本流は止めない。
        from backend.services.instruments.tier2 import record_response_smells
        record_response_smells(
            state.sqlite, clean_text, character_name=used_char_name, feature="chat",
        )

        state.sqlite.update_chat_session(session_id, title=effective_title, model_id=effective_model_id)

        done_data = json.dumps({
            "type": "done",
            "log_message_id": log_msg_id,
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
