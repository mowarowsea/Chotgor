"""SELF_DRIFT API。

キャラクターがチャット内で自分自身に課した一時的な行動指針の取得・toggle・リセットを担当する。
セッション管理・メッセージ送信: chat.py
画像管理: chat_images.py
"""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/chat", tags=["chat_drifts"])


def _drift_to_dict(drift) -> dict:
    """SessionDrift レコードをAPIレスポンス用dictに変換する。"""
    return {
        "id": drift.id,
        "session_id": drift.session_id,
        "character_id": drift.character_id,
        "content": drift.content,
        "enabled": bool(drift.enabled),
        "created_at": drift.created_at.isoformat() if drift.created_at else None,
    }


@router.get("/sessions/{session_id}/drifts")
async def list_drifts(request: Request, session_id: str):
    """セッションの全キャラのSELF_DRIFT一覧を返す。"""
    session = request.app.state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    drifts = request.app.state.drift_manager.list_drifts(session_id)
    return [_drift_to_dict(d) for d in drifts]


@router.patch("/sessions/{session_id}/drifts/{drift_id}/toggle")
async def toggle_drift(request: Request, session_id: str, drift_id: str):
    """SELF_DRIFT の enabled フラグを反転する。"""
    session = request.app.state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    drift = request.app.state.drift_manager.toggle_drift(drift_id)
    if not drift:
        raise HTTPException(status_code=404, detail="Drift not found")
    return _drift_to_dict(drift)


@router.delete("/sessions/{session_id}/drifts", status_code=204)
async def reset_drifts(request: Request, session_id: str, character_id: str):
    """指定キャラの全SELF_DRIFTを削除する。

    Query params:
        character_id: リセット対象のキャラクターID。
    """
    session = request.app.state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    request.app.state.drift_manager.reset_drifts(session_id, character_id)
