"""チャット添付ファイル API。

添付（画像・音声）のアップロード・配信を担当する。
受け入れ可否の判定は lib/attachments に一本化されている。
セッション管理・メッセージ送信: chat.py
"""

import os
import uuid
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend.lib.attachments import attachment_kind

router = APIRouter(prefix="/api/chat", tags=["chat_attachments"])


@router.post("/sessions/{session_id}/attachments", status_code=201)
async def upload_attachments(
    request: Request,
    session_id: str,
    files: list[UploadFile] = File(...),
):
    """複数の添付ファイルをアップロードしてセッションに紐づける。

    受け付けるMIMEタイプ: attachment_kind が種別を導出できるもの（画像・音声）。
    ファイルは uploads_dir/{attachment_id} として保存される。

    ここで見るのは「Chotgor が扱える添付か」だけ。プロバイダー適合（音声を渡せるか）
    はセッションのプリセット次第で変わるため、送信時（chat.py）に別途判定する。

    Returns:
        [{"id": attachment_id, "url": "/api/chat/attachments/{attachment_id}",
          "mime_type": ..., "kind": "image"|"audio"}] の形式で返す。
    """
    state = request.app.state
    session = state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    results = []
    for file in files:
        mime = file.content_type or ""
        kind = attachment_kind(mime)
        if kind is None:
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' は扱えない形式です（{mime or '不明'}）",
            )
        attachment_id = str(uuid.uuid4())
        data = await file.read()
        att_path = os.path.join(state.uploads_dir, attachment_id)
        with open(att_path, "wb") as f:
            f.write(data)
        state.sqlite.create_chat_attachment(
            attachment_id=attachment_id,
            session_id=session_id,
            mime_type=mime,
        )
        results.append({
            "id": attachment_id,
            "url": f"/api/chat/attachments/{attachment_id}",
            "mime_type": mime,
            "kind": kind,
        })
    return results


@router.get("/attachments/{attachment_id}")
async def get_attachment(request: Request, attachment_id: str):
    """添付ファイルを配信する。"""
    att = request.app.state.sqlite.get_chat_attachment(attachment_id)
    if not att:
        raise HTTPException(status_code=404, detail="Attachment not found")
    att_path = os.path.join(request.app.state.uploads_dir, attachment_id)
    if not os.path.exists(att_path):
        raise HTTPException(status_code=404, detail="Attachment file not found")
    return FileResponse(att_path, media_type=att.mime_type)
