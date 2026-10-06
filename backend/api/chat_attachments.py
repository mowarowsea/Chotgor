"""チャット添付ファイル API。

添付（画像・音声・テキスト）のアップロード・配信を担当する。
受け入れ可否の判定は lib/attachments に一本化されている。
セッション管理・メッセージ送信: chat.py
"""

import os
import uuid
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend.lib.attachments import (
    MAX_TEXT_ATTACHMENT_BYTES,
    attachment_kind,
    resolve_upload_mime,
)

router = APIRouter(prefix="/api/chat", tags=["chat_attachments"])


@router.post("/sessions/{session_id}/attachments", status_code=201)
async def upload_attachments(
    request: Request,
    session_id: str,
    files: list[UploadFile] = File(...),
):
    """複数の添付ファイルをアップロードしてセッションに紐づける。

    受け付けるMIMEタイプ: attachment_kind が種別を導出できるもの（画像・音声・テキスト）。
    MIME が当てにならないテキストは拡張子から text/plain へ正規化して保存する。
    ファイルは uploads_dir/{attachment_id} として保存される。

    ここで見るのは「Chotgor が扱える添付か」だけ。プロバイダー適合（音声を渡せるか）
    はセッションのプリセット次第で変わるため、送信時（chat.py）に別途判定する。

    Returns:
        [{"id": attachment_id, "url": "/api/chat/attachments/{attachment_id}",
          "mime_type": ..., "kind": "image"|"audio"|"text", "filename": ...}] の形式で返す。
    """
    state = request.app.state
    session = state.sqlite.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    results = []
    for file in files:
        mime = resolve_upload_mime(file.content_type, file.filename)
        kind = attachment_kind(mime)
        if kind is None:
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' は扱えない形式です（{mime or '不明'}）",
            )
        data = await file.read()
        if kind == "text" and len(data) > MAX_TEXT_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' は大きすぎます（テキストは 1 MB まで）",
            )
        attachment_id = str(uuid.uuid4())
        att_path = os.path.join(state.uploads_dir, attachment_id)
        with open(att_path, "wb") as f:
            f.write(data)
        state.sqlite.create_chat_attachment(
            attachment_id=attachment_id,
            session_id=session_id,
            mime_type=mime,
            filename=file.filename,
        )
        results.append({
            "id": attachment_id,
            "url": f"/api/chat/attachments/{attachment_id}",
            "mime_type": mime,
            "kind": kind,
            "filename": file.filename,
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
    media_type = att.mime_type
    # テキストは charset が無いとブラウザが日本語を化かす。保存は原文のバイト列のまま。
    if attachment_kind(media_type) == "text" and "charset" not in media_type.lower():
        media_type = f"{media_type}; charset=utf-8"
    return FileResponse(att_path, media_type=media_type)
