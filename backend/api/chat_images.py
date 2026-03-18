"""チャット添付画像 API。

画像のアップロード・配信を担当する。
セッション管理・メッセージ送信: chat.py
SELF_DRIFT: chat_drifts.py
"""

import os
import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/chat", tags=["chat_images"])


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
