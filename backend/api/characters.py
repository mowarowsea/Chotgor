"""Character CRUD REST API."""

import base64
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response

from .schemas import CharacterCreate, CharacterUpdate
from .utils import char_to_dict

router = APIRouter(prefix="/api/characters", tags=["characters"])


@router.get("/")
async def list_characters(request: Request):
    chars = request.app.state.sqlite.list_characters()
    return [char_to_dict(c) for c in chars]


@router.post("/", status_code=201)
async def create_character(request: Request, body: CharacterCreate):
    char_id = str(uuid.uuid4())
    char = request.app.state.sqlite.create_character(
        character_id=char_id,
        name=body.name,
        system_prompt_block1=body.system_prompt_block1,
        inner_narrative=body.inner_narrative,
        cleanup_config=body.cleanup_config,
        ghost_model=body.ghost_model,
    )
    return char_to_dict(char)


@router.get("/{character_id}")
async def get_character(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return char_to_dict(char)


@router.patch("/{character_id}")
async def update_character(request: Request, character_id: str, body: CharacterUpdate):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    char = request.app.state.sqlite.update_character(character_id, **updates)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return char_to_dict(char)


@router.get("/{character_id}/image")
async def get_character_image(request: Request, character_id: str):
    """キャラクターのアバター画像をバイナリで返す。画像未設定の場合は404を返す。"""
    char = request.app.state.sqlite.get_character(character_id)
    if not char or not char.image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        # "data:{mime_type};base64,{b64}" 形式をデコードする
        header, b64_data = char.image_data.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        image_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(status_code=500, detail="Image data is corrupted")
    return Response(content=image_bytes, media_type=mime_type)


@router.delete("/{character_id}", status_code=204)
async def delete_character(request: Request, character_id: str):
    # Also clean up ChromaDB memories
    request.app.state.chroma.delete_all_memories(character_id)
    ok = request.app.state.sqlite.delete_character(character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Character not found")
