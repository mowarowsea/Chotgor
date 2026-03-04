"""Character CRUD REST API."""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from .schemas import CharacterCreate, CharacterUpdate

router = APIRouter(prefix="/api/characters", tags=["characters"])


def _char_to_dict(char) -> dict:
    return {
        "id": char.id,
        "name": char.name,
        "system_prompt_block1": char.system_prompt_block1,
        "meta_instructions": char.meta_instructions,
        "cleanup_config": char.cleanup_config,
        "created_at": char.created_at.isoformat() if char.created_at else None,
        "updated_at": char.updated_at.isoformat() if char.updated_at else None,
    }


@router.get("/")
async def list_characters(request: Request):
    chars = request.app.state.sqlite.list_characters()
    return [_char_to_dict(c) for c in chars]


@router.post("/", status_code=201)
async def create_character(request: Request, body: CharacterCreate):
    char_id = str(uuid.uuid4())
    char = request.app.state.sqlite.create_character(
        character_id=char_id,
        name=body.name,
        system_prompt_block1=body.system_prompt_block1,
        meta_instructions=body.meta_instructions,
        cleanup_config=body.cleanup_config,
    )
    return _char_to_dict(char)


@router.get("/{character_id}")
async def get_character(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return _char_to_dict(char)


@router.patch("/{character_id}")
async def update_character(request: Request, character_id: str, body: CharacterUpdate):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    char = request.app.state.sqlite.update_character(character_id, **updates)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return _char_to_dict(char)


@router.delete("/{character_id}", status_code=204)
async def delete_character(request: Request, character_id: str):
    # Also clean up ChromaDB memories
    request.app.state.chroma.delete_all_memories(character_id)
    ok = request.app.state.sqlite.delete_character(character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Character not found")
