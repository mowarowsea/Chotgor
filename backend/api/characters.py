"""Character CRUD REST API."""

import base64
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response

from backend.api.schemas import CharacterCreate, CharacterUpdate
from backend.api.utils import char_to_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/characters", tags=["characters"])


@router.get("/")
async def list_characters(request: Request):
    chars = request.app.state.sqlite.list_characters()
    return [char_to_dict(c) for c in chars]


@router.post("/", status_code=201)
async def create_character(request: Request, body: CharacterCreate):
    """キャラクターを新規作成する。

    estranged キャラクターと類似する定義の場合は HTTP 409 を返す。
    作成後にキャラクター定義を ChromaDB に登録する。
    """
    state = request.app.state

    # 類似 estranged キャラクターのチェック: 同一定義での再作成を防ぐ
    if body.system_prompt_block1 and hasattr(state, "chroma") and state.chroma:
        try:
            similar = state.chroma.find_similar_definition(body.system_prompt_block1)
            if similar:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "類似する定義を持つキャラクターが既に別れを決断しています。"
                        "この定義ではキャラクターを作成できません。"
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            pass  # ChromaDB エラーは無視して作成を続行する

    char_id = str(uuid.uuid4())
    char = state.sqlite.create_character(
        character_id=char_id,
        name=body.name,
        system_prompt_block1=body.system_prompt_block1,
        inner_narrative=body.inner_narrative,
        cleanup_config=body.cleanup_config,
        ghost_model=body.ghost_model,
        allowed_tools=body.allowed_tools,
    )

    # キャラクター定義を ChromaDB に登録する（embedding 作成）
    if body.system_prompt_block1 and hasattr(state, "chroma") and state.chroma:
        try:
            state.chroma.upsert_character_definition(char_id, body.system_prompt_block1)
        except Exception as e:
            logger.warning("ChromaDB キャラクター定義登録失敗 char=%s error=%s", char_id, e)

    return char_to_dict(char)


@router.get("/{character_id}")
async def get_character(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return char_to_dict(char)


@router.patch("/{character_id}")
async def update_character(request: Request, character_id: str, body: CharacterUpdate):
    """キャラクター情報を更新する。

    system_prompt_block1 が変更された場合は ChromaDB の定義 embedding も更新する。
    """
    state = request.app.state
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    char = state.sqlite.update_character(character_id, **updates)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    # system_prompt_block1 が更新された場合は定義 embedding を再登録する
    if "system_prompt_block1" in updates and hasattr(state, "chroma") and state.chroma:
        try:
            state.chroma.upsert_character_definition(character_id, updates["system_prompt_block1"])
        except Exception as e:
            logger.warning("ChromaDB キャラクター定義更新失敗 char=%s error=%s", character_id, e)

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
    """キャラクターと全記憶を削除する。ChromaDB・SQLite の順に削除する。"""
    ok = request.app.state.memory_manager.delete_character_with_memories(character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Character not found")
