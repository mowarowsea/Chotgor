"""Memory management REST API."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from backend.batch.chronicle_job import run_chronicle
from backend.lib.log_context import new_message_id

router = APIRouter(prefix="/api/memories", tags=["memories"])


@router.get("/{character_id}")
async def list_memories(
    request: Request,
    character_id: str,
    category: Optional[str] = Query(None),
    include_deleted: bool = Query(False),
    sort_by: str = Query("created_at", description="ソートキー: created_at / updated_at"),
):
    """キャラクターの記憶一覧を返す。"""
    # Verify character exists
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    memories = request.app.state.memory_manager.list_memories(
        character_id=character_id,
        category=category,
        include_deleted=include_deleted,
        sort_by=sort_by,
    )
    return memories


@router.delete("/{character_id}/{memory_id}", status_code=204)
async def delete_memory(request: Request, character_id: str, memory_id: str):
    """Soft-delete a memory."""
    ok = request.app.state.memory_manager.delete_memory(memory_id, character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")


@router.post("/{character_id}/{memory_id}/restore", status_code=200)
async def restore_memory(request: Request, character_id: str, memory_id: str):
    """Restore a soft-deleted memory."""
    ok = request.app.state.memory_manager.restore_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "restored"}


@router.post("/{character_id}/chronicle", status_code=200)
async def trigger_chronicle(
    request: Request,
    character_id: str,
    target_date: Optional[str] = Query(None, description="YYYY-MM-DD — 省略時は chronicled_at IS NULL のメッセージ全件を対象とする"),
):
    """指定キャラクターの chronicle 処理を手動実行する。

    self_history / relationship_state の更新をキャラクター自身に判断させる。
    target_date を省略した場合、未処理（chronicled_at IS NULL）のメッセージをすべて対象とする。
    """
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    new_message_id()
    result = await run_chronicle(
        character_id=character_id,
        sqlite=request.app.state.sqlite,
        target_date=target_date,
    )
    return result
