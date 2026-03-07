"""Memory management REST API."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..core.memory.digest import run_daily_digest

router = APIRouter(prefix="/api/memories", tags=["memories"])


@router.get("/{character_id}")
async def list_memories(
    request: Request,
    character_id: str,
    category: Optional[str] = Query(None),
    include_deleted: bool = Query(False),
):
    """List all memories for a character."""
    # Verify character exists
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    memories = request.app.state.memory_manager.list_memories(
        character_id=character_id,
        category=category,
        include_deleted=include_deleted,
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
    ok = request.app.state.sqlite.restore_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "restored"}


@router.post("/{character_id}/digest", status_code=200)
async def trigger_digest(
    request: Request,
    character_id: str,
    target_date: Optional[str] = Query(None, description="YYYY-MM-DD (defaults to yesterday)"),
):
    """Manually trigger a daily digest for a character."""
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).date().isoformat()

    cleanup_config = char.cleanup_config or {}
    delete_originals = cleanup_config.get("digest_delete_originals", False)

    result = await run_daily_digest(
        character_id=character_id,
        character_name=char.name,
        character_system_prompt=char.system_prompt_block1,
        target_date=target_date,
        memory_manager=request.app.state.memory_manager,
        sqlite=request.app.state.sqlite,
        delete_originals=delete_originals,
        ghost_model=char.ghost_model,
    )
    return result


@router.get("/{character_id}/digest-logs", status_code=200)
async def get_digest_logs(
    request: Request,
    character_id: str,
    limit: int = Query(50, ge=1, le=500),
):
    """Return digest log entries for a character."""
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    logs = request.app.state.sqlite.get_digest_logs(character_id, limit=limit)
    return [
        {
            "id": log.id,
            "character_id": log.character_id,
            "digest_date": log.digest_date,
            "status": log.status,
            "memory_id": log.memory_id,
            "memory_count": log.memory_count,
            "message": log.message,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]
