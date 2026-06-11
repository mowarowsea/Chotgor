"""InscribedMemory 管理 REST API。"""

from fastapi import APIRouter, HTTPException, Query, Request

from backend.batch.chronicle_job import run_chronicle, run_pending_chronicles
from backend.batch.forget_job import run_forget_process, run_pending_forget
from backend.lib.log_context import new_message_id

router = APIRouter(prefix="/api/inscribed_memories", tags=["inscribed_memories"])


@router.get("/{character_id}")
async def list_memories(
    request: Request,
    character_id: str,
    category: str | None = Query(None),
    include_deleted: bool = Query(False),
    sort_by: str = Query("created_at", description="ソートキー: created_at / updated_at"),
):
    """キャラクターの記憶一覧を返す。"""
    # キャラクター存在確認
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    memories = request.app.state.memory_manager.list_inscribed_memories(
        character_id=character_id,
        category=category,
        include_deleted=include_deleted,
        sort_by=sort_by,
    )
    return memories


@router.delete("/{character_id}/{memory_id}", status_code=204)
async def delete_memory(request: Request, character_id: str, memory_id: str):
    """Soft-delete a memory."""
    ok = request.app.state.memory_manager.delete_inscribed_memory(memory_id, character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="InscribedMemory not found")


@router.post("/{character_id}/{memory_id}/restore", status_code=200)
async def restore_memory(request: Request, character_id: str, memory_id: str):
    """Restore a soft-deleted memory."""
    ok = request.app.state.memory_manager.restore_inscribed_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="InscribedMemory not found")
    return {"status": "restored"}


# 静的パス（/batch/...）は動的パス（/{character_id}/...）より先に定義してルーティング衝突を防ぐ。

@router.post("/batch/chronicle", status_code=200)
async def trigger_batch_chronicle(request: Request):
    """全キャラクターを対象に chronicle 処理を一括実行する。

    スケジューラーと同じ run_pending_chronicles を呼び出す。
    各キャラクターの未処理メッセージをすべて対象とする。
    """
    await run_pending_chronicles(
        sqlite=request.app.state.sqlite,
        vector_store=request.app.state.vector_store,
        memory_manager=request.app.state.memory_manager,
        working_memory_manager=request.app.state.working_memory_manager,
    )
    return {"status": "ok"}


@router.post("/batch/forget", status_code=200)
async def trigger_batch_forget(request: Request):
    """全キャラクターを対象に forget 処理を一括実行する。

    スケジューラーと同じ run_pending_forget を呼び出す。
    ghost_model 未設定のキャラクターはスキップされる。
    """
    await run_pending_forget(
        sqlite=request.app.state.sqlite,
        memory_manager=request.app.state.memory_manager,
    )
    return {"status": "ok"}


@router.post("/{character_id}/chronicle", status_code=200)
async def trigger_chronicle(
    request: Request,
    character_id: str,
    target_date: str | None = Query(None, description="YYYY-MM-DD — 省略時は chronicled_at IS NULL のメッセージ全件を対象とする"),
):
    """指定キャラクターの chronicle 処理を手動実行する。

    ワーキングメモリの棚卸し（スレッド更新・新規・統合・Close）と、
    長期記憶・inner_narrative への蒸留をキャラクター自身に判断させる。
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
        vector_store=request.app.state.vector_store,
        memory_manager=request.app.state.memory_manager,
        working_memory_manager=request.app.state.working_memory_manager,
    )
    return result


@router.post("/{character_id}/forget", status_code=200)
async def trigger_forget(
    request: Request,
    character_id: str,
    threshold: float = Query(0.2, description="忘却判定の閾値（decayed_score がこれ未満の記憶が対象）"),
):
    """指定キャラクターの forget 処理を手動実行する。

    decayed_score が閾値を下回る長期記憶を対象に、キャラクター自身が「芯に残るもの」を
    inner_narrative へ昇華（carve_narrative）し、残りを手放す（soft-delete）。
    ghost_model が未設定のキャラクターはエラーを返す。
    """
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    new_message_id()
    settings = request.app.state.sqlite.get_all_settings()
    result = await run_forget_process(
        character_id=character_id,
        character_name=char.name,
        memory_manager=request.app.state.memory_manager,
        sqlite=request.app.state.sqlite,
        settings=settings,
        threshold=threshold,
        ghost_model=char.ghost_model,
    )
    return result
