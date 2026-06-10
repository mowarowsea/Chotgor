"""設定 UI — 保存記憶＆ワーキングメモリ閲覧ページ。"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from backend.api.ui.common import get_templates

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/inscribed_memories/{character_id}", response_class=HTMLResponse)
async def memories_view(
    request: Request,
    character_id: str,
    category: str | None = None,
    deleted_only: bool = False,
):
    """記憶一覧ページ。deleted_only=True のときは論理削除済み記憶のみ表示する。"""
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)

    memories = request.app.state.memory_manager.list_inscribed_memories(
        character_id=character_id,
        category=category,
        include_deleted=deleted_only,
    )
    # deleted_only モードでは削除済みレコードのみに絞る
    if deleted_only:
        memories = [m for m in memories if m["deleted_at"]]

    # プリセットID→名前マップを構築し、記憶作成元プリセット名をテンプレートで表示できるようにする
    presets = request.app.state.sqlite.list_model_presets()
    preset_name_map = {p.id: p.name for p in presets}

    categories = ["identity", "user", "semantic", "contextual"]
    return get_templates().TemplateResponse(
        "memories.html",
        {
            "request": request,
            "character": char,
            "memories": memories,
            "categories": categories,
            "selected_category": category,
            "deleted_only": deleted_only,
            "preset_name_map": preset_name_map,
        },
    )


@router.post("/inscribed_memories/{character_id}/{memory_id}/delete")
async def delete_memory(request: Request, character_id: str, memory_id: str):
    request.app.state.memory_manager.delete_inscribed_memory(memory_id, character_id)
    return RedirectResponse(url=f"/ui/inscribed_memories/{character_id}", status_code=303)


@router.get("/working-memory/{character_id}", response_class=HTMLResponse)
async def working_memory_view(
    request: Request,
    character_id: str,
    type: str | None = None,
    archived: bool = False,
):
    """ワーキングメモリのスレッド一覧ページ（読み取り専用）。

    type 指定でスレッド種別を絞り込み、archived=True でアーカイブ済みのみ表示する。
    記憶の取捨選択はキャラクター自身が行うため、UI からの編集機能は設けない。
    """
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)

    wm = request.app.state.working_memory_manager
    threads = wm.list_threads_by_type(
        character_id,
        type=type or None,
        is_open=(False if archived else None),
    )
    types = ["emotion", "body", "task", "topic", "relation"]
    return get_templates().TemplateResponse(
        "working_memory.html",
        {
            "request": request,
            "character": char,
            "threads": threads,
            "types": types,
            "selected_type": type,
            "archived": archived,
        },
    )


@router.get("/working-memory/{character_id}/{thread_id}", response_class=HTMLResponse)
async def working_memory_thread_view(request: Request, character_id: str, thread_id: str):
    """ワーキングメモリのスレッド詳細ページ（全ポストを時系列表示）。"""
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)

    wm = request.app.state.working_memory_manager
    thread = wm.get_thread_detail(thread_id)
    if thread is None or thread.get("character_id") != character_id:
        return RedirectResponse(url=f"/ui/working-memory/{character_id}", status_code=303)

    return get_templates().TemplateResponse(
        "working_memory_thread.html",
        {
            "request": request,
            "character": char,
            "thread": thread,
        },
    )


# --- Model Presets ---

