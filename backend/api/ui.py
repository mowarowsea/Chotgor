"""Settings UI routes using Jinja2 templates."""

import uuid
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/ui", tags=["ui"])


def setup_templates(templates_dir: str) -> Jinja2Templates:
    return Jinja2Templates(directory=templates_dir)


# Templates instance is set during app startup
templates: Optional[Jinja2Templates] = None


def get_templates() -> Jinja2Templates:
    if templates is None:
        raise RuntimeError("Templates not initialized")
    return templates


# --- Dashboard ---

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    chars = request.app.state.sqlite.list_characters()
    return get_templates().TemplateResponse(
        "dashboard.html",
        {"request": request, "characters": chars},
    )


# --- Characters ---

@router.get("/characters/new", response_class=HTMLResponse)
async def new_character_form(request: Request):
    return get_templates().TemplateResponse(
        "character_edit.html",
        {"request": request, "character": None, "action": "/ui/characters/new"},
    )


@router.post("/characters/new")
async def create_character(
    request: Request,
    name: str = Form(...),
    system_prompt_block1: str = Form(""),
    meta_instructions: str = Form(""),
    digest_delete_originals: bool = Form(False),
):
    char_id = str(uuid.uuid4())
    request.app.state.sqlite.create_character(
        character_id=char_id,
        name=name,
        system_prompt_block1=system_prompt_block1,
        meta_instructions=meta_instructions,
        cleanup_config={"digest_delete_originals": digest_delete_originals},
    )
    return RedirectResponse(url="/ui/", status_code=303)


@router.get("/characters/{character_id}", response_class=HTMLResponse)
async def edit_character_form(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)
    return get_templates().TemplateResponse(
        "character_edit.html",
        {
            "request": request,
            "character": char,
            "action": f"/ui/characters/{character_id}",
        },
    )


@router.post("/characters/{character_id}")
async def update_character(
    request: Request,
    character_id: str,
    name: str = Form(...),
    system_prompt_block1: str = Form(""),
    meta_instructions: str = Form(""),
    digest_delete_originals: bool = Form(False),
):
    char = request.app.state.sqlite.get_character(character_id)
    existing_config = (char.cleanup_config or {}) if char else {}
    existing_config["digest_delete_originals"] = digest_delete_originals
    request.app.state.sqlite.update_character(
        character_id,
        name=name,
        system_prompt_block1=system_prompt_block1,
        meta_instructions=meta_instructions,
        cleanup_config=existing_config,
    )
    return RedirectResponse(url="/ui/", status_code=303)


@router.post("/characters/{character_id}/delete")
async def delete_character(request: Request, character_id: str):
    request.app.state.chroma.delete_all_memories(character_id)
    request.app.state.sqlite.delete_character(character_id)
    return RedirectResponse(url="/ui/", status_code=303)


# --- Memories ---

@router.get("/memories/{character_id}", response_class=HTMLResponse)
async def memories_view(
    request: Request,
    character_id: str,
    category: Optional[str] = None,
    include_deleted: bool = False,
):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)

    memories = request.app.state.memory_manager.list_memories(
        character_id=character_id,
        category=category,
        include_deleted=include_deleted,
    )
    categories = ["general", "user_preference", "relationship", "event", "fact"]
    return get_templates().TemplateResponse(
        "memories.html",
        {
            "request": request,
            "character": char,
            "memories": memories,
            "categories": categories,
            "selected_category": category,
            "include_deleted": include_deleted,
        },
    )


@router.post("/memories/{character_id}/{memory_id}/delete")
async def delete_memory(request: Request, character_id: str, memory_id: str):
    request.app.state.memory_manager.delete_memory(memory_id, character_id)
    return RedirectResponse(url=f"/ui/memories/{character_id}", status_code=303)


# --- Settings ---

@router.get("/settings", response_class=HTMLResponse)
async def settings_form(request: Request):
    settings = request.app.state.sqlite.get_all_settings()
    return get_templates().TemplateResponse(
        "settings.html",
        {"request": request, "settings": settings},
    )


@router.post("/settings")
async def save_settings(
    request: Request,
    tavily_api_key: str = Form(""),
    digest_time: str = Form("03:00"),
):
    if tavily_api_key:
        request.app.state.sqlite.set_setting("tavily_api_key", tavily_api_key)
    # Validate HH:MM format
    try:
        h, m = map(int, digest_time.split(":"))
        assert 0 <= h <= 23 and 0 <= m <= 59
        request.app.state.sqlite.set_setting("digest_time", digest_time)
    except Exception:
        pass
    return RedirectResponse(url="/ui/settings?saved=1", status_code=303)
