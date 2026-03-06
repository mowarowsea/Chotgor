"""Settings UI routes using Jinja2 templates."""

import base64
import uuid
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..core.providers.registry import (
    PROVIDER_LABELS,
    PROVIDER_ORDER,
    get_default_model,
)

router = APIRouter(prefix="/ui", tags=["ui"])


MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2MB


async def _read_image_data(form) -> Optional[str]:
    """フォームから画像を読み込みbase64 data URIとして返す。画像がなければNone。"""
    image_file = form.get("image")
    if not image_file or not hasattr(image_file, "read"):
        return None
    content = await image_file.read()
    if not content:
        return None
    if len(content) > MAX_IMAGE_BYTES:
        return None  # サイズ超過は無視
    content_type = image_file.content_type or "image/png"
    b64 = base64.b64encode(content).decode()
    return f"data:{content_type};base64,{b64}"


def _available_providers(settings: dict) -> list[str]:
    """Return providers usable in the UI (claude_cli always, others if API key set)."""
    result = ["claude_cli"]
    for p in ("anthropic", "openai", "xai", "google"):
        if settings.get(f"{p}_api_key"):
            result.append(p)
    return result


def setup_templates(templates_dir: str) -> Jinja2Templates:
    return Jinja2Templates(directory=templates_dir)


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
    settings = request.app.state.sqlite.get_all_settings()
    return get_templates().TemplateResponse(
        "character_edit.html",
        {
            "request": request,
            "character": None,
            "action": "/ui/characters/new",
            "available_providers": _available_providers(settings),
            "provider_labels": PROVIDER_LABELS,
            "provider_default_models": {p: get_default_model(p) for p in PROVIDER_ORDER},
        },
    )


@router.post("/characters/new")
async def create_character(request: Request):
    form = await request.form()
    name = (form.get("name") or "").strip()
    if not name:
        return RedirectResponse(url="/ui/characters/new", status_code=303)

    settings = request.app.state.sqlite.get_all_settings()
    available = _available_providers(settings)

    enabled_providers = {}
    for p in available:
        if form.get(f"provider_{p}_enabled"):
            enabled_providers[p] = {
                "model": form.get(f"provider_{p}_model", ""),
                "additional_instructions": form.get(f"provider_{p}_additional_instructions", ""),
            }

    image_data = await _read_image_data(form)

    char_id = str(uuid.uuid4())
    request.app.state.sqlite.create_character(
        character_id=char_id,
        name=name,
        system_prompt_block1=form.get("system_prompt_block1", ""),
        meta_instructions=form.get("meta_instructions", ""),
        cleanup_config={"digest_delete_originals": bool(form.get("digest_delete_originals"))},
        enabled_providers=enabled_providers,
        image_data=image_data,
    )
    return RedirectResponse(url="/ui/", status_code=303)


@router.get("/characters/{character_id}", response_class=HTMLResponse)
async def edit_character_form(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)
    settings = request.app.state.sqlite.get_all_settings()
    return get_templates().TemplateResponse(
        "character_edit.html",
        {
            "request": request,
            "character": char,
            "action": f"/ui/characters/{character_id}",
            "available_providers": _available_providers(settings),
            "provider_labels": PROVIDER_LABELS,
            "provider_default_models": {p: get_default_model(p) for p in PROVIDER_ORDER},
        },
    )


@router.post("/characters/{character_id}")
async def update_character(request: Request, character_id: str):
    form = await request.form()
    settings = request.app.state.sqlite.get_all_settings()
    available = _available_providers(settings)

    enabled_providers = {}
    for p in available:
        if form.get(f"provider_{p}_enabled"):
            enabled_providers[p] = {
                "model": form.get(f"provider_{p}_model", ""),
                "additional_instructions": form.get(f"provider_{p}_additional_instructions", ""),
            }

    char = request.app.state.sqlite.get_character(character_id)
    existing_config = (char.cleanup_config or {}) if char else {}
    existing_config["digest_delete_originals"] = bool(form.get("digest_delete_originals"))

    update_kwargs: dict = dict(
        name=(form.get("name") or "").strip(),
        system_prompt_block1=form.get("system_prompt_block1", ""),
        meta_instructions=form.get("meta_instructions", ""),
        cleanup_config=existing_config,
        enabled_providers=enabled_providers,
    )
    new_image = await _read_image_data(form)
    if new_image:
        update_kwargs["image_data"] = new_image

    request.app.state.sqlite.update_character(character_id, **update_kwargs)
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
    categories = ["identity", "user", "semantic", "contextual"]
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
    anthropic_api_key: str = Form(""),
    openai_api_key: str = Form(""),
    xai_api_key: str = Form(""),
    google_api_key: str = Form(""),
    tavily_api_key: str = Form(""),
    digest_time: str = Form("03:00"),
):
    store = request.app.state.sqlite
    for key, value in [
        ("anthropic_api_key", anthropic_api_key),
        ("openai_api_key", openai_api_key),
        ("xai_api_key", xai_api_key),
        ("google_api_key", google_api_key),
        ("tavily_api_key", tavily_api_key),
    ]:
        if value and set(value) != {"●"}:
            store.set_setting(key, value)
    try:
        h, m = map(int, digest_time.split(":"))
        assert 0 <= h <= 23 and 0 <= m <= 59
        store.set_setting("digest_time", digest_time)
    except Exception:
        pass
    return RedirectResponse(url="/ui/settings?saved=1", status_code=303)
