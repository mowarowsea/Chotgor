"""Settings UI routes using Jinja2 templates."""

import base64
import uuid
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.services.memory.migration_service import migrate_embeddings
from backend.providers.registry import (
    PROVIDER_LABELS,
    PROVIDER_ORDER,
)

router = APIRouter(prefix="/ui", tags=["ui"])


MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2MB


def _extract_self_reflection_params(form) -> tuple:
    """フォームから自己参照ループパラメータを抽出する。

    create_character・update_character の両方で使用する。

    Args:
        form: FastAPI のフォームデータ。

    Returns:
        (self_reflection_mode, self_reflection_preset_id, self_reflection_n_turns) のタプル。
    """
    return (
        form.get("self_reflection_mode") or "disabled",
        form.get("self_reflection_preset_id") or None,
        int(form.get("self_reflection_n_turns") or 5),
    )


def _build_allowed_tools(form) -> dict:
    """フォームから allowed_tools 辞書を構築する。

    チェックボックスがONの場合のみ True、未チェックは False になる。

    Args:
        form: await request.form() の結果。

    Returns:
        {web_search, google_calendar, gmail, google_drive} の bool dict。
    """
    return {
        "web_search":       bool(form.get("tool_web_search")),
        "google_calendar":  bool(form.get("tool_google_calendar")),
        "gmail":            bool(form.get("tool_gmail")),
        "google_drive":     bool(form.get("tool_google_drive")),
    }


def _build_enabled_providers(form) -> dict:
    """フォームから enabled_providers 辞書を構築する。

    create_character・update_character の両方で同じロジックが必要なため一元化する。
    preset_ids は複数値フォームフィールドで、各 preset_id に対して
    additional_instructions と when_to_switch を取得して辞書に格納する。

    Args:
        form: await request.form() の結果。

    Returns:
        {preset_id: {additional_instructions, when_to_switch}} の辞書。
    """
    enabled_providers = {}
    for pid in form.getlist("preset_ids"):
        enabled_providers[pid] = {
            "additional_instructions": form.get(f"ai_{pid}", ""),
            "when_to_switch": form.get(f"wts_{pid}", ""),
        }
    return enabled_providers


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
    model_presets = request.app.state.sqlite.list_model_presets()
    return get_templates().TemplateResponse(
        "character_edit.html",
        {
            "request": request,
            "character": None,
            "action": "/ui/characters/new",
            "model_presets": model_presets,
            "provider_labels": PROVIDER_LABELS,
        },
    )


@router.post("/characters/new")
async def create_character(request: Request):
    form = await request.form()
    name = (form.get("name") or "").strip()
    if not name:
        return RedirectResponse(url="/ui/characters/new", status_code=303)

    enabled_providers = _build_enabled_providers(form)
    allowed_tools = _build_allowed_tools(form)

    image_data = await _read_image_data(form)

    char_id = str(uuid.uuid4())
    ghost_model = form.get("ghost_model") or None
    switch_angle_enabled = bool(form.get("switch_angle_enabled"))
    afterglow_default = 1 if form.get("afterglow_default") else 0
    self_reflection_mode, self_reflection_preset_id, self_reflection_n_turns = (
        _extract_self_reflection_params(form)
    )

    request.app.state.sqlite.create_character(
        character_id=char_id,
        name=name,
        system_prompt_block1=form.get("system_prompt_block1", ""),
        enabled_providers=enabled_providers,
        ghost_model=ghost_model,
        image_data=image_data,
        switch_angle_enabled=switch_angle_enabled,
        afterglow_default=afterglow_default,
        self_reflection_mode=self_reflection_mode,
        self_reflection_preset_id=self_reflection_preset_id,
        self_reflection_n_turns=self_reflection_n_turns,
        allowed_tools=allowed_tools,
    )
    return RedirectResponse(url="/ui/", status_code=303)


@router.get("/characters/{character_id}", response_class=HTMLResponse)
async def edit_character_form(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)
    model_presets = request.app.state.sqlite.list_model_presets()
    return get_templates().TemplateResponse(
        "character_edit.html",
        {
            "request": request,
            "character": char,
            "action": f"/ui/characters/{character_id}",
            "model_presets": model_presets,
            "provider_labels": PROVIDER_LABELS,
        },
    )


@router.post("/characters/{character_id}")
async def update_character(request: Request, character_id: str):
    form = await request.form()

    enabled_providers = _build_enabled_providers(form)
    allowed_tools = _build_allowed_tools(form)

    ghost_model = form.get("ghost_model") or None
    switch_angle_enabled = 1 if form.get("switch_angle_enabled") else 0
    afterglow_default = 1 if form.get("afterglow_default") else 0
    self_reflection_mode, self_reflection_preset_id, self_reflection_n_turns = (
        _extract_self_reflection_params(form)
    )

    update_kwargs: dict = dict(
        name=(form.get("name") or "").strip(),
        system_prompt_block1=form.get("system_prompt_block1", ""),
        enabled_providers=enabled_providers,
        ghost_model=ghost_model,
        switch_angle_enabled=switch_angle_enabled,
        afterglow_default=afterglow_default,
        self_reflection_mode=self_reflection_mode,
        self_reflection_preset_id=self_reflection_preset_id,
        self_reflection_n_turns=self_reflection_n_turns,
        allowed_tools=allowed_tools,
    )
    new_image = await _read_image_data(form)
    if new_image:
        update_kwargs["image_data"] = new_image
    elif form.get("remove_image"):
        # 削除フラグが立っている場合は画像をクリアする
        update_kwargs["image_data"] = None

    request.app.state.sqlite.update_character(character_id, **update_kwargs)
    return RedirectResponse(url="/ui/", status_code=303)


@router.post("/characters/{character_id}/delete")
async def delete_character(request: Request, character_id: str):
    """キャラクターと全記憶を削除する。ChromaDB・SQLite の順に削除する。"""
    request.app.state.memory_manager.delete_character_with_memories(character_id)
    return RedirectResponse(url="/ui/", status_code=303)


# --- Memories ---

@router.get("/memories/{character_id}", response_class=HTMLResponse)
async def memories_view(
    request: Request,
    character_id: str,
    category: Optional[str] = None,
    deleted_only: bool = False,
):
    """記憶一覧ページ。deleted_only=True のときは論理削除済み記憶のみ表示する。"""
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)

    memories = request.app.state.memory_manager.list_memories(
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


@router.post("/memories/{character_id}/{memory_id}/delete")
async def delete_memory(request: Request, character_id: str, memory_id: str):
    request.app.state.memory_manager.delete_memory(memory_id, character_id)
    return RedirectResponse(url=f"/ui/memories/{character_id}", status_code=303)


# --- Model Presets ---

@router.get("/model-presets", response_class=HTMLResponse)
async def model_presets_list(request: Request):
    presets = request.app.state.sqlite.list_model_presets()
    return get_templates().TemplateResponse(
        "model_presets.html",
        {
            "request": request,
            "presets": presets,
            "provider_labels": PROVIDER_LABELS,
            "provider_order": PROVIDER_ORDER,
        },
    )


@router.post("/model-presets/new")
async def create_model_preset(request: Request):
    form = await request.form()
    name = (form.get("name") or "").strip()
    provider = (form.get("provider") or "").strip()
    model_id = (form.get("model_id") or "").strip()
    if not name or not provider:
        return RedirectResponse(url="/ui/model-presets", status_code=303)
    preset_id = str(uuid.uuid4())
    thinking_level = form.get("thinking_level") or "default"
    request.app.state.sqlite.create_model_preset(
        preset_id=preset_id,
        name=name,
        provider=provider,
        model_id=model_id,
        thinking_level=thinking_level,
    )
    return RedirectResponse(url="/ui/model-presets", status_code=303)


@router.post("/model-presets/{preset_id}/edit")
async def update_model_preset(request: Request, preset_id: str):
    form = await request.form()
    name = (form.get("name") or "").strip()
    provider = (form.get("provider") or "").strip()
    model_id = (form.get("model_id") or "").strip()
    thinking_level = form.get("thinking_level") or "default"
    if name and provider:
        request.app.state.sqlite.update_model_preset(
            preset_id,
            name=name,
            provider=provider,
            model_id=model_id,
            thinking_level=thinking_level,
        )
    return RedirectResponse(url="/ui/model-presets", status_code=303)


@router.post("/model-presets/{preset_id}/delete")
async def delete_model_preset(request: Request, preset_id: str):
    request.app.state.sqlite.delete_model_preset(preset_id)
    return RedirectResponse(url="/ui/model-presets", status_code=303)


# --- Settings ---

@router.get("/settings", response_class=HTMLResponse)
async def settings_form(request: Request):
    """設定ページを表示する。Google APIキー設定状況とモデルプリセット一覧をテンプレートに渡す。"""
    settings = request.app.state.sqlite.get_all_settings()
    model_presets = request.app.state.sqlite.list_model_presets()
    return get_templates().TemplateResponse(
        "settings.html",
        {
            "request": request,
            "settings": settings,
            "has_google_key": bool(settings.get("google_api_key")),
            "model_presets": model_presets,
        },
    )


@router.post("/settings")
async def save_settings(
    request: Request,
    user_name: str = Form("ユーザ"),
    anthropic_api_key: str = Form(""),
    openai_api_key: str = Form(""),
    xai_api_key: str = Form(""),
    google_api_key: str = Form(""),
    openrouter_api_key: str = Form(""),
    tavily_api_key: str = Form(""),
    chronicle_time: str = Form("03:00"),
    enable_time_awareness: Optional[str] = Form(None),
    context_window_max_chronicled: int = Form(10),
    embedding_provider: str = Form("default"),
    embedding_model: str = Form(""),
    translation_preset_id: str = Form(""),
):
    """設定を保存し、embeddingモデルが変更された場合は記憶を再インデックスする。"""
    store = request.app.state.sqlite

    # embedding設定の変更を検出（APIキー更新前に現在値を取得）
    old_embedding_provider = store.get_setting("embedding_provider", "default")
    old_embedding_model = store.get_setting("embedding_model", "")

    # ユーザ名は常に保存（マスク不要）
    store.set_setting("user_name", user_name.strip() or "ユーザ")

    # APIキーはマスク値（●のみ）でなければ更新する
    for key, value in [
        ("anthropic_api_key", anthropic_api_key),
        ("openai_api_key", openai_api_key),
        ("xai_api_key", xai_api_key),
        ("google_api_key", google_api_key),
        ("openrouter_api_key", openrouter_api_key),
        ("tavily_api_key", tavily_api_key),
    ]:
        if value and set(value) != {"●"}:
            store.set_setting(key, value)

    try:
        h, m = map(int, chronicle_time.split(":"))
        assert 0 <= h <= 23 and 0 <= m <= 59
        store.set_setting("chronicle_time", chronicle_time)
    except Exception:
        pass

    store.set_setting("enable_time_awareness", bool(enable_time_awareness))
    store.set_setting("context_window_max_chronicled", str(max(0, min(200, context_window_max_chronicled))))

    # embedding設定の保存
    store.set_setting("embedding_provider", embedding_provider)
    store.set_setting("embedding_model", embedding_model)

    # 翻訳モデル設定の保存
    store.set_setting("translation_preset_id", translation_preset_id)

    # embeddingモデルが変更された場合は全記憶を再インデックスする
    embedding_changed = (
        embedding_provider != old_embedding_provider
        or embedding_model != old_embedding_model
    )
    if embedding_changed:
        # APIキー保存後の最新値を使用する
        current_google_key = store.get_setting("google_api_key", "")
        try:
            state = request.app.state
            new_chroma, new_memory_manager, new_chat_service = await migrate_embeddings(
                sqlite=state.sqlite,
                old_chroma=state.chroma,
                chroma_db_path=state.chroma_db_path,
                drift_manager=state.drift_manager,
                new_provider=embedding_provider,
                new_model=embedding_model,
                new_api_key=current_google_key,
            )
            # マイグレーション後に app.state を新しいインスタンスで更新する
            state.chroma = new_chroma
            state.memory_manager = new_memory_manager
            state.chat_service = new_chat_service
        except Exception:
            return RedirectResponse(url="/ui/settings?saved=1&migration_error=1", status_code=303)

    return RedirectResponse(url="/ui/settings?saved=1", status_code=303)


# --- Provider helpers ---

@router.get("/providers/{provider_id}/models")
async def get_provider_models(request: Request, provider_id: str):
    """指定プロバイダーの利用可能なモデル一覧を取得して返す。

    各プロバイダーの list_models() を呼び出す。
    APIキー未設定・取得不可の場合は空リストを返す。
    """
    from backend.providers.registry import PROVIDER_REGISTRY
    cls = PROVIDER_REGISTRY.get(provider_id)
    if cls is None:
        return JSONResponse({"models": []})
    settings = request.app.state.sqlite.get_all_settings()
    models = await cls.list_models(settings)
    return JSONResponse({"models": models})


@router.get("/providers/{provider_id}/embedding-models")
async def get_provider_embedding_models(request: Request, provider_id: str):
    """指定プロバイダーの利用可能な Embedding モデル一覧を取得して返す。

    各プロバイダーの list_embedding_models() を呼び出す。
    APIキー未設定・取得不可の場合は空リストを返す。
    """
    from backend.providers.registry import PROVIDER_REGISTRY
    cls = PROVIDER_REGISTRY.get(provider_id)
    if cls is None:
        return JSONResponse({"models": []})
    settings = request.app.state.sqlite.get_all_settings()
    models = await cls.list_embedding_models(settings)
    return JSONResponse({"models": models})
