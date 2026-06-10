"""設定 UI — ダッシュボード＆キャラクター管理ページ。

キャラクターの作成・編集・削除フォームと、それらが使う
フォーム解釈ヘルパーを提供する。
"""

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from backend.api.ui.common import _read_image_data, _save_response, get_templates
from backend.providers.registry import PROVIDER_LABELS

router = APIRouter(prefix="/ui", tags=["ui"])


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
        {google_calendar, gmail, google_drive} の bool dict。
    """
    return {
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


# --- Characters ---

@router.get("/characters", response_class=HTMLResponse)
async def characters_list(request: Request):
    """キャラクター一覧ページ（旧 index。index はダッシュボードに譲った）。"""
    chars = request.app.state.sqlite.list_characters()
    return get_templates().TemplateResponse(
        "characters.html",
        {"request": request, "characters": chars},
    )

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
        self_reflection_mode=self_reflection_mode,
        self_reflection_preset_id=self_reflection_preset_id,
        self_reflection_n_turns=self_reflection_n_turns,
        allowed_tools=allowed_tools,
    )
    return RedirectResponse(url="/ui/characters", status_code=303)


@router.get("/characters/{character_id}", response_class=HTMLResponse)
async def edit_character_form(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/characters", status_code=303)
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
    self_reflection_mode, self_reflection_preset_id, self_reflection_n_turns = (
        _extract_self_reflection_params(form)
    )

    update_kwargs: dict = dict(
        system_prompt_block1=form.get("system_prompt_block1", ""),
        enabled_providers=enabled_providers,
        ghost_model=ghost_model,
        switch_angle_enabled=switch_angle_enabled,
        self_reflection_mode=self_reflection_mode,
        self_reflection_preset_id=self_reflection_preset_id,
        self_reflection_n_turns=self_reflection_n_turns,
        allowed_tools=allowed_tools,
    )
    # 名前は空欄なら更新しない（自動保存中の一時的な空入力で名前を消さない）。
    name = (form.get("name") or "").strip()
    if name:
        update_kwargs["name"] = name
    new_image = await _read_image_data(form)
    if new_image:
        update_kwargs["image_data"] = new_image
    elif form.get("remove_image"):
        # 削除フラグが立っている場合は画像をクリアする
        update_kwargs["image_data"] = None

    request.app.state.sqlite.update_character(character_id, **update_kwargs)
    return _save_response(request, "/ui/characters")


@router.post("/characters/{character_id}/delete")
async def delete_character(request: Request, character_id: str):
    """キャラクターと、紐づく全データをカスケード削除する（SQLite → LanceDB の順）。"""
    request.app.state.memory_manager.delete_character_with_inscribed_memories(character_id)
    return RedirectResponse(url="/ui/characters", status_code=303)


# --- Memories ---

