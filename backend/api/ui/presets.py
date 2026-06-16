"""設定 UI — LLM モデルプリセット管理ページ。"""

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from backend.api.ui.common import get_templates
from backend.providers.registry import PROVIDER_LABELS, PROVIDER_ORDER

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/model-presets", response_class=HTMLResponse)
async def model_presets_list(request: Request):
    presets = request.app.state.sqlite.list_model_presets()
    return get_templates().TemplateResponse(
        request,
        "model_presets.html",
        {
            "presets": presets,
            "provider_labels": PROVIDER_LABELS,
            "provider_order": PROVIDER_ORDER,
        },
    )


def _parse_timeout_seconds(raw: str | None) -> int:
    """フォーム入力からタイムアウト秒数を解釈する。空欄・不正値はデフォルト300秒（5分）。"""
    if not raw:
        return 300
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 300
    return value if value > 0 else 300


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
    timeout_seconds = _parse_timeout_seconds(form.get("timeout_seconds"))
    request.app.state.sqlite.create_model_preset(
        preset_id=preset_id,
        name=name,
        provider=provider,
        model_id=model_id,
        thinking_level=thinking_level,
        timeout_seconds=timeout_seconds,
    )
    return RedirectResponse(url="/ui/model-presets", status_code=303)


@router.post("/model-presets/{preset_id}/edit")
async def update_model_preset(request: Request, preset_id: str):
    form = await request.form()
    name = (form.get("name") or "").strip()
    provider = (form.get("provider") or "").strip()
    model_id = (form.get("model_id") or "").strip()
    thinking_level = form.get("thinking_level") or "default"
    timeout_seconds = _parse_timeout_seconds(form.get("timeout_seconds"))
    if name and provider:
        request.app.state.sqlite.update_model_preset(
            preset_id,
            name=name,
            provider=provider,
            model_id=model_id,
            thinking_level=thinking_level,
            timeout_seconds=timeout_seconds,
        )
    return RedirectResponse(url="/ui/model-presets", status_code=303)


@router.post("/model-presets/{preset_id}/delete")
async def delete_model_preset(request: Request, preset_id: str):
    request.app.state.sqlite.delete_model_preset(preset_id)
    return RedirectResponse(url="/ui/model-presets", status_code=303)


# --- Scenarios (シナリオテンプレート管理) ---


