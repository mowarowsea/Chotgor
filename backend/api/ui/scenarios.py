"""設定 UI — シナリオテンプレート＆NPC 管理ページ。"""

import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from backend.api.ui.common import _read_image_data, _save_response, get_templates

router = APIRouter(prefix="/ui", tags=["ui"])


def _coalesce_optional_int(form, key: str):
    """フォームから任意の int 値を取り出す。空欄は None、不正値も None。"""
    raw = (form.get(key) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _parse_pc_slots(value: str | None) -> list[dict] | None:
    """pc_slots フォーム入力（JSON テキスト）を正規化済みリストへ変換する。

    例:
        '[{"slot_id":"pc1","name":"アリス","description":"..."}]' → そのまま
        '' / None / 不正 JSON / 非 list → None
        slot_id/name 欠落要素はスキップ。
        image_data は data:image/ で始まる文字列のみ保持（表示専用アバター）。
    """
    if not value or not value.strip():
        return None
    import json as _json
    try:
        parsed = _json.loads(value)
    except _json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    out: list[dict] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("slot_id", "")).strip()
        name = str(entry.get("name", "")).strip()
        if not sid or not name:
            continue
        normalized = {
            "slot_id": sid,
            "name": name,
            "description": str(entry.get("description", "") or "").strip(),
        }
        image_data = entry.get("image_data")
        if isinstance(image_data, str) and image_data.startswith("data:image/"):
            normalized["image_data"] = image_data
        out.append(normalized)
    return out or None


def _parse_dice_pool_spec(value: str | None) -> dict | None:
    """dice_pool_spec フォーム入力（JSON テキスト）を辞書に正規化する。

    例:
        '{"d6": 10, "d100": 5}' → {"d6": 10, "d100": 5}
        '' / None / 不正 JSON / 非 dict → None
        値が int でない・0 以下・キーが `d` で始まらないエントリは無視する。
    """
    if not value or not value.strip():
        return None
    import json as _json
    try:
        parsed = _json.loads(value)
    except _json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    result: dict[str, int] = {}
    for k, v in parsed.items():
        if not isinstance(k, str) or not k.startswith("d"):
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        result[k] = n
    return result or None


@router.get("/scenarios", response_class=HTMLResponse)
async def scenarios_list(request: Request):
    """シナリオテンプレート一覧ページ。"""
    scenarios = request.app.state.sqlite.list_scenarios()
    return get_templates().TemplateResponse(
        "scenarios.html",
        {"request": request, "scenarios": scenarios},
    )


@router.get("/scenarios/new", response_class=HTMLResponse)
async def new_scenario_form(request: Request):
    """シナリオテンプレート新規作成フォーム。"""
    return get_templates().TemplateResponse(
        "scenario_edit.html",
        {
            "request": request,
            "scenario": None,
            "npcs": [],
            "action": "/ui/scenarios/new",
        },
    )


@router.post("/scenarios/new")
async def create_scenario(request: Request):
    """シナリオテンプレートを作成する。

    GM の LLM プリセットはテンプレートではなくセッション単位で持つため、
    本フォームでは扱わない（フロントの「新しい会話」モーダルで選ばせる）。
    """
    form = await request.form()
    title = (form.get("title") or "").strip()
    if not title:
        return RedirectResponse(url="/ui/scenarios/new", status_code=303)

    sid = str(uuid.uuid4())
    request.app.state.sqlite.create_scenario(
        scenario_id=sid,
        title=title,
        scenario=(form.get("scenario") or "") or None,
        intro=(form.get("intro") or "") or None,
        history_max_turns=_coalesce_optional_int(form, "history_max_turns"),
        history_max_chars=_coalesce_optional_int(form, "history_max_chars"),
        custom_system_prompt=(form.get("custom_system_prompt") or "") or None,
        dice_pool_spec=_parse_dice_pool_spec(form.get("dice_pool_spec")),
        pc_slots=_parse_pc_slots(form.get("pc_slots")),
        banner_data=await _read_image_data(form, field="banner"),
    )
    return RedirectResponse(url=f"/ui/scenarios/{sid}/edit", status_code=303)


@router.get("/scenarios/{scenario_id}/edit", response_class=HTMLResponse)
async def edit_scenario_form(request: Request, scenario_id: str):
    """シナリオテンプレート編集ページ（NPC 編集を含む）。"""
    sqlite = request.app.state.sqlite
    scenario = sqlite.get_scenario(scenario_id)
    if not scenario:
        return RedirectResponse(url="/ui/scenarios", status_code=303)
    npcs = sqlite.list_scenario_npcs(scenario_id)
    return get_templates().TemplateResponse(
        "scenario_edit.html",
        {
            "request": request,
            "scenario": scenario,
            "npcs": npcs,
            "action": f"/ui/scenarios/{scenario_id}/edit",
        },
    )


@router.post("/scenarios/{scenario_id}/edit")
async def update_scenario(request: Request, scenario_id: str):
    """シナリオテンプレートを更新する（プレイ中のセッションには影響しない）。"""
    form = await request.form()
    update_kwargs = {
        "scenario": (form.get("scenario") or "") or None,
        "intro": (form.get("intro") or "") or None,
        "history_max_turns": _coalesce_optional_int(form, "history_max_turns"),
        "history_max_chars": _coalesce_optional_int(form, "history_max_chars"),
        "custom_system_prompt": (form.get("custom_system_prompt") or "") or None,
        "dice_pool_spec": _parse_dice_pool_spec(form.get("dice_pool_spec")),
        "pc_slots": _parse_pc_slots(form.get("pc_slots")),
    }
    # タイトルは必須項目。空欄なら更新しない（自動保存の空入力対策）。
    title = (form.get("title") or "").strip()
    if title:
        update_kwargs["title"] = title
    # バナーは新規アップロード時のみ更新、remove_banner チェック時はクリア。
    new_banner = await _read_image_data(form, field="banner")
    if new_banner:
        update_kwargs["banner_data"] = new_banner
    elif form.get("remove_banner"):
        update_kwargs["banner_data"] = None
    request.app.state.sqlite.update_scenario(scenario_id, **update_kwargs)
    return _save_response(request, f"/ui/scenarios/{scenario_id}/edit")


@router.post("/scenarios/{scenario_id}/delete")
async def delete_scenario(request: Request, scenario_id: str):
    """シナリオテンプレートを削除する（紐づくセッション・ターンも一括削除）。"""
    request.app.state.sqlite.delete_scenario(scenario_id)
    return RedirectResponse(url="/ui/scenarios", status_code=303)


@router.post("/scenarios/{scenario_id}/npcs/new")
async def add_npc_form(request: Request, scenario_id: str):
    """NPC を追加する。multipart/form-data 経由でアバター画像も受け取る。"""
    form = await request.form()
    name = (form.get("name") or "").strip()
    if not name:
        return RedirectResponse(url=f"/ui/scenarios/{scenario_id}/edit", status_code=303)
    sqlite = request.app.state.sqlite
    existing = [n for n in sqlite.list_scenario_npcs(scenario_id) if n.name == name]
    if existing:
        return RedirectResponse(url=f"/ui/scenarios/{scenario_id}/edit", status_code=303)
    image_data = await _read_image_data(form)
    sqlite.create_scenario_npc(
        npc_id=str(uuid.uuid4()),
        scenario_id=scenario_id,
        name=name,
        description=(form.get("description") or "") or None,
        image_data=image_data,
    )
    return RedirectResponse(url=f"/ui/scenarios/{scenario_id}/edit", status_code=303)


@router.post("/scenarios/{scenario_id}/npcs/{npc_id}/edit")
async def edit_npc_form(request: Request, scenario_id: str, npc_id: str):
    """NPC を更新する。画像未指定なら既存画像を維持、remove_image チェック時はクリア。"""
    form = await request.form()
    update_kwargs: dict = {
        "description": (form.get("description") or "") or None,
    }
    # 名前は必須項目。空欄なら名前以外だけ保存する（自動保存の空入力対策）。
    name = (form.get("name") or "").strip()
    if name:
        update_kwargs["name"] = name

    new_image = await _read_image_data(form)
    if new_image:
        update_kwargs["image_data"] = new_image
    elif form.get("remove_image"):
        update_kwargs["image_data"] = None

    request.app.state.sqlite.update_scenario_npc(npc_id, **update_kwargs)
    return _save_response(request, f"/ui/scenarios/{scenario_id}/edit")


@router.post("/scenarios/{scenario_id}/npcs/{npc_id}/delete")
async def delete_npc_form(request: Request, scenario_id: str, npc_id: str):
    """NPC を削除する（過去の発話履歴は残る）。"""
    request.app.state.sqlite.delete_scenario_npc(npc_id)
    return RedirectResponse(url=f"/ui/scenarios/{scenario_id}/edit", status_code=303)


# --- Settings ---

