"""シナリオチャット API — シナリオテンプレート＆NPC の CRUD。

GM プリセット選択用のプリセット一覧エンドポイントも提供する。
"""

import uuid

from fastapi import APIRouter, HTTPException, Request

from backend.api.scenario_chat.schemas import (
    NpcCreate,
    NpcUpdate,
    ScenarioCreate,
    ScenarioUpdate,
)
from backend.services.scenario_chat.service import (
    scenario_npc_to_dict,
    scenario_to_dict,
)

router = APIRouter(prefix="/api/scenario_chat", tags=["scenario_chat"])

# ─── プリセット一覧（GM プリセット選択用） ──────────────────────────────────


@router.get("/presets")
async def list_presets(request: Request):
    """シナリオ作成画面の GM プリセット選択用に、全プリセット一覧を返す。"""
    presets = request.app.state.sqlite.list_model_presets()
    return [
        {
            "id": p.id,
            "name": p.name,
            "provider": p.provider,
            "model_id": p.model_id,
            "thinking_level": p.thinking_level,
        }
        for p in presets
    ]


# ─── シナリオテンプレート CRUD ──────────────────────────────────────────────


@router.post("/scenarios", status_code=201)
async def create_scenario(request: Request, body: ScenarioCreate):
    """シナリオテンプレートを新規作成する。"""
    sqlite = request.app.state.sqlite
    sid = str(uuid.uuid4())
    sqlite.create_scenario(
        scenario_id=sid,
        title=body.title,
        scenario=body.scenario,
        intro=body.intro,
        history_max_turns=body.history_max_turns,
        history_max_chars=body.history_max_chars,
        custom_system_prompt=body.custom_system_prompt,
        dice_pool_spec=body.dice_pool_spec,
        pc_slots=_normalize_pc_slots_input(body.pc_slots),
    )
    return scenario_to_dict(sqlite.get_scenario(sid))


def _normalize_pc_slots_input(raw: list[dict] | None) -> list[dict] | None:
    """API 入力の pc_slots を最低限正規化する。

    - slot_id / name が必須。description は任意。
    - slot_id は trim + 空チェック。重複は呼出側のバリデーションに任せる。
    - image_data は data:image/ で始まる文字列のみ保持（表示専用アバター）。
    - None なら None を返す（DB 上 NULL のまま）。
    """
    if raw is None:
        return None
    out: list[dict] = []
    for entry in raw:
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
    return out


@router.get("/scenarios")
async def list_scenarios(request: Request, limit: int = 100):
    """シナリオテンプレート一覧を新しい順で返す。"""
    sqlite = request.app.state.sqlite
    return [scenario_to_dict(s) for s in sqlite.list_scenarios(limit=limit)]


@router.get("/scenarios/{scenario_id}")
async def get_scenario(request: Request, scenario_id: str):
    """シナリオテンプレート詳細（NPC を含む）を返す。"""
    sqlite = request.app.state.sqlite
    sc = sqlite.get_scenario(scenario_id)
    if sc is None:
        raise HTTPException(status_code=404, detail="シナリオが見つかりません")
    npcs = sqlite.list_scenario_npcs(scenario_id)
    result = scenario_to_dict(sc)
    result["npcs"] = [scenario_npc_to_dict(n) for n in npcs]
    return result


@router.patch("/scenarios/{scenario_id}")
async def update_scenario(request: Request, scenario_id: str, body: ScenarioUpdate):
    """シナリオテンプレートを部分更新する。紐づくプレイセッションは残る。"""
    sqlite = request.app.state.sqlite
    if sqlite.get_scenario(scenario_id) is None:
        raise HTTPException(status_code=404, detail="シナリオが見つかりません")
    update_fields = body.model_dump(exclude_unset=True)
    if "pc_slots" in update_fields:
        update_fields["pc_slots"] = _normalize_pc_slots_input(update_fields["pc_slots"])
    updated = sqlite.update_scenario(scenario_id, **update_fields)
    return scenario_to_dict(updated)


@router.delete("/scenarios/{scenario_id}")
async def delete_scenario(request: Request, scenario_id: str):
    """シナリオテンプレートを削除する。紐づく NPC・セッション・ターンも一括削除する。"""
    sqlite = request.app.state.sqlite
    ok = sqlite.delete_scenario(scenario_id)
    if not ok:
        raise HTTPException(status_code=404, detail="シナリオが見つかりません")
    return {"deleted": True}


# ─── NPC CRUD（シナリオに紐づく） ───────────────────────────────────────────


@router.post("/scenarios/{scenario_id}/npcs", status_code=201)
async def add_npc(request: Request, scenario_id: str, body: NpcCreate):
    """シナリオテンプレートに NPC を追加する。"""
    sqlite = request.app.state.sqlite
    if sqlite.get_scenario(scenario_id) is None:
        raise HTTPException(status_code=404, detail="シナリオが見つかりません")
    existing = [n for n in sqlite.list_scenario_npcs(scenario_id) if n.name == body.name]
    if existing:
        raise HTTPException(
            status_code=400, detail=f"同名 NPC が既に存在します: {body.name}"
        )
    npc_id = str(uuid.uuid4())
    sqlite.create_scenario_npc(
        npc_id=npc_id,
        scenario_id=scenario_id,
        name=body.name,
        description=body.description,
        image_data=body.image_data,
    )
    return scenario_npc_to_dict(sqlite.get_scenario_npc(npc_id))


@router.patch("/scenarios/{scenario_id}/npcs/{npc_id}")
async def edit_npc(request: Request, scenario_id: str, npc_id: str, body: NpcUpdate):
    """NPC を部分更新する。"""
    sqlite = request.app.state.sqlite
    npc = sqlite.get_scenario_npc(npc_id)
    if npc is None or npc.scenario_id != scenario_id:
        raise HTTPException(status_code=404, detail="NPC が見つかりません")
    update_fields = body.model_dump(exclude_unset=True)
    if "name" in update_fields and update_fields["name"] != npc.name:
        existing = [
            n for n in sqlite.list_scenario_npcs(scenario_id)
            if n.name == update_fields["name"] and n.id != npc_id
        ]
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"同名 NPC が既に存在します: {update_fields['name']}",
            )
    updated = sqlite.update_scenario_npc(npc_id, **update_fields)
    return scenario_npc_to_dict(updated)


@router.delete("/scenarios/{scenario_id}/npcs/{npc_id}")
async def remove_npc(request: Request, scenario_id: str, npc_id: str):
    """NPC を削除する。発話履歴は残る。"""
    sqlite = request.app.state.sqlite
    npc = sqlite.get_scenario_npc(npc_id)
    if npc is None or npc.scenario_id != scenario_id:
        raise HTTPException(status_code=404, detail="NPC が見つかりません")
    sqlite.delete_scenario_npc(npc_id)
    return {"deleted": True}


