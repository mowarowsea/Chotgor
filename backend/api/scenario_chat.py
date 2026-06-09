"""シナリオチャット API ルータ。

`/api/scenario_chat/...` 配下に、シナリオテンプレートとプレイインスタンスの
2 層を扱う CRUD と SSE ストリームを提供する。

エンドポイント:
    -- シナリオテンプレート（CRUD・NPC 編集） --
    POST   /api/scenario_chat/scenarios                                   作成
    GET    /api/scenario_chat/scenarios                                   一覧
    GET    /api/scenario_chat/scenarios/{sid}                             詳細
    PATCH  /api/scenario_chat/scenarios/{sid}                             更新
    DELETE /api/scenario_chat/scenarios/{sid}                             削除（紐づくセッションも）
    POST   /api/scenario_chat/scenarios/{sid}/npcs                        NPC 追加
    PATCH  /api/scenario_chat/scenarios/{sid}/npcs/{nid}                  NPC 編集
    DELETE /api/scenario_chat/scenarios/{sid}/npcs/{nid}                  NPC 削除

    -- プレイインスタンス（起動・履歴・SSE） --
    POST   /api/scenario_chat/sessions                                    シナリオから起動
    GET    /api/scenario_chat/sessions                                    一覧
    GET    /api/scenario_chat/sessions/{sid}                              詳細（シナリオ・NPC込み）
    PATCH  /api/scenario_chat/sessions/{sid}                              更新（title / status）
    DELETE /api/scenario_chat/sessions/{sid}                              削除（テンプレ非影響）
    POST   /api/scenario_chat/sessions/{sid}/end                          終了
    GET    /api/scenario_chat/sessions/{sid}/turns                        履歴
    POST   /api/scenario_chat/sessions/{sid}/stream                       SSE

    -- 補助 --
    GET    /api/scenario_chat/presets                                     GM プリセット選択用一覧
"""

import json
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.lib.debug_logger import logger as debug_logger
from backend.lib.log_context import (
    current_log_feature,
    current_log_session_id,
    current_log_target,
    current_log_turn_sequence,
    current_message_id,
    new_message_id,
)
from backend.services.scenario_chat.service import (
    compute_synopsis_progress,
    maybe_update_auto_synopsis,
    run_scenario_turn,
    seed_intro_turns,
    scenario_npc_to_dict,
    scenario_to_dict,
    scenario_session_to_dict,
    scenario_turn_to_dict,
)

router = APIRouter(prefix="/api/scenario_chat", tags=["scenario_chat"])


# ─── Pydantic スキーマ ──────────────────────────────────────────────────────


class ScenarioCreate(BaseModel):
    """シナリオテンプレート作成リクエスト。

    GM プリセットはテンプレートではなくセッション単位（SessionStart.gm_preset_id）で
    指定するため、本リクエストには含めない。
    custom_system_prompt を設定するとGMシステムプロンプトをカスタマイズできる。
    dice_pool_spec / pc_slots は ensemble_pc エンジンのセッション専用フィールド。
    """

    title: str = Field(min_length=1)
    scenario: str | None = None
    intro: str | None = None
    history_max_turns: int | None = None
    history_max_chars: int | None = None
    custom_system_prompt: str | None = None
    dice_pool_spec: dict | None = None
    # 全PC（ユーザPC含む）を定義する単一ソース。旧 user_alias は廃止し、ユーザPCも
    # この pc_slots の 1 枠として表現する（セッション開始時に player_type="user" を割り当てる）。
    pc_slots: list[dict] | None = None  # [{slot_id, name, description}]


class ScenarioUpdate(BaseModel):
    """シナリオテンプレート更新リクエスト（部分更新）。"""

    title: str | None = None
    scenario: str | None = None
    intro: str | None = None
    history_max_turns: int | None = None
    history_max_chars: int | None = None
    custom_system_prompt: str | None = None
    dice_pool_spec: dict | None = None
    pc_slots: list[dict] | None = None


class NpcCreate(BaseModel):
    """NPC 追加リクエスト。

    description には人物像・口調・話し方を自由テキストで全部詰め込む。
    image_data はアバター画像の base64 data URI（オプション）。
    """

    name: str = Field(min_length=1)
    description: str | None = None
    image_data: str | None = None


class NpcUpdate(BaseModel):
    """NPC 更新リクエスト（部分更新）。"""

    name: str | None = None
    description: str | None = None
    image_data: str | None = None


class SessionStart(BaseModel):
    """プレイセッション起動リクエスト。

    gm_preset_id / synopsis_preset_id はそれぞれセッション単位で必須。
    同一シナリオから複数セッションを起動した際にそれぞれ別モデルを選べる設計で、
    `synopsis_preset_id` はあらすじ蒸留専用（レートリミット節約のため別モデル指定可能）。
    UI 上は両方明示指定させる（同じプリセットでもよい）。

    engine_type は "ensemble"（既存・GMのみ）または "ensemble_pc"（GM + PC配役）。
    "ensemble_pc" の場合、親シナリオの `pc_slots` の各 slot_id について
    pc_assignments を 1 件以上指定する。形式:
        [{"slot_id":"pc1","player_type":"user"|"character",
          "character_id":"...","preset_id":"..."}]
    """

    scenario_id: str = Field(min_length=1)
    gm_preset_id: str = Field(min_length=1)
    synopsis_preset_id: str = Field(min_length=1)
    title: str | None = None
    engine_type: str = "ensemble"
    pc_assignments: list[dict] | None = None


class SessionUpdate(BaseModel):
    """プレイセッション更新リクエスト（タイトル / status / GM モデル / あらすじモデル）。"""

    title: str | None = None
    status: str | None = None
    gm_preset_id: str | None = None
    synopsis_preset_id: str | None = None


class SynopsisUpdate(BaseModel):
    """あらすじ部分更新リクエスト。

    `auto` と `manual` はそれぞれ独立に更新できる（リクエストに含めなければ触らない）。
    自動更新フローと違い、本 API では `auto` も上書き可能（ユーザが捏造記述を
    削除・修正するため）。
    """

    auto: str | None = None
    manual: str | None = None


class SynopsisRegenerate(BaseModel):
    """あらすじ手動作成（強制蒸留）リクエスト。

    `synopsis_preset_id` を指定すると、その preset をセッションへ永続化（記憶）した上で
    蒸留に使う。フロントの「あらすじ作成モーダル」で選んだモデルが次回以降の既定にも
    なる。省略時はセッション既定の `synopsis_preset_id` を使う。
    """

    synopsis_preset_id: str | None = None


class StreamRequest(BaseModel):
    """ストリーム発火リクエスト。

    auto_advance=True なら「ユーザは無言で続きを促す」モード。
    content は無視され、user turn は保存されない（履歴に痕跡を残さない）。

    regenerate_request_id を指定すると、そのログエントリに追記する形で
    再生成ログをまとめる。フロントは再生成ボタン押下時のみ前ターンの
    log_request_id を渡す（過去ターン編集時は渡さない）。
    """

    content: str = ""
    auto_advance: bool = False
    regenerate_request_id: str | None = None


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
        out.append({
            "slot_id": sid,
            "name": name,
            "description": str(entry.get("description", "") or "").strip(),
        })
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


# ─── プレイセッション CRUD（起動・履歴） ───────────────────────────────────


@router.post("/sessions", status_code=201)
async def start_session(request: Request, body: SessionStart):
    """シナリオテンプレートから新しいプレイセッションを起動する。

    起動時に GM プリセットを必須で受け取り、セッションへ紐付ける（チャット中も変更可）。
    engine_type="ensemble_pc" の場合は pc_assignments を 1 件以上必須とし、
    PC本名・PC配役名・NPC名・user_alias・narrator 名の三方衝突を検知する。
    """
    sqlite = request.app.state.sqlite
    scenario = sqlite.get_scenario(body.scenario_id)
    if scenario is None:
        raise HTTPException(
            status_code=400, detail=f"シナリオが見つかりません: {body.scenario_id}"
        )
    if sqlite.get_model_preset(body.gm_preset_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"指定された gm_preset_id が見つかりません: {body.gm_preset_id}",
        )
    if sqlite.get_model_preset(body.synopsis_preset_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"指定された synopsis_preset_id が見つかりません: {body.synopsis_preset_id}",
        )

    engine_type = (body.engine_type or "ensemble").strip()
    if engine_type not in {"ensemble", "ensemble_pc"}:
        raise HTTPException(
            status_code=400,
            detail=f"未知の engine_type: {engine_type}（ensemble | ensemble_pc）",
        )

    pc_assignments_validated: list[dict] | None = None
    if engine_type == "ensemble_pc":
        from backend.services.scenario_chat.mention import (
            detect_name_conflicts,
            normalize_pc_assignments,
            normalize_pc_slots,
        )
        pc_slots_norm = normalize_pc_slots(getattr(scenario, "pc_slots", None))
        if not pc_slots_norm:
            raise HTTPException(
                status_code=400,
                detail=(
                    "engine_type=ensemble_pc にはシナリオ側に PC枠（pc_slots）を"
                    "1 件以上定義してください"
                ),
            )
        slot_by_id = {s.slot_id: s for s in pc_slots_norm}
        raw_assignments = body.pc_assignments or []
        if not raw_assignments:
            raise HTTPException(
                status_code=400,
                detail="engine_type=ensemble_pc には pc_assignments を 1 件以上指定してください",
            )
        seen_slot_ids: set[str] = set()
        normalized: list[dict] = []
        for entry in raw_assignments:
            if not isinstance(entry, dict):
                raise HTTPException(
                    status_code=400,
                    detail="pc_assignments の各要素は dict（slot_id, player_type など）が必要です",
                )
            sid = str(entry.get("slot_id", "")).strip()
            ptype = str(entry.get("player_type", "")).strip()
            if not sid or ptype not in {"user", "character"}:
                raise HTTPException(
                    status_code=400,
                    detail="pc_assignments の各要素には slot_id と player_type(user|character) が必要です",
                )
            if sid not in slot_by_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"pc_assignments の slot_id がシナリオの pc_slots に存在しません: {sid}",
                )
            if sid in seen_slot_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"pc_assignments に同じ slot_id が重複しています: {sid}",
                )
            seen_slot_ids.add(sid)
            normalized_entry: dict = {"slot_id": sid, "player_type": ptype}
            if ptype == "character":
                cid = str(entry.get("character_id", "")).strip()
                if not cid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"player_type=character の slot には character_id が必要です: slot_id={sid}",
                    )
                if sqlite.get_character(cid) is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"PC枠のキャラクターが見つかりません: slot_id={sid} character_id={cid}",
                    )
                normalized_entry["character_id"] = cid
                preset_ref = str(entry.get("preset_id", "") or "").strip()
                if preset_ref:
                    # /v1/models 経由のフロントは preset 名を送ってくるが、
                    # SQLite では LLMModelPreset.id が正規キー。ID で引けなければ
                    # 名前で再検索して ID へ正規化する。
                    preset_obj = sqlite.get_model_preset(preset_ref) or sqlite.get_model_preset_by_name(preset_ref)
                    if preset_obj is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"指定されたPCプリセットが見つかりません: slot_id={sid} preset_id={preset_ref}",
                        )
                    normalized_entry["preset_id"] = preset_obj.id
            normalized.append(normalized_entry)
        # 全 slot_id を割り当て必須（未割り当てがあると挙動が読みづらい）
        unassigned = [s.slot_id for s in pc_slots_norm if s.slot_id not in seen_slot_ids]
        if unassigned:
            raise HTTPException(
                status_code=400,
                detail=(
                    "全 PC枠を割り当ててください。未割当: " + ", ".join(unassigned)
                ),
            )
        pc_assignments_validated = normalized

        # 名前衝突検知（NPC名 × PC枠名 × PCキャラ本名 × Narrator）
        npcs = sqlite.list_scenario_npcs(body.scenario_id)
        pcs_norm = normalize_pc_assignments(pc_assignments_validated, pc_slots_norm, sqlite)
        conflicts = detect_name_conflicts(
            pcs_norm,
            npc_names={n.name for n in npcs if getattr(n, "name", None)},
        )
        if conflicts:
            raise HTTPException(
                status_code=400,
                detail=(
                    "PC枠名・キャラ本名・NPC名・Narrator の中に重複があります: "
                    + ", ".join(conflicts)
                    + "（重複を解消してから再度起動してください）"
                ),
            )
    elif engine_type == "ensemble":
        # ensemble はユーザPC 1 枠のみ。シナリオ pc_slots の先頭枠をユーザ割当にする。
        # （旧 user_alias 廃止後、ユーザPC名はこの user 割当スロットから解決される）
        from backend.services.scenario_chat.mention import normalize_pc_slots

        pc_slots_norm = normalize_pc_slots(getattr(scenario, "pc_slots", None))
        if pc_slots_norm:
            pc_assignments_validated = [
                {"slot_id": pc_slots_norm[0].slot_id, "player_type": "user"}
            ]

    sid = str(uuid.uuid4())
    title = body.title or scenario.title
    sqlite.create_scenario_session(
        session_id=sid,
        scenario_id=body.scenario_id,
        title=title,
        gm_preset_id=body.gm_preset_id,
        synopsis_preset_id=body.synopsis_preset_id,
        engine_type=engine_type,
        pc_assignments=pc_assignments_validated,
    )
    # シナリオ設定の導入部（intro）があれば固定ターンとして先頭挿入する
    seed_intro_turns(sqlite, sid, scenario)
    return scenario_session_to_dict(sqlite.get_scenario_session(sid))


@router.get("/sessions")
async def list_sessions(request: Request, limit: int = 100):
    """プレイセッション一覧を新しい順で返す。"""
    sqlite = request.app.state.sqlite
    return [scenario_session_to_dict(s) for s in sqlite.list_scenario_sessions(limit=limit)]


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    """プレイセッション詳細を返す。元シナリオの基本情報と NPC も含める。"""
    sqlite = request.app.state.sqlite
    sess = sqlite.get_scenario_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    scenario = sqlite.get_scenario(sess.scenario_id)
    npcs = sqlite.list_scenario_npcs(sess.scenario_id) if scenario else []
    result = scenario_session_to_dict(sess)
    result["scenario"] = scenario_to_dict(scenario) if scenario else None
    result["npcs"] = [scenario_npc_to_dict(n) for n in npcs]
    return result


@router.patch("/sessions/{session_id}")
async def update_session(request: Request, session_id: str, body: SessionUpdate):
    """プレイセッションを部分更新する（タイトル変更 / status 変更 / GM モデル変更）。"""
    sqlite = request.app.state.sqlite
    if sqlite.get_scenario_session(session_id) is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    update_fields = body.model_dump(exclude_unset=True)
    if "gm_preset_id" in update_fields and update_fields["gm_preset_id"]:
        if sqlite.get_model_preset(update_fields["gm_preset_id"]) is None:
            raise HTTPException(
                status_code=400,
                detail=f"指定された gm_preset_id が見つかりません: {update_fields['gm_preset_id']}",
            )
    if "synopsis_preset_id" in update_fields and update_fields["synopsis_preset_id"]:
        if sqlite.get_model_preset(update_fields["synopsis_preset_id"]) is None:
            raise HTTPException(
                status_code=400,
                detail=f"指定された synopsis_preset_id が見つかりません: {update_fields['synopsis_preset_id']}",
            )
    updated = sqlite.update_scenario_session(session_id, **update_fields)
    return scenario_session_to_dict(updated)


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str):
    """プレイセッションを削除する。テンプレには影響しない。"""
    sqlite = request.app.state.sqlite
    ok = sqlite.delete_scenario_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    return {"deleted": True}


@router.post("/sessions/{session_id}/end")
async def end_session(request: Request, session_id: str):
    """プレイセッションを終了状態にする。"""
    sqlite = request.app.state.sqlite
    if sqlite.get_scenario_session(session_id) is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    updated = sqlite.update_scenario_session(session_id, status="ended")
    return scenario_session_to_dict(updated)


@router.get("/sessions/{session_id}/turns")
async def list_turns(request: Request, session_id: str):
    """セッションの全ターンを時系列昇順で返す。"""
    sqlite = request.app.state.sqlite
    if sqlite.get_scenario_session(session_id) is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    return [scenario_turn_to_dict(t) for t in sqlite.list_scenario_turns(session_id)]


@router.delete("/sessions/{session_id}/turns/from/{turn_id}")
async def delete_turns_from(request: Request, session_id: str, turn_id: str):
    """指定ターン以降（自身を含む）をすべて削除する。

    ユーザ発話の編集・GM ターンの再生成の前処理として呼ぶ。
    """
    sqlite = request.app.state.sqlite
    if sqlite.get_scenario_session(session_id) is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    ok = sqlite.delete_scenario_turns_from(session_id, turn_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ターンが見つかりません")
    return {"deleted": True}


@router.get("/sessions/{session_id}/synopsis")
async def get_session_synopsis(request: Request, session_id: str):
    """セッションのあらすじ（auto / manual / last_turn_index）を取得する。"""
    sqlite = request.app.state.sqlite
    if sqlite.get_scenario_session(session_id) is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    synopsis = sqlite.get_scenario_session_synopsis(session_id)
    if synopsis is None:
        # セッションが存在する以上、None は想定外（防御的に空を返す）
        return {"auto": "", "manual": "", "last_turn_index": -1}
    return synopsis


@router.patch("/sessions/{session_id}/synopsis")
async def patch_session_synopsis(
    request: Request, session_id: str, body: SynopsisUpdate
):
    """セッションのあらすじを部分更新する。

    リクエストに含めたフィールドだけを書き換える。auto も書き換え可能なので、
    ユーザは UI 上で混入した捏造記述を直接削除・修正できる。

    特例: auto を明示的に空文字列にする場合、`synopsis_last_turn_index` を
    -1 に連動リセットする。auto を「白紙」にする意図は通常「最初から蒸留し直したい」
    なので、進捗ポインタを残したままにすると次回 `maybe_update_auto_synopsis` で
    new_dropped が永久に空判定になり、再生成不能になる。
    """
    sqlite = request.app.state.sqlite
    if sqlite.get_scenario_session(session_id) is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    update_fields = body.model_dump(exclude_unset=True)
    if not update_fields:
        # 何も更新指示が無ければ現状を返すだけ
        return sqlite.get_scenario_session_synopsis(session_id) or {
            "auto": "",
            "manual": "",
            "last_turn_index": -1,
        }
    if "auto" in update_fields and (update_fields.get("auto") or "") == "":
        # auto を空にしたら蒸留進捗もゼロに戻す（再生成可能な状態へ）
        update_fields["last_turn_index"] = -1
    updated = sqlite.update_scenario_session_synopsis(session_id, **update_fields)
    if updated is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    return updated


@router.post("/sessions/{session_id}/synopsis/regenerate")
async def regenerate_session_synopsis(
    request: Request, session_id: str, body: SynopsisRegenerate | None = None
):
    """あらすじを強制的に再蒸留する（ユーザ起動の「あらすじ作成」フロー）。

    通常チャットでは閾値（SYNOPSIS_AUTO_TRIGGER_RATIO × history 上限）に達すると
    UI 側で作成を促すが、蒸留自体は走らせない。本 API がその蒸留本体を force=True で
    実行する。既存 auto は**書き換えず**、新規分のみが末尾に追記される。

    `body.synopsis_preset_id` が指定されていれば、その preset をセッションへ永続化
    （記憶）してから蒸留に使う。これにより、モーダルで選んだモデルが次回以降の
    既定にもなり、「あらすじモデル切り替え忘れ」を防ぐ。

    レスポンスは `{"synopsis": {...}, "progress": {...}}`。progress は蒸留後の
    最新進捗（前回蒸留以降のターン数・文字数と上限）で、フロントが作成バーを
    即時に更新（多くの場合 0 になり非表示化）できるようにするため。
    """
    # 通常の stream エンドポイントと同じく、リクエスト ID と feature タグを設定する。
    # こうしないと synopsis 蒸留の debug ログが 1on1 の "chat" 扱いで出力され、
    # 「どのモデルで蒸留されたか」を debug フォルダ名から追えなくなる。
    new_message_id()
    current_log_feature.set("synopsis")
    current_log_session_id.set(session_id)

    state = request.app.state
    sqlite = state.sqlite
    sess = sqlite.get_scenario_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    # preset 指定があれば検証してセッションへ永続化し、最新の sess を取り直す。
    requested_preset_id = body.synopsis_preset_id if body else None
    if requested_preset_id:
        if sqlite.get_model_preset(requested_preset_id) is None:
            raise HTTPException(
                status_code=400,
                detail=f"指定された synopsis_preset_id が見つかりません: {requested_preset_id}",
            )
        sqlite.update_scenario_session(
            session_id, synopsis_preset_id=requested_preset_id
        )
        sess = sqlite.get_scenario_session(session_id)
    scenario = sqlite.get_scenario(sess.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=400, detail="元シナリオが見つかりません")
    settings = sqlite.get_all_settings()
    history = sqlite.list_scenario_turns(session_id)
    updated = await maybe_update_auto_synopsis(
        sqlite=sqlite,
        settings=settings,
        scenario=scenario,
        history=history,
        session_id=session_id,
        synopsis_preset_id=(
            getattr(sess, "synopsis_preset_id", "")
            or getattr(sess, "gm_preset_id", "")
            or ""
        ),
        force=True,
    )
    if updated is None:
        # 追記対象が無い、または LLM 呼び出し失敗（best-effort）。現状を返す。
        synopsis = sqlite.get_scenario_session_synopsis(session_id) or {
            "auto": "",
            "manual": "",
            "last_turn_index": -1,
        }
    else:
        synopsis = updated
    progress = compute_synopsis_progress(sqlite, settings, scenario, session_id) or {
        "turns": 0,
        "max_turns": 0,
        "chars": 0,
        "max_chars": 0,
    }
    return {"synopsis": synopsis, "progress": progress}


@router.post("/sessions/{session_id}/stream")
async def stream_turn(request: Request, session_id: str, body: StreamRequest):
    """プレイヤー発話を入力としてシナリオ 1 ターン分を SSE で返す。

    既存 chat と同じく、リクエストごとに log_message_id を発行して
    `debug/<8 桁hex>/` フォルダ内に各種ログを保存できるようにする。
    """
    # リクエスト識別子を発行。再生成時は前ターンの log_request_id を引き継ぐ。
    new_message_id()
    current_log_session_id.set(session_id)
    current_log_feature.set("scenario")

    state = request.app.state
    sqlite = state.sqlite
    sess = sqlite.get_scenario_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    if sess.status != "active":
        raise HTTPException(status_code=400, detail="セッションは終了しています")

    # シナリオタイトルとターン番号をログコンテキストに設定する
    _scenario = sqlite.get_scenario(sess.scenario_id)
    if _scenario:
        current_log_target.set(_scenario.title)
    _next_turn = sqlite.get_next_scenario_turn_index(session_id)
    current_log_turn_sequence.set(_next_turn)

    # 再生成時は前ターンの log_request_id を引き継いで同一ログエントリにまとめる。
    # 過去ターン編集や新規ターンの場合は引き継がない（新規 ID のまま）。
    if body.regenerate_request_id:
        current_message_id.set(body.regenerate_request_id)

    debug_logger.log_front_input(body.model_dump())

    settings = sqlite.get_all_settings()

    async def sse_generator():
        # GM の発話内容を収集して最後に log_front_output() に渡す
        _gm_parts: list[str] = []
        # chat_service は app.state に必ず存在する想定だが、テスト用 fixture では未注入のことが
        # あるため getattr で防御する。None でも ensemble モードは動作する（ensemble_pc 時のみ
        # PC ターンがスキップされる）。
        _chat_service = getattr(state, "chat_service", None)
        async for event_type, payload in run_scenario_turn(
            session_id=session_id,
            user_message=body.content,
            sqlite=sqlite,
            settings=settings,
            auto_advance=body.auto_advance,
            chat_service=_chat_service,
        ):
            data = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
            yield f"data: {data}\n\n"
            if event_type == "speaker_end":
                turn = payload.get("turn", {})
                if turn.get("speaker_type") != "user":
                    name = turn.get("speaker_name", "")
                    content = turn.get("content", "")
                    if content:
                        _gm_parts.append(content)
        # ストリーム完了後に DB の response カラムを更新する
        if _gm_parts:
            debug_logger.log_front_output("\n\n".join(_gm_parts))
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/default-system-prompt-template")
async def get_default_system_prompt_template():
    """デフォルトのGMシステムプロンプト（テンプレートタグ版）を返す。

    「デフォルトに戻す」ボタンで custom_system_prompt 欄に自動入力される。
    実行時にはテンプレートタグが実際の値に置き換えられる。
    """
    from backend.services.scenario_chat.prompt_builder import DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE
    return {
        "template": DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE
    }
