"""シナリオチャット API — プレイセッションの CRUD・履歴・あらすじ。"""

import uuid

from fastapi import APIRouter, HTTPException, Request

from backend.api.scenario_chat.schemas import (
    SessionStart,
    SessionUpdate,
    SynopsisRegenerate,
    SynopsisUpdate,
)
from backend.lib.log_context import (
    current_log_feature,
    current_log_session_id,
    current_log_target,
    new_message_id,
)
from backend.services.scenario_chat.service import (
    compute_synopsis_progress,
    maybe_update_auto_synopsis,
    scenario_npc_to_dict,
    scenario_session_to_dict,
    scenario_to_dict,
    scenario_turn_to_dict,
    seed_intro_turns,
)

router = APIRouter(prefix="/api/scenario_chat", tags=["scenario_chat"])

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
    # シナリオ名をログターゲットに設定（Logs画面の Scenario タブでシナリオ名を表示するため）
    _scen = sqlite.get_scenario(sess.scenario_id)
    if _scen:
        current_log_target.set(_scen.title)
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


