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
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.lib.debug_logger import logger as debug_logger
from backend.lib.log_context import current_log_feature, new_message_id
from backend.services.scenario_chat.service import (
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
    """

    title: str = Field(min_length=1)
    user_alias: str = Field(min_length=1)
    scenario: Optional[str] = None
    intro: Optional[str] = None
    history_max_turns: Optional[int] = None
    history_max_chars: Optional[int] = None


class ScenarioUpdate(BaseModel):
    """シナリオテンプレート更新リクエスト（部分更新）。"""

    title: Optional[str] = None
    user_alias: Optional[str] = None
    scenario: Optional[str] = None
    intro: Optional[str] = None
    history_max_turns: Optional[int] = None
    history_max_chars: Optional[int] = None


class NpcCreate(BaseModel):
    """NPC 追加リクエスト。

    description には人物像・口調・話し方を自由テキストで全部詰め込む。
    image_data はアバター画像の base64 data URI（オプション）。
    """

    name: str = Field(min_length=1)
    description: Optional[str] = None
    image_data: Optional[str] = None


class NpcUpdate(BaseModel):
    """NPC 更新リクエスト（部分更新）。"""

    name: Optional[str] = None
    description: Optional[str] = None
    image_data: Optional[str] = None


class SessionStart(BaseModel):
    """プレイセッション起動リクエスト。

    gm_preset_id はセッション単位で必須。同一シナリオから複数セッションを起動した際に
    それぞれ別の GM モデルを選べる設計。
    """

    scenario_id: str = Field(min_length=1)
    gm_preset_id: str = Field(min_length=1)
    title: Optional[str] = None  # 省略時はシナリオ title を使う


class SessionUpdate(BaseModel):
    """プレイセッション更新リクエスト（タイトル変更 / status 変更 / GM モデル変更）。"""

    title: Optional[str] = None
    status: Optional[str] = None
    gm_preset_id: Optional[str] = None


class SynopsisUpdate(BaseModel):
    """あらすじ部分更新リクエスト。

    `auto` と `manual` はそれぞれ独立に更新できる（リクエストに含めなければ触らない）。
    自動更新フローと違い、本 API では `auto` も上書き可能（ユーザが捏造記述を
    削除・修正するため）。
    """

    auto: Optional[str] = None
    manual: Optional[str] = None


class StreamRequest(BaseModel):
    """ストリーム発火リクエスト。

    auto_advance=True なら「ユーザは無言で続きを促す」モード。
    content は無視され、user turn は保存されない（履歴に痕跡を残さない）。
    """

    content: str = ""
    auto_advance: bool = False


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
        user_alias=body.user_alias,
        scenario=body.scenario,
        intro=body.intro,
        history_max_turns=body.history_max_turns,
        history_max_chars=body.history_max_chars,
    )
    return scenario_to_dict(sqlite.get_scenario(sid))


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
    sid = str(uuid.uuid4())
    title = body.title or scenario.title
    sqlite.create_scenario_session(
        session_id=sid,
        scenario_id=body.scenario_id,
        title=title,
        gm_preset_id=body.gm_preset_id,
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
async def regenerate_session_synopsis(request: Request, session_id: str):
    """`synopsis_auto` への自動追記フローを手動起動する。

    通常チャットでは閾値（SYNOPSIS_AUTO_TRIGGER_CHARS）未満では自動更新を
    抑制しているため、ユーザが任意のタイミングで追記更新を強制したいときに使う。
    既存 auto は**書き換えず**、新規分のみが末尾に追記される。
    """
    # 通常の stream エンドポイントと同じく、リクエスト ID と feature タグを設定する。
    # こうしないと synopsis 蒸留の debug ログが 1on1 の "chat" 扱いで出力され、
    # 「どのモデルで蒸留されたか」を debug フォルダ名から追えなくなる。
    new_message_id()
    current_log_feature.set("scenario_chat")

    state = request.app.state
    sqlite = state.sqlite
    sess = sqlite.get_scenario_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
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
        gm_preset_id=getattr(sess, "gm_preset_id", "") or "",
        force=True,
    )
    if updated is None:
        # 追記対象が無い、または LLM 呼び出し失敗（best-effort）。現状を返す。
        return sqlite.get_scenario_session_synopsis(session_id) or {
            "auto": "",
            "manual": "",
            "last_turn_index": -1,
        }
    return updated


@router.post("/sessions/{session_id}/stream")
async def stream_turn(request: Request, session_id: str, body: StreamRequest):
    """プレイヤー発話を入力としてシナリオ 1 ターン分を SSE で返す。

    既存 chat と同じく、リクエストごとに log_message_id を発行して
    `debug/<8 桁hex>/` フォルダ内に各種ログを保存できるようにする。
    """
    # リクエスト識別子を発行（debug ログのフォルダ名・ログ行のトレース ID に使う）。
    # 既存 1on1 chat の stream エンドポイントと同じパターン。
    new_message_id()
    debug_logger.log_front_input(body.model_dump())

    state = request.app.state
    sqlite = state.sqlite
    sess = sqlite.get_scenario_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    if sess.status != "active":
        raise HTTPException(status_code=400, detail="セッションは終了しています")

    settings = sqlite.get_all_settings()

    async def sse_generator():
        async for event_type, payload in run_scenario_turn(
            session_id=session_id,
            user_message=body.content,
            sqlite=sqlite,
            settings=settings,
            auto_advance=body.auto_advance,
        ):
            data = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
            yield f"data: {data}\n\n"
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
