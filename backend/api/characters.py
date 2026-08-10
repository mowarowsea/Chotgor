"""キャラクター CRUD REST API。"""

import base64
import logging
import uuid
from fastapi import APIRouter, HTTPException, Request, Response

from backend.api.schemas import CharacterCreate, CharacterUpdate, FaceToFaceModeUpdate
from backend.api.utils import char_to_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/characters", tags=["characters"])


@router.get("/")
async def list_characters(request: Request):
    chars = request.app.state.sqlite.list_characters()
    return [char_to_dict(c) for c in chars]


@router.post("/", status_code=201)
async def create_character(request: Request, body: CharacterCreate):
    """キャラクターを新規作成する。

    estranged キャラクターと類似する定義の場合は HTTP 409 を返す。
    作成後にキャラクター定義をベクトルストアに登録する。
    """
    state = request.app.state

    # 類似 estranged キャラクターのチェック: 同一定義での再作成を防ぐ
    if body.system_prompt_block1 and hasattr(state, "vector_store") and state.vector_store:
        try:
            similar = state.vector_store.find_similar_definition(body.system_prompt_block1)
            if similar:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "類似する定義を持つキャラクターが既に別れを決断しています。"
                        "この定義ではキャラクターを作成できません。"
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            pass  # ベクトルストアエラーは無視して作成を続行する

    char_id = str(uuid.uuid4())
    char = state.sqlite.create_character(
        character_id=char_id,
        name=body.name,
        system_prompt_block1=body.system_prompt_block1,
        inner_narrative=body.inner_narrative,
        cleanup_config=body.cleanup_config,
        ghost_model=body.ghost_model,
        allowed_tools=body.allowed_tools,
    )

    # キャラクター定義をベクトルストアに登録する（embedding 作成）
    if body.system_prompt_block1 and hasattr(state, "vector_store") and state.vector_store:
        try:
            state.vector_store.upsert_character_definition(char_id, body.system_prompt_block1)
        except Exception as e:
            logger.warning("ベクトルストア キャラクター定義登録失敗 char=%s error=%s", char_id, e)

    return char_to_dict(char)


@router.get("/{character_id}")
async def get_character(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return char_to_dict(char)


@router.patch("/{character_id}")
async def update_character(request: Request, character_id: str, body: CharacterUpdate):
    """キャラクター情報を更新する。

    system_prompt_block1 が変更された場合はベクトルストアの定義 embedding も更新する。
    """
    state = request.app.state
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    char = state.sqlite.update_character(character_id, **updates)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    # system_prompt_block1 が更新された場合は定義 embedding を再登録する
    if "system_prompt_block1" in updates and hasattr(state, "vector_store") and state.vector_store:
        try:
            state.vector_store.upsert_character_definition(character_id, updates["system_prompt_block1"])
        except Exception as e:
            logger.warning("ベクトルストア キャラクター定義更新失敗 char=%s error=%s", character_id, e)

    return char_to_dict(char)


@router.get("/{character_id}/image")
async def get_character_image(request: Request, character_id: str):
    """キャラクターのアバター画像をバイナリで返す。画像未設定の場合は404を返す。"""
    char = request.app.state.sqlite.get_character(character_id)
    if not char or not char.image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        # "data:{mime_type};base64,{b64}" 形式をデコードする
        header, b64_data = char.image_data.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        image_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(status_code=500, detail="Image data is corrupted")
    return Response(content=image_bytes, media_type=mime_type)


@router.get("/{character_id}/face_to_face_bg_image")
async def get_face_to_face_bg_image(
    request: Request, character_id: str,
    label: str | None = None, index: int | None = None,
):
    """対面モード時に ChatView 背景へ表示する画像をバイナリで返す。

    face_to_face_bg_images 配列から `label`（一致）または `index`（位置）で
    1件を解決する。両方省略時は先頭。該当なし・未設定なら 404。
    """
    char = request.app.state.sqlite.get_character(character_id)
    entries = (getattr(char, "face_to_face_bg_images", None) or []) if char else []
    entry = None
    if label is not None:
        entry = next((e for e in entries if e.get("label") == label), None)
    elif index is not None:
        if 0 <= index < len(entries):
            entry = entries[index]
    elif entries:
        entry = entries[0]
    bg = (entry or {}).get("image")
    if not bg:
        raise HTTPException(status_code=404, detail="Background image not found")
    try:
        header, b64_data = bg.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        image_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(status_code=500, detail="Background image data is corrupted")
    return Response(content=image_bytes, media_type=mime_type)


@router.put("/{character_id}/face_to_face_mode")
async def update_face_to_face_mode(request: Request, character_id: str, body: FaceToFaceModeUpdate):
    """対面モードの現在値だけを軽量に切り替える専用エンドポイント。

    1on1チャット画面のトグルから叩く。Settings UI のフル更新（PATCH）を毎クリック
    走らせないため独立させている。enabled=true で 1、false で 0 を書き込む。

    生活カレンダー有効キャラの**起動ガード**（schedule_plan.md §7 (b)）: 就寝中・超繁忙中
    （offline / busy）には対面を始められない。理由（「いま仕事中」等）を返して DB は変更しない。
    無効キャラは従来どおり無条件で切り替わる（従来挙動を変えない）。解除（enabled=false）は
    常に許可する。
    """
    state = request.app.state
    char = state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    # 起動ガード: 生活カレンダー有効キャラを、active/OnTime 以外の時間帯には対面起動させない
    if body.enabled and int(getattr(char, "living_schedule_enabled", 0) or 0):
        from backend.services.gate import check_availability

        availability = check_availability(char, sqlite=state.sqlite)
        if availability.state not in ("OnTime", "active"):
            return {
                "character_id": character_id,
                "face_to_face_mode": int(getattr(char, "face_to_face_mode", 0) or 0),
                "blocked": True,
                "reason": availability.reason or availability.state,
            }

    state.sqlite.update_character(
        character_id,
        face_to_face_mode=1 if body.enabled else 0,
    )
    return {"character_id": character_id, "face_to_face_mode": 1 if body.enabled else 0}


@router.post("/{character_id}/pressure_interview")
async def run_pressure_interview(request: Request, character_id: str):
    """体質インタビュー（めぐり Phase 3）を実施して pressure_profile を初期化する。

    ask_character（1on1 同等のシステムプロンプト・WM ブロック込み）で本人に
    体験の質問を投げ、固定ルーブリックで係数へ決定論写像して保存する。
    機能有効化時に一度呼ぶ想定。再実行すると上書きされる（本人の言葉も
    interview ペイロードに残るため、ルーブリック改良時の再導出が可能）。
    """
    from backend.services.pressure import run_constitution_interview

    state = request.app.state
    result = await run_constitution_interview(
        character_id=character_id,
        sqlite=state.sqlite,
        settings=state.sqlite.get_all_settings(),
        working_memory_manager=getattr(state, "working_memory_manager", None),
    )
    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=result.get("error", "インタビュー失敗"))
    return result


@router.post("/{character_id}/weekly_schedule/rebuild")
async def rebuild_weekly_schedule(request: Request, character_id: str, week: str = "current"):
    """生活カレンダーの週次バッチ（①GM→②本人）を手動で回し直す。

    生活時間割（テンプレ層・例外日）を直した直後に、その週の実現層を作り直すための口。
    バッチ本体が「対象週の template を消して入れ直す」冪等設計なので再実行がそのまま
    成立する（③④の adhoc エントリは温存される）。自動実行との二重生成を避けるため、
    冪等キー `weekly_schedule_done_{id}` も対象週まで進める。

    Args:
        week: "current"（今週・既定）か "next"（来週）。

    Returns:
        生成件数と経路（{"world":n, "haru":n, "world_mode":..., "events":n}）＋対象週。
    """
    from datetime import date, timedelta

    from backend.services.schedule import (
        run_weekly_schedule_batch,
        week_key,
        week_start_of,
    )

    state = request.app.state
    char = state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    if not int(getattr(char, "living_schedule_enabled", 0) or 0):
        raise HTTPException(status_code=400, detail="生活カレンダーが無効です")
    if week not in ("current", "next"):
        raise HTTPException(status_code=400, detail="week は current / next のみ")

    target = week_start_of(date.today())
    if week == "next":
        target += timedelta(days=7)
    try:
        summary = await run_weekly_schedule_batch(state, char, target)
    except Exception as e:
        logger.exception("週次バッチ手動再生成に失敗 char=%s", character_id)
        raise HTTPException(status_code=500, detail=f"再生成に失敗: {e}")

    done_key = f"weekly_schedule_done_{character_id}"
    wkey = week_key(target)
    # 自動実行の冪等キーは「最後に生成した週」。手動で先へ進めた分だけ前進させる
    if str(state.sqlite.get_setting(done_key, "") or "") < wkey:
        state.sqlite.set_setting(done_key, wkey)
    state.sqlite.record_scheduler_decision(
        "weekly_schedule", "fired", character_id=character_id,
        reason=f"手動再生成（{wkey}）", details={"week": wkey, "manual": True, **(summary or {})},
    )
    return {"character_id": character_id, "week": wkey, **(summary or {})}


@router.get("/{character_id}/pressures")
async def get_pressures(request: Request, character_id: str):
    """現在の圧力3変数（読み取り時計算）を返す。デバッグ・管理UI用。"""
    from backend.services.pressure import compute_pressures

    state = request.app.state
    char = state.sqlite.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return {
        "character_id": character_id,
        "pressures": compute_pressures(state.sqlite, character_id),
        "profile": getattr(char, "pressure_profile", None),
    }


@router.delete("/{character_id}", status_code=204)
async def delete_character(request: Request, character_id: str):
    """キャラクターと、紐づく全データをカスケード削除する（SQLite → LanceDB の順）。"""
    ok = request.app.state.memory_manager.delete_character_with_inscribed_memories(character_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Character not found")
