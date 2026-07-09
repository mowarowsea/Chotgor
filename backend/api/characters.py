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
async def get_face_to_face_bg_image(request: Request, character_id: str):
    """対面モード時に ChatView 背景へ表示する画像をバイナリで返す。未設定なら 404。"""
    char = request.app.state.sqlite.get_character(character_id)
    bg = getattr(char, "face_to_face_bg_image", None) if char else None
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
