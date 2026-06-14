"""設定 UI — ダッシュボード＆キャラクター管理ページ。

キャラクターの作成・編集・削除フォームと、それらが使う
フォーム解釈ヘルパーを提供する。
"""

import json
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


def _usual_time_grid_text(usual_config: dict) -> str:
    """うつつの time_grid を、日本語が読める JSON 文字列に整形する（テキストエリア表示用）。

    Jinja の ``| tojson`` フィルタは ``ensure_ascii=True`` で日本語を ``\\uXXXX`` に
    エスケープしてしまい、編集画面のテキストエリアで内容が読めなくなる。
    そのためビュー側で ``ensure_ascii=False`` の JSON 文字列へ変換して渡す。

    Args:
        usual_config: うつつ運用設定 dict（None / 空可）。

    Returns:
        time_grid を表す JSON 文字列。空なら空文字列。
    """
    time_grid = (usual_config or {}).get("time_grid") or {}
    if not time_grid:
        return ""
    return json.dumps(time_grid, ensure_ascii=False)


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
            # 新規作成時はまだ うつつ 世界が存在しない（空の既定値でフォームを描画する）
            "usual_scenario": None,
            "usual_config": {},
            "usual_time_grid_text": "",
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
    # 同一フォームに同梱された うつつ（生活世界）設定も併せて保存する。
    _persist_usual_world(request.app.state.sqlite, char_id, name, form)
    return RedirectResponse(url="/ui/characters", status_code=303)


@router.get("/characters/{character_id}", response_class=HTMLResponse)
async def edit_character_form(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/characters", status_code=303)
    model_presets = request.app.state.sqlite.list_model_presets()
    # うつつ（Usual Days）世界の現在設定。未作成なら None。
    usual_scenario = request.app.state.sqlite.get_usual_scenario(character_id)
    usual_config = (getattr(usual_scenario, "usual_config", None) or {}) if usual_scenario else {}
    return get_templates().TemplateResponse(
        "character_edit.html",
        {
            "request": request,
            "character": char,
            "action": f"/ui/characters/{character_id}",
            "model_presets": model_presets,
            "provider_labels": PROVIDER_LABELS,
            "usual_scenario": usual_scenario,
            "usual_config": usual_config,
            "usual_time_grid_text": _usual_time_grid_text(usual_config),
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
    # 同一フォームに同梱された うつつ（生活世界）設定も併せて保存する。
    # PC 枠の名前には更新後の最新キャラ名を使う。
    char = request.app.state.sqlite.get_character(character_id)
    if char:
        _persist_usual_world(request.app.state.sqlite, character_id, char.name, form)
    return _save_response(request, "/ui/characters")


def _parse_usual_form(form, character_name: str) -> tuple[dict, dict]:
    """うつつ（生活世界）フォームを scenario 更新用 dict と usual_config dict に変換する。

    Args:
        form: FastAPI のフォームデータ。
        character_name: 主人公 PC 枠の名前に使うキャラ名。

    Returns:
        (scenario_kwargs, usual_config) のタプル。
        scenario_kwargs は create_scenario / update_scenario に渡すフィールド群。
    """
    # スロット時刻: カンマ/改行区切りの "HH:MM" を配列化
    slots_raw = (form.get("usual_slots") or "").replace("\n", ",")
    slots = [s.strip() for s in slots_raw.split(",") if s.strip()]

    # 偶発イベントカテゴリ: 1 行 1 カテゴリ
    cats = [c.strip() for c in (form.get("usual_event_categories") or "").splitlines() if c.strip()]

    # 時間グリッド: 自由 JSON（空・不正なら無視）
    time_grid = {}
    tg_raw = (form.get("usual_time_grid") or "").strip()
    if tg_raw:
        try:
            parsed = json.loads(tg_raw)
            if isinstance(parsed, dict):
                time_grid = parsed
        except json.JSONDecodeError:
            pass

    def _num(key: str, default):
        """フォーム値を数値化（空・不正なら default）。"""
        raw = (form.get(key) or "").strip()
        if not raw:
            return default
        try:
            return type(default)(raw)
        except (TypeError, ValueError):
            return default

    usual_config = {
        "enabled": bool(form.get("usual_enabled")),
        "slots": slots,
        "time_grid": time_grid,
        "event_categories": cats,
        "event_probability": _num("usual_event_probability", 0.0),
        "max_turns_per_scene": _num("usual_max_turns", 8),
        "gm_preset_id": (form.get("usual_gm_preset_id") or "").strip(),
        "pc_preset_id": (form.get("usual_pc_preset_id") or "").strip(),
    }

    # 主人公 PC 枠（1 枠固定）。description はこの世界での人物像・知っていること。
    pc_slots = [{
        "slot_id": "pc1",
        "name": character_name,
        "description": (form.get("usual_pc_description") or "").strip(),
    }]
    scenario_kwargs = {
        "scenario": (form.get("usual_world") or "").strip() or None,
        "pc_slots": pc_slots,
        "usual_config": usual_config,
    }
    return scenario_kwargs, usual_config


def _usual_has_content(scenario_kwargs: dict, usual_config: dict) -> bool:
    """うつつフォームに保存すべき実体があるかを判定する。

    キャラ作成・編集のたびに空のうつつシナリオが量産されるのを防ぐためのガード。
    有効化トグル・世界設定・スロット・イベント種・主人公像のいずれかが入力されていれば
    実体ありとみなす。

    Args:
        scenario_kwargs: _parse_usual_form が返す scenario 更新用フィールド群。
        usual_config: _parse_usual_form が返す usual_config dict。

    Returns:
        保存すべき内容があれば True。
    """
    pc_slots = scenario_kwargs.get("pc_slots") or []
    pc_description = pc_slots[0].get("description") if pc_slots else ""
    return bool(
        usual_config.get("enabled")
        or scenario_kwargs.get("scenario")
        or usual_config.get("slots")
        or usual_config.get("event_categories")
        or pc_description
    )


def _persist_usual_world(sqlite, character_id: str, character_name: str, form) -> None:
    """うつつ（生活世界）設定を find-or-create で保存する。

    キャラクター作成・更新フォームに同梱された うつつ フィールドを解釈し、
    owner_character_id 付きのうつつシナリオを作成・更新する（1 キャラ 1 世界、plan §2）。
    汎用シナリオ一覧には出ない（owner 付き除外）。既存シナリオがあれば常に更新し、
    無ければ実体がある場合のみ作成する（未設定キャラの編集で空シナリオを作らないため）。

    Args:
        sqlite: SQLite ストア。
        character_id: オーナーキャラクターID。
        character_name: 主人公 PC 枠の名前に使うキャラ名。
        form: await request.form() の結果。
    """
    scenario_kwargs, usual_config = _parse_usual_form(form, character_name)
    existing = sqlite.get_usual_scenario(character_id)
    if existing:
        sqlite.update_scenario(existing.id, **scenario_kwargs)
    elif _usual_has_content(scenario_kwargs, usual_config):
        sqlite.create_scenario(
            scenario_id=str(uuid.uuid4()),
            title=f"{character_name} のうつつ",
            owner_character_id=character_id,
            **scenario_kwargs,
        )


@router.post("/characters/{character_id}/delete")
async def delete_character(request: Request, character_id: str):
    """キャラクターと、紐づく全データをカスケード削除する（SQLite → LanceDB の順）。"""
    request.app.state.memory_manager.delete_character_with_inscribed_memories(character_id)
    return RedirectResponse(url="/ui/characters", status_code=303)


# --- Memories ---

