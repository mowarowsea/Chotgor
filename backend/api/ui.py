"""Settings UI routes using Jinja2 templates."""

import base64
import uuid
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.services.memory.reindex_service import reindex_with_new_embeddings
from backend.providers.registry import (
    PROVIDER_LABELS,
    PROVIDER_ORDER,
)

router = APIRouter(prefix="/ui", tags=["ui"])


MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2MB


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
        {web_search, google_calendar, gmail, google_drive} の bool dict。
    """
    return {
        "web_search":       bool(form.get("tool_web_search")),
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


async def _read_image_data(form) -> Optional[str]:
    """フォームから画像を読み込みbase64 data URIとして返す。画像がなければNone。"""
    image_file = form.get("image")
    if not image_file or not hasattr(image_file, "read"):
        return None
    content = await image_file.read()
    if not content:
        return None
    if len(content) > MAX_IMAGE_BYTES:
        return None  # サイズ超過は無視
    content_type = image_file.content_type or "image/png"
    b64 = base64.b64encode(content).decode()
    return f"data:{content_type};base64,{b64}"


def _is_ajax(request: Request) -> bool:
    """fetch / XHR からのリクエストかどうかを X-Requested-With ヘッダで判定する。"""
    return request.headers.get("x-requested-with", "").lower() in (
        "fetch",
        "xmlhttprequest",
    )


def _save_response(request: Request, redirect_url: str):
    """保存完了レスポンスを返す。

    自動保存（AJAX）の場合は JSON を、通常のフォーム送信の場合は
    従来どおりリダイレクトを返す。これにより 1 つのハンドラが
    「Save ボタン送信」と「フィールド変更ごとの自動保存」の両方に対応する。

    Args:
        request: リクエスト。AJAX 判定に使う。
        redirect_url: 通常送信時のリダイレクト先。

    Returns:
        JSONResponse または RedirectResponse。
    """
    if _is_ajax(request):
        return JSONResponse({"ok": True})
    return RedirectResponse(url=redirect_url, status_code=303)


def setup_templates(templates_dir: str) -> Jinja2Templates:
    return Jinja2Templates(directory=templates_dir)


templates: Optional[Jinja2Templates] = None


def get_templates() -> Jinja2Templates:
    if templates is None:
        raise RuntimeError("Templates not initialized")
    return templates


# --- Dashboard ---

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    chars = request.app.state.sqlite.list_characters()
    return get_templates().TemplateResponse(
        "dashboard.html",
        {"request": request, "characters": chars},
    )


# --- Characters ---

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
    afterglow_default = 1 if form.get("afterglow_default") else 0
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
        afterglow_default=afterglow_default,
        self_reflection_mode=self_reflection_mode,
        self_reflection_preset_id=self_reflection_preset_id,
        self_reflection_n_turns=self_reflection_n_turns,
        allowed_tools=allowed_tools,
    )
    return RedirectResponse(url="/ui/", status_code=303)


@router.get("/characters/{character_id}", response_class=HTMLResponse)
async def edit_character_form(request: Request, character_id: str):
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)
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
    afterglow_default = 1 if form.get("afterglow_default") else 0
    self_reflection_mode, self_reflection_preset_id, self_reflection_n_turns = (
        _extract_self_reflection_params(form)
    )

    update_kwargs: dict = dict(
        system_prompt_block1=form.get("system_prompt_block1", ""),
        enabled_providers=enabled_providers,
        ghost_model=ghost_model,
        switch_angle_enabled=switch_angle_enabled,
        afterglow_default=afterglow_default,
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
    return _save_response(request, "/ui/")


@router.post("/characters/{character_id}/delete")
async def delete_character(request: Request, character_id: str):
    """キャラクターと全保存記憶を削除する。LanceDB・SQLite の順に削除する。"""
    request.app.state.memory_manager.delete_character_with_inscribed_memories(character_id)
    return RedirectResponse(url="/ui/", status_code=303)


# --- Memories ---

@router.get("/inscribed_memories/{character_id}", response_class=HTMLResponse)
async def memories_view(
    request: Request,
    character_id: str,
    category: Optional[str] = None,
    deleted_only: bool = False,
):
    """記憶一覧ページ。deleted_only=True のときは論理削除済み記憶のみ表示する。"""
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)

    memories = request.app.state.memory_manager.list_inscribed_memories(
        character_id=character_id,
        category=category,
        include_deleted=deleted_only,
    )
    # deleted_only モードでは削除済みレコードのみに絞る
    if deleted_only:
        memories = [m for m in memories if m["deleted_at"]]

    # プリセットID→名前マップを構築し、記憶作成元プリセット名をテンプレートで表示できるようにする
    presets = request.app.state.sqlite.list_model_presets()
    preset_name_map = {p.id: p.name for p in presets}

    categories = ["identity", "user", "semantic", "contextual"]
    return get_templates().TemplateResponse(
        "memories.html",
        {
            "request": request,
            "character": char,
            "memories": memories,
            "categories": categories,
            "selected_category": category,
            "deleted_only": deleted_only,
            "preset_name_map": preset_name_map,
        },
    )


@router.post("/inscribed_memories/{character_id}/{memory_id}/delete")
async def delete_memory(request: Request, character_id: str, memory_id: str):
    request.app.state.memory_manager.delete_inscribed_memory(memory_id, character_id)
    return RedirectResponse(url=f"/ui/inscribed_memories/{character_id}", status_code=303)


@router.get("/working-memory/{character_id}", response_class=HTMLResponse)
async def working_memory_view(
    request: Request,
    character_id: str,
    type: Optional[str] = None,
    archived: bool = False,
):
    """ワーキングメモリのスレッド一覧ページ（読み取り専用）。

    type 指定でスレッド種別を絞り込み、archived=True でアーカイブ済みのみ表示する。
    記憶の取捨選択はキャラクター自身が行うため、UI からの編集機能は設けない。
    """
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)

    wm = request.app.state.working_memory_manager
    threads = wm.list_threads_by_type(
        character_id,
        type=type or None,
        is_open=(False if archived else None),
    )
    types = ["emotion", "body", "task", "topic", "relation"]
    return get_templates().TemplateResponse(
        "working_memory.html",
        {
            "request": request,
            "character": char,
            "threads": threads,
            "types": types,
            "selected_type": type,
            "archived": archived,
        },
    )


@router.get("/working-memory/{character_id}/{thread_id}", response_class=HTMLResponse)
async def working_memory_thread_view(request: Request, character_id: str, thread_id: str):
    """ワーキングメモリのスレッド詳細ページ（全ポストを時系列表示）。"""
    char = request.app.state.sqlite.get_character(character_id)
    if not char:
        return RedirectResponse(url="/ui/", status_code=303)

    wm = request.app.state.working_memory_manager
    thread = wm.get_thread_detail(thread_id)
    if thread is None or thread.get("character_id") != character_id:
        return RedirectResponse(url=f"/ui/working-memory/{character_id}", status_code=303)

    return get_templates().TemplateResponse(
        "working_memory_thread.html",
        {
            "request": request,
            "character": char,
            "thread": thread,
        },
    )


# --- Model Presets ---

@router.get("/model-presets", response_class=HTMLResponse)
async def model_presets_list(request: Request):
    presets = request.app.state.sqlite.list_model_presets()
    return get_templates().TemplateResponse(
        "model_presets.html",
        {
            "request": request,
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


def _coalesce_optional_int(form, key: str):
    """フォームから任意の int 値を取り出す。空欄は None、不正値も None。"""
    raw = (form.get(key) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


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
    user_alias = (form.get("user_alias") or "").strip() or "プレイヤー"
    if not title:
        return RedirectResponse(url="/ui/scenarios/new", status_code=303)

    sid = str(uuid.uuid4())
    request.app.state.sqlite.create_scenario(
        scenario_id=sid,
        title=title,
        user_alias=user_alias,
        scenario=(form.get("scenario") or "") or None,
        intro=(form.get("intro") or "") or None,
        history_max_turns=_coalesce_optional_int(form, "history_max_turns"),
        history_max_chars=_coalesce_optional_int(form, "history_max_chars"),
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
        "user_alias": (form.get("user_alias") or "").strip() or "プレイヤー",
        "scenario": (form.get("scenario") or "") or None,
        "intro": (form.get("intro") or "") or None,
        "history_max_turns": _coalesce_optional_int(form, "history_max_turns"),
        "history_max_chars": _coalesce_optional_int(form, "history_max_chars"),
    }
    # タイトルは必須項目。空欄なら更新しない（自動保存の空入力対策）。
    title = (form.get("title") or "").strip()
    if title:
        update_kwargs["title"] = title
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

@router.get("/settings", response_class=HTMLResponse)
async def settings_form(request: Request):
    """設定ページを表示する。Google APIキー設定状況とモデルプリセット一覧をテンプレートに渡す。"""
    settings = request.app.state.sqlite.get_all_settings()
    model_presets = request.app.state.sqlite.list_model_presets()
    return get_templates().TemplateResponse(
        "settings.html",
        {
            "request": request,
            "settings": settings,
            "has_google_key": bool(settings.get("google_api_key")),
            "model_presets": model_presets,
        },
    )


@router.post("/settings/general")
async def save_general_settings(request: Request):
    """embedding 以外の設定を保存する（即時反映・再インデックスを伴わない）。

    フィールド変更ごとの自動保存対象。embedding 設定はミス操作で全記憶の
    再インデックスを招くため、ここでは扱わず save_embedding_settings に分離する。
    """
    form = await request.form()
    store = request.app.state.sqlite

    # ユーザ名は常に保存（マスク不要）
    store.set_setting("user_name", (form.get("user_name") or "").strip() or "ユーザ")

    # APIキーはマスク値（●のみ）でなければ更新する
    for key in (
        "anthropic_api_key",
        "openai_api_key",
        "xai_api_key",
        "google_api_key",
        "openrouter_api_key",
        "tavily_api_key",
    ):
        value = form.get(key) or ""
        if value and set(value) != {"●"}:
            store.set_setting(key, value)

    chronicle_time = form.get("chronicle_time") or "03:00"
    try:
        h, m = map(int, chronicle_time.split(":"))
        assert 0 <= h <= 23 and 0 <= m <= 59
        store.set_setting("chronicle_time", chronicle_time)
    except Exception:
        pass

    store.set_setting("enable_time_awareness", bool(form.get("enable_time_awareness")))

    try:
        cw = int(form.get("context_window_max_chronicled") or 10)
    except (TypeError, ValueError):
        cw = 10
    store.set_setting("context_window_max_chronicled", str(max(0, min(200, cw))))

    # 翻訳モデル設定の保存
    store.set_setting("translation_preset_id", form.get("translation_preset_id") or "")

    # 司会モデル設定の保存（グループチャットの次発言者判断に使用）
    store.set_setting("group_director_preset_id", form.get("group_director_preset_id") or "")

    # Ollama 設定の保存
    store.set_setting("ollama_base_url", (form.get("ollama_base_url") or "http://localhost:11434").strip())
    store.set_setting("ollama_no_think", "true" if form.get("ollama_no_think") else "false")

    return _save_response(request, "/ui/settings?saved=1")


@router.post("/settings/embedding")
async def save_embedding_settings(
    request: Request,
    embedding_provider: str = Form("default"),
    embedding_model: str = Form(""),
    infinity_base_url: str = Form("http://localhost:7997"),
):
    """embedding 設定を保存し、変更時は全記憶を再インデックスする。

    再インデックスは重い処理かつ後戻りしにくいため、自動保存ではなく
    明示的なボタン操作（通常のフォーム送信）でのみ実行する。
    """
    store = request.app.state.sqlite

    # embedding設定の変更を検出（保存前に現在値を取得）
    old_embedding_provider = store.get_setting("embedding_provider", "default")
    old_embedding_model = store.get_setting("embedding_model", "")

    store.set_setting("embedding_provider", embedding_provider)
    store.set_setting("embedding_model", embedding_model)
    if infinity_base_url.strip():
        store.set_setting("infinity_base_url", infinity_base_url.strip())

    # embeddingモデルが変更された場合は全記憶を再インデックスする
    embedding_changed = (
        embedding_provider != old_embedding_provider
        or embedding_model != old_embedding_model
    )
    if embedding_changed:
        current_google_key = store.get_setting("google_api_key", "")
        current_infinity_url = store.get_setting("infinity_base_url", "http://localhost:7997")
        try:
            state = request.app.state
            new_vector_store, new_memory_manager, new_chat_service = await reindex_with_new_embeddings(
                sqlite=state.sqlite,
                vector_store=state.vector_store,
                working_memory_manager=state.working_memory_manager,
                new_provider=embedding_provider,
                new_model=embedding_model,
                new_api_key=current_google_key,
                new_base_url=current_infinity_url,
            )
            # 再インデックス後に app.state を新しいインスタンスで更新する
            state.vector_store = new_vector_store
            state.memory_manager = new_memory_manager
            state.chat_service = new_chat_service
        except Exception:
            return RedirectResponse(url="/ui/settings?saved=1&reindex_error=1", status_code=303)

    return RedirectResponse(url="/ui/settings?saved=1", status_code=303)


@router.post("/settings/reindex")
async def reindex_memories(request: Request):
    """現在のembedding設定で全キャラクターの記憶を強制再インデックスする。

    Geminiモデル変更直後にコレクションが空になった場合の緊急修復用。
    SQLiteに記憶データが残っていれば復元できる。
    設定は変更せず、現在のDBに保存されているembedding設定をそのまま使用する。
    """
    store = request.app.state.sqlite
    current_provider = store.get_setting("embedding_provider", "default")
    current_model = store.get_setting("embedding_model", "")
    current_api_key = store.get_setting("google_api_key", "")
    current_infinity_url = store.get_setting("infinity_base_url", "http://localhost:7997")
    try:
        state = request.app.state
        new_vector_store, new_memory_manager, new_chat_service = await reindex_with_new_embeddings(
            sqlite=state.sqlite,
            vector_store=state.vector_store,
            working_memory_manager=state.working_memory_manager,
            new_provider=current_provider,
            new_model=current_model,
            new_api_key=current_api_key,
            new_base_url=current_infinity_url,
        )
        state.vector_store = new_vector_store
        state.memory_manager = new_memory_manager
        state.chat_service = new_chat_service
    except Exception:
        return RedirectResponse(url="/ui/settings?reindex_error=1", status_code=303)
    return RedirectResponse(url="/ui/settings?reindex_done=1", status_code=303)


# --- Provider helpers ---

@router.get("/providers/{provider_id}/models")
async def get_provider_models(request: Request, provider_id: str):
    """指定プロバイダーの利用可能なモデル一覧を取得して返す。

    各プロバイダーの list_models() を呼び出す。
    APIキー未設定・取得不可の場合は空リストを返す。
    """
    from backend.providers.registry import PROVIDER_REGISTRY
    cls = PROVIDER_REGISTRY.get(provider_id)
    if cls is None:
        return JSONResponse({"models": []})
    settings = request.app.state.sqlite.get_all_settings()
    models = await cls.list_models(settings)
    return JSONResponse({"models": models})


@router.get("/providers/{provider_id}/embedding-models")
async def get_provider_embedding_models(request: Request, provider_id: str):
    """指定プロバイダーの利用可能な Embedding モデル一覧を取得して返す。

    各プロバイダーの list_embedding_models() を呼び出す。
    APIキー未設定・取得不可の場合は空リストを返す。
    """
    from backend.providers.registry import PROVIDER_REGISTRY
    cls = PROVIDER_REGISTRY.get(provider_id)
    if cls is None:
        return JSONResponse({"models": []})
    settings = request.app.state.sqlite.get_all_settings()
    models = await cls.list_embedding_models(settings)
    return JSONResponse({"models": models})
