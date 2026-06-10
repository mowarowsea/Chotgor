"""設定 UI — グローバル設定（API キー・embedding・再インデックス）ページ。"""

import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from backend.api.ui.common import _save_response, get_templates
from backend.services.memory.reindex_service import reindex_with_new_embeddings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])


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
    """embedding 設定を保存し、明示的に全記憶を再インデックスする。

    再インデックスは重い処理かつ後戻りしにくいため、自動保存ではなく
    明示的なボタン操作（通常のフォーム送信）でのみ実行する。
    ボタン名が「保存して再インデックス」のため、保存値の差分有無に関わらず
    ボタンが押された時点で常に再インデックスを実行する（UI 文言との一貫性）。
    """
    store = request.app.state.sqlite

    store.set_setting("embedding_provider", embedding_provider)
    store.set_setting("embedding_model", embedding_model)
    if infinity_base_url.strip():
        store.set_setting("infinity_base_url", infinity_base_url.strip())

    # 「保存して再インデックス」ボタン押下＝常に再インデックス。
    # default プロバイダーは EmbeddingFunction を提供できないため skip する。
    if embedding_provider == "default":
        return RedirectResponse(url="/ui/settings?saved=1", status_code=303)

    current_google_key = store.get_setting("google_api_key", "")
    current_infinity_url = store.get_setting("infinity_base_url", "http://localhost:7997")
    current_ollama_url = store.get_setting("ollama_base_url", "http://localhost:11434")
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
            new_ollama_base_url=current_ollama_url,
        )
        # 再インデックス後に app.state を新しいインスタンスで更新する
        state.vector_store = new_vector_store
        state.memory_manager = new_memory_manager
        state.chat_service = new_chat_service
    except Exception:
        logger.exception("embedding 設定変更時の再インデックスに失敗")
        return RedirectResponse(url="/ui/settings?saved=1&reindex_error=1", status_code=303)

    return RedirectResponse(url="/ui/settings?saved=1&reindex_done=1", status_code=303)


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
    current_ollama_url = store.get_setting("ollama_base_url", "http://localhost:11434")
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
            new_ollama_base_url=current_ollama_url,
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
