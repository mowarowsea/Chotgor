"""Chotgor バックエンド — FastAPI アプリケーションエントリーポイント。"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.adapters.openai import router as openai_router
from backend.api import characters, inscribed_memories, chat as chat_module, chat_images as chat_images_module, group_chat as group_chat_module, scenario_chat as scenario_chat_module
from backend.api import ui as ui_module
from backend.api import logs_ui as logs_ui_module
from backend.api import translation as translation_module
from backend.api import mcp_tools as mcp_tools_module
from backend.services.chat.service import ChatService
from backend.lib.log_context import setup_logging
from backend.repositories.lance.store import LanceStore
from backend.batch.chronicle_job import run_pending_chronicles
from backend.batch.forget_job import run_pending_forget
from backend.services.memory.manager import InscribedMemoryManager
from backend.services.memory.working_memory_manager import WorkingMemoryManager
from backend.repositories.sqlite.store import SQLiteStore

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.parent
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(_PROJECT_ROOT / "data" / "chotgor.db"))
LANCE_DB_PATH = os.getenv("LANCE_DB_PATH", str(_PROJECT_ROOT / "data" / "lancedb"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", str(_PROJECT_ROOT / "data" / "uploads"))
TEMPLATES_DIR = str(Path(__file__).parent / "templates")
STATIC_DIR = str(Path(__file__).parent / "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理。起動・終了処理を担う。"""
    # ロギング設定を最初に適用する
    setup_logging()
    _log = logging.getLogger(__name__)

    # 起動時: ストアを初期化する
    os.makedirs(os.path.dirname(os.path.abspath(SQLITE_DB_PATH)), exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    sqlite = SQLiteStore(SQLITE_DB_PATH)

    # embedding モデル設定を SQLite から一括取得してベクトルストアに渡す
    all_settings = sqlite.get_all_settings()
    embedding_provider = all_settings.get("embedding_provider", "infinity")
    embedding_model = all_settings.get("embedding_model", "")
    api_key = all_settings.get("google_api_key", "")
    base_url = all_settings.get("infinity_base_url", "http://localhost:7997")

    vector_store = LanceStore(
        LANCE_DB_PATH,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        api_key=api_key,
        base_url=base_url,
    )
    _log.info("ベクトルストア: LanceStore (path=%s)", LANCE_DB_PATH)

    memory_manager = InscribedMemoryManager(sqlite=sqlite, vector_store=vector_store)
    working_memory_manager = WorkingMemoryManager(sqlite=sqlite, vector_store=vector_store)

    app.state.sqlite = sqlite
    app.state.vector_store = vector_store
    app.state.memory_manager = memory_manager
    app.state.working_memory_manager = working_memory_manager
    app.state.chat_service = ChatService(
        memory_manager=memory_manager,
        working_memory_manager=working_memory_manager,
    )
    app.state.uploads_dir = UPLOADS_DIR

    # 環境変数から Tavily API キーを設定に反映する（未設定時のみ）
    if not sqlite.get_setting("tavily_api_key"):
        env_key = os.getenv("TAVILY_API_KEY", "")
        if env_key:
            sqlite.set_setting("tavily_api_key", env_key)

    if not sqlite.get_setting("claude_model"):
        sqlite.set_setting("claude_model", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"))

    # UI テンプレートを初期化する
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    # CSS キャッシュバスティング：サーバ起動時のタイムスタンプを全テンプレートに注入
    templates.env.globals["css_version"] = str(int(time.time()))
    ui_module.set_templates(templates)

    # ログUIにも同じテンプレートインスタンスを共有する
    logs_ui_module.set_templates(templates)

    # debug_logger に SQLiteStore をセットして DB ログ書き込みを有効化する
    from backend.lib.debug_logger import logger as debug_logger
    debug_logger.set_store(sqlite)
    # logs_ui に SQLiteStore をセットして DB からログを読み込めるようにする
    logs_ui_module.set_sqlite_store(sqlite)
    # usage_recorder に SQLiteStore をセットして LLM 使用量の記録を有効化する
    from backend.lib import usage_recorder
    usage_recorder.set_store(sqlite)
    # tool_event_recorder に SQLiteStore をセットしてツール実行イベントの記録を有効化する
    from backend.lib import tool_event_recorder
    tool_event_recorder.set_store(sqlite)

    _log.info("Chotgor backend 起動 sqlite=%s", SQLITE_DB_PATH)

    asyncio.create_task(_chronicle_scheduler(app))
    asyncio.create_task(_forget_scheduler(app))

    yield

    # Shutdown — InscribedMemoryManager は LanceStore 移行後はバックグラウンドリトライスレッドを
    # 持たないため stop() 不要。
    _log.info("Chotgor backend 終了")


async def _chronicle_scheduler(app: FastAPI) -> None:
    """Background task: 毎日設定時刻に chronicle を実行する。"""
    _log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(60)
        now = datetime.now()
        chronicle_time_str = app.state.sqlite.get_setting("chronicle_time", "03:00")
        try:
            h, m = map(int, chronicle_time_str.split(":"))
        except Exception:
            h, m = 3, 0
        scheduled = now.replace(hour=h, minute=m, second=0, microsecond=0)
        today_str = now.date().isoformat()
        last_run = app.state.sqlite.get_setting("chronicle_last_run_date", "")
        if now >= scheduled and last_run != today_str:
            _log.info("chronicle スケジューラー 起動 設定時刻=%s", chronicle_time_str)
            app.state.sqlite.set_setting("chronicle_last_run_date", today_str)
            try:
                await run_pending_chronicles(
                    app.state.sqlite,
                    vector_store=app.state.vector_store,
                    memory_manager=app.state.memory_manager,
                    working_memory_manager=app.state.working_memory_manager,
                )
            except Exception:
                _log.exception("chronicle スケジューラー 実行エラー")


async def _forget_scheduler(app: FastAPI) -> None:
    """Background task: 毎日 04:00 に forget プロセスを実行する。"""
    _log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(60)
        now = datetime.now()
        # デフォルト 04:00（chronicle 実行後）
        h, m = 4, 0
        scheduled = now.replace(hour=h, minute=m, second=0, microsecond=0)
        today_str = now.date().isoformat()
        last_run = app.state.sqlite.get_setting("forget_last_run_date", "")
        if now >= scheduled and last_run != today_str:
            _log.info("forget スケジューラー 起動 設定時刻=%02d:%02d", h, m)
            app.state.sqlite.set_setting("forget_last_run_date", today_str)
            try:
                await run_pending_forget(app.state.sqlite, app.state.memory_manager)
            except Exception:
                _log.exception("forget スケジューラー 実行エラー")


app = FastAPI(
    title="Chotgor Backend",
    description="AI character memory management system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImmutableStaticFiles(StaticFiles):
    """静的ファイルに長期 immutable キャッシュを付与する StaticFiles。

    chotgor.css / autosave.js は base.html から `?v={css_version}` 付きで参照される。
    css_version はサーバ起動時刻のタイムスタンプ（main.py の lifespan で設定）なので、
    内容を変更してサーバを再起動すれば URL のクエリが変わり、ブラウザは確実に再取得する。
    そのためファイル本体には「1年・immutable」のキャッシュを安全に付与でき、
    通常 StaticFiles が毎ナビゲーションで行う条件付きGET（304確認）の往復を無くせる。
    これにより初回以降のページ表示で CSS 読込待ちのカクつきが解消される。
    """

    async def get_response(self, path, scope):
        """ファイル応答に Cache-Control: immutable ヘッダを付与して返す。"""
        response = await super().get_response(path, scope)
        # 200/304 いずれの応答でも長期キャッシュを宣言する（URL のクエリでバスティング済み）
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


# 静的ファイルをマウントする（immutable キャッシュ付き）
if os.path.exists(STATIC_DIR):
    app.mount("/static", ImmutableStaticFiles(directory=STATIC_DIR), name="static")

# ルーターをアプリへ登録する
app.include_router(openai_router.router)
app.include_router(characters.router)
app.include_router(inscribed_memories.router)
app.include_router(ui_module.router)
app.include_router(chat_module.router)
app.include_router(chat_images_module.router)
app.include_router(group_chat_module.router)
app.include_router(scenario_chat_module.router)
app.include_router(logs_ui_module.router)
app.include_router(logs_ui_module.json_router)
app.include_router(translation_module.router)
app.include_router(mcp_tools_module.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
