"""Chotgor backend — FastAPI application entry point."""

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
from backend.api import characters, memories, chat as chat_module, chat_images as chat_images_module, chat_drifts as chat_drifts_module, group_chat as group_chat_module
from backend.api import ui as ui_module
from backend.api import logs_ui as logs_ui_module
from backend.services.chat.service import ChatService
from backend.lib.log_context import setup_logging
from backend.repositories.chroma.store import ChromaStore
from backend.batch.chronicle_job import run_pending_chronicles
from backend.services.memory.drift_manager import DriftManager
from backend.batch.forget_job import run_pending_forget
from backend.services.memory.manager import MemoryManager
from backend.repositories.sqlite.store import SQLiteStore

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/chotgor.db")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./data/uploads")
TEMPLATES_DIR = str(Path(__file__).parent / "templates")
STATIC_DIR = str(Path(__file__).parent / "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理。起動・終了処理を担う。"""
    # ロギング設定を最初に適用する
    setup_logging()
    _log = logging.getLogger(__name__)

    # Startup: initialize stores
    os.makedirs(os.path.dirname(os.path.abspath(SQLITE_DB_PATH)), exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    sqlite = SQLiteStore(SQLITE_DB_PATH)

    # embeddingモデル設定をSQLiteから一括取得してChromaStoreに渡す
    all_settings = sqlite.get_all_settings()
    chroma = ChromaStore(
        CHROMA_DB_PATH,
        embedding_provider=all_settings.get("embedding_provider", "default"),
        embedding_model=all_settings.get("embedding_model", ""),
        api_key=all_settings.get("google_api_key", ""),
    )

    memory_manager = MemoryManager(sqlite=sqlite, chroma=chroma)
    drift_manager = DriftManager(sqlite=sqlite)

    app.state.sqlite = sqlite
    app.state.chroma = chroma
    app.state.chroma_db_path = CHROMA_DB_PATH
    app.state.memory_manager = memory_manager
    app.state.drift_manager = drift_manager
    app.state.chat_service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
    app.state.uploads_dir = UPLOADS_DIR

    # Seed optional Tavily key from environment if not already set
    if not sqlite.get_setting("tavily_api_key"):
        env_key = os.getenv("TAVILY_API_KEY", "")
        if env_key:
            sqlite.set_setting("tavily_api_key", env_key)

    if not sqlite.get_setting("claude_model"):
        sqlite.set_setting("claude_model", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"))

    # Initialize UI templates
    from fastapi.templating import Jinja2Templates

    ui_module.templates = Jinja2Templates(directory=TEMPLATES_DIR)
    # CSS キャッシュバスティング：サーバ起動時のタイムスタンプを全テンプレートに注入
    ui_module.templates.env.globals["css_version"] = str(int(time.time()))

    # ログUIにも同じテンプレートインスタンスを共有する
    logs_ui_module.templates = ui_module.templates

    _log.info("Chotgor backend 起動 sqlite=%s", SQLITE_DB_PATH)

    asyncio.create_task(_chronicle_scheduler(app))
    asyncio.create_task(_forget_scheduler(app))

    yield

    # Shutdown
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
                await run_pending_chronicles(app.state.sqlite)
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

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include routers
app.include_router(openai_router.router)
app.include_router(characters.router)
app.include_router(memories.router)
app.include_router(ui_module.router)
app.include_router(chat_module.router)
app.include_router(chat_images_module.router)
app.include_router(chat_drifts_module.router)
app.include_router(group_chat_module.router)
app.include_router(logs_ui_module.router)
app.include_router(logs_ui_module.json_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
