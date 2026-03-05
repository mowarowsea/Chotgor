"""Chotgor backend — FastAPI application entry point."""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import characters, memories, openai_compat
from .api import ui as ui_module
from .core.memory.chroma_store import ChromaStore
from .core.memory.digest import run_pending_digests
from .core.memory.forget import run_pending_forget
from .core.memory.manager import MemoryManager
from .core.memory.sqlite_store import SQLiteStore

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/chotgor.db")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")
TEMPLATES_DIR = str(Path(__file__).parent / "templates")
STATIC_DIR = str(Path(__file__).parent / "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize stores
    os.makedirs(os.path.dirname(os.path.abspath(SQLITE_DB_PATH)), exist_ok=True)

    sqlite = SQLiteStore(SQLITE_DB_PATH)
    chroma = ChromaStore(CHROMA_DB_PATH)
    memory_manager = MemoryManager(sqlite=sqlite, chroma=chroma)

    app.state.sqlite = sqlite
    app.state.chroma = chroma
    app.state.memory_manager = memory_manager

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

    asyncio.create_task(_digest_scheduler(app))
    asyncio.create_task(_forget_scheduler(app))

    yield
    # Shutdown: nothing to clean up for SQLite/ChromaDB


async def _digest_scheduler(app: FastAPI) -> None:
    """Background task: run pending digests once per day at the configured time."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now()
        digest_time_str = app.state.sqlite.get_setting("digest_time", "03:00")
        try:
            h, m = map(int, digest_time_str.split(":"))
        except Exception:
            h, m = 3, 0
        scheduled = now.replace(hour=h, minute=m, second=0, microsecond=0)
        today_str = now.date().isoformat()
        last_run = app.state.sqlite.get_setting("digest_last_run_date", "")
        if now >= scheduled and last_run != today_str:
            app.state.sqlite.set_setting("digest_last_run_date", today_str)
            try:
                await run_pending_digests(app.state.sqlite, app.state.memory_manager)
            except Exception:
                pass


async def _forget_scheduler(app: FastAPI) -> None:
    """Background task: run pending forget process once per day at 04:00."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now()
        # Default to 04:00 (after digest usually runs)
        h, m = 4, 0
        scheduled = now.replace(hour=h, minute=m, second=0, microsecond=0)
        today_str = now.date().isoformat()
        last_run = app.state.sqlite.get_setting("forget_last_run_date", "")
        if now >= scheduled and last_run != today_str:
            app.state.sqlite.set_setting("forget_last_run_date", today_str)
            try:
                await run_pending_forget(app.state.sqlite, app.state.memory_manager)
            except Exception:
                pass


app = FastAPI(
    title="Chotgor Backend",
    description="AI character memory management system",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include routers
app.include_router(openai_compat.router)
app.include_router(characters.router)
app.include_router(memories.router)
app.include_router(ui_module.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
