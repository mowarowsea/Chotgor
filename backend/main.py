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
from backend.api import characters, inscribed_memories, chat as chat_module, chat_images as chat_images_module, scenario_chat as scenario_chat_module
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
    # instrument_recorder に SQLiteStore をセットして計器アラームの記録を有効化する
    from backend.lib import instrument_recorder
    instrument_recorder.set_store(sqlite)
    # 計器の稼働開始時刻を記録する（静音期間の起点。初回のみ）
    if not sqlite.get_setting("instruments_started_at"):
        sqlite.set_setting("instruments_started_at", datetime.now().isoformat())

    _log.info("Chotgor backend 起動 sqlite=%s", SQLITE_DB_PATH)

    asyncio.create_task(_chronicle_scheduler(app))
    asyncio.create_task(_forget_scheduler(app))
    asyncio.create_task(_usual_days_scheduler(app))
    asyncio.create_task(_instruments_scheduler(app))
    asyncio.create_task(_action_scheduler(app))
    asyncio.create_task(_escrow_delivery_scheduler(app))
    asyncio.create_task(_weekly_schedule_scheduler(app))

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


async def _instruments_scheduler(app: FastAPI) -> None:
    """Background task: 毎日 05:00（設定可）に計器の巡回チェックを実行する。

    巡回時刻は Chronicle（03:00）→ Forget（04:00）の後に置く
    （night_batch_heartbeat が当日の夜間バッチ完了を前提とするため）。
    実行内容:
        - Tier 1 巡回インバリアント（run_patrol_checks）
        - Tier 2 肥大メーターの日次スナップショット（record_bloat_meters）
        - Tier 3 判定巡回（run_judgement_patrol。判定プリセット未設定ならスキップ）
    冪等キーは chronicle スケジューラと同型（日付文字列）。
    """
    _log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(60)
        now = datetime.now()
        patrol_time_str = app.state.sqlite.get_setting("instruments_patrol_time", "05:00")
        try:
            h, m = map(int, patrol_time_str.split(":"))
        except Exception:
            h, m = 5, 0
        scheduled = now.replace(hour=h, minute=m, second=0, microsecond=0)
        today_str = now.date().isoformat()
        last_run = app.state.sqlite.get_setting("instruments_last_run_date", "")
        if now >= scheduled and last_run != today_str:
            _log.info("計器巡回スケジューラー 起動 設定時刻=%s", patrol_time_str)
            app.state.sqlite.set_setting("instruments_last_run_date", today_str)
            try:
                from backend.services.instruments import (
                    record_bloat_meters,
                    run_judgement_patrol,
                    run_patrol_checks,
                )
                from backend.services.pressure import record_pressure_meters
                summary = run_patrol_checks(app.state.sqlite)
                meters = record_bloat_meters(app.state.sqlite)
                # 圧力は保存しない純関数だが、傾向観測の日次スナップショットだけ残す
                pressure_meters = record_pressure_meters(app.state.sqlite)
                judge_result = await run_judgement_patrol(
                    app.state.sqlite, app.state.sqlite.get_all_settings()
                )
                _log.info(
                    "計器巡回 完了 patrol=%s meters=%d pressure=%d judge=%s",
                    summary, meters, pressure_meters, judge_result.get("status"),
                )
            except Exception:
                _log.exception("計器巡回スケジューラー 実行エラー")


async def _action_scheduler(app: FastAPI) -> None:
    """Background task: 会話外行動権の周期評価（めぐり Phase 6）。

    行動メニューが1つでも ON のキャラクターについて、2時間格子＋キャラ別ジッター
    （決定論・乱数は世界に置く）のタイミングで評価する:

        availability 確認 → 閾値評価（純関数・無料）→ 閾値超えのときだけ
        本人問い合わせ → 本人の選択で実行 or 見送り → 帰還

    unavailable のスロットは流す（通過したスロットは捨てる — うつつと同じ思想）。
    冪等キーはキャラごと1本（最後に評価したスロットの ISO 文字列）。
    """
    from backend.services.actions import jittered_slot_time, run_action_cycle
    from backend.services.gate import check_availability, is_usual_scene_running

    _log = logging.getLogger(__name__)
    period_minutes = 120
    while True:
        await asyncio.sleep(60)
        try:
            sqlite = app.state.sqlite
            now = datetime.now()
            # 現在のスロット開始時刻（period_minutes 格子に床合わせ）
            minutes_of_day = now.hour * 60 + now.minute
            slot_index = minutes_of_day // period_minutes
            slot_start = now.replace(
                hour=(slot_index * period_minutes) // 60,
                minute=(slot_index * period_minutes) % 60,
                second=0, microsecond=0,
            )
            for char in sqlite.list_characters():
                menu = getattr(char, "action_menu", None) or {}
                if not isinstance(menu, dict) or not any(
                    menu.get(k) for k in ("push", "research", "impromptu_scene")
                ):
                    continue
                key = f"action_eval_{char.id}"
                if sqlite.get_setting(key, "") == slot_start.isoformat():
                    continue  # このスロットは評価済み
                if now < jittered_slot_time(char.id, slot_start):
                    continue  # ジッター時刻がまだ来ていない
                # 評価を試みた時点で冪等キーを立てる（unavailable でも流す）
                sqlite.set_setting(key, slot_start.isoformat())
                availability = check_availability(
                    char, now,
                    usual_scene_running=is_usual_scene_running(sqlite, char.id, now),
                    sqlite=sqlite,
                )
                if not availability.available:
                    _log.debug(
                        "行動権: unavailable のためスロットを流す char=%s reason=%s",
                        char.name, availability.reason,
                    )
                    continue
                result = await run_action_cycle(
                    char.id, sqlite, sqlite.get_all_settings(),
                    chat_service=app.state.chat_service,
                    memory_manager=app.state.memory_manager,
                    working_memory_manager=app.state.working_memory_manager,
                )
                if result.get("status") in ("executed", "declined", "error"):
                    _log.info("行動権: char=%s result=%s", char.name, result)
        except Exception:
            _log.exception("行動権スケジューラー 実行エラー")


def _parse_slot_time(now: datetime, slot: str) -> datetime | None:
    """"HH:MM" 形式のスロット文字列を、今日のその時刻の datetime に変換する。

    パースできなければ None。うつつスケジューラのスロット到来判定に使う。
    """
    try:
        h, m = map(int, str(slot).strip().split(":"))
    except Exception:
        return None
    if not (0 <= h < 24 and 0 <= m < 60):
        return None
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


async def _run_due_usual_scenes(app: FastAPI) -> None:
    """到来済みのうつつスロットを 1 回ずつ無人進行させる（冪等キー＝日付+スロット）。

    - 有効（usual_config.enabled）なうつつ世界ごとに、今日すでに到来したスロットを処理。
    - 冪等キー ``usual_days_last_run_{owner}_{slot}`` に当日日付を立てて二重起動を防ぐ。
    - 日次コストガード ``usual_days_daily_cap``（既定 24）を超えたら以降は当日スキップ。
    - セッションは永続1本（ensure_usual_session が find-or-create）。
    - GM へ前回シーンからの経過時間メモを添える。
    """
    from backend.services.scenario_chat.service import (
        ensure_usual_session,
        run_usual_days_scene,
        usual_elapsed_note,
    )

    _log = logging.getLogger(__name__)
    sqlite = app.state.sqlite
    now = datetime.now()
    today_str = now.date().isoformat()

    daily_cap = int(sqlite.get_setting("usual_days_daily_cap", "24") or 24)
    count_key = f"usual_days_scene_count_{today_str}"
    ran_today = int(sqlite.get_setting(count_key, "0") or 0)

    for scenario in sqlite.list_usual_scenarios():
        cfg = getattr(scenario, "usual_config", None) or {}
        if not cfg.get("enabled"):
            continue
        owner_id = getattr(scenario, "owner_character_id", None)
        # 対面モード中はそのキャラのうつつを止める。確定方針として「通過したスロットは捨てる」ため、
        # スキップ時にも冪等キーを当日日付で立てて、対面解除後の駆け込み実行を防ぐ。
        # （次のスロット時刻まで自然に待ち、そこからスケジュールどおりに再開する）。
        owner_char = sqlite.get_character(owner_id) if owner_id else None
        owner_in_face_to_face = bool(getattr(owner_char, "face_to_face_mode", 0)) if owner_char else False
        slots = cfg.get("slots") or []
        for slot in slots:
            scheduled = _parse_slot_time(now, slot)
            if scheduled is None or now < scheduled:
                continue  # まだ到来していない / 不正スロット
            key = f"usual_days_last_run_{owner_id}_{slot}"
            if sqlite.get_setting(key, "") == today_str:
                continue  # 当日このスロットは実行済み
            if owner_in_face_to_face:
                # 通過分は捨てる：冪等キーを立てて当日この時刻は再評価しない。
                sqlite.set_setting(key, today_str)
                _log.info(
                    "うつつ: 対面モード中につきスキップ owner=%s slot=%s",
                    owner_id, slot,
                )
                continue
            if ran_today >= daily_cap:
                _log.warning(
                    "うつつ: 日次上限 %d 到達。当日は以降スキップ owner=%s slot=%s",
                    daily_cap, owner_id, slot,
                )
                return
            session = ensure_usual_session(sqlite, scenario)
            # 冪等キーは「実行を試みた」時点で立てる（設定不備でも毎分叩かない）。
            sqlite.set_setting(key, today_str)
            if session is None:
                _log.error("うつつ: セッション未解決のためスキップ owner=%s", owner_id)
                continue
            elapsed_note = usual_elapsed_note(sqlite, session.id, now)
            _log.info(
                "うつつ: シーン起動 owner=%s slot=%s session=%s",
                owner_id, slot, session.id,
            )
            try:
                result = await run_usual_days_scene(
                    session_id=session.id,
                    sqlite=sqlite,
                    settings=sqlite.get_all_settings(),
                    chat_service=app.state.chat_service,
                    extra_first_gm_ooc=elapsed_note,
                    slot=slot,
                )
                ran_today += 1
                sqlite.set_setting(count_key, str(ran_today))
                # 無人運転なので結果を必ずログに残す（GM/PC のエラーは観測できるように）。
                # responses = LLM 呼出回数（GM + PC）、turns = 保存された話者ブロック数。
                if result.get("error"):
                    _log.warning(
                        "うつつ: シーンがエラーで中断 owner=%s slot=%s responses=%d turns=%d error=%s",
                        owner_id, slot,
                        result.get("fired_responses", 0), result.get("fired_turns", 0),
                        result["error"],
                    )
                else:
                    _log.info(
                        "うつつ: シーン完了 owner=%s slot=%s responses=%d turns=%d scene_closed=%s",
                        owner_id, slot,
                        result.get("fired_responses", 0), result.get("fired_turns", 0),
                        result.get("scene_closed"),
                    )
            except Exception:
                _log.exception("うつつ: シーン実行エラー owner=%s slot=%s", owner_id, slot)


async def _usual_days_scheduler(app: FastAPI) -> None:
    """Background task: 有効なうつつ世界を各スロット時刻に1シーンずつ無人進行させる。

    _chronicle_scheduler と同型（while True: sleep(60)）。冪等キーを日付+スロットにして
    1日複数スロットへ対応する。SSE 配信は不要なので run_usual_days_scene を await 回収する。
    """
    _log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(60)
        try:
            await _run_due_usual_scenes(app)
        except Exception:
            _log.exception("うつつスケジューラー 実行エラー")


async def _escrow_delivery_scheduler(app: FastAPI) -> None:
    """Background task: 預かり（escrow）メッセージの能動配達（めぐり §5.1 フォローアップ）。

    毎分、未配達メッセージ（delivered_at=NULL）を持つセッションを走査し、
    キャラの availability が戻っていれば決定論ジッター（0〜10分）を挟んで
    本人へ配達し、返信を生成・保存する。判定・ガードの詳細は
    services/gate/delivery.py 参照。
    """
    from backend.services.gate.delivery import run_pending_escrow_deliveries

    _log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(60)
        try:
            await run_pending_escrow_deliveries(app.state)
        except Exception:
            _log.exception("能動配達スケジューラー 実行エラー")


async def _weekly_schedule_scheduler(app: FastAPI) -> None:
    """Background task: 生活カレンダーの週次バッチ①②（schedule_plan.md §3 / Phase 3）。

    毎分、生活カレンダー有効キャラを走査し、未生成の週があればバッチを実行する。
    日曜夜（weekly_schedule_time・既定 20:00）に翌週分、コールドスタート
    （機能有効化・取りこぼし復旧）では当週分を即時生成する。
    冪等キー = キャラごとの対象 ISO 週（判定の詳細は services/schedule/weekly_batch.py）。
    """
    from backend.services.schedule import run_pending_weekly_batches

    _log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(60)
        try:
            await run_pending_weekly_batches(app.state)
        except Exception:
            _log.exception("週次スケジュールスケジューラー 実行エラー")


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
app.include_router(scenario_chat_module.router)
app.include_router(logs_ui_module.router)
app.include_router(logs_ui_module.json_router)
app.include_router(translation_module.router)
app.include_router(mcp_tools_module.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
