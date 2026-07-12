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

    # 日次スケジューラ（毎日 HH:MM に1回。冪等キー={name}_last_run_date）
    asyncio.create_task(_run_daily(
        app, name="chronicle", label="chronicle", fn=_chronicle_batch,
        default_time="03:00", time_setting_key="chronicle_time",
    ))
    asyncio.create_task(_run_daily(
        app, name="forget", label="forget", fn=_forget_batch,
        default_time="04:00",
    ))
    asyncio.create_task(_run_daily(
        app, name="instruments", label="計器巡回", fn=_instruments_patrol,
        default_time="05:00", time_setting_key="instruments_patrol_time",
    ))
    # 毎分スケジューラ（冪等キーの管理は各 tick の責務）
    asyncio.create_task(_run_every_minute(app, name="usual_days", label="うつつ", fn=_usual_days_tick))
    asyncio.create_task(_run_every_minute(app, name="action", label="行動権", fn=_action_tick))
    asyncio.create_task(_run_every_minute(app, name="escrow_delivery", label="能動配達", fn=_escrow_delivery_tick))
    asyncio.create_task(_run_every_minute(app, name="weekly_schedule", label="週次スケジュール", fn=_weekly_schedule_tick))
    asyncio.create_task(_run_every_minute(app, name="sudden_event", label="突発イベント", fn=_sudden_event_tick))

    yield

    # Shutdown — InscribedMemoryManager は LanceStore 移行後はバックグラウンドリトライスレッドを
    # 持たないため stop() 不要。
    _log.info("Chotgor backend 終了")


def _beat_scheduler(sqlite, name: str) -> None:
    """スケジューラの生存痕を settings に上書き記録する（ループ1周ごと）。

    決定ログ（scheduler_decisions）は「意味のある評価」だけを追記するため、
    発火機会が乏しい機構では最終行が古くなり「死んでいる」と区別できない。
    生存確認はこの settings キー（scheduler_heartbeat_*）の鮮度で行う
    （上書きなので行は増えない）。予報パネルの診断ヘッダと Tier 1
    `scheduler_heartbeat` インバリアントが読む。

    Args:
        sqlite: SQLiteStore。
        name: 機構名（action / usual_days / sudden_event / escrow_delivery /
            weekly_schedule / chronicle / forget / instruments）。
    """
    try:
        sqlite.set_setting(f"scheduler_heartbeat_{name}", datetime.now().isoformat())
    except Exception:
        logging.getLogger(__name__).exception("heartbeat 記録に失敗 name=%s", name)


async def _run_daily(
    app: FastAPI,
    *,
    name: str,
    label: str,
    fn,
    default_time: str,
    time_setting_key: str | None = None,
) -> None:
    """毎日 HH:MM に1回 fn(app) を実行する日次スケジューラの共通ループ。

    - 60秒周期で判定し、heartbeat（scheduler_heartbeat_{name}）を毎周上書きする。
    - 実行時刻は time_setting_key の設定値（未指定・パース不能時は default_time）。
    - 冪等キーは {name}_last_run_date。実行を試みた時点で当日日付を立てる
      （失敗しても当日は再実行しない）。

    Args:
        name: 機構名（heartbeat・冪等キーの接頭辞になる英名）。
        label: ログ表示名。
        fn: 実行内容 ``async fn(app)``。
        default_time: "HH:MM"。設定が無い・壊れているときの実行時刻。
        time_setting_key: 実行時刻を保持する settings キー（None なら固定時刻）。
    """
    _log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(60)
        _beat_scheduler(app.state.sqlite, name)
        now = datetime.now()
        time_str = (
            app.state.sqlite.get_setting(time_setting_key, default_time)
            if time_setting_key
            else default_time
        )
        try:
            h, m = map(int, str(time_str).split(":"))
        except Exception:
            h, m = map(int, default_time.split(":"))
        scheduled = now.replace(hour=h, minute=m, second=0, microsecond=0)
        today_str = now.date().isoformat()
        last_run = app.state.sqlite.get_setting(f"{name}_last_run_date", "")
        if now >= scheduled and last_run != today_str:
            _log.info("%s スケジューラー 起動 設定時刻=%02d:%02d", label, h, m)
            app.state.sqlite.set_setting(f"{name}_last_run_date", today_str)
            try:
                await fn(app)
            except Exception:
                _log.exception("%s スケジューラー 実行エラー", label)


async def _run_every_minute(app: FastAPI, *, name: str, label: str, fn) -> None:
    """60秒周期で fn(app) を実行する常時巡回スケジューラの共通ループ。

    heartbeat（scheduler_heartbeat_{name}）を毎周上書きし、fn の例外は
    ログへ残してループを継続する。冪等キーの管理は fn 側の責務。
    """
    _log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(60)
        _beat_scheduler(app.state.sqlite, name)
        try:
            await fn(app)
        except Exception:
            _log.exception("%s スケジューラー 実行エラー", label)


async def _chronicle_batch(app: FastAPI) -> None:
    """chronicle 日次バッチの実行内容（_run_daily から呼ばれる）。"""
    await run_pending_chronicles(
        app.state.sqlite,
        vector_store=app.state.vector_store,
        memory_manager=app.state.memory_manager,
        working_memory_manager=app.state.working_memory_manager,
    )


async def _forget_batch(app: FastAPI) -> None:
    """forget 日次バッチの実行内容（_run_daily から呼ばれる。04:00 固定＝chronicle 後）。"""
    await run_pending_forget(app.state.sqlite, app.state.memory_manager)


async def _instruments_patrol(app: FastAPI) -> None:
    """計器の日次巡回チェック（_run_daily から呼ばれる）。

    巡回時刻は Chronicle（03:00）→ Forget（04:00）の後に置く
    （night_batch_heartbeat が当日の夜間バッチ完了を前提とするため）。
    実行内容:
        - Tier 1 巡回インバリアント（run_patrol_checks）
        - Tier 2 肥大メーターの日次スナップショット（record_bloat_meters）
        - Tier 3 判定巡回（run_judgement_patrol。判定プリセット未設定ならスキップ）
    """
    from backend.services.instruments import (
        record_bloat_meters,
        run_judgement_patrol,
        run_patrol_checks,
    )
    from backend.services.pressure import record_pressure_meters

    _log = logging.getLogger(__name__)
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


# 行動権評価の格子幅（分）。2時間格子＋キャラ別ジッターで評価する。
_ACTION_PERIOD_MINUTES = 120


async def _action_tick(app: FastAPI) -> None:
    """会話外行動権の周期評価（めぐり Phase 6。_run_every_minute から呼ばれる）。

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
    sqlite = app.state.sqlite
    now = datetime.now()
    # 現在のスロット開始時刻（_ACTION_PERIOD_MINUTES 格子に床合わせ）
    minutes_of_day = now.hour * 60 + now.minute
    slot_index = minutes_of_day // _ACTION_PERIOD_MINUTES
    slot_start = now.replace(
        hour=(slot_index * _ACTION_PERIOD_MINUTES) // 60,
        minute=(slot_index * _ACTION_PERIOD_MINUTES) % 60,
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
            # 「流れたスロット」も決定ログへ — 飽和（評価機会の枯渇）を予報パネルで可視化する
            sqlite.record_scheduler_decision(
                "action", "skipped", character_id=char.id,
                reason=f"unavailable: {availability.reason}",
                details={"slot": slot_start.isoformat()},
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


# 生活カレンダー有効キャラで usual_scenes_per_day 未設定時の既定シーン回数（Phase 4・§8）
_DEFAULT_SCENES_PER_DAY = 3


def _slot_scene_descriptors(now: datetime, owner_id, cfg: dict) -> list[dict]:
    """従来 slots からの到来済みシーン記述子を返す（生活カレンダー無効キャラ）。

    各記述子は {key, slot, topic_ooc}。topic_ooc は空（従来は題材を指定しない）。
    まだ到来していないスロット・不正スロットは含めない。

    Args:
        now: 基準時刻。
        owner_id: うつつ世界の所有者キャラ ID。
        cfg: usual_config。

    Returns:
        到来済みスロットの記述子リスト。
    """
    descriptors: list[dict] = []
    for slot in cfg.get("slots") or []:
        scheduled = _parse_slot_time(now, slot)
        if scheduled is None or now < scheduled:
            continue  # まだ到来していない / 不正スロット
        descriptors.append({
            "key": f"usual_days_last_run_{owner_id}_{slot}",
            "slot": slot,
            "topic_ooc": "",
        })
    return descriptors


def _schedule_scene_descriptors(
    sqlite, owner_char, cfg: dict, now: datetime
) -> list[dict]:
    """②導出（実現層エントリ）からの到来済みシーン記述子を返す（生活カレンダー有効キャラ）。

    その日の planned エントリから select_daily_scenes で決定論選出し、起動時刻が到来した
    ものだけを記述子化する。冪等キーはエントリ単位（entry_id）。題材（ラベル）を GM への
    framing OOC として topic_ooc に載せる（§8）。

    Args:
        sqlite: SQLiteStore。
        owner_char: うつつ世界の所有者 Character（living_schedule_enabled=1）。
        cfg: usual_config（scenes_per_day を読む）。
        now: 基準時刻。

    Returns:
        起動時刻到来済みシーンの記述子リスト。
    """
    from datetime import timedelta

    from backend.services.schedule import format_scene_framing, select_daily_scenes

    owner_id = owner_char.id
    scenes_per_day = int(cfg.get("scenes_per_day") or 0) or _DEFAULT_SCENES_PER_DAY
    day = now.date()
    day_start = datetime(day.year, day.month, day.day)
    day_end = day_start + timedelta(days=1)
    entries = sqlite.list_schedule_entries(
        owner_id, since=day_start, until=day_end, statuses=["planned"],
    )
    scenes = select_daily_scenes(
        entries, character_id=owner_id, day=day, scenes_per_day=scenes_per_day,
    )
    descriptors: list[dict] = []
    for scene in scenes:
        if now < scene.fire_at:
            continue  # 起動時刻がまだ来ていない
        descriptors.append({
            "key": f"usual_days_entry_run_{scene.entry_id}",
            "slot": f"{scene.fire_at:%H:%M}",
            "topic_ooc": format_scene_framing(
                getattr(owner_char, "name", "") or "", scene.label
            ),
        })
    return descriptors


async def _run_due_usual_scenes(app: FastAPI) -> None:
    """到来済みのうつつシーンを 1 回ずつ無人進行させる（冪等キー＝シーン単位）。

    2系統のシーン起動タイミングを持つ（Phase 4・§8）:
        - 生活カレンダー無効: 従来どおり ``usual_config.slots`` の時刻で起動。
        - 生活カレンダー有効: ②はる固定（＋①世界固定）エントリから決定論選出して起動。
          手動 slots に依らず「はるが自分で決めた予定」がそのままシーンになる。

    共通:
        - 有効（usual_config.enabled）なうつつ世界ごとに、今日すでに到来したシーンを処理。
        - 冪等キーに当日日付を立てて二重起動を防ぐ（対面スキップ時も立てる＝通過分は捨てる）。
        - 日次コストガード ``usual_days_daily_cap``（既定 24）を超えたら以降は当日スキップ。
        - セッションは永続1本（ensure_usual_session が find-or-create）。
        - GM へ前回シーンからの経過時間メモ＋（②導出なら）題材の framing を添える。
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
        # reach_out によるポーズ待機中は新規シーンを起動しない（再開は
        # _run_pending_push_resumes の領分。二重進行で物語が枝分かれする事故を防ぐ）。
        if owner_id:
            from backend.character_actions.messenger import read_push_pause
            if read_push_pause(sqlite, owner_id) is not None:
                _log.info("うつつ: push待機中のため新規シーンを起動しない owner=%s", owner_id)
                continue
        # 生活カレンダー有効キャラは②導出、無効キャラは従来 slots でシーンを起動する。
        living_enabled = bool(
            owner_char and int(getattr(owner_char, "living_schedule_enabled", 0) or 0)
        )
        if living_enabled:
            descriptors = _schedule_scene_descriptors(sqlite, owner_char, cfg, now)
        else:
            descriptors = _slot_scene_descriptors(now, owner_id, cfg)
        for desc in descriptors:
            key, slot, topic_ooc = desc["key"], desc["slot"], desc["topic_ooc"]
            if sqlite.get_setting(key, "") == today_str:
                continue  # 当日このシーンは実行済み
            if owner_in_face_to_face:
                # 通過分は捨てる：冪等キーを立てて当日この時刻は再評価しない（§7 聖域化）。
                sqlite.set_setting(key, today_str)
                _log.info(
                    "うつつ: 対面モード中につきスキップ owner=%s slot=%s",
                    owner_id, slot,
                )
                sqlite.record_scheduler_decision(
                    "usual_days", "skipped", character_id=owner_id,
                    reason="対面モード中（聖域化・通過分は捨てる）",
                    details={"slot": slot},
                )
                continue
            if ran_today >= daily_cap:
                _log.warning(
                    "うつつ: 日次上限 %d 到達。当日は以降スキップ owner=%s slot=%s",
                    daily_cap, owner_id, slot,
                )
                sqlite.record_scheduler_decision(
                    "usual_days", "skipped", character_id=owner_id,
                    reason=f"日次上限（{ran_today}/{daily_cap}）",
                    details={"slot": slot},
                )
                return
            session = ensure_usual_session(sqlite, scenario)
            # 冪等キーは「実行を試みた」時点で立てる（設定不備でも毎分叩かない）。
            sqlite.set_setting(key, today_str)
            if session is None:
                _log.error("うつつ: セッション未解決のためスキップ owner=%s", owner_id)
                sqlite.record_scheduler_decision(
                    "usual_days", "error", character_id=owner_id,
                    reason="セッション未解決", details={"slot": slot},
                )
                continue
            elapsed_note = usual_elapsed_note(sqlite, session.id, now)
            # 経過時間メモ＋題材 framing（②導出時のみ非空）を結合して GM へ渡す。
            extra_ooc = "\n\n".join(p for p in (elapsed_note, topic_ooc) if p)
            _log.info(
                "うつつ: シーン起動 owner=%s slot=%s session=%s living=%s",
                owner_id, slot, session.id, living_enabled,
            )
            try:
                result = await run_usual_days_scene(
                    session_id=session.id,
                    sqlite=sqlite,
                    settings=sqlite.get_all_settings(),
                    chat_service=app.state.chat_service,
                    extra_first_gm_ooc=extra_ooc,
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
                    sqlite.record_scheduler_decision(
                        "usual_days", "error", character_id=owner_id,
                        reason=f"シーン中断: {result['error']}",
                        details={"slot": slot,
                                 "turns": result.get("fired_turns", 0)},
                    )
                else:
                    _log.info(
                        "うつつ: シーン完了 owner=%s slot=%s responses=%d turns=%d scene_closed=%s",
                        owner_id, slot,
                        result.get("fired_responses", 0), result.get("fired_turns", 0),
                        result.get("scene_closed"),
                    )
                    sqlite.record_scheduler_decision(
                        "usual_days", "fired", character_id=owner_id,
                        reason="シーン完了",
                        details={"slot": slot,
                                 "turns": result.get("fired_turns", 0),
                                 "scene_closed": result.get("scene_closed")},
                    )
            except Exception:
                _log.exception("うつつ: シーン実行エラー owner=%s slot=%s", owner_id, slot)
                sqlite.record_scheduler_decision(
                    "usual_days", "error", character_id=owner_id,
                    reason="シーン実行例外", details={"slot": slot},
                )


async def _run_pending_push_resumes(app: FastAPI) -> None:
    """reach_out で一時停止したうつつシーンを、待ち時間経過後に GM 継続で再開する。

    本人がうつつの中から現実へメッセージを送る（reach_out）と、シーンは本人の発言
    終了で一時停止し、ポーズ要求キー（usual_push_pause_{character_id}）が立つ。
    ここでは resume_at の到来を待って GM へターンを渡す — 「連絡してから15分経った」
    という現実の時間経過を GM に伝え、その間にユーザからの返事が**あったかどうかは
    既存の履歴合流（external シーン統合・不在ユーザブロック）から GM が読み取る**。

    - 対面モード ON（reach_out visit=true / ユーザ操作）の間は再開しない — 対面は
      聖域（うつつを走らせない既存方針）。キーは消して自然消滅させる。
    - 日次上限（usual_days_daily_cap）到達時も再開を見送る（キーは消す）。物語は
      次のスロットのシーンが同一セッションで自然に続きを拾う。
    """
    from backend.character_actions.messenger import clear_push_pause, read_push_pause
    from backend.services.scenario_chat.service import (
        ensure_usual_session,
        run_usual_days_scene,
    )

    _log = logging.getLogger(__name__)
    sqlite = app.state.sqlite
    now = datetime.now()
    today_str = now.date().isoformat()

    for scenario in sqlite.list_usual_scenarios():
        cfg = getattr(scenario, "usual_config", None) or {}
        if not cfg.get("enabled"):
            continue
        owner_id = getattr(scenario, "owner_character_id", None)
        if not owner_id:
            continue
        pause = read_push_pause(sqlite, owner_id)
        if pause is None:
            continue
        try:
            resume_at = datetime.fromisoformat(str(pause.get("resume_at", "")))
            sent_at = datetime.fromisoformat(str(pause.get("sent_at", "")))
        except (TypeError, ValueError):
            # 壊れたポーズ要求は捨てる（永久ポーズを防ぐ）
            _log.warning("うつつ: ポーズ要求が不正のため破棄 owner=%s raw=%r", owner_id, pause)
            clear_push_pause(sqlite, owner_id)
            continue
        if now < resume_at:
            continue  # まだ待ち時間内（本人の連絡から15分待つ）

        # 再開確定 or 見送り確定 — どちらでもキーは消す（毎分の再評価を止める）
        clear_push_pause(sqlite, owner_id)

        owner_char = sqlite.get_character(owner_id)
        if owner_char is not None and int(getattr(owner_char, "face_to_face_mode", 0) or 0):
            # 会いに行った（visit）/ユーザが対面を開いた — 対面は聖域、シーンは再開しない
            sqlite.record_scheduler_decision(
                "usual_days", "skipped", character_id=owner_id,
                reason="push再開: 対面モード中（聖域化）",
            )
            continue

        daily_cap = int(sqlite.get_setting("usual_days_daily_cap", "24") or 24)
        count_key = f"usual_days_scene_count_{today_str}"
        ran_today = int(sqlite.get_setting(count_key, "0") or 0)
        if ran_today >= daily_cap:
            sqlite.record_scheduler_decision(
                "usual_days", "skipped", character_id=owner_id,
                reason=f"push再開: 日次上限（{ran_today}/{daily_cap}）",
            )
            continue

        session = ensure_usual_session(sqlite, scenario)
        if session is None:
            sqlite.record_scheduler_decision(
                "usual_days", "error", character_id=owner_id,
                reason="push再開: セッション未解決",
            )
            continue

        minutes = max(1, int((now - sent_at).total_seconds() // 60))
        char_name = getattr(owner_char, "name", None) or "本人"
        resume_ooc = (
            f"（{char_name}が約{minutes}分前にユーザへメッセージを送った。"
            "その後この時間が経過している。ユーザからの返事があったかどうかは"
            "これまでの履歴に現れている（履歴に無ければ返事はまだ来ていない）。"
            "その事実を踏まえて場面の続きを描写すること。"
            "ユーザの言動を捏造してはならない）"
        )
        sqlite.set_setting(count_key, str(ran_today + 1))
        _log.info(
            "うつつ: push再開 owner=%s session=%s 経過=%d分", owner_id, session.id, minutes,
        )
        try:
            result = await run_usual_days_scene(
                session_id=session.id,
                sqlite=sqlite,
                settings=sqlite.get_all_settings(),
                chat_service=app.state.chat_service,
                extra_first_gm_ooc=resume_ooc,
                slot="push_resume",
            )
            sqlite.record_scheduler_decision(
                "usual_days",
                "error" if result.get("error") else "fired",
                character_id=owner_id,
                reason=(
                    f"push再開: シーン中断: {result['error']}" if result.get("error")
                    else f"push再開（連絡から{minutes}分後にGM継続）"
                ),
                details={"turns": result.get("fired_turns", 0),
                         "scene_closed": result.get("scene_closed")},
            )
        except Exception:
            _log.exception("うつつ: push再開の実行エラー owner=%s", owner_id)
            sqlite.record_scheduler_decision(
                "usual_days", "error", character_id=owner_id,
                reason="push再開: 実行例外",
            )


async def _usual_days_tick(app: FastAPI) -> None:
    """うつつの毎分処理（_run_every_minute から呼ばれる）: push再開 → 到来済みシーンの無人進行。

    冪等キーは日付+スロットで1日複数スロットへ対応する。SSE 配信は不要なので
    run_usual_days_scene を await 回収する。push再開の例外はここで握って
    シーン起動を続行する（片方の事故でもう片方を止めない）。
    """
    try:
        await _run_pending_push_resumes(app)
    except Exception:
        logging.getLogger(__name__).exception("うつつ push再開スケジューラー 実行エラー")
    await _run_due_usual_scenes(app)


async def _escrow_delivery_tick(app: FastAPI) -> None:
    """預かり（escrow）メッセージの能動配達（めぐり §5.1 フォローアップ）。

    未配達メッセージ（delivered_at=NULL）を持つセッションを走査し、
    キャラの availability が戻っていれば決定論ジッター（0〜10分）を挟んで
    本人へ配達し、返信を生成・保存する。判定・ガードの詳細は
    services/gate/delivery.py 参照。
    """
    from backend.services.gate.delivery import run_pending_escrow_deliveries

    await run_pending_escrow_deliveries(app.state)


async def _weekly_schedule_tick(app: FastAPI) -> None:
    """生活カレンダーの週次バッチ①②（schedule_plan.md §3 / Phase 3）。

    生活カレンダー有効キャラを走査し、未生成の週があればバッチを実行する。
    日曜夜（weekly_schedule_time・既定 20:00）に翌週分、コールドスタート
    （機能有効化・取りこぼし復旧）では当週分を即時生成する。
    冪等キー = キャラごとの対象 ISO 週（判定の詳細は services/schedule/weekly_batch.py）。
    """
    from backend.services.schedule import run_pending_weekly_batches

    await run_pending_weekly_batches(app.state)


async def _sudden_event_tick(app: FastAPI) -> None:
    """生活カレンダーの③世界突発イベントの発火（schedule_plan.md §5 / Phase 5）。

    生活カレンダー有効キャラの未発火伏せ枠（週次バッチで確率配置済み）を走査し、
    発火時刻が到来したものを GM 具体化 → 轢き判定 → insert → シーン → 玉突き裁定の順で
    処理する。判定・ガードの詳細は services/schedule/events.py 参照。
    """
    from backend.services.schedule import run_pending_sudden_events

    await run_pending_sudden_events(app.state)


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
