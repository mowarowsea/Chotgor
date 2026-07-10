"""予報パネルの集約層 — 無人機構の「確定予定・無風外挿・診断・実績」を一括計算する。

docs/planned/forecast_panel_plan.md の中核。すべて既存の決定論純関数
（compute_pressures / intent_pressure / jittered_slot_time / select_daily_scenes /
resolve_delivery_due / check_availability）の合成であり、LLM を一切呼ばない。

予報の原理:
    圧力は封筒（timeline_events）の導関数であり、未来の封筒は存在しない。
    したがって未来時刻での評価は自動的に「今後新しいイベントが何も起きなければ」
    という無風仮定の外挿になる。1on1 会話が起きればカーブはズレるが、
    それは予報の失敗ではなく前提の更新（天気予報と同じ）。

予報の限界（設計上の証明）:
    予報できるのは物理（圧力・スロット・ゲート）まで。「本人がやるかどうか」は
    意志であり、乱数も関数も置かない設計のため原理的に予報できない。
    パネルの予測点は「問い合わせ予報」であって「行動予報」ではない。
"""

import logging
from datetime import date, datetime, timedelta

from backend.services.actions.runner import (
    _URGE_THRESHOLD,
    action_urge_snapshot,
    jittered_slot_time,
)
from backend.services.gate import check_availability, is_usual_scene_running
from backend.services.gate.delivery import (
    _delivery_jitter_seconds,
    resolve_delivery_due,
)
from backend.services.intents.lifecycle import intent_pressure
from backend.services.pressure.engine import (
    _EVENT_WINDOW_DAYS,
    _make_relation_weight_fn,
    compute_boredom,
    compute_body,
    compute_social,
    merge_profile,
    rhythm_component,
)

logger = logging.getLogger(__name__)

# 圧力予報の格子間隔（分）。72h × 30分 = 145点。純関数なのでゼロ円。
_PRESSURE_GRID_MINUTES = 30
# 行動権の評価周期（分）— actions/runner.py の _EVAL_PERIOD_MINUTES と対（値の正は runner）
_ACTION_PERIOD_MINUTES = 120
# 配達シミュレータの走査刻み（分）と上限時間
_DELIVERY_SIM_STEP_MINUTES = 5
_DELIVERY_SIM_MAX_HOURS = 48
# 意図圧カーブを描く意図の上限（系列色4枠と対 — 5本目からはグラフが読めない）
_MAX_INTENT_CURVES = 4
# 揺れ監査の実績参照期間（日）
_VARIANCE_WINDOW_DAYS = 14

# 診断ヘッダに出す heartbeat の機構名（main.py の _beat_scheduler と対）
_HEARTBEAT_SCHEDULERS = [
    "action", "usual_days", "sudden_event", "escrow_delivery",
    "weekly_schedule", "chronicle", "forget", "instruments",
]

# 日次コストガードのカウンタ定義: (表示名, カウンタ settings キーの接頭辞, cap キー, cap 既定)
_CAP_DEFS = [
    ("行動問い合わせ", "action_inquiry_count_", "action_inquiry_daily_cap", 6),
    ("行動実行", "action_exec_count_", "action_exec_daily_cap", 3),
    ("うつつシーン", "usual_days_scene_count_", "usual_days_daily_cap", 24),
    ("③突発発火", "sudden_event_fire_count_", "sudden_event_daily_cap", 3),
    ("能動配達", "escrow_delivery_count_", "escrow_delivery_daily_cap", 12),
]


def _iso(dt: datetime | None) -> str | None:
    """datetime を ISO 文字列へ（None 透過）。JSON 埋め込み用。"""
    return dt.isoformat() if dt is not None else None


def _availability_dict(avail) -> dict:
    """Availability dataclass を JSON 化可能な辞書へ変換する。"""
    return {
        "available": bool(avail.available),
        "reason": avail.reason or "",
        "state": avail.state,
        "occupancy": round(float(avail.occupancy), 2),
        "reply_rate": round(float(avail.reply_rate), 2),
        "check_interval": avail.check_interval,
    }


def _collect_heartbeats(sqlite, now: datetime) -> list[dict]:
    """各スケジューラの heartbeat（生存痕）を鮮度付きで集める。

    heartbeat は settings キー scheduler_heartbeat_*（main.py の _beat_scheduler が
    ループ1周ごとに上書き）。鮮度が数分を超えていたらそのループは止まっている。

    Args:
        sqlite: SQLiteStore。
        now: 基準時刻。

    Returns:
        [{"scheduler", "at", "age_minutes"}] のリスト（未記録は at=None）。
    """
    beats = []
    for name in _HEARTBEAT_SCHEDULERS:
        raw = sqlite.get_setting(f"scheduler_heartbeat_{name}", "")
        at = None
        age = None
        if raw and isinstance(raw, str):
            try:
                at = datetime.fromisoformat(raw)
                age = max(0.0, (now - at).total_seconds() / 60.0)
            except ValueError:
                at = None
        beats.append({
            "scheduler": name,
            "at": _iso(at),
            "age_minutes": round(age, 1) if age is not None else None,
        })
    return beats


def _collect_caps(sqlite, now: datetime) -> list[dict]:
    """日次コストガードの当日消費量を集める（診断ヘッダ用）。"""
    today = now.date().isoformat()
    rows = []
    for label, count_prefix, cap_key, default_cap in _CAP_DEFS:
        try:
            cap = int(sqlite.get_setting(cap_key, ""))
        except (TypeError, ValueError):
            cap = default_cap
        used = int(sqlite.get_setting(f"{count_prefix}{today}", "0") or 0)
        rows.append({"label": label, "used": used, "cap": cap})
    return rows


def _find_sim_session_id(sqlite, char) -> str:
    """配達シミュレータの決定論シードに使うセッション ID を返す。

    実際の配達判定は session_id をシードに含むため、そのキャラの最新 1on1
    セッションを使うと「本当にそのセッションに今送った場合」の予測になる。
    セッションが無ければ予報専用の擬似シード（実在しないので参考値）。
    """
    name = getattr(char, "name", "") or ""
    for session in sqlite.list_chat_sessions(limit=200):
        model_id = getattr(session, "model_id", "") or ""
        if model_id.rsplit("@", 1)[0] == name:
            return str(session.id)
    return f"forecast-sim:{char.id}"


def _simulate_delivery(sqlite, char, now: datetime) -> dict | None:
    """「いまメッセージを送ったら、いつ配達されるか」を決定論シミュレートする。

    実運用の _maybe_deliver_session と同じ判定（生活カレンダー経路＝チェック間隔
    格子×決定論 reply_rate、従来経路＝availability 復帰＋決定論ジッター）を、
    未来時刻を注入して最初に配達 due になる時刻を探す。無風仮定
    （away/対面などの現在状態が続く前提）で、走査は5分刻み・最長48時間。

    Args:
        sqlite: SQLiteStore。
        char: 対象 Character ORM。
        now: 「送った」時刻。

    Returns:
        {"session_id", "delivered_at", "wait_minutes", "mode"}。
        48時間以内に配達点が見つからなければ delivered_at=None。
    """
    session_id = _find_sim_session_id(sqlite, char)
    living = bool(int(getattr(char, "living_schedule_enabled", 0) or 0))
    step = timedelta(minutes=_DELIVERY_SIM_STEP_MINUTES)
    limit = now + timedelta(hours=_DELIVERY_SIM_MAX_HOURS)

    t = now
    while t <= limit:
        avail = check_availability(
            char, t, usual_scene_running=False, sqlite=sqlite,
        )
        if living:
            # 生活カレンダー経路: 毎分の再評価と同じ式に未来時刻を入れるだけ
            if resolve_delivery_due(
                session_id, now, avail.reply_rate, avail.check_interval, t,
            ):
                delivered = t
                break
        else:
            # 従来経路: 復帰を初観測した時点＋決定論ジッターで配達される
            if avail.available:
                delivered = t + timedelta(
                    seconds=_delivery_jitter_seconds(session_id, t)
                )
                break
        t += step
    else:
        return {
            "session_id": session_id, "delivered_at": None,
            "wait_minutes": None, "mode": "living" if living else "legacy",
        }
    return {
        "session_id": session_id,
        "delivered_at": _iso(delivered),
        "wait_minutes": round((delivered - now).total_seconds() / 60.0),
        "mode": "living" if living else "legacy",
    }


def _usual_config_for(sqlite, character_id: str) -> dict | None:
    """キャラのうつつ世界設定（enabled な usual_config）を返す。無ければ None。"""
    for scenario in sqlite.list_usual_scenarios():
        if getattr(scenario, "owner_character_id", None) != character_id:
            continue
        cfg = getattr(scenario, "usual_config", None) or {}
        if cfg.get("enabled"):
            return cfg
    return None


def _calendar_scenes(
    sqlite, char, cfg: dict | None, entries: list, days: list[date],
) -> list[dict]:
    """カレンダー期間内のうつつシーン起動予定を決定論計算する。

    生活カレンダー有効キャラは②導出（select_daily_scenes — main.py のスケジューラと
    同じ関数・同じシードなので、ここで出す時刻がそのまま実際の起動時刻になる）。
    無効キャラは従来 slots（HH:MM が毎日繰り返し）。

    Args:
        sqlite: SQLiteStore。
        char: 対象 Character ORM。
        cfg: usual_config（None ならうつつ無効 → 空リスト）。
        entries: カレンダー期間の planned な ScheduleEntry リスト。
        days: 対象日のリスト。

    Returns:
        [{"label", "fire_at", "entry_id"}] の起動時刻昇順リスト。
    """
    if cfg is None:
        return []
    from backend.services.schedule import select_daily_scenes

    scenes: list[dict] = []
    living = bool(int(getattr(char, "living_schedule_enabled", 0) or 0))
    if living:
        scenes_per_day = int(cfg.get("scenes_per_day") or 0) or 3
        for day in days:
            for slot in select_daily_scenes(
                entries, character_id=char.id, day=day,
                scenes_per_day=scenes_per_day,
            ):
                scenes.append({
                    "label": slot.label,
                    "fire_at": _iso(slot.fire_at),
                    "entry_id": slot.entry_id,
                })
    else:
        for day in days:
            for slot in cfg.get("slots") or []:
                try:
                    h, m = map(int, str(slot).strip().split(":"))
                except (ValueError, AttributeError):
                    continue
                if not (0 <= h < 24 and 0 <= m < 60):
                    continue
                scenes.append({
                    "label": f"うつつスロット {slot}",
                    "fire_at": _iso(datetime(day.year, day.month, day.day, h, m)),
                    "entry_id": None,
                })
    scenes.sort(key=lambda s: s["fire_at"] or "")
    return scenes


def _action_slots(
    sqlite, char, days: list[date], now: datetime, intents: list, events: list,
    profile: dict, weight_fn,
) -> list[dict]:
    """カレンダー期間内の行動権評価スロットと発火予報を計算する。

    評価時刻は jittered_slot_time（スケジューラと同じシード）なので分単位で確定。
    未来スロットには無風仮定の予報を付ける:
        - unavailable: そのスロットは流れる（飽和の可視化）
        - fires: 閾値超えの意図があり問い合わせが起きる見込み
        - quiet: 評価はされるが閾値未達（正常な沈黙）

    Args:
        sqlite: SQLiteStore。
        char: 対象 Character ORM。
        days: 対象日のリスト。
        now: 基準時刻。
        intents: active な Intent ORM のリスト。
        events: 圧力計算用の封筒リスト（[now-窓, now]）。
        profile: merge_profile 済みの体質。
        weight_fn: 社会圧の相手別重み関数。

    Returns:
        [{"slot_start", "eval_at", "forecast"}] の昇順リスト（過去分は forecast="past"）。
    """
    slots: list[dict] = []
    for day in days:
        day_start = datetime(day.year, day.month, day.day)
        for i in range(0, 24 * 60, _ACTION_PERIOD_MINUTES):
            slot_start = day_start + timedelta(minutes=i)
            eval_at = jittered_slot_time(char.id, slot_start)
            if eval_at <= now:
                forecast = "past"
            else:
                avail = check_availability(
                    char, eval_at, usual_scene_running=False, sqlite=sqlite,
                )
                if not avail.available:
                    forecast = "unavailable"
                else:
                    pressures = _pressures_at(
                        events, eval_at, profile, char.id, weight_fn
                    )
                    hot = any(
                        intent_pressure(it, pressures, now=eval_at) >= _URGE_THRESHOLD
                        for it in intents
                    )
                    forecast = "fires" if hot else "quiet"
            slots.append({
                "slot_start": _iso(slot_start),
                "eval_at": _iso(eval_at),
                "forecast": forecast,
            })
    return slots


def _pressures_at(
    events: list, t: datetime, profile: dict, character_id: str, weight_fn,
) -> dict:
    """封筒リストから時刻 t の圧力3変数を計算する（無風外挿の1点）。

    compute_pressures と同じ合成だが、封筒を1度だけ取得して使い回すために
    構成関数を直接呼ぶ（格子145点で毎回 SQL を引かない）。
    窓（30日）は t 基準で切り直す。

    Args:
        events: [now-窓, now] の封筒リスト（occurred_at 昇順・降順は不問）。
        t: 評価時刻（now 以降なら無風外挿になる）。
        profile: merge_profile 済みの体質。
        character_id: 対象キャラ（体調圧リズム成分のシード）。
        weight_fn: 社会圧の相手別重み関数。

    Returns:
        {"social", "boredom", "body"}（各 0.0〜1.0）。
    """
    window_start = t - timedelta(days=_EVENT_WINDOW_DAYS)
    windowed = [e for e in events if e.occurred_at >= window_start]
    return {
        "social": compute_social(windowed, t, profile, weight_fn),
        "boredom": compute_boredom(windowed, t, profile),
        "body": compute_body(windowed, t, profile, character_id),
    }


def _pressure_forecast(
    sqlite, char, intents: list, events: list, profile: dict, weight_fn,
    now: datetime, horizon_hours: int,
) -> dict:
    """圧力3変数＋意図圧の無風外挿カーブと「問い合わせ予報点」を計算する。

    Args:
        sqlite: SQLiteStore。
        char: 対象 Character ORM。
        intents: active な Intent ORM のリスト。
        events: 圧力計算用の封筒リスト。
        profile: merge_profile 済みの体質。
        weight_fn: 社会圧の相手別重み関数。
        now: 予報の起点。
        horizon_hours: 外挿時間。

    Returns:
        {"grid": [iso...], "social": [...], "boredom": [...], "body": [...],
         "threshold": float,
         "intents": [{"intent_id", "description", "series": [...]}],
         "fire_points": [{"at", "descriptions": [...]}]}
    """
    grid: list[datetime] = []
    t = now
    end = now + timedelta(hours=horizon_hours)
    while t <= end:
        grid.append(t)
        t += timedelta(minutes=_PRESSURE_GRID_MINUTES)

    social, boredom, body = [], [], []
    # 意図圧カーブは現在圧の高い順に上位のみ（グラフの可読性）
    curve_intents = sorted(
        intents,
        key=lambda it: -intent_pressure(
            it, _pressures_at(events, now, profile, char.id, weight_fn), now=now
        ),
    )[:_MAX_INTENT_CURVES]
    intent_series: dict = {it.id: [] for it in curve_intents}

    for t in grid:
        p = _pressures_at(events, t, profile, char.id, weight_fn)
        social.append(round(p["social"], 3))
        boredom.append(round(p["boredom"], 3))
        body.append(round(p["body"], 3))
        for it in curve_intents:
            intent_series[it.id].append(
                round(intent_pressure(it, p, now=t), 3)
            )

    # 問い合わせ予報点: 未来の行動権評価時刻 × available × 閾値超え
    fire_points: list[dict] = []
    slot_cursor = now.replace(
        hour=(now.hour // (_ACTION_PERIOD_MINUTES // 60)) * (_ACTION_PERIOD_MINUTES // 60),
        minute=0, second=0, microsecond=0,
    )
    while slot_cursor <= end:
        eval_at = jittered_slot_time(char.id, slot_cursor)
        slot_cursor += timedelta(minutes=_ACTION_PERIOD_MINUTES)
        if eval_at <= now or eval_at > end:
            continue
        avail = check_availability(
            char, eval_at, usual_scene_running=False, sqlite=sqlite,
        )
        if not avail.available:
            continue
        p = _pressures_at(events, eval_at, profile, char.id, weight_fn)
        hot = [
            it.description for it in intents
            if intent_pressure(it, p, now=eval_at) >= _URGE_THRESHOLD
        ]
        if hot:
            fire_points.append({"at": _iso(eval_at), "descriptions": hot})

    return {
        "grid": [_iso(t) for t in grid],
        "social": social,
        "boredom": boredom,
        "body": body,
        "threshold": _URGE_THRESHOLD,
        "intents": [
            {
                "intent_id": str(it.id),
                "description": it.description,
                "series": intent_series[it.id],
            }
            for it in curve_intents
        ],
        "fire_points": fire_points,
    }


def _variance_audit(sqlite, character_id: str, now: datetime) -> dict:
    """揺れ監査 — 発火実績の散布と体調圧リズム成分の波形を返す。

    「揺れがあるはずなのに固定化していないか」を目視できる材料:
        - fired_scatter: 過去14日の発火実績（日×時刻）。ジッターが死んで
          毎日同時刻に発火していれば縦一直線に見える。
        - rhythm: 誰も設計していない決定論の波（過去7日〜先72h）。
          平坦なら周期導出が壊れている。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラ。
        now: 基準時刻。

    Returns:
        {"fired_scatter": [{"at", "scheduler"}], "rhythm": {"grid", "values"}}
    """
    since = now - timedelta(days=_VARIANCE_WINDOW_DAYS)
    fired = sqlite.list_scheduler_decisions(
        character_id=character_id, outcome="fired", since=since, limit=500,
    )
    scatter = [
        {"at": _iso(d.occurred_at), "scheduler": d.scheduler}
        for d in fired
    ]

    rhythm_grid: list[datetime] = []
    t = now - timedelta(days=7)
    end = now + timedelta(hours=72)
    while t <= end:
        rhythm_grid.append(t)
        t += timedelta(hours=3)
    return {
        "fired_scatter": scatter,
        "rhythm": {
            "grid": [_iso(t) for t in rhythm_grid],
            "values": [
                round(rhythm_component(character_id, t), 3) for t in rhythm_grid
            ],
        },
    }


def build_forecast(
    sqlite,
    character_id: str,
    *,
    now: datetime | None = None,
    horizon_hours: int = 72,
    calendar_days: int = 7,
) -> dict:
    """予報パネルの全セクションを一括計算する（LLM 不使用・副作用なし）。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター。
        now: 基準時刻（テスト注入用。省略時は現在時刻）。
        horizon_hours: 圧力予報の外挿時間（既定72h）。
        calendar_days: カレンダーの表示日数（今日から。既定7日）。

    Returns:
        {"character", "now", "diagnosis", "calendar", "pressure_forecast",
         "decisions", "variance"} の辞書（datetime はすべて ISO 文字列）。
        キャラクターが見つからなければ {"error": ...}。
    """
    now = now or datetime.now()
    char = sqlite.get_character(character_id)
    if char is None:
        return {"error": f"キャラクターが見つかりません: {character_id}"}

    # --- 共有材料（1回だけ取得して各セクションで使い回す）---
    profile = merge_profile(getattr(char, "pressure_profile", None))
    weight_fn = _make_relation_weight_fn(sqlite, character_id)
    events = sqlite.list_timeline_events(
        character_id,
        since=now - timedelta(days=_EVENT_WINDOW_DAYS),
        until=now,
    )
    intents = sqlite.list_intents(character_id, status="active")

    # --- 診断ヘッダ「いま」---
    availability = check_availability(
        char, now,
        usual_scene_running=is_usual_scene_running(sqlite, character_id, now),
        sqlite=sqlite,
    )
    diagnosis = {
        "availability": _availability_dict(availability),
        "urge": action_urge_snapshot(sqlite, character_id, now=now),
        "heartbeats": _collect_heartbeats(sqlite, now),
        "caps": _collect_caps(sqlite, now),
        "delivery_sim": _simulate_delivery(sqlite, char, now),
    }

    # --- 週間カレンダー（確定層）---
    days = [now.date() + timedelta(days=i) for i in range(calendar_days)]
    cal_start = datetime(days[0].year, days[0].month, days[0].day)
    cal_end = cal_start + timedelta(days=calendar_days)
    raw_entries = sqlite.list_schedule_entries(
        character_id, since=cal_start, until=cal_end,
        statuses=["planned", "pending", "done", "cancelled"],
    )
    entries = [
        {
            "id": str(e.id),
            "label": getattr(e, "label", None) or "",
            "start_at": _iso(e.start_at),
            "end_at": _iso(e.end_at),
            "state": getattr(e, "state", "") or "",
            "occupancy": round(float(getattr(e, "occupancy", 0.0) or 0.0), 2),
            "origin": getattr(e, "origin", "") or "",
            "source": getattr(e, "source", "") or "",
            "status": getattr(e, "status", "") or "",
            # ③伏せ枠（発火前の種）は全開示方針でそのまま見せる
            "is_seed": (
                (getattr(e, "payload", None) or {}).get("kind")
                == "sudden_event_seed"
            ),
        }
        for e in raw_entries
    ]
    cfg = _usual_config_for(sqlite, character_id)
    planned_orm = [e for e in raw_entries if getattr(e, "status", "") == "planned"]
    calendar = {
        "days": [d.isoformat() for d in days],
        "entries": entries,
        "scenes": _calendar_scenes(sqlite, char, cfg, planned_orm, days),
        "action_slots": _action_slots(
            sqlite, char, days, now, intents, events, profile, weight_fn,
        ),
        "usual_enabled": cfg is not None,
    }

    # --- 圧力予報（無風外挿）---
    pressure_forecast = _pressure_forecast(
        sqlite, char, intents, events, profile, weight_fn, now, horizon_hours,
    )

    # --- 決定ログ（実績と不発理由）---
    decisions = [
        {
            "occurred_at": _iso(d.occurred_at),
            "scheduler": d.scheduler,
            "outcome": d.outcome,
            "reason": d.reason or "",
            "details": d.details,
        }
        for d in sqlite.list_scheduler_decisions(
            character_id=character_id, limit=100,
        )
    ]

    # --- 揺れ監査 ---
    variance = _variance_audit(sqlite, character_id, now)

    return {
        "character": {
            "id": str(char.id),
            "name": char.name,
            "living_schedule_enabled": bool(
                int(getattr(char, "living_schedule_enabled", 0) or 0)
            ),
        },
        "now": _iso(now),
        "diagnosis": diagnosis,
        "calendar": calendar,
        "pressure_forecast": pressure_forecast,
        "decisions": decisions,
        "variance": variance,
    }
