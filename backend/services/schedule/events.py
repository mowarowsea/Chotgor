"""③世界突発イベント — 伏せ枠の確率配置と発火時 GM 具体化（Phase 5・§3/§4/§5）。

生成順序（②→③）により秘匿の仕掛けが不要になる（②生成時点で③はまだ存在しない・§3）。
③の流れは2段:

    1. **伏せ枠の確率配置**（週次バッチ内・LLM 不使用・決定論乱数）:
       偶発イベントカテゴリ（種＋発生確率）から、決定論乱数で時間枠だけを伏せて配置する。
       status="pending" で置くため availability には出ない（発火するまで生活を侵さない）。
    2. **発火時 GM 具体化**（`run_pending_sudden_events`・LLM あり）:
       伏せ枠の時刻が到来したら、GM が [EVENT] 行で拘束時間・状態・返信率・間隔を定義する。
       その占有圧で轢き判定（§4）— 既存の重なる planned の占有圧最大を上回れば insert して
       シーンを走らせる。同値以下なら「本人の予定が僅差で守られる」＝発火せず流れる。

轢いても既存エントリは物理的に触らない（分割・削除しない）。availability は読み取り時に
占有圧最大が勝つため、insert だけで侵食が表現される（§4）。轢かれた側の裁定は Phase 6。
"""

import logging
import random
from datetime import date, datetime, timedelta

from backend.lib.log_context import current_log_feature
from backend.providers.registry import create_provider
from backend.services.schedule.plan_parser import parse_event_line
from backend.services.schedule.weekly_batch import (
    _find_usual_scenario,
    resolve_gm_preset,
    week_key,
)

logger = logging.getLogger(__name__)

# 伏せ枠を配置する時間帯（日中の発火時刻の範囲・時）。深夜就寝帯を避け、
# 起きている時間に突発が飛び込む自然さを出す（就寝を轢く 激強 は GM が個別に定義しうる）。
_EVENT_HOUR_MIN = 8
_EVENT_HOUR_MAX = 21
# 発火の猶予（時）。プロセス停止等で時刻を過ぎた伏せ枠は、これを超えて古ければ
# 「通過分は捨てる」で発火せず done にする（生活は流れる・§7 と同じ思想）。
_EVENT_FIRE_GRACE_HOURS = 6
# ③発火の日次上限（GM 具体化＋玉突き裁定の LLM 呼び出しを抑える・§8）。
# 超過分は捨てる（通過分は捨てる思想）。0 は「③突発を止める」有効設定。
_DEFAULT_SUDDEN_EVENT_CAP = 3


def _living_enabled(char) -> bool:
    """生活カレンダーが有効なキャラかを返す。"""
    return bool(int(getattr(char, "living_schedule_enabled", 0) or 0))


def place_weekly_hidden_events(sqlite, char, week_start: date) -> int:
    """③の伏せ枠を対象週へ決定論配置する（週次バッチから呼ぶ・LLM 不使用）。

    偶発イベントカテゴリ（usual_config.event_categories）と発生率（event_probability）を
    使い、各曜日ごとに1回抽選して発生日を決め、日中のランダム時刻に伏せ枠を置く。
    冪等性のため、対象週の未発火（pending）adhoc 伏せ枠を先に消してから配置する
    （発火済み adhoc = 実イベントは温存）。乱数は決定論（seed=キャラ+週）。

    Args:
        sqlite: SQLiteStore。
        char: 対象 Character ORM。
        week_start: 対象週の月曜日の日付。

    Returns:
        配置した伏せ枠の件数（カテゴリ・確率未設定なら 0）。
    """
    scenario = _find_usual_scenario(sqlite, char.id)
    cfg = getattr(scenario, "usual_config", None) or {}
    # うつつのイベント設定を流用（種＋確率。§10 既存資産マッピング）
    from backend.services.scenario_chat.service import (
        _usual_event_categories,
        _usual_event_probability,
    )

    prob = _usual_event_probability(cfg)
    cats = _usual_event_categories(cfg)
    span_start = datetime.combine(week_start, datetime.min.time())
    span_end = span_start + timedelta(days=7)
    # 再配置冪等: 未発火の伏せ枠だけ消す（発火済みの実イベントは残す）
    sqlite.delete_schedule_entries(
        character_id=char.id, since=span_start, until=span_end,
        origins=["adhoc"], statuses=["pending"],
    )
    if prob <= 0.0 or not cats:
        return 0

    wkey = week_key(week_start)
    rng = random.Random(f"sudden-event:{char.id}:{wkey}")
    placed = 0
    for day_idx in range(7):
        if rng.random() >= prob:
            continue
        category = rng.choice(cats)
        hour = rng.randint(_EVENT_HOUR_MIN, _EVENT_HOUR_MAX - 1)
        minute = rng.choice((0, 15, 30, 45))
        fire_at = span_start + timedelta(days=day_idx, hours=hour, minutes=minute)
        # 伏せ枠: status="pending" なので get_active_schedule_entries には出ず、
        # availability も②の生成も侵さない。拘束時間・状態・占有圧は発火時に GM が定義する。
        sqlite.create_schedule_entry(
            character_id=char.id,
            start_at=fire_at,
            end_at=fire_at + timedelta(hours=1),  # 仮枠（実値は発火時 GM 定義）
            state="busy",
            source="world",
            origin="adhoc",
            occupancy=0.0,
            status="pending",
            label=category,
            payload={"kind": "sudden_event_seed", "category": category, "week": wkey},
        )
        placed += 1
    if placed:
        logger.info("③伏せ枠を配置 char=%s week=%s 件数=%d", char.name, wkey, placed)
    return placed


def max_overlapping_occupancy(
    sqlite, character_id: str, start: datetime, end: datetime
) -> float:
    """[start, end) と重なる planned エントリの占有圧の最大を返す（轢き判定・§4）。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター。
        start: 判定区間の開始。
        end: 判定区間の終了。

    Returns:
        重なる planned エントリの占有圧最大。重なりが無ければ 0.0。
    """
    rows = sqlite.list_schedule_entries(
        character_id, since=start, until=end, statuses=["planned"],
    )
    return max(
        (float(getattr(r, "occupancy", 0.0) or 0.0) for r in rows), default=0.0
    )


# ---------------------------------------------------------------------------
# 発火（GM 具体化 → 轢き判定 → insert → シーン）
# ---------------------------------------------------------------------------

async def run_pending_sudden_events(state, now: datetime | None = None) -> None:
    """発火時刻が到来した③伏せ枠を発火させる（毎分のスケジューラから呼ぶ）。

    生活カレンダー有効キャラの未発火伏せ枠を走査し、発火時刻を過ぎたものを1件ずつ
    処理する。冪等のため発火を試みる前に status を done にする（失敗しても再発火しない
    — 生活は流れる）。1件の失敗が他を止めないよう個別に握る。

    Args:
        state: FastAPI の app.state（sqlite / chat_service）。
        now: 基準時刻（テスト注入用）。
    """
    sqlite = state.sqlite
    now = now or datetime.now()
    today_str = now.date().isoformat()
    try:
        cap = int(sqlite.get_setting("sudden_event_daily_cap", ""))
    except (TypeError, ValueError):
        cap = _DEFAULT_SUDDEN_EVENT_CAP
    count_key = f"sudden_event_fire_count_{today_str}"
    fired_today = int(sqlite.get_setting(count_key, "0") or 0)

    for char in sqlite.list_characters():
        if not _living_enabled(char):
            continue
        # 聖域化（§7）: 対面中は③の発火を保留する。伏せ枠には触れず（done にせず）スキップし、
        # 対面が終わってから後続の tick で発火させる（溜まった突発をまとめて反映）。
        if int(getattr(char, "face_to_face_mode", 0) or 0):
            continue
        # pending の adhoc 伏せ枠を拾い、発火時刻（start_at）が現在以前のものを処理する。
        # 期間 until フィルタは start_at < until の排他境界なので、ちょうど now に置かれた
        # 伏せ枠を取りこぼさないよう Python 側で start_at <= now を判定する。
        seeds = sqlite.list_schedule_entries(
            char.id, statuses=["pending"], origins=["adhoc"],
        )
        for seed in seeds:
            if seed.start_at > now:
                continue  # 発火時刻がまだ来ていない
            payload = getattr(seed, "payload", None) or {}
            if payload.get("kind") != "sudden_event_seed":
                continue
            # 発火の冪等: 試みる前に done にする（LLM 失敗でも再発火しない）
            sqlite.set_schedule_entry_status(seed.id, "done")
            # 猶予を超えて古い伏せ枠は発火せず捨てる（通過分は捨てる・§7）
            if now - seed.start_at > timedelta(hours=_EVENT_FIRE_GRACE_HOURS):
                logger.info(
                    "③発火: 猶予超過につき伏せ枠を捨てる char=%s label=%s start=%s",
                    char.name, seed.label, seed.start_at,
                )
                continue
            # 日次上限（§8）: 超過分は捨てる（done 済みなので再発火しない）
            if fired_today >= cap:
                logger.warning(
                    "③発火: 日次上限 %d 到達につき捨てる char=%s label=%s",
                    cap, char.name, seed.label,
                )
                continue
            fired_today += 1
            sqlite.set_setting(count_key, str(fired_today))
            try:
                await _fire_sudden_event(state, char, seed, now)
            except Exception:
                logger.exception(
                    "③発火に失敗 char=%s seed=%s", char.name, seed.id,
                )


async def _fire_sudden_event(state, char, seed, now: datetime) -> None:
    """1件の伏せ枠を発火させる — GM 具体化 → 轢き判定 → insert → シーン → 玉突き裁定。

    Args:
        state: FastAPI の app.state。
        char: 対象 Character ORM。
        seed: 発火する伏せ枠 ScheduleEntry（既に status=done 済み）。
        now: 基準時刻。
    """
    sqlite = state.sqlite
    current_log_feature.set("sudden_event")
    category = str((getattr(seed, "payload", None) or {}).get("category") or seed.label)

    text = await _ask_gm_to_concretize(state, char, category, seed.start_at)
    if not text:
        logger.warning("③発火: GM 具体化が空 char=%s category=%s", char.name, category)
        return
    event = parse_event_line(text, seed.start_at.date())
    if event is None:
        logger.warning(
            "③発火: [EVENT] 行が取れず発火中止 char=%s category=%s", char.name, category,
        )
        return

    # 轢き判定（§4）: 占有圧が重なる既存の最大を上回らなければ insert しない（本人の予定が守られる）
    existing_max = max_overlapping_occupancy(
        sqlite, char.id, event.start_at, event.end_at
    )
    if event.occupancy <= existing_max:
        logger.info(
            "③発火: 轢けず流れる char=%s event=%s occ=%.2f<=既存%.2f",
            char.name, event.label, event.occupancy, existing_max,
        )
        return

    # insert（実イベント = adhoc/world/planned）。既存は物理的に触らない（読み取り時に占有圧最大が勝つ）
    entry = sqlite.create_schedule_entry(
        character_id=char.id,
        start_at=event.start_at,
        end_at=event.end_at,
        state=event.state,
        source="world",
        origin="adhoc",
        occupancy=event.occupancy,
        reply_rate=event.reply_rate,
        check_interval=event.check_interval,
        status="planned",
        label=event.label,
        payload={"kind": "sudden_event", "category": category, "seed_id": seed.id},
    )
    logger.info(
        "③発火: 突発を insert char=%s event=%s occ=%.2f (>既存%.2f)",
        char.name, event.label, event.occupancy, existing_max,
    )

    # シーン実行（③発火 = うつつシーン）。世界の突発が生活に割り込む場面を GM が回す。
    result = await _run_event_scene(state, char, event, category)

    # Phase 6: 玉突き裁定（轢かれた予定を本人が裁く＋内圧確認）。シーン完走時のみ。
    if result is not None and not result.get("error"):
        try:
            from backend.services.schedule.dilemma import run_collision_ruling
            await run_collision_ruling(state, char, entry, now=now)
        except Exception:
            logger.exception("玉突き裁定に失敗 char=%s event=%s", char.name, event.label)


async def _ask_gm_to_concretize(
    state, char, category: str, fire_at: datetime
) -> str | None:
    """GM に③突発の具体化（[EVENT] 行）をリクエストする。

    種（カテゴリ名）と発火日時だけを渡し、拘束時間・状態・占有圧・配達値を GM が定義する
    （世界＝GM が外的フレームを与える思想）。プリセットは①週次バッチと共通の解決順。

    Args:
        state: FastAPI の app.state。
        char: 対象 Character ORM。
        category: イベントカテゴリ（種）。
        fire_at: 伏せ枠の発火時刻。

    Returns:
        GM の出力テキスト。プリセット未解決・LLM 失敗時は None。
    """
    sqlite = state.sqlite
    preset = resolve_gm_preset(sqlite, char)
    if preset is None:
        logger.warning("③発火: GM プリセット未解決 char=%s", char.name)
        return None

    system_prompt = (
        f"あなたは「{char.name}」の生活世界を運営する進行役（GM）です。\n"
        f"{char.name} の本人の意思に依らず、世界の側から突発的な出来事が今まさに起きます。\n\n"
        "# 出力形式\n"
        "次の1行を必ず含めてください（自由文の説明を添えてよい）:\n"
        "[EVENT: HH:MM-HH:MM | 具体的な出来事 | 状態 | 圧 | reply=返信率 | check=間隔分]\n\n"
        "- 時刻: 発火時刻を含む拘束時間帯（24時超え表記可）\n"
        "- 状態: OnTime / active / busy / offline（手が離せないなら busy、意識が飛ぶなら offline）\n"
        "- 圧: 弱 / 中 / 強 / 激強（突発の割り込みの強さ。本人の予定を潰すなら 強〜激強）\n"
        "- reply= / check= は任意（省略時は状態の既定。特別に返信が鈍る/途切れるなら指定）\n\n"
        "# ルール\n"
        f"- 種「{category}」を具体的な1つの出来事に落とす（中身はあなたの即興）\n"
        f"- {char.name} の内面・選択には踏み込まない（世界の側の出来事だけを定義する）"
    )
    user_content = (
        f"種: {category}\n"
        f"発火時刻の目安: {fire_at:%m/%d %H:%M}\n\n"
        "この突発を [EVENT] 行として具体化してください。"
    )
    try:
        provider = create_provider(
            preset.provider,
            preset.model_id or "",
            sqlite.get_all_settings(),
            thinking_level=getattr(preset, "thinking_level", None) or "default",
            preset_name=preset.name,
            timeout_seconds=getattr(preset, "timeout_seconds", None),
        )
        return await provider.generate(
            system_prompt, [{"role": "user", "content": user_content}]
        )
    except Exception as e:
        logger.warning("③発火: GM 具体化リクエスト失敗 char=%s error=%s", char.name, e)
        return None


async def _run_event_scene(state, char, event, category: str) -> dict | None:
    """③突発を題材にうつつシーンを1本走らせる（発火の体験化）。

    日次コストガード（usual_days_scene_count）はうつつスケジューラと共有する。
    うつつ世界が未設定・上限到達なら None（シーンは走らせないが insert 済みの予定は残る）。

    Args:
        state: FastAPI の app.state。
        char: 対象 Character ORM。
        event: 具体化済み EventEntry。
        category: イベントカテゴリ（ログ・framing 用）。

    Returns:
        run_usual_days_scene の結果 dict。走らせなかった場合は None。
    """
    sqlite = state.sqlite
    from backend.services.scenario_chat.service import (
        ensure_usual_session,
        run_usual_days_scene,
    )

    scenario = _find_usual_scenario(sqlite, char.id)
    if scenario is None or not (getattr(scenario, "usual_config", None) or {}).get("enabled"):
        return None  # うつつ世界が無い/無効 — 予定は残るが体験は生成しない

    today_str = datetime.now().date().isoformat()
    daily_cap = int(sqlite.get_setting("usual_days_daily_cap", "24") or 24)
    count_key = f"usual_days_scene_count_{today_str}"
    ran_today = int(sqlite.get_setting(count_key, "0") or 0)
    if ran_today >= daily_cap:
        logger.warning("③発火: 日次上限につきシーンは省略 char=%s", char.name)
        return None

    session = ensure_usual_session(sqlite, scenario)
    if session is None:
        return None
    sqlite.set_setting(count_key, str(ran_today + 1))
    framing = (
        f"[OOC] 世界の側から突発的な出来事が起きる: {event.label}"
        f"（種:「{category}」・{event.start_at:%H:%M}〜{event.end_at:%H:%M}）。"
        f"{char.name} の予定に割り込むかたちで、この出来事を場面として立ち上げてよい。"
    )
    return await run_usual_days_scene(
        session_id=session.id,
        sqlite=sqlite,
        settings=sqlite.get_all_settings(),
        chat_service=state.chat_service,
        extra_first_gm_ooc=framing,
        slot=f"event:{category}",
    )
