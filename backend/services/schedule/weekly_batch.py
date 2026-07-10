"""週次バッチ①② — 生活カレンダー実現層の生成（schedule_plan.md §3・Phase 3）。

順序を固定することで、競合と秘匿の複雑さを機構ではなく順序で解決する:

    ① 世界固定予定（GMリクエスト）
        入力: availability_schedule（テンプレ層シード）＋ 先週うつつのあらすじ
        → 実現層に world/template エントリとして確定
    ② はる固定予定（本人問い合わせ・ask_character_with_tools・WM込み）
        入力: ①を見せる ＋ はる自身のプライベート
        → 実現層に haru/template エントリとして確定

②が①を織り込んで確定するので①②はぶつからない（占有圧による①②間の競合解決は不要）。
③世界突発（確率配置）は Phase 5 で本バッチに追加される。

パース失敗のフォールバック（§3・層ごと・前週優先）:
    ① 失敗 → 前週の① → それも無ければ availability_schedule を裸で変換
    ② 失敗 → 前週の② → それも無ければ ①をそのまま採用（はる層なしの週）
    層失敗 = offline（就寝）が1件も取れなかった（就寝が消えると深夜も OnTime になるため）。
    最悪でも「availability_schedule が裸で出る」まで縮退し、週次バッチの失敗で生活は止まらない。

実行契機（§13 裁定）:
    - 日曜夜（weekly_schedule_time・既定 20:00）に翌週分を生成。冪等キー = 対象 ISO 週。
    - コールドスタート（機能有効化・取りこぼし復旧）: 当週分の実現層が未生成なら即時バッチ。
    - 冪等キーは実行**前**に立てる（他スケジューラと同型 — 失敗時に毎分 LLM を叩かない。
      フォールバック連鎖が最悪でもテンプレ裸変換まで保証する）。
"""

import logging
from datetime import date, datetime, time, timedelta

from backend.lib.log_context import current_log_feature
from backend.providers.registry import create_provider
from backend.services.schedule.plan_parser import (
    PlanEntry,
    entries_from_template,
    format_plan_lines,
    layer_has_offline,
    parse_plan_lines,
)

logger = logging.getLogger(__name__)

# 週次バッチの実行時刻（日曜のこの時刻以降に翌週分を生成）。夜間バッチ群（03:00〜05:00）と分離。
_DEFAULT_BATCH_TIME = "20:00"

# [PLAN] 行の書式説明（①GM・②はる共通で提示する）
_PLAN_FORMAT_NOTE = """予定は次の形式の行で書く（1行1予定。自由文と混ぜてよい）:
[PLAN: 曜日 | HH:MM-HH:MM | ラベル | 状態 | 圧]

- 曜日: 月 / 火 / 水 / 木 / 金 / 土 / 日
- 時刻: 24時を超える表記で翌日跨ぎを表せる（25:00 = 翌1:00。例 25:00-31:30 の就寝）
- 状態: OnTime（完全に手が空いている）/ active（何かしているが連絡は取れる）/
  busy（手が離せない）/ offline（意識がない・連絡がつかない。就寝など）
- 圧: 弱 / 中 / 強 / 激強（その予定の重さ。強いほど突発の割り込みに潰されにくい）"""


def week_start_of(d: date) -> date:
    """日付を含む ISO 週の月曜日を返す。"""
    return d - timedelta(days=d.weekday())


def week_key(week_start: date) -> str:
    """週の冪等キー（ISO 週文字列・例 "2026-W29"）を返す。

    年→週の辞書順が時系列順と一致する（年跨ぎも "2026-W53" < "2027-W01" で正しく並ぶ）
    ため、settings に保存した前回実行キーとの大小比較で「未生成の週」を判定できる。
    """
    iso = week_start.isocalendar()
    return f"{iso[0]:04d}-W{iso[1]:02d}"


def _week_span(week_start: date) -> tuple[datetime, datetime]:
    """週の [開始, 終了) を datetime で返す（月曜 0:00 〜 翌週月曜 0:00）。"""
    start = datetime.combine(week_start, time())
    return start, start + timedelta(days=7)


def _format_week_days(week_start: date) -> str:
    """対象週の曜日と日付の一覧テキスト（プロンプト用・例 "月=7/13 火=7/14 ..."）。"""
    day_labels = ("月", "火", "水", "木", "金", "土", "日")
    return " ".join(
        f"{day_labels[i]}={d.month}/{d.day}"
        for i, d in ((i, week_start + timedelta(days=i)) for i in range(7))
    )


# ---------------------------------------------------------------------------
# スケジューラ入口
# ---------------------------------------------------------------------------

async def run_pending_weekly_batches(state, now: datetime | None = None) -> None:
    """生活カレンダー有効キャラを走査し、未生成の週があればバッチを実行する。

    スケジューラ（main.py の _weekly_schedule_scheduler）から毎分呼ばれる。
    キャラごとの冪等キー ``weekly_schedule_done_{character_id}``（最後に生成した
    対象週の ISO 週文字列）と比較して、次の2種類の実行を判定する:

    - **コールドスタート／取りこぼし復旧**: 当週がまだ生成されていなければ即時バッチ
      （機能有効化直後・日曜夜にプロセスが落ちていた場合の月曜起動を含む）。
    - **日曜夜の定時実行**: weekly_schedule_time（既定 20:00）以降に翌週分を生成。

    Args:
        state: FastAPI の app.state（sqlite / memory_manager / working_memory_manager）。
        now: 基準時刻（テスト注入用。省略時は現在時刻）。
    """
    sqlite = state.sqlite
    now = now or datetime.now()
    current_week = week_start_of(now.date())

    # 日曜の定時を過ぎていれば翌週分も対象になる
    batch_time_str = sqlite.get_setting("weekly_schedule_time", _DEFAULT_BATCH_TIME)
    try:
        h, m = map(int, str(batch_time_str).split(":"))
    except (ValueError, AttributeError):
        h, m = map(int, _DEFAULT_BATCH_TIME.split(":"))
    sunday_due = now.weekday() == 6 and now >= now.replace(
        hour=h, minute=m, second=0, microsecond=0
    )

    for char in sqlite.list_characters():
        if not int(getattr(char, "living_schedule_enabled", 0) or 0):
            continue
        done_key = f"weekly_schedule_done_{char.id}"
        done = str(sqlite.get_setting(done_key, "") or "")
        targets: list[date] = []
        if done < week_key(current_week):
            targets.append(current_week)  # コールドスタート／取りこぼし復旧
        next_week = current_week + timedelta(days=7)
        if sunday_due and done < week_key(next_week) and next_week not in targets:
            targets.append(next_week)
        for target in targets:
            # 冪等キーは実行前に立てる（失敗しても毎分 LLM を叩かない）
            sqlite.set_setting(done_key, week_key(target))
            try:
                summary = await run_weekly_schedule_batch(state, char, target)
                sqlite.record_scheduler_decision(
                    "weekly_schedule", "fired", character_id=char.id,
                    reason=f"週次バッチ完了（{week_key(target)}）",
                    details={"week": week_key(target), **(summary or {})},
                )
            except Exception:
                logger.exception(
                    "週次バッチ失敗 char=%s week=%s", char.name, week_key(target),
                )
                sqlite.record_scheduler_decision(
                    "weekly_schedule", "error", character_id=char.id,
                    reason="週次バッチで例外", details={"week": week_key(target)},
                )


# ---------------------------------------------------------------------------
# バッチ本体
# ---------------------------------------------------------------------------

async def run_weekly_schedule_batch(state, char, week_start: date) -> dict:
    """1キャラ・1週分の実現層を生成する（①GM → ②はる → 入れ替え保存）。

    対象週の template エントリを削除してから入れ直す（再生成冪等）。adhoc（③④）は
    触らない。①②の層フォールバックはそれぞれの生成関数が抱え、本関数は
    「必ず何らかのエントリ集合が返る」前提で保存だけ行う。

    Args:
        state: FastAPI の app.state。
        char: 対象 Character ORM（living_schedule_enabled=1）。
        week_start: 対象週の月曜日の日付。

    Returns:
        {"world": 件数, "haru": 件数, "world_mode": 生成経路, "haru_mode": 生成経路}。
    """
    sqlite = state.sqlite
    current_log_feature.set("weekly_schedule")
    wkey = week_key(week_start)

    world_entries, world_mode = await _generate_world_layer(state, char, week_start)
    haru_entries, haru_mode = await _generate_haru_layer(
        state, char, week_start, world_entries
    )

    span_start, span_end = _week_span(week_start)
    deleted = sqlite.delete_schedule_entries(
        character_id=char.id, since=span_start, until=span_end, origins=["template"],
    )
    for source, entries, mode in (
        ("world", world_entries, world_mode),
        ("haru", haru_entries, haru_mode),
    ):
        for e in entries:
            sqlite.create_schedule_entry(
                character_id=char.id,
                start_at=e.start_at,
                end_at=e.end_at,
                state=e.state,
                source=source,
                origin="template",
                occupancy=e.occupancy,
                label=e.label,
                payload={"week": wkey, "generated_by": mode},
            )
    # ③世界突発の伏せ枠を確率配置する（②→③の順序で秘匿の仕掛けが不要・§3）。
    # 発火時 GM 具体化・轢き判定は events.run_pending_sudden_events が担う（Phase 5）。
    # 循環 import（events → weekly_batch）を起動時に作らないため遅延 import。
    from backend.services.schedule.events import place_weekly_hidden_events

    events_placed = place_weekly_hidden_events(sqlite, char, week_start)
    logger.info(
        "週次バッチ完了 char=%s week=%s world=%d(%s) haru=%d(%s) 旧template削除=%d 伏せ枠=%d",
        char.name, wkey, len(world_entries), world_mode,
        len(haru_entries), haru_mode, deleted, events_placed,
    )
    return {
        "world": len(world_entries), "haru": len(haru_entries),
        "world_mode": world_mode, "haru_mode": haru_mode,
        "events": events_placed,
    }


# ---------------------------------------------------------------------------
# ① 世界固定予定（GM）
# ---------------------------------------------------------------------------

async def _generate_world_layer(
    state, char, week_start: date
) -> tuple[list[PlanEntry], str]:
    """①世界固定予定を生成する（GMリクエスト → 前週① → テンプレ裸変換）。

    Returns:
        (エントリ群, 生成経路)。経路は "gm" / "prev_week" / "template" のいずれか。
    """
    sqlite = state.sqlite
    text = await _ask_gm_for_world_plan(state, char, week_start)
    if text:
        entries = parse_plan_lines(text, week_start)
        if layer_has_offline(entries):
            return entries, "gm"
        logger.warning(
            "週次バッチ①: offline（就寝）が取れず層失敗 char=%s 有効行=%d",
            char.name, len(entries),
        )
    prev = _previous_week_entries(sqlite, char.id, week_start, source="world")
    if layer_has_offline(prev):
        return prev, "prev_week"
    # 最終縮退: テンプレ層を裸で変換（offline 要求は課さない — これ以上落ちる先がない）
    bare = entries_from_template(
        getattr(char, "availability_schedule", None), week_start
    )
    if not layer_has_offline(bare):
        logger.warning(
            "週次バッチ①: テンプレ裸変換にも offline が無い（深夜も OnTime になる）"
            " char=%s entries=%d", char.name, len(bare),
        )
    return bare, "template"


async def _ask_gm_for_world_plan(state, char, week_start: date) -> str | None:
    """GM（人格なき環境）へ今週の世界側予定の確定をリクエストする。

    入力はテンプレ層シード（availability_schedule）＋先週うつつのあらすじ（§3 ①の
    フィードバックは時間差 — 先週の結果を今週の入力として畳む）。プリセットは
    うつつ世界の gm_preset_id を最優先し、無ければ本人の ghost_model で代用する。

    Returns:
        GM の出力テキスト。プリセット未解決・LLM 失敗時は None（層フォールバックへ）。
    """
    sqlite = state.sqlite
    settings = sqlite.get_all_settings()
    scenario = _find_usual_scenario(sqlite, char.id)
    preset = resolve_gm_preset(sqlite, char, scenario)
    if preset is None:
        logger.warning("週次バッチ①: GM プリセット未解決 char=%s", char.name)
        return None

    template_entries = entries_from_template(
        getattr(char, "availability_schedule", None), week_start
    )
    template_text = format_plan_lines(template_entries) or "（固定時間割は未設定）"
    synopsis_text = _usual_synopsis_text(sqlite, scenario)

    system_prompt = (
        f"あなたは「{char.name}」の生活世界を運営する進行役（GM）です。\n"
        f"{char.name} の生活カレンダーのうち、本人の意思に依らず世界側で確定している予定"
        "（仕事・来客・約束など）を今週分として確定させます。\n\n"
        f"# 出力形式\n{_PLAN_FORMAT_NOTE}\n\n"
        "# ルール\n"
        "- 固定時間割を骨格として今週の各曜日へ展開する\n"
        "- 就寝時間帯（offline）を毎日必ず1行含める\n"
        "- 先週の生活のあらすじに今週の予定として話題へ上がったもの（来客・約束など）が"
        "あれば織り込む\n"
        f"- 世界側の確定予定だけを出す（{char.name} が自分の意思で決める過ごし方は"
        "本人が後で決める — ここでは決めない）\n"
        "- 段取りの説明は自由に書いてよいが、確定する予定は必ず [PLAN] 行にする"
    )
    user_content = (
        f"今週（{_format_week_days(week_start)}）の世界側予定を確定してください。\n\n"
        f"# 固定時間割（テンプレート）\n{template_text}\n"
    )
    if synopsis_text:
        user_content += f"\n# 先週の生活のあらすじ\n{synopsis_text}\n"

    try:
        provider = create_provider(
            preset.provider,
            preset.model_id or "",
            settings,
            thinking_level=getattr(preset, "thinking_level", None) or "default",
            preset_name=preset.name,
            timeout_seconds=getattr(preset, "timeout_seconds", None),
        )
        return await provider.generate(
            system_prompt, [{"role": "user", "content": user_content}]
        )
    except Exception as e:
        logger.warning(
            "週次バッチ①: GM リクエスト失敗 char=%s preset=%s error=%s",
            char.name, gm_preset_id, e,
        )
        return None


def _find_usual_scenario(sqlite, character_id: str):
    """キャラクターが所有するうつつ（生活世界）シナリオを返す。無ければ None。"""
    for scenario in sqlite.list_usual_scenarios():
        if getattr(scenario, "owner_character_id", None) == character_id:
            return scenario
    return None


def resolve_gm_preset(sqlite, char, scenario=None):
    """GM（人格なき環境）を動かすモデルプリセットを解決する。

    優先順: うつつ世界の usual_config.gm_preset_id → 本人の ghost_model。①週次バッチと
    ③突発の具体化で共有する（どちらも「世界の側」を GM に喋らせるため）。

    Args:
        sqlite: SQLiteStore。
        char: 対象 Character ORM。
        scenario: 所有うつつシナリオ（省略時はここで探す）。

    Returns:
        ModelPreset ORM。プリセット未解決・未発見なら None。
    """
    if scenario is None:
        scenario = _find_usual_scenario(sqlite, char.id)
    gm_preset_id = ""
    if scenario is not None:
        cfg = getattr(scenario, "usual_config", None) or {}
        gm_preset_id = (cfg.get("gm_preset_id") or "").strip()
    gm_preset_id = gm_preset_id or (getattr(char, "ghost_model", None) or "").strip()
    if not gm_preset_id:
        return None
    return sqlite.get_model_preset(gm_preset_id)


def _usual_synopsis_text(sqlite, scenario) -> str:
    """うつつ永続セッションのあらすじ（auto＋manual）を返す。取れなければ空文字列。

    週次バッチは日曜夜に走るため、この時点のあらすじ＝「先週までの生活」。
    同一週内で①は不変（§3 フィードバックの時間差）なので、これで循環は切れている。
    """
    if scenario is None:
        return ""
    try:
        for s in sqlite.list_scenario_sessions_by_scenario(scenario.id):
            if (
                getattr(s, "engine_type", "") == "usual_days"
                and getattr(s, "status", "") == "active"
            ):
                synopsis = sqlite.get_scenario_session_synopsis(s.id) or {}
                parts = [
                    t for t in (synopsis.get("manual", ""), synopsis.get("auto", "")) if t
                ]
                return "\n\n".join(parts)
    except Exception as e:
        logger.warning("週次バッチ①: あらすじ取得失敗 scenario=%s error=%s", scenario.id, e)
    return ""


# ---------------------------------------------------------------------------
# ② はる固定予定（本人問い合わせ）
# ---------------------------------------------------------------------------

async def _generate_haru_layer(
    state, char, week_start: date, world_entries: list[PlanEntry]
) -> tuple[list[PlanEntry], str]:
    """②本人固定予定を生成する（本人問い合わせ → 前週② → はる層なし）。

    ①を見せた上で「今週どう暮らすか」を本人に聞く（キャラクター問い合わせ原則 —
    ask_character_with_tools・WM込み）。この答えは予定であると同時にうつつの題材でもある
    （§8 — シーン導出は Phase 4）。

    Returns:
        (エントリ群, 生成経路)。経路は "haru" / "prev_week" / "none"。
        "none" は①をそのまま採用する週（はる層なし — フォールバック最終段）。
    """
    sqlite = state.sqlite
    response = await _ask_character_for_week_plan(state, char, week_start, world_entries)
    if response:
        entries = parse_plan_lines(response, week_start)
        if layer_has_offline(entries):
            return entries, "haru"
        logger.warning(
            "週次バッチ②: offline（就寝）が取れず層失敗 char=%s 有効行=%d",
            char.name, len(entries),
        )
    prev = _previous_week_entries(sqlite, char.id, week_start, source="haru")
    if layer_has_offline(prev):
        return prev, "prev_week"
    return [], "none"  # ①をそのまま採用（はる層なしの週）


async def _ask_character_for_week_plan(
    state, char, week_start: date, world_entries: list[PlanEntry]
) -> str | None:
    """本人へ「今週どう暮らすか」を問いかける（1on1 同等の WM ブロック込み）。

    Returns:
        本人の応答テキスト。ghost_model 未設定・LLM 失敗時は None（層フォールバックへ）。
    """
    from backend.services.character_query import ask_character_with_tools

    sqlite = state.sqlite
    ghost_model = (getattr(char, "ghost_model", None) or "").strip()
    if not ghost_model:
        logger.warning("週次バッチ②: ghost_model 未設定 char=%s", char.name)
        return None

    world_text = format_plan_lines(world_entries) or "（今週、世界側の確定予定はない）"
    question = (
        f"今週（{_format_week_days(week_start)}）の、世界側で確定している予定はこうなってる:\n\n"
        f"{world_text}\n\n"
        "このすき間で、今週をどう暮らすか教えてほしい。没頭したいこと、調べもの、片付け、"
        "だらだらする時間——あなたが自分で決めた予定が、そのまま今週の生活になる。\n\n"
        "考えは自由に書いていい。そのうえで、予定として確定したい部分を次の形式の行に"
        "してほしい。\n\n"
        f"{_PLAN_FORMAT_NOTE}\n\n"
        "- 圧は「その予定をどれだけ守りたいか」。強いほど突発の割り込みに潰されにくい\n"
        "- 寝る時間も毎日 [PLAN] 行で入れてほしい（offline）。"
        "深夜1時に寝るなら 25:00 のように書いていい"
    )
    return await ask_character_with_tools(
        character_id=char.id,
        preset_id=ghost_model,
        messages=[{"role": "user", "content": question}],
        sqlite=sqlite,
        settings=sqlite.get_all_settings(),
        memory_manager=getattr(state, "memory_manager", None),
        feature_label="weekly_schedule",
        working_memory_manager=getattr(state, "working_memory_manager", None),
        recall_query="今週の予定・暮らし方",
        return_response=True,
    )


# ---------------------------------------------------------------------------
# 前週フォールバック
# ---------------------------------------------------------------------------

def _previous_week_entries(
    sqlite, character_id: str, week_start: date, *, source: str
) -> list[PlanEntry]:
    """前週の同一層（template・planned）を +7 日シフトして返す（層フォールバック用）。

    cancelled/done（玉突き裁定で消えた予定）は写さない。前週が存在しなければ空リスト。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター ID。
        week_start: 対象週（フォールバック先ではなく生成したい週）の月曜日。
        source: 写す層（"world" = ① / "haru" = ②）。

    Returns:
        対象週の時刻へシフト済みの PlanEntry リスト。
    """
    prev_start, prev_end = _week_span(week_start - timedelta(days=7))
    rows = sqlite.list_schedule_entries(
        character_id,
        since=prev_start,
        until=prev_end,
        statuses=["planned"],
        origins=["template"],
    )
    shifted: list[PlanEntry] = []
    for row in rows:
        # 期間フィルタは「重なり」判定なので、前週開始より前に始まる跨ぎエントリも
        # 返りうる。開始が前週内のものだけを写す（週の複製として自然な範囲）。
        if getattr(row, "source", "") != source:
            continue
        if not (prev_start <= row.start_at < prev_end):
            continue
        shifted.append(
            PlanEntry(
                start_at=row.start_at + timedelta(days=7),
                end_at=row.end_at + timedelta(days=7),
                label=str(getattr(row, "label", None) or "予定"),
                state=str(getattr(row, "state", None) or "active"),
                occupancy=float(getattr(row, "occupancy", 0.5) or 0.5),
            )
        )
    return shifted
