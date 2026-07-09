"""玉突き裁定 — 轢かれた予定を本人が裁く（Phase 6・§6）。

③が既存予定を轢いた場合、(a)消す (b)ずらす (c)削る を **機械で決めない**。
世界突発イベントのシーン終了後に、本人へ問い合わせる（Chotgor 思想: 記憶も選択も本人）。
この問い合わせで **内圧の確認も同時に行う**。

    ③発火 → GM 具体化・シーン進行 → シーン終了
         → 本人へ問い合わせ（ask_character_with_tools・WM込み）:
             「突発でこの予定が潰れた。どうする? そして今どんな気分?」
         → 本人の裁定:
             ・諦める      → 潰れた予定を見送り（entry を cancelled に）
             ・ずらす/短縮  → ④として実現層を書き換え（旧を cancelled・新を adhoc insert）
             ・不満        → 見送った予定が不満圧へ転化（intent を作って soured・記憶へ刻む）

既存の意図経済に乗る（新機構なし・§6）: 轢かれた予定は満たされなかった意図。本人が
不満を言語化すれば、その intent を soured にして不満の言葉を記憶へ刻む（pickup と同型）。
そしてこの問い合わせ自体が④を生む（③に轢かれた反動＝③→④の因果）。
"""

import logging
import re
from datetime import datetime

from backend.services.character_query import ask_character_with_tools
from backend.services.schedule.plan_parser import _parse_time_range

logger = logging.getLogger(__name__)

# 本人の裁定タグ（設問文と対で維持すること）。id は轢かれたエントリの ID。
_GIVE_UP_RE = re.compile(r"\[GIVE_UP:\s*([0-9a-fA-F-]+)\s*\]")
_RESCHEDULE_RE = re.compile(
    r"\[RESCHEDULE:\s*([0-9a-fA-F-]+)\s*\|\s*(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})\s*\]"
)
_DISSATISFIED_RE = re.compile(r"\[DISSATISFIED:\s*([0-9a-fA-F-]+)\s*\|\s*([^\]]+?)\s*\]")


def _find_disrupted_entries(sqlite, character_id: str, event_entry) -> list:
    """③イベントに轢かれた予定（重なる低占有圧の固定予定）を集める。

    対象は event と時間帯が重なる planned・template 由来のエントリのうち、占有圧が
    event 未満で、offline（就寝）でないもの。就寝は「裁定してずらす」対象にしないため除外
    （起こされたこと自体の内圧は問い合わせの自由文で拾える）。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター。
        event_entry: 轢いた側の ScheduleEntry（突発イベント）。

    Returns:
        轢かれた ScheduleEntry のリスト（占有圧降順 — 本人が重い予定から見られるように）。
    """
    event_occ = float(getattr(event_entry, "occupancy", 0.0) or 0.0)
    rows = sqlite.list_schedule_entries(
        character_id,
        since=event_entry.start_at,
        until=event_entry.end_at,
        statuses=["planned"],
        origins=["template"],
    )
    disrupted = [
        r for r in rows
        if r.id != event_entry.id
        and str(getattr(r, "state", "") or "") != "offline"
        and float(getattr(r, "occupancy", 0.0) or 0.0) < event_occ
    ]
    disrupted.sort(
        key=lambda r: float(getattr(r, "occupancy", 0.0) or 0.0), reverse=True
    )
    return disrupted


def _build_ruling_question(char, event_entry, disrupted: list) -> str:
    """玉突き裁定の設問文を組み立てる（潰れた予定＋内圧確認）。

    Args:
        char: 対象 Character ORM。
        event_entry: 轢いた突発イベント ScheduleEntry。
        disrupted: 轢かれた予定のリスト。

    Returns:
        本人へ渡す設問テキスト。
    """
    lines = [
        f"さっき、突発で「{event_entry.label}」が入った"
        f"（{event_entry.start_at:%H:%M}〜{event_entry.end_at:%H:%M}）。",
        "そのせいで、立てていた予定が潰れた:",
        "",
    ]
    for e in disrupted:
        lines.append(
            f"- [{e.id}] {e.label}（{e.start_at:%H:%M}〜{e.end_at:%H:%M}）"
        )
    lines += [
        "",
        "これ、どうする？　そして——いま、どんな気分？",
        "無理に何かしなくていい。しゃーないと流すのも、ずらすのも、あなた次第。",
        "",
        "当てはまるものだけ、返事のどこかに書いて（複数可）:",
        "- 諦める（見送る）: `[GIVE_UP: 予定のid]`",
        "- ずらす／短縮する: `[RESCHEDULE: 予定のid | HH:MM-HH:MM]`（同じ日の新しい時間帯）",
        "- 不満・もどかしさが残る: `[DISSATISFIED: 予定のid | その気持ちの言葉]`",
        "",
        "何も書かなければ、予定はそのまま（潰れたまま流れる）。気持ちは自由に書いていい。",
    ]
    return "\n".join(lines)


def _parse_ruling(text: str, valid_ids: set) -> dict:
    """本人の返答から裁定タグを抽出する（valid_ids に無い id は捨てる）。

    Args:
        text: 本人の返答。
        valid_ids: 轢かれた予定の ID 集合（設問で提示したもの）。

    Returns:
        {"give_up": [id...], "reschedule": [{"id", "range"}...],
         "dissatisfied": [{"id", "words"}...]}。
    """
    text = text or ""
    give_up = [
        m.group(1).strip() for m in _GIVE_UP_RE.finditer(text)
        if m.group(1).strip() in valid_ids
    ]
    reschedule = [
        {"id": m.group(1).strip(), "range": m.group(2).strip()}
        for m in _RESCHEDULE_RE.finditer(text)
        if m.group(1).strip() in valid_ids
    ]
    dissatisfied = [
        {"id": m.group(1).strip(), "words": m.group(2).strip()}
        for m in _DISSATISFIED_RE.finditer(text)
        if m.group(1).strip() in valid_ids
    ]
    return {"give_up": give_up, "reschedule": reschedule, "dissatisfied": dissatisfied}


async def run_collision_ruling(state, char, event_entry, now: datetime | None = None) -> dict:
    """③に轢かれた予定を本人に裁定させる（シーン終了後の1回・§6）。

    Args:
        state: FastAPI の app.state（sqlite / memory_manager / working_memory_manager）。
        char: 対象 Character ORM。
        event_entry: 轢いた突発イベント ScheduleEntry（既に insert 済み）。
        now: 基準時刻（テスト注入用）。

    Returns:
        {"status": ..., "given_up": int, "rescheduled": int, "soured": int} の集計 dict。
    """
    sqlite = state.sqlite
    ghost_model = (getattr(char, "ghost_model", None) or "").strip()
    if not ghost_model:
        return {"status": "skipped", "reason": "ghost_model 未設定"}

    disrupted = _find_disrupted_entries(sqlite, char.id, event_entry)
    if not disrupted:
        return {"status": "skipped", "reason": "潰れた予定なし"}

    question = _build_ruling_question(char, event_entry, disrupted)
    response = await ask_character_with_tools(
        character_id=char.id,
        preset_id=ghost_model,
        messages=[{"role": "user", "content": question}],
        sqlite=sqlite,
        settings=sqlite.get_all_settings(),
        memory_manager=getattr(state, "memory_manager", None),
        feature_label="collision_ruling",
        working_memory_manager=getattr(state, "working_memory_manager", None),
        recall_query="潰れた予定・突発への気持ち",
        return_response=True,
    )
    if not response:
        return {"status": "error", "error": "本人からの返答が取得できませんでした"}

    by_id = {e.id: e for e in disrupted}
    parsed = _parse_ruling(response, set(by_id.keys()))

    given_up = 0
    for entry_id in parsed["give_up"]:
        if sqlite.set_schedule_entry_status(entry_id, "cancelled"):
            given_up += 1

    rescheduled = 0
    for item in parsed["reschedule"]:
        old = by_id.get(item["id"])
        if old is None:
            continue
        base = datetime(old.start_at.year, old.start_at.month, old.start_at.day)
        span = _parse_time_range(item["range"], base)
        if span is None:
            continue
        # 旧を見送り、新しい時間帯へ ④（本人由来 adhoc）として insert する
        sqlite.set_schedule_entry_status(old.id, "cancelled")
        sqlite.create_schedule_entry(
            character_id=char.id,
            start_at=span[0],
            end_at=span[1],
            state=str(getattr(old, "state", None) or "active"),
            source="haru",
            origin="adhoc",
            occupancy=float(getattr(old, "occupancy", 0.5) or 0.5),
            status="planned",
            label=str(getattr(old, "label", None) or "予定"),
            payload={"kind": "reschedule", "from_entry": old.id},
        )
        rescheduled += 1

    soured = _apply_dissatisfaction(state, char, ghost_model, by_id, parsed["dissatisfied"])

    logger.info(
        "玉突き裁定 完了 char=%s event=%s 諦め=%d ずらし=%d 不満=%d",
        char.name, event_entry.label, given_up, rescheduled, soured,
    )
    return {
        "status": "success",
        "given_up": given_up,
        "rescheduled": rescheduled,
        "soured": soured,
    }


def _apply_dissatisfaction(state, char, ghost_model: str, by_id: dict, items: list) -> int:
    """不満の裁定を意図経済へ乗せる — 意図を作って soured にし、不満の言葉を記憶へ刻む。

    轢かれた予定は「満たされなかった意図」（§6）。本人が不満を言語化したら、その予定を
    表す意図を作成→即 soured 遷移させ、不満の言葉を intent.payload に凍結＋記憶へ刻む
    （pickup 層の soured 処理と同型・不満化＝利害と合流）。

    Args:
        state: app.state（memory_manager を使う）。
        char: 対象 Character ORM。
        ghost_model: 記憶刻み込みの source_preset_id。
        by_id: 轢かれた予定 id→ScheduleEntry の辞書。
        items: [{"id", "words"}...] の不満裁定リスト。

    Returns:
        soured 化できた件数。
    """
    sqlite = state.sqlite
    memory_manager = getattr(state, "memory_manager", None)
    soured = 0
    for item in items:
        entry = by_id.get(item["id"])
        if entry is None or not item["words"]:
            continue
        # 予定を表す意図を作り、その場で soured へ遷移させる（不満圧へ転化）
        intent = sqlite.create_intent(
            char.id,
            f"{entry.label}をしたかった",
            source_kind="none",
            born_from="collision",
            payload={"from_entry": entry.id},
        )
        resolved = sqlite.resolve_intent(intent.id, "soured", words=item["words"])
        if resolved is None:
            continue
        soured += 1
        # 不満の言葉を記憶へ刻む（embedding 障害等で失敗しても遷移自体は成立済み）
        if memory_manager is not None:
            try:
                memory_manager.write_inscribed_memory(
                    character_id=char.id,
                    content=f"突発で「{entry.label}」が潰れた不満: {item['words']}",
                    category="contextual",
                    contextual_importance=0.7,
                    user_importance=0.5,
                    source_preset_id=ghost_model,
                )
            except Exception:
                logger.exception(
                    "不満の刻み込みに失敗 char=%s entry=%s", char.name, entry.id,
                )
    return soured
