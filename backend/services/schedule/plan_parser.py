"""[PLAN] 行パーサとテンプレ層変換 — 生活カレンダー週次バッチの純関数群。

docs/planned/schedule_plan.md §3「生成プロトコル」の実装。①GM・②はるの出力本文から
行単位タグ（既存 [ACTION:] / [SCENE_CLOSE] と同じ流儀）で予定行を抽出し、
実現層エントリの素材（PlanEntry）へ変換する。LLM は使わない。

```
[PLAN: 曜日 | HH:MM-HH:MM | ラベル | 状態 | 圧]

[PLAN: 月 | 09:00-17:30 | 仕事 | active | 強]
[PLAN: 月 | 25:00-31:30 | 就寝 | offline | 中]   ← 24時超え表記 = 翌日跨ぎ（25:00 = 翌1:00）
```

- 曜日: 月〜日（mon〜sun も受理）。状態: OnTime / active / busy / offline。
- 24時超え表記と `from > to`（日跨ぎ）の両方を受理する（既存 _schedule_block と同じ流儀）。
- 不正行はスキップして有効行だけ採用する（§3 パース失敗のフォールバック — 部分成功は
  エントリの隙間 = OnTime なので安全側に縮退する）。
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# 占有圧の4段ラベル → 内部値（schedule_plan.md §4。初期値・計器で調整）
PRESSURE_VALUES: dict[str, float] = {"弱": 0.25, "中": 0.5, "強": 0.75, "激強": 1.0}

# 状態の正規化テーブル（小文字キー → 正準表記）
_STATE_CANONICAL: dict[str, str] = {
    "ontime": "OnTime",
    "active": "active",
    "busy": "busy",
    "offline": "offline",
}

# 曜日表記 → 週内インデックス（月=0 〜 日=6）
_DAY_INDEX: dict[str, int] = {
    "月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

# [PLAN: ...] 行の抽出（本文の自由文と共存できる行単位タグ方式）
_PLAN_RE = re.compile(r"\[PLAN:\s*([^\]]*?)\s*\]")

# [EVENT: ...] 行の抽出（③突発の発火時 GM 具体化・§5）
_EVENT_RE = re.compile(r"\[EVENT:\s*([^\]]*?)\s*\]")

# "HH:MM-HH:MM"。時は 0〜47 を受理（24 以上 = 翌日跨ぎ表記）
_TIME_RANGE_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$")

# 個別上書き（reply=0.2 / check=90）の key=value 抽出（§5「個別値を吐ける穴」）
_OVERRIDE_RE = re.compile(r"^(reply|check)\s*=\s*([0-9.]+)$")


@dataclass
class PlanEntry:
    """パース済みの予定1件 — schedule_entries へ insert する素材。

    実現層（ScheduleEntry ORM）と違い、source / origin / status は持たない
    （どの層として保存するかは呼び出し側 = 週次バッチが決める）。

    Attributes:
        start_at: 予定の開始時刻（絶対時刻へ展開済み）。
        end_at: 予定の終了時刻。
        label: 表示ラベル（"仕事" 等）。
        state: 配達プリセット状態（OnTime / active / busy / offline）。
        occupancy: 占有圧 0.0–1.0。
    """

    start_at: datetime
    end_at: datetime
    label: str
    state: str
    occupancy: float


def _parse_time_range(
    text: str, base: datetime
) -> tuple[datetime, datetime] | None:
    """"HH:MM-HH:MM" を (開始, 終了) の絶対時刻に変換する。不正なら None。

    Args:
        text: 時刻範囲文字列。時は 0〜47（24 以上は翌日跨ぎ表記）。
        base: その曜日の 0:00（週開始 + 曜日オフセット）。

    Returns:
        (start_at, end_at)。終了 <= 開始のときは終了を翌日に送る（日跨ぎ）。
    """
    m = _TIME_RANGE_RE.match(text.strip())
    if m is None:
        return None
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    if not (0 <= h1 < 48 and 0 <= h2 < 48 and 0 <= m1 < 60 and 0 <= m2 < 60):
        return None
    start = base + timedelta(hours=h1, minutes=m1)
    end = base + timedelta(hours=h2, minutes=m2)
    if end <= start:
        end += timedelta(days=1)  # from > to = 日跨ぎ（例 23:00-06:00）
    return start, end


def canonical_state(raw: str, default: str = "active") -> str:
    """状態表記を正準形へ正規化する（大文字小文字ゆらぎ耐性）。

    Args:
        raw: 生成側が吐いた状態文字列。
        default: 未知表記のときのフォールバック。

    Returns:
        OnTime / active / busy / offline のいずれか（未知は default）。
    """
    return _STATE_CANONICAL.get((raw or "").strip().lower(), default)


def parse_plan_lines(text: str, week_start: date) -> list[PlanEntry]:
    """本文から [PLAN] 行を抽出し、対象週の絶対時刻エントリへ展開する。

    不正行（フィールド数不足・未知曜日・時刻不正・未知の圧ラベル）はスキップし、
    有効行だけを返す（部分成功の安全側縮退・§3）。状態の未知表記だけは
    active へ寄せる（タイポで行ごと失うより配達値が既定に寄る方が害が小さい）。

    Args:
        text: ①GM／②はるの出力本文（自由文と [PLAN] 行が混在してよい）。
        week_start: 対象週の月曜日の日付。

    Returns:
        PlanEntry のリスト（出現順）。
    """
    if not text:
        return []
    entries: list[PlanEntry] = []
    for body in _PLAN_RE.findall(text):
        parts = [p.strip() for p in body.split("|")]
        if len(parts) != 5:
            continue
        day_raw, time_raw, label, state_raw, pressure_raw = parts
        day_idx = _DAY_INDEX.get(day_raw.lower())
        if day_idx is None:
            continue
        base = datetime(week_start.year, week_start.month, week_start.day) + timedelta(
            days=day_idx
        )
        span = _parse_time_range(time_raw, base)
        if span is None:
            continue
        occupancy = PRESSURE_VALUES.get(pressure_raw)
        if occupancy is None:
            continue
        entries.append(
            PlanEntry(
                start_at=span[0],
                end_at=span[1],
                label=label or "予定",
                state=canonical_state(state_raw),
                occupancy=occupancy,
            )
        )
    return entries


@dataclass
class EventEntry:
    """③突発イベントの発火時 GM 具体化1件（[EVENT] 行のパース結果・§5）。

    PlanEntry に配達値の個別上書き（reply_rate / check_interval）を加えたもの。
    ③は「拘束時間・返信率・間隔は発火時に GM が定義」なので、プリセットに縛られない
    個別値を持てる（None なら state プリセット既定を使う）。

    Attributes:
        start_at: イベントの開始時刻（絶対時刻）。
        end_at: イベントの終了時刻。
        label: 表示ラベル（"客先トラブルの電話対応" 等）。
        state: 配達プリセット状態（OnTime / active / busy / offline）。
        occupancy: 占有圧 0.0–1.0（轢き判定に使う侵食力）。
        reply_rate: 返信率の個別上書き（None = state プリセット既定）。
        check_interval: チェック間隔 [分] の個別上書き（None = state プリセット既定）。
    """

    start_at: datetime
    end_at: datetime
    label: str
    state: str
    occupancy: float
    reply_rate: float | None = None
    check_interval: int | None = None


def parse_event_line(text: str, base_date: date) -> EventEntry | None:
    """本文から最初の [EVENT] 行を抽出し、その日の絶対時刻イベントへ展開する（§5）。

    書式: ``[EVENT: HH:MM-HH:MM | ラベル | 状態 | 圧 | reply=0.2 | check=90]``
    末尾の ``reply=`` / ``check=`` は任意（配達値の個別上書き）。PLAN と違い曜日を持たず、
    base_date（発火日）を基準に時刻を展開する。24時超え表記・日跨ぎは PLAN と同流儀。

    複数の [EVENT] 行があっても最初の1件だけ採用する（1発火＝1イベント）。

    Args:
        text: GM の具体化出力（自由文と [EVENT] 行が混在してよい）。
        base_date: イベント発火日（この日の 0:00 を時刻の基準にする）。

    Returns:
        EventEntry。有効な [EVENT] 行が無い・パース不能なら None。
    """
    if not text:
        return None
    base = datetime(base_date.year, base_date.month, base_date.day)
    for body in _EVENT_RE.findall(text):
        parts = [p.strip() for p in body.split("|")]
        if len(parts) < 4:
            continue
        time_raw, label, state_raw, pressure_raw = parts[:4]
        span = _parse_time_range(time_raw, base)
        if span is None:
            continue
        occupancy = PRESSURE_VALUES.get(pressure_raw)
        if occupancy is None:
            continue
        reply_rate: float | None = None
        check_interval: int | None = None
        for extra in parts[4:]:
            m = _OVERRIDE_RE.match(extra.lower())
            if m is None:
                continue
            if m.group(1) == "reply":
                try:
                    reply_rate = max(0.0, min(1.0, float(m.group(2))))
                except ValueError:
                    pass
            else:  # check
                try:
                    check_interval = max(0, int(float(m.group(2))))
                except ValueError:
                    pass
        return EventEntry(
            start_at=span[0],
            end_at=span[1],
            label=label or "突発",
            state=canonical_state(state_raw, default="busy"),
            occupancy=occupancy,
            reply_rate=reply_rate,
            check_interval=check_interval,
        )
    return None


def layer_has_offline(entries: list[PlanEntry]) -> bool:
    """層に offline（就寝）エントリが1件でもあるかを返す。

    §3 の層失敗判定: offline が1件も取れないと深夜も OnTime になってしまうため、
    その層はパース失敗として前週優先のフォールバックへ落とす。
    """
    return any(e.state == "offline" for e in entries)


def entries_from_template(
    schedule: dict | None, week_start: date
) -> list[PlanEntry]:
    """テンプレ層（characters.availability_schedule）を対象週へ裸で展開する。

    §3 の最終フォールバック「availability_schedule を裸で変換」。state はブロック指定
    （既定 active）・隙間は OnTime。占有圧は初期値として offline=中(0.5)・
    それ以外=強(0.75) を割り当てる（固定予定 = 世界の拘束力高めの流儀・§4 表）。

    Args:
        schedule: {"mon": [{"from": "09:00", "to": "18:00", "label": "仕事",
            "state": "offline"?}], ...} 形式（Phase 0 で任意 state 欄を追加済み）。
        week_start: 対象週の月曜日の日付。

    Returns:
        PlanEntry のリスト。schedule が空・不正なら空リスト。
    """
    if not schedule or not isinstance(schedule, dict):
        return []
    day_keys = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    entries: list[PlanEntry] = []
    for day_idx, day_key in enumerate(day_keys):
        blocks = schedule.get(day_key) or []
        if not isinstance(blocks, list):
            continue
        base = datetime(week_start.year, week_start.month, week_start.day) + timedelta(
            days=day_idx
        )
        for block in blocks:
            if not isinstance(block, dict):
                continue
            span = _parse_time_range(
                f"{block.get('from', '')}-{block.get('to', '')}", base
            )
            if span is None:
                continue
            state = canonical_state(str(block.get("state") or ""), default="active")
            entries.append(
                PlanEntry(
                    start_at=span[0],
                    end_at=span[1],
                    label=str(block.get("label") or "予定"),
                    state=state,
                    occupancy=0.5 if state == "offline" else 0.75,
                )
            )
    return entries


def _pressure_label(occupancy: float) -> str:
    """内部値 0.0–1.0 を最も近い4段ラベルへ写す（プロンプト表示用）。"""
    return min(PRESSURE_VALUES.items(), key=lambda kv: abs(kv[1] - occupancy))[0]


def format_plan_lines(entries: list[PlanEntry]) -> str:
    """PlanEntry 群を [PLAN] 行テキストへ整形する（②へ①を見せる・ログ用）。

    翌日跨ぎは 24時超え表記へ戻す（例 火 01:00 終了の月曜就寝 → 25:00-31:30 ではなく、
    エントリ自身の開始曜日を基準に end を +24h 表記する）。

    Args:
        entries: 整形対象（start_at 昇順である必要はない — 与えられた順で出す）。

    Returns:
        1行1エントリのテキスト。空リストなら空文字列。
    """
    day_labels = ("月", "火", "水", "木", "金", "土", "日")
    lines = []
    for e in entries:
        day = day_labels[e.start_at.weekday()]
        start_txt = f"{e.start_at:%H:%M}"
        end_hour = e.end_at.hour + 24 * max(0, (e.end_at.date() - e.start_at.date()).days)
        end_txt = f"{end_hour:02d}:{e.end_at.minute:02d}"
        lines.append(
            f"[PLAN: {day} | {start_txt}-{end_txt} | {e.label} | {e.state} | "
            f"{_pressure_label(e.occupancy)}]"
        )
    return "\n".join(lines)
