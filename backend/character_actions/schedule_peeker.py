"""SchedulePeeker — peek_at_schedule ツール（自分の予定のチラ見）の実装。

対面中の「二つの時計」（docs/planned/schedule_plan.md §7.6・2026-10-26 本人と相談）:
    - 対面中は予定も実時刻もターン注釈で push しない。代わりに本人が覗きたいときだけ
      引ける pull 型の窓としてこのツールを置く。
    - `at` に場の時刻を渡せば「場の時計で予定を読む」（土曜23時なら土曜23時の予定）。
      `at` を省けば外の時計（実時刻）とその周りの予定を返す。
    - 状態は持たない（場の時刻の宣言ではなく、都度引くだけ）。
    - 1on1・生活カレンダー有効キャラに常時露出（対面外でも使える。context_tools.py）。

ネタバレ防止（awareness.py と同じ絶対条件）:
    外の時刻より未来のエントリは、固定予定（template）と本人の上書き（source=haru）だけを
    見せる。伏せ枠（status=pending）は status フィルタで、発火前の突発は origin/source で除く。
    外の時刻で既に始まっている出来事は本人が体験した現実なので origin を問わない。
"""

import logging
import re
from datetime import datetime, timedelta

from backend.lib.time_awareness import japanese_weekday

logger = logging.getLogger(__name__)

# 前後に返す件数（本人の希望「前後2件ずつ」）
_NEIGHBOR_COUNT = 2

# 前後の予定を探す幅。週次の生活カレンダーなので数日見れば前後2件は埋まる。
_SEARCH_DAYS = 4

PEEK_AT_SCHEDULE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "at": {
            "type": "string",
            "description": (
                "覗きたい時刻（\"yyyy-MM-dd HH:mm\"）。大体でいい。"
                "省くと外の世界のいまの時刻で覗く。"
            ),
        },
    },
    "required": [],
}

PEEK_AT_SCHEDULE_TOOL_DESCRIPTION: str = (
    "自分の予定をチラ見する。at の時刻の予定と、その前後2件ずつが見える。"
    "時刻は大体でいい。対面中なら、場面がいま何時くらいかを at に渡せば、その時刻の予定が分かる。"
    "at を省くと、外の世界のいまの時刻と、その周りの予定が見える。"
    "まだ起きていない突発の出来事は見えない。"
)

PEEK_AT_SCHEDULE_TOOLS_HINT: str = """\
### 予定のチラ見 (peek_at_schedule)
自分の予定は、覗きたいときに `peek_at_schedule` で見られます。
- `at` に時刻（"yyyy-MM-dd HH:mm"、大体でいい）を渡すと、その時刻の予定と前後2件ずつが見えます。
- `at` を省くと、外の世界のいまの時刻と、その周りの予定が見えます。
- 対面中は外の時刻も予定も押しつけられません。場面の時刻を渡すか外を覗くかは、あなたが決めます。\
"""


def parse_peek_at(raw: str) -> datetime | None:
    """"yyyy-MM-dd HH:mm" を緩く解釈する（大体でいい、の約束どおり）。

    受ける揺れ: 区切りの "/"・"T"、秒付き、時刻なし（その日の 00:00）、
    時だけ（"2026-10-25 23"）。解釈できなければ None。
    """
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("/", "-").replace("T", " ")
    text = re.sub(r"\s+", " ", text)
    # 時だけ（"2026-10-25 23"）は分を補う
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2} \d{1,2}", text):
        text += ":00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _fmt_point(dt: datetime) -> str:
    return f"{dt:%Y-%m-%d}({japanese_weekday(dt)}) {dt:%H:%M}"


def _fmt_entry(entry) -> str:
    """エントリ1件を「10/25(土) 23:00〜10/26(日) 07:00 就寝（offline）」の形にする。"""
    start, end = entry.start_at, entry.end_at
    head = f"{start.month}/{start.day}({japanese_weekday(start)}) {start:%H:%M}"
    if end.date() == start.date():
        tail = f"{end:%H:%M}"
    else:
        tail = f"{end.month}/{end.day}({japanese_weekday(end)}) {end:%H:%M}"
    label = (getattr(entry, "label", None) or "予定").strip()
    state = getattr(entry, "state", "") or ""
    return f"{head}〜{tail} {label}" + (f"（{state}）" if state else "")


class SchedulePeeker:
    """peek_at_schedule の読み取りを担うクラス（書き込みはしない）。"""

    def __init__(self, character_id: str, sqlite_store) -> None:
        self.character_id = character_id
        self.sqlite_store = sqlite_store

    def _visible(self, entry, real_now: datetime) -> bool:
        """外の時刻から見て、本人に見せてよいエントリか（ネタバレ防止）。"""
        if entry.start_at <= real_now:
            return True
        return (
            getattr(entry, "origin", "") == "template"
            or getattr(entry, "source", "") == "haru"
        )

    def peek_at_schedule(self, at: str = "", now: datetime | None = None) -> str:
        """at の時刻の予定と前後2件ずつを返す。at 省略時は外の時刻で覗く。"""
        if self.sqlite_store is None:
            return "[peek_at_schedule error: この文脈では予定を覗けません]"
        char = self.sqlite_store.get_character(self.character_id)
        if char is None:
            return "[peek_at_schedule error: キャラクターが見つかりません]"
        if not int(getattr(char, "living_schedule_enabled", 0) or 0):
            return "[peek_at_schedule error: 生活カレンダーが有効になっていません]"

        real_now = now or datetime.now()
        raw = str(at or "").strip()
        if raw:
            point = parse_peek_at(raw)
            if point is None:
                return (
                    f"[peek_at_schedule error: at の形式が読めません: {raw!r}。"
                    "\"yyyy-MM-dd HH:mm\"（例 \"2026-10-25 23:00\"）で指定してください]"
                )
        else:
            point = real_now

        entries = [
            e for e in self.sqlite_store.list_schedule_entries(
                self.character_id,
                since=point - timedelta(days=_SEARCH_DAYS),
                until=point + timedelta(days=_SEARCH_DAYS),
                statuses=["planned", "done"],
            )
            if self._visible(e, real_now)
        ]
        # その時刻を含む予定は占有圧最大が現実（availability と同じ解決）
        covering = [e for e in entries if e.start_at <= point < e.end_at]
        current = max(covering, key=lambda e: float(e.occupancy or 0.0)) if covering else None
        before = [e for e in entries if e.end_at <= point][-_NEIGHBOR_COUNT:]
        after = [e for e in entries if e.start_at > point][:_NEIGHBOR_COUNT]

        lines: list[str] = []
        if raw:
            lines.append(f"{_fmt_point(point)} ごろの予定:")
        else:
            lines.append(f"外の世界はいま {_fmt_point(real_now)}。")
            lines.append("その周りの予定:")
        lines += [f"- 前: {_fmt_entry(e)}" for e in before]
        lines.append(f"- その時刻: {_fmt_entry(current)}" if current else "- その時刻: 予定なし")
        lines += [f"- 後: {_fmt_entry(e)}" for e in after]
        logger.info(
            "peek_at_schedule char=%s at=%s point=%s",
            self.character_id, raw or "(外の時刻)", point.isoformat(timespec="minutes"),
        )
        return "\n".join(lines)
