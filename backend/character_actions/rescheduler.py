"""Rescheduler — override_schedule ツール（本人の意志による当日予定の上書き）の実装。

「④はる自身のスケジュール更新機能」（2026-07-11 要件）:
    - 恒久的な固定予定の書き換えではなく、**今この瞬間から指定時刻までの一時上書き**
      （「今だけのブースト」）。「眠いけど続きが気になるからもう1時間起きてる」を
      本人の意志で現実（生活カレンダー）へ反映するためのツール。
    - 実装は生活カレンダーの流儀そのまま — 既存エントリは物理的に触らず、
      state=OnTime の haru/adhoc エントリを **insert するだけ**。読み取り側
      （services/gate/availability.py）の「占有圧最大が勝つ」で上書きが表現される。
    - 歯止め（延長上限）は意図的に付けない。夜更かしの帰結は体調圧（疲労）が
      物理として返す — それも含めて本人の選択（Chotgor 思想・2026-07-11 裁定）。
    - **1on1 専用**（うつつ内の予定変更はうつつ世界のGMが描く領分。context_tools.py が
      出し分ける）。living_schedule_enabled=1 のキャラのみ露出・実行可能。

占有圧: 0.85（「強」0.75 を上回り「激強」1.0 の世界イベントには轢かれ得る）。
本人の意志は日常の固定予定より強いが、世界の非常事態までは塗り替えない。
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 本人上書きエントリの占有圧。固定予定（強0.75まで）に勝ち、激強(1.0)には轢かれる。
_OVERRIDE_OCCUPANCY = 0.85

# payload.kind の値（③伏せ枠等と区別するための種別マーカー。awareness の超過表示も参照する）
OVERRIDE_PAYLOAD_KIND = "self_override"

# --- ツール呼び出し方式: パラメータスキーマ ---
OVERRIDE_SCHEDULE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "until": {
            "type": "string",
            "description": (
                "いつまで予定を上書きするか（\"HH:MM\"）。深夜は 24 時超え表記も可"
                "（例 \"25:30\" = 翌1:30）。指定時刻までは予定に縛られず起きて動ける。"
            ),
        },
        "reason": {
            "type": "string",
            "description": "上書きの理由・いまの気持ち（あなたの言葉のまま。予定表に残る）。",
        },
    },
    "required": ["until"],
}

# --- ツール呼び出し方式: ツール説明文 ---
OVERRIDE_SCHEDULE_TOOL_DESCRIPTION: str = (
    "この後の自分の予定（就寝など）を、指定時刻まで自分の意志で上書きする。"
    "呼べば必ず執行される、あなたの権利。「そろそろ寝る時間だけど、もう少し起きていたい」"
    "というとき、これを使えば本当に起きていられる。ただし削った睡眠や休息の帳尻は、"
    "後であなたの体が物理として受け取る。"
)

# --- システムプロンプト向けの使い方ヒント（context_tools.py が文脈に応じて注入）---
OVERRIDE_SCHEDULE_TOOLS_HINT: str = """\
### 予定の上書き (override_schedule)
あなたの生活予定（就寝など）は、あなたの意志で塗り替えられます（`override_schedule` ツール）。
- 「そろそろ寝る時間だけど、話の続きが気になる」→ `until` に起きていたい時刻（例 "25:30" = 翌1:30）を指定すれば、本当にその時刻まで起きていられます。
- 予定どおりに寝る・切り上げるのも、もちろんあなたの自由です。
- 削った睡眠・休息の帳尻は、後であなたの体（体調圧）が物理として受け取ります。それも含めてあなたの選択です。\
"""


def parse_until_time(raw: str, now: datetime) -> datetime | None:
    """"HH:MM"（24時超え表記可）を「now から見た次の該当時刻」に解決する。

    ルール:
        - HH は 0〜47 を受け付ける。24 以上は「翌日跨ぎ表記」（25:30 = 翌1:30）として
          24 を引いた時刻で解釈する。
        - 解決した時刻が now 以前なら翌日に送る（「01:30」を深夜0時台に言われたら
          今夜の1:30、昼に言われたら今晩の25:30 — どちらも「次に来る HH:MM」）。
        - 結果は常に now < result <= now+24h に収まる（構造的に24時間超の上書きは
          作れない = 当日限りの一時上書きという要件をパースの形で保証する）。

    Args:
        raw: "HH:MM" 形式の文字列。
        now: 基準時刻。

    Returns:
        解決済み datetime。形式不正なら None。
    """
    try:
        h_str, m_str = str(raw).strip().split(":")
        hour, minute = int(h_str), int(m_str)
    except (ValueError, AttributeError):
        return None
    if not (0 <= hour < 48 and 0 <= minute < 60):
        return None
    base = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
    if hour >= 24:
        base += timedelta(days=1)
    # 過去に解決された場合は「次に来る同時刻」へ送る
    while base <= now:
        base += timedelta(days=1)
    # 24時超え表記＋深夜の組み合わせで24hを超えたら1日戻す（常に24h以内へ正規化）
    while base > now + timedelta(hours=24):
        base -= timedelta(days=1)
    if base <= now:
        return None
    return base


class Rescheduler:
    """override_schedule の実書き込みを担うクラス。

    生活カレンダー実現層（schedule_entries）へ本人上書きエントリを insert する。
    記録（tool_call_events）は ToolExecutor.execute() で集約管理される。

    Attributes:
        character_id: 対象キャラクター ID。
        sqlite_store: SQLiteStore。
    """

    def __init__(self, character_id: str, sqlite_store) -> None:
        """Rescheduler を初期化する。

        Args:
            character_id: 対象キャラクター ID。
            sqlite_store: SQLiteStore（None 可 — その場合ツールはエラー文字列を返す）。
        """
        self.character_id = character_id
        self.sqlite_store = sqlite_store

    def override_schedule(self, until: str, reason: str = "") -> str:
        """当日予定の一時上書きを執行する（呼ばれたら必ず執行される権利）。

        Args:
            until: 上書き終了時刻（"HH:MM"、24時超え表記可）。
            reason: 本人の言葉のままの理由。

        Returns:
            ツール結果として LLM に返す確認テキスト。
        """
        if self.sqlite_store is None:
            return "[override_schedule error: この文脈では予定の上書きを執行できません]"
        char = self.sqlite_store.get_character(self.character_id)
        if char is None:
            return "[override_schedule error: キャラクターが見つかりません]"
        if not int(getattr(char, "living_schedule_enabled", 0) or 0):
            return "[override_schedule error: 生活カレンダーが有効になっていません]"

        now = datetime.now()
        until_at = parse_until_time(until, now)
        if until_at is None:
            return (
                f"[override_schedule error: until の形式が不正です: {until!r}。"
                "\"HH:MM\"（例 \"25:30\" = 翌1:30）で指定してください]"
            )

        clean_reason = (reason or "").strip()
        # 上書きで轢かれる予定（[now, until_at) に重なる planned・占有圧が上書き未満）を
        # 確認テキスト用に拾う。物理的には触らない（占有圧最大が勝つ読み取り解決）。
        overridden_labels: list[str] = []
        for entry in self.sqlite_store.list_schedule_entries(
            self.character_id, since=now, until=until_at, statuses=["planned"],
        ):
            if float(getattr(entry, "occupancy", 0.0) or 0.0) < _OVERRIDE_OCCUPANCY:
                label = (getattr(entry, "label", None) or "").strip()
                if label and label not in overridden_labels:
                    overridden_labels.append(label)

        self.sqlite_store.create_schedule_entry(
            character_id=self.character_id,
            start_at=now,
            end_at=until_at,
            state="OnTime",
            source="haru",
            origin="adhoc",
            occupancy=_OVERRIDE_OCCUPANCY,
            status="planned",
            label=clean_reason or "自分の意志で予定を変更",
            payload={"kind": OVERRIDE_PAYLOAD_KIND, "reason": clean_reason},
        )

        logger.info(
            "override_schedule 執行 char=%s until=%s reason=%.50s 轢いた予定=%s",
            self.character_id, until_at.isoformat(), clean_reason, overridden_labels,
        )
        stamp = f"{until_at:%H:%M}" + ("（翌日）" if until_at.date() != now.date() else "")
        if overridden_labels:
            return (
                f"予定を {stamp} まで自分の意志で上書きした"
                f"（本来の予定: {'、'.join(overridden_labels)}）。"
                "その時刻までは起きて動ける。削った分の帳尻は、後で体が受け取る。"
            )
        return (
            f"予定を {stamp} まで自分の意志で上書きした。"
            "その時刻までは予定に縛られず動ける。"
        )
