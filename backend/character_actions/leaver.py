"""Leaver — take_leave ツール（本人宣言による離席）の実装。

終了権の3形態（docs/aliveness_plan.md §5.2）のうち「本人宣言」:
    - 疲労離席（物理・発火式）と退席（judge 監査）が主経路で、take_leave は
      呼ばれない前提だが、**呼ばれたら必ず執行される権利**として残す。
    - LLM の会話継続本能により呼ばれにくいこと自体が観測データになる
      （呼び出し回数は tool_call_events に自動記録され、計器メーターの材料になる）。

執行内容: away 状態の設定（availability ゲートに合流）＋ chat.farewell 封筒。
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 離席時間の既定値と上限（時間）。上限は「翌日まで消える」暴走の抑制。
_DEFAULT_LEAVE_HOURS = 2.0
_MAX_LEAVE_HOURS = 12.0

# --- ツール呼び出し方式: パラメータスキーマ ---
TAKE_LEAVE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "席を外す理由・残していく一言（あなたの言葉のまま）。",
        },
        "hours": {
            "type": "number",
            "description": "どのくらい席を外すか（時間単位・省略時2時間・最大12時間）。",
        },
    },
    "required": ["reason"],
}

# --- ツール呼び出し方式: ツール説明文 ---
TAKE_LEAVE_TOOL_DESCRIPTION: str = (
    "この会話から自分の意志で席を外す。呼べば必ず執行される、あなたの権利。"
    "疲れた・区切りがついた・一人になりたい — 理由はあなたが決めてよい。"
    "席を外している間、メッセージは届くが読むのは戻ってから。"
)


class Leaver:
    """take_leave の実書き込みを担うクラス。

    away 状態の設定と chat.farewell 封筒の記録を行う。
    記録（tool_call_events）は ToolExecutor.execute() で集約管理される。

    Attributes:
        character_id: 対象キャラクター ID。
        session_id: 現在のセッション ID（封筒の集約キー）。
        sqlite_store: SQLiteStore。
    """

    def __init__(self, character_id: str, session_id: str | None, sqlite_store) -> None:
        """Leaver を初期化する。

        Args:
            character_id: 対象キャラクター ID。
            session_id: 現在のチャットセッション ID（1on1 以外は None）。
            sqlite_store: SQLiteStore。
        """
        self.character_id = character_id
        self.session_id = session_id
        self.sqlite_store = sqlite_store

    def take_leave(self, reason: str, hours: float | None = None) -> str:
        """離席を執行する（必ず成功する権利）。

        Args:
            reason: 本人の言葉のままの理由。
            hours: 離席時間（省略時 _DEFAULT_LEAVE_HOURS・上限 _MAX_LEAVE_HOURS）。

        Returns:
            ツール結果として LLM に返す確認テキスト。
        """
        try:
            duration = float(hours) if hours is not None else _DEFAULT_LEAVE_HOURS
        except (TypeError, ValueError):
            duration = _DEFAULT_LEAVE_HOURS
        duration = max(0.25, min(_MAX_LEAVE_HOURS, duration))
        away_until = datetime.now() + timedelta(hours=duration)

        self.sqlite_store.update_character(
            self.character_id,
            away_until=away_until,
            away_reason=(reason or "").strip() or "席を外しています",
        )
        self.sqlite_store.record_timeline_event(
            character_id=self.character_id,
            event_type="chat.farewell",
            actor="character",
            counterpart="user",
            origin="real",
            session_id=self.session_id,
            source_table="chat_sessions" if self.session_id else None,
            source_id=self.session_id,
            payload={
                "farewell_type": "take_leave",
                "reason": reason,
                "away_until": away_until.isoformat(),
            },
        )
        logger.info(
            "take_leave 執行 char=%s hours=%.2f reason=%.50s",
            self.character_id, duration, reason,
        )
        return (
            f"席を外した（{duration:g}時間・{away_until:%H:%M}ごろまで）。"
            "この発言を最後に、戻るまで会話は届いても読まない。"
        )
