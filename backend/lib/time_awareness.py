"""時刻認識パラメータの計算ユーティリティ。

1on1チャット・OpenAI互換 API・グループチャットで共通する
「現在時刻」「前回インタラクションからの経過時間」の計算ロジックを集約する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .utils import format_time_delta


@dataclass
class TimeAwareness:
    """時刻認識パラメータのまとまり。"""

    enabled: bool
    current_time_str: str
    time_since_last_interaction: str


def compute_time_awareness(
    settings: dict,
    character_id: str,
    sqlite,
    now: datetime | None = None,
) -> TimeAwareness:
    """時刻認識パラメータを計算して返す。

    enable_time_awareness 設定が "false" の場合は enabled=False で空文字列を返す。
    計算後の last_interaction 更新は呼び出し元が行うこと（副作用分離のため）。

    Args:
        settings: get_all_settings() の結果辞書。
        character_id: キャラクターID（last_interaction_{id} キーの参照に使用）。
        sqlite: SQLiteStore インスタンス（last_interaction 値の読み取りに使用）。
        now: 基準時刻。省略時は datetime.now() を使用。

    Returns:
        TimeAwareness データクラス。
    """
    enabled = settings.get("enable_time_awareness", "true") == "true"
    if not enabled:
        return TimeAwareness(enabled=False, current_time_str="", time_since_last_interaction="")

    if now is None:
        now = datetime.now()

    current_time_str = now.isoformat(timespec="seconds")
    time_since_last = ""

    last_str = settings.get(f"last_interaction_{character_id}")
    if last_str:
        try:
            last_dt = datetime.fromisoformat(last_str)
            time_since_last = format_time_delta(now - last_dt)
        except Exception:
            pass

    return TimeAwareness(
        enabled=True,
        current_time_str=current_time_str,
        time_since_last_interaction=time_since_last,
    )
