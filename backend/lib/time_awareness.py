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


# 曜日の日本語表記（datetime.weekday() は月=0..日=6）。
_WEEKDAY_JA = ("月", "火", "水", "木", "金", "土", "日")


def japanese_weekday(now: datetime) -> str:
    """曜日を日本語 1 文字（月〜日）で返す。"""
    return _WEEKDAY_JA[now.weekday()]


def japanese_time_of_day(now: datetime) -> str:
    """時刻を日本語の時間帯ラベル（早朝/朝/昼/午後/夕方/夜/深夜）に変換する。"""
    h = now.hour
    if 5 <= h < 7:
        return "早朝"
    if 7 <= h < 11:
        return "朝"
    if 11 <= h < 14:
        return "昼"
    if 14 <= h < 17:
        return "午後"
    if 17 <= h < 19:
        return "夕方"
    if 19 <= h < 23:
        return "夜"
    return "深夜"


def japanese_season(now: datetime) -> str:
    """月から日本語の季節（春/夏/秋/冬）を返す。"""
    m = now.month
    if 3 <= m <= 5:
        return "春"
    if 6 <= m <= 8:
        return "夏"
    if 9 <= m <= 11:
        return "秋"
    return "冬"


def format_time_context(now: datetime | None = None) -> str:
    """GM プロンプト向けに、現在の日付・曜日・時間帯・季節を日本語 1 文にまとめて返す。

    うつつ（Usual Days）の GM が「いつの・どんな時間か」を把握するための外的フレーム。
    1on1 の時刻ブロックとも語彙を共有する（曜日・時間帯・季節の日本語算出）。

    例: "いまは 2026年6月14日（日）の夕方。季節は夏。"

    Args:
        now: 基準時刻。省略時は datetime.now()。

    Returns:
        日本語の時間文脈 1 文。
    """
    if now is None:
        now = datetime.now()
    return (
        f"いまは {now.year}年{now.month}月{now.day}日"
        f"（{japanese_weekday(now)}）の{japanese_time_of_day(now)}。"
        f"季節は{japanese_season(now)}。"
    )


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
