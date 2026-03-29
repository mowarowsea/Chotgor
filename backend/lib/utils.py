"""バックエンド共通ユーティリティ。"""

from datetime import timedelta


def format_time_delta(diff: timedelta) -> str:
    """timedeltaを日本語の経過時間文字列に変換する。

    例:
        数分以内 / 約 15 分 / 約 2.5 時間 / 約 3 日
    """
    hours = diff.total_seconds() / 3600
    if hours < 1:
        m = int(hours * 60)
        return f"約 {m} 分" if m > 0 else "数分以内"
    elif hours < 24:
        return f"約 {hours:.1f} 時間"
    else:
        days = int(hours / 24)
        return f"約 {days} 日"
