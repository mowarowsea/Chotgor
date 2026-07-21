"""Spontaneous Initiative — キャラ自発リクエストの日次予算（単一定義点）。

cap の目的は「**ユーザが意図しないリクエストの暴走を防ぐ**」こと
（aliveness_plan.md §5.1「日次コストガードの定義」・2026-07-21 裁定）。
したがって消費するのは**ユーザ発話を起点としない** LLM 呼び出しだけである。

| 機構 | 起点 | 消費 |
|---|---|---|
| `reach_out`（うつつからのプッシュ） | キャラ自発 | する |
| 発話予約（speak_later）の発火 | キャラ自発 | する |
| 預かりの能動配達（services/gate/delivery.py） | ユーザ発話への返信 | **しない** |
| 行動権（調べもの等） | キャラ自発 | する（`action_*_daily_cap` の別枠） |

行動権が別枠なのは、問い合わせ／実行の2段で別々に絞る設計意図があるため
（1本化すると「調べもの1回で発話予約が枯れる」干渉が起きる）。将来まとめる場合も
この名前が受け皿になる。

旧名は `escrow_delivery_daily_cap` / `escrow_delivery_count_{date}`。escrow 配達が
消費対象から外れた時点で名前が実態と食い違うため、リネームした
（設定値の移行は `_migrate_rename_initiative_cap`）。
"""

from datetime import datetime

# 日次上限の settings キー（未設定・不正値のときは DEFAULT_DAILY_CAP）
CAP_SETTING_KEY = "spontaneous_initiative_daily_cap"
# 日次カウンタの settings キー接頭辞（`{prefix}{YYYY-MM-DD}`）
COUNT_KEY_PREFIX = "spontaneous_initiative_count_"
# 日次上限の既定値（キャラ自発リクエスト = LLM 1呼び出し/回）
DEFAULT_DAILY_CAP = 12


def count_key(now: datetime | None = None) -> str:
    """その日の消費カウンタの settings キーを返す（書き手と読み手で共有する唯一の定義）。"""
    return f"{COUNT_KEY_PREFIX}{(now or datetime.now()).date().isoformat()}"


def read_cap(sqlite_store) -> int:
    """日次上限を読む。

    cap=0 は「自発リクエストを止める」意味で有効な設定値なので、
    or フォールバックではなく明示的にパースする（未設定・不正値のみ既定へ倒す）。
    """
    try:
        return int(sqlite_store.get_setting(CAP_SETTING_KEY, ""))
    except (TypeError, ValueError):
        return DEFAULT_DAILY_CAP


def read_count(sqlite_store, now: datetime | None = None) -> int:
    """その日の消費数を読む。"""
    return int(sqlite_store.get_setting(count_key(now), "0") or 0)


def initiative_cap_reached(sqlite_store, now: datetime | None = None) -> bool:
    """キャラ自発リクエストの日次上限に達しているかを返す純関数。

    Args:
        sqlite_store: SQLiteStore。
        now: 基準時刻（テスト注入用）。

    Returns:
        上限到達なら True。cap=0 は「自発リクエストを止める」有効設定として True になる。
    """
    return read_count(sqlite_store, now) >= read_cap(sqlite_store)


def consume_initiative(sqlite_store, now: datetime | None = None) -> int:
    """消費カウンタを1つ進める（実行が確定した側が呼ぶ）。

    呼び出しは生成（LLM）より**前**に行う — 生成失敗で毎分叩き直さないため。

    Returns:
        消費後のその日の合計。
    """
    consumed = read_count(sqlite_store, now) + 1
    sqlite_store.set_setting(count_key(now), str(consumed))
    return consumed
