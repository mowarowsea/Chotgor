"""設定 UI — ダッシュボード（index）ページ。

LLM 使用量（リクエスト数・トークン In/Cache/Out・キャッシュ率）の日次・週次・月次集計と、
Characters / Scenarios / Logs 等への入口リンクを表示する。

キャッシュ率は prompt_cache_plan.md C案（計測の見える化）の表示値。
プロンプトキャッシュが効いているかを運用中に数字で判断するためのもので、
集計行（期間合計・プロバイダー内訳・直近イベント）すべてに付ける。
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from backend.api.ui.common import get_templates

router = APIRouter(prefix="/ui", tags=["ui"])


def cache_hit_ratio(input_tokens: int, cache_tokens: int) -> float | None:
    """キャッシュ率 = キャッシュ読出 ÷ 入力全長（0.0〜1.0）を返す。

    入力全長 = input（キャッシュ外＋キャッシュ作成）＋ cache（キャッシュ読出）。
    usage_store の表示規約（3列が互いに素で合計＝入力全長）に乗る。
    入力が 0 のとき（記録なし・出力のみ）は None（表示側で「–」にする）。
    """
    total = int(input_tokens or 0) + int(cache_tokens or 0)
    if total <= 0:
        return None
    return int(cache_tokens or 0) / total


def _with_cache_ratio(row: dict) -> dict:
    """集計行へ `cache_ratio` を書き足して同じ dict を返す。"""
    row["cache_ratio"] = cache_hit_ratio(row.get("input_tokens", 0), row.get("cache_tokens", 0))
    return row


def _merge_period_rows(rows: list[dict], key: str) -> list[dict]:
    """日次/週次のプロバイダー別行を期間単位にまとめ、プロバイダー内訳を抱き合わせる。

    Args:
        rows: get_usage_daily / get_usage_weekly の結果（期間×プロバイダー別）。
        key: 期間ラベルのキー名（"day" または "week"）。

    Returns:
        [{"label": str, "requests": int, "input_tokens": int, "cache_tokens": int,
          "output_tokens": int, "cost_usd": float, "cache_ratio": float | None,
          "providers": [row, ...]}]
        を新しい期間順で返す。期間合計・内訳行の双方に `cache_ratio` を付ける。
    """
    merged: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        label = r[key]
        if label not in merged:
            merged[label] = {
                "label": label,
                "requests": 0,
                "input_tokens": 0,
                "cache_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "providers": [],
            }
            order.append(label)
        m = merged[label]
        m["requests"] += r["requests"]
        m["input_tokens"] += r["input_tokens"]
        m["cache_tokens"] += r["cache_tokens"]
        m["output_tokens"] += r["output_tokens"]
        m["cost_usd"] += r["cost_usd"]
        m["providers"].append(_with_cache_ratio(r))
    return [_with_cache_ratio(merged[label]) for label in order]


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """ダッシュボード（index）。使用量サマリーと各ページへのリンクを表示する。"""
    sqlite = request.app.state.sqlite

    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)
    # 今週 = 月曜起点（SQLite 集計の %W と揃える）
    week_start = today_start - timedelta(days=today_start.weekday())
    # 今月 = 月初（SQLite 集計の %Y-%m と揃える）
    month_start = datetime(now.year, now.month, 1)

    usage_today = _with_cache_ratio(sqlite.get_usage_totals_since(today_start))
    usage_week = _with_cache_ratio(sqlite.get_usage_totals_since(week_start))
    usage_month = _with_cache_ratio(sqlite.get_usage_totals_since(month_start))
    daily = _merge_period_rows(sqlite.get_usage_daily(days=14), "day")
    weekly = _merge_period_rows(sqlite.get_usage_weekly(weeks=8), "week")
    monthly = _merge_period_rows(sqlite.get_usage_monthly(months=6), "month")
    recent_events = [_with_cache_ratio(e) for e in sqlite.get_usage_recent_events(limit=30)]

    characters = sqlite.list_characters()
    scenarios = sqlite.list_scenarios()

    return get_templates().TemplateResponse(
        request,
        "dashboard.html",
        {
            "usage_today": usage_today,
            "usage_week": usage_week,
            "usage_month": usage_month,
            "daily": daily,
            "weekly": weekly,
            "monthly": monthly,
            "recent_events": recent_events,
            "character_count": len(characters),
            "scenario_count": len(scenarios),
        },
    )
