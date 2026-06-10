"""設定 UI — ダッシュボード（index）ページ。

LLM 使用量（リクエスト数・トークン In/Out）の日次・週次集計と、
Characters / Scenarios / Logs 等への入口リンクを表示する。
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from backend.api.ui.common import get_templates

router = APIRouter(prefix="/ui", tags=["ui"])


def _merge_period_rows(rows: list[dict], key: str) -> list[dict]:
    """日次/週次のプロバイダー別行を期間単位にまとめ、プロバイダー内訳を抱き合わせる。

    Args:
        rows: get_usage_daily / get_usage_weekly の結果（期間×プロバイダー別）。
        key: 期間ラベルのキー名（"day" または "week"）。

    Returns:
        [{"label": str, "requests": int, "input_tokens": int, "output_tokens": int,
          "cost_usd": float, "providers": [row, ...]}] を新しい期間順で返す。
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
                "output_tokens": 0,
                "cost_usd": 0.0,
                "providers": [],
            }
            order.append(label)
        m = merged[label]
        m["requests"] += r["requests"]
        m["input_tokens"] += r["input_tokens"]
        m["output_tokens"] += r["output_tokens"]
        m["cost_usd"] += r["cost_usd"]
        m["providers"].append(r)
    return [merged[label] for label in order]


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """ダッシュボード（index）。使用量サマリーと各ページへのリンクを表示する。"""
    sqlite = request.app.state.sqlite

    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)
    # 今週 = 月曜起点（SQLite 集計の %W と揃える）
    week_start = today_start - timedelta(days=today_start.weekday())

    usage_today = sqlite.get_usage_totals_since(today_start)
    usage_week = sqlite.get_usage_totals_since(week_start)
    daily = _merge_period_rows(sqlite.get_usage_daily(days=14), "day")
    weekly = _merge_period_rows(sqlite.get_usage_weekly(weeks=8), "week")
    recent_events = sqlite.get_usage_recent_events(limit=30)

    characters = sqlite.list_characters()
    scenarios = sqlite.list_scenarios()

    return get_templates().TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "usage_today": usage_today,
            "usage_week": usage_week,
            "daily": daily,
            "weekly": weekly,
            "recent_events": recent_events,
            "character_count": len(characters),
            "scenario_count": len(scenarios),
        },
    )
