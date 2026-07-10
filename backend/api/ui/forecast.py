"""設定 UI — 予報パネル（forecast）ページ。

無人機構（うつつ・行動権・突発・配達）の「確定予定・無風外挿・診断・実績」を
一枚で見せる読み取り専用パネル（docs/planned/forecast_panel_plan.md）。
計算はすべて services/timeline/forecast.build_forecast（決定論純関数・LLM 不使用）。

計器（instruments）が「事故が起きていないか」の監査なのに対し、こちらは
「これから何が起きる見込みか・沈黙は正常か」の観測。どちらも監査者の窓であり、
キャラクターの世界には一切現れない。
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from backend.api.ui.common import get_templates

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/forecast", response_class=HTMLResponse)
async def forecast_panel(request: Request, character_id: str = ""):
    """予報パネル。キャラ選択＋予報データ一式をテンプレートへ渡す。

    Args:
        request: FastAPI リクエスト（app.state.sqlite を使う）。
        character_id: 対象キャラクター ID。未指定なら最初のキャラ。
    """
    from backend.services.timeline.forecast import build_forecast

    sqlite = request.app.state.sqlite
    characters = sqlite.list_characters()
    selected = character_id or (characters[0].id if characters else "")

    forecast = build_forecast(sqlite, selected) if selected else {"error": "キャラクターがいません"}

    return get_templates().TemplateResponse(
        request,
        "forecast.html",
        {
            "characters": characters,
            "selected_id": selected,
            "forecast": forecast,
        },
    )
