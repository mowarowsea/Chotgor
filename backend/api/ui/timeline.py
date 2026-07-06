"""設定 UI — タイムライン閲覧ページ（めぐり Phase 7・ユーザダイヤルの適用面）。

キャラクターのタイムライン正本を **user_ui 観測者の投影**（ダイヤル適用済み）で
表示する。ダイヤルはここで切り替える（キャラ単位・v1 は手動）:

    0 = 全開（開発期） / 1 = 生活の秘匿 / 2 = 内面の秘匿 / 3 = 最終形（チャットのみ）

段階を上げるほどユーザから見えるものが減る — 非対称性（秘密）の成立。
計器（/ui/instruments）はダイヤル非依存で常時稼働し、静音期間の数字が
「窓を閉じてよい」確信を支える。
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from backend.api.ui.common import get_templates

router = APIRouter(prefix="/ui", tags=["ui"])

# イベント種別の表示ラベル
_EVENT_LABELS = {
    "chat.message": "会話",
    "chat.farewell": "退席",
    "scene.turn": "生活のひとコマ",
    "scene.closed": "シーン完走",
    "night.chronicle": "夢（棚卸し）",
    "night.forget": "夢（忘却）",
    "memory.inscribed": "記憶を刻んだ",
    "memory.forgotten": "記憶を手放した",
    "memory.carved": "内的叙述を彫った",
    "memory.recalled": "思い出した",
    "intent.created": "意図が生まれた",
    "intent.fulfilled": "意図が満ちた",
    "intent.expired": "意図を手放した",
    "intent.soured": "意図が不満になった",
    "action.performed": "行動した",
}

# ダイヤル段階の表示名
_DIAL_LABELS = {
    0: "0 — 全開（開発期）",
    1: "1 — 生活の秘匿",
    2: "2 — 内面の秘匿",
    3: "3 — 最終形（チャットのみ）",
}


@router.get("/timeline", response_class=HTMLResponse)
async def timeline_page(request: Request, character_id: str = "", days: int = 7):
    """タイムライン閲覧ページ。user_ui 投影（キャラのダイヤル適用済み）を表示する。"""
    from backend.services.timeline import project

    sqlite = request.app.state.sqlite
    characters = sqlite.list_characters()
    selected = None
    if character_id:
        selected = sqlite.get_character(character_id)
    elif characters:
        selected = characters[0]

    events = []
    dial = 0
    if selected is not None:
        dial = int(getattr(selected, "timeline_dial", 0) or 0)
        days = max(1, min(90, days))
        projected = project(
            character_id=selected.id,
            observer="user_ui",
            sqlite=sqlite,
            since=datetime.now() - timedelta(days=days),
            user_dial=dial,
        )
        for ev in reversed(projected):  # 新しい順で表示
            events.append({
                "occurred_at": ev.occurred_at,
                "label": _EVENT_LABELS.get(ev.event_type, ev.event_type),
                "event_type": ev.event_type,
                "actor": ev.actor or "",
                "origin": ev.origin,
                "disclosure": ev.disclosure,
                "content": ev.content or "",
            })

    return get_templates().TemplateResponse(
        request,
        "timeline.html",
        {
            "characters": characters,
            "selected": selected,
            "events": events,
            "dial": dial,
            "dial_labels": _DIAL_LABELS,
            "days": days,
        },
    )


@router.post("/timeline/{character_id}/dial")
async def set_dial(request: Request, character_id: str):
    """ユーザダイヤルを切り替えてタイムラインページへ戻る（守護者の操作）。"""
    form = await request.form()
    try:
        dial = max(0, min(3, int(form.get("dial", 0))))
    except (TypeError, ValueError):
        dial = 0
    request.app.state.sqlite.update_character(character_id, timeline_dial=dial)
    return RedirectResponse(
        url=f"/ui/timeline?character_id={character_id}", status_code=303,
    )
