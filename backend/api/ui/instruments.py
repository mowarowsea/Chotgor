"""設定 UI — 計器パネル（instruments）ページ。

めぐり（巡り / Aliveness）の計器層（docs/planned/aliveness_plan.md §3）の表示口。
3層別（Tier 1 アラーム / Tier 2 スメル / Tier 3 判定巡回）＋静音期間（無事故N日）＋
メーターの最新スナップショットを表示する。計器はダイヤル非依存で常時稼働し、
このパネルの数字が「窓を閉じてよい」というユーザの確信を支える。
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from backend.api.ui.common import get_templates

router = APIRouter(prefix="/ui", tags=["ui"])

# Tier 1 インバリアントの表示名（パネルの見出し用）
_INVARIANT_LABELS = {
    "fabrication_backstop": "GMのユーザ捏造（バックストップ発火）",
    "usual_scene_error": "生活の中断（うつつシーンエラー）",
    "embedding_degraded": "記憶の縮退（embedding 障害）",
    "night_batch_heartbeat": "夜の営みの停止",
    "usual_slot_completion": "生活の連続性",
    "scheduler_heartbeat": "無人ループの停止（heartbeat 鮮度）",
    "chronicle_backlog": "蒸留漏れ",
    "envelope_integrity": "正本性の破れ（封筒突合）",
    "judgement_patrol": "判定巡回（Tier 3）",
    "instrument_error": "計器自体の故障",
}

# メーターの表示名
_METER_LABELS = {
    "inner_narrative_len": "内的叙述の長さ",
    "self_history_len": "経緯テキストの長さ",
    "relationship_state_len": "関係テキストの長さ",
    "wm_thread_count": "WMスレッド数",
    "memory_count": "保存記憶件数",
}


@router.get("/instruments", response_class=HTMLResponse)
async def instruments_panel(request: Request):
    """計器パネル。静音期間・3層別アラーム・メーターの最新値を表示する。"""
    sqlite = request.app.state.sqlite
    since_30d = datetime.now() - timedelta(days=30)

    alarms = sqlite.list_alarms(severity="alarm", since=since_30d, limit=100)
    smells = sqlite.list_alarms(severity="smell", since=since_30d, limit=100)
    quiet_days = sqlite.quiet_period_days()
    unacknowledged = sum(1 for a in alarms if a.acknowledged_at is None)

    # メーター: キャラ×メーターの最新値だけを抽出する（一覧は新しい順で来る）
    latest_meters: dict[tuple, object] = {}
    for snap in sqlite.list_meter_snapshots(since=since_30d, limit=500):
        key = (snap.character_id, snap.meter_id)
        if key not in latest_meters:
            latest_meters[key] = snap
    char_names = {c.id: c.name for c in sqlite.list_characters()}
    meter_rows = []
    for (char_id, meter_id), snap in sorted(
        latest_meters.items(),
        key=lambda kv: (char_names.get(kv[0][0], ""), kv[0][1]),
    ):
        meter_rows.append({
            "character": char_names.get(char_id, char_id or "（全体）"),
            "meter": _METER_LABELS.get(meter_id, meter_id),
            "meter_id": meter_id,
            "value": snap.value,
            "occurred_at": snap.occurred_at,
        })

    last_patrol = sqlite.get_setting("instruments_last_run_date", "")
    judge_preset_id = sqlite.get_setting("instruments_judge_preset_id", "")

    return get_templates().TemplateResponse(
        request,
        "instruments.html",
        {
            "quiet_days": quiet_days,
            "unacknowledged": unacknowledged,
            "alarms": alarms,
            "smells": smells,
            "meter_rows": meter_rows,
            "invariant_labels": _INVARIANT_LABELS,
            "last_patrol": last_patrol,
            "judge_preset_id": judge_preset_id,
        },
    )


@router.post("/instruments/alarms/{alarm_id}/ack")
async def acknowledge_alarm(request: Request, alarm_id: int):
    """アラームを確認済みにしてパネルへ戻る。"""
    request.app.state.sqlite.acknowledge_alarm(alarm_id)
    return RedirectResponse(url="/ui/instruments", status_code=303)
