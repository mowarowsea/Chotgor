"""計器 Tier 1 — インバリアント（機械・真偽確定）の巡回チェック。

即時系（fabrication_backstop / usual_scene_error / embedding_degraded）は
発生点のフックが lib/instrument_recorder.fire_alarm を直接呼ぶため、
本モジュールは **巡回系** のチェックだけを持つ:

    - night_batch_heartbeat : night.chronicle / night.forget が前日〜当日に発生済みか
    - usual_slot_completion : うつつ有効キャラの生活が前日丸ごと止まっていないか
    - chronicle_backlog     : chronicled_at IS NULL の3日超滞留がないか
    - envelope_integrity    : 源テーブルと封筒の件数突合（ID 突合はしない）

巡回は 05:00（Chronicle 03:00 → Forget 04:00 の後）に main.py のスケジューラから
呼ばれる。各チェックは真偽確定 — 発火したら幻想の穴が開いた証拠であり調査する。
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _check_night_batch_heartbeat(sqlite) -> list[dict]:
    """夜の営みの停止を見張る — ghost_model 持ちキャラに当日の night.* 封筒があるか。

    巡回（05:00）は Chronicle（03:00）・Forget（04:00）の後に走るため、
    正常なら当日 0 時以降に night.chronicle / night.forget 封筒が存在するはず。
    ghost_model 未設定キャラは夜間バッチの対象外なのでチェックしない。
    Forget は候補ゼロで skip される日があるため、night.forget は「chronicle があるのに
    forget が3日以上皆無」のときだけ発火する（誤検知の抑制）。

    Returns:
        発火したアラームの details リスト（発火なしなら空）。
    """
    fired: list[dict] = []
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    three_days_ago = datetime.now() - timedelta(days=3)
    for char in sqlite.list_characters():
        if not getattr(char, "ghost_model", None):
            continue
        chronicles = sqlite.list_timeline_events(
            char.id, since=today_start, event_type_prefixes=["night.chronicle"], limit=1,
        )
        if not chronicles:
            fired.append({
                "character": char.name,
                "missing": "night.chronicle",
                "note": "当日の Chronicle 封筒が無い（夜の営みが止まっている疑い）",
            })
            continue
        forgets = sqlite.list_timeline_events(
            char.id, since=three_days_ago, event_type_prefixes=["night.forget"], limit=1,
        )
        if not forgets:
            fired.append({
                "character": char.name,
                "missing": "night.forget",
                "note": "3日以上 Forget 封筒が無い（候補ゼロ skip では説明できない停止疑い）",
            })
    return fired


def _check_usual_slot_completion(sqlite) -> list[dict]:
    """生活の連続性を見張る — うつつ有効キャラの前日が丸ごと無音でないか。

    個々のスロットには正当なスキップ（対面中・日次上限）があるため、
    スロット単位ではなく「前日、スロットが設定されているのに scene.closed 封筒が
    1件も無い」ことを発火条件にする（誤検知の抑制を優先した粗い判定）。

    Returns:
        発火したアラームの details リスト。
    """
    fired: list[dict] = []
    now = datetime.now()
    yesterday_start = (now - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for scenario in sqlite.list_usual_scenarios():
        cfg = getattr(scenario, "usual_config", None) or {}
        if not cfg.get("enabled") or not (cfg.get("slots") or []):
            continue
        owner_id = getattr(scenario, "owner_character_id", None)
        if not owner_id:
            continue
        closed = sqlite.list_timeline_events(
            owner_id,
            since=yesterday_start,
            until=today_start,
            event_type_prefixes=["scene.closed"],
            limit=1,
        )
        if not closed:
            owner = sqlite.get_character(owner_id)
            fired.append({
                "character": getattr(owner, "name", owner_id),
                "slots": cfg.get("slots"),
                "note": "前日にうつつシーンが1件も完走していない（生活の中断疑い）",
            })
    return fired


def _check_chronicle_backlog(sqlite) -> list[dict]:
    """蒸留漏れを見張る — chronicled_at IS NULL の3日超滞留がないか。

    Returns:
        発火したアラームの details リスト。
    """
    counts = sqlite.count_chronicle_backlog(days=3)
    if counts["chat_messages"] > 0 or counts["usual_turns"] > 0:
        return [{
            "backlog": counts,
            "note": "3日を超えて未蒸留のまま滞留している発話がある",
        }]
    return []


def _check_envelope_integrity(sqlite) -> list[dict]:
    """正本性の破れを見張る — 源テーブルと封筒の件数突合。

    封筒は削除されない（retracted マークのみ）ため、正常なら常に
    封筒件数 >= 源件数。封筒件数 < 源件数 は dual-write 漏れの証拠。

    Returns:
        発火したアラームの details リスト。
    """
    fired: list[dict] = []
    for table, counts in sqlite.envelope_integrity_counts().items():
        if counts["envelope"] < counts["source"]:
            fired.append({
                "source_table": table,
                "source_count": counts["source"],
                "envelope_count": counts["envelope"],
                "note": "封筒件数が源テーブル件数を下回っている（dual-write 漏れ疑い）",
            })
    return fired


# 巡回チェックの一覧: invariant_id → チェック関数
_PATROL_CHECKS = {
    "night_batch_heartbeat": _check_night_batch_heartbeat,
    "usual_slot_completion": _check_usual_slot_completion,
    "chronicle_backlog": _check_chronicle_backlog,
    "envelope_integrity": _check_envelope_integrity,
}


def run_patrol_checks(sqlite) -> dict:
    """巡回インバリアントを全件実行し、違反をアラームとして記録する。

    1つのチェックの失敗（例外）が他のチェックを止めないよう、個別に隔離する。
    チェック機構自体の故障もまた異常なので、例外は instrument_error アラームにする。

    Args:
        sqlite: SQLiteStore。

    Returns:
        {invariant_id: 発火件数} の集計 dict（発火ゼロのチェックも 0 で含む）。
    """
    summary: dict[str, int] = {}
    for invariant_id, check in _PATROL_CHECKS.items():
        try:
            violations = check(sqlite)
        except Exception as e:
            logger.exception("巡回チェック自体が失敗 invariant=%s", invariant_id)
            sqlite.fire_alarm(
                "instrument_error",
                details={"invariant_id": invariant_id, "error": str(e)},
            )
            summary[invariant_id] = -1
            continue
        for details in violations:
            sqlite.fire_alarm(invariant_id, details=details)
        summary[invariant_id] = len(violations)
        if violations:
            logger.warning(
                "計器アラーム発火 invariant=%s count=%d", invariant_id, len(violations)
            )
    return summary
