"""計器アラームの記録リレー — どこからでも安全にアラームを発火するための薄い口。

usage_recorder / tool_event_recorder と同じ思想のモジュールレベル・シングルトン。
main.py の起動時に set_store() で SQLiteStore を注入し、以降は
backend 内のどこからでも fire_alarm() を呼べる（store 未設定・例外時は no-op）。

即時系インバリアントのフック（fabrication_backstop / usual_scene_error /
embedding_degraded）と Tier 2 スメル記録がこれを使う。
計器は監査者であり、記録の失敗が本処理（チャット・シーン進行）を
止めてはならないため、例外はすべて握り潰してログに残すだけにする。
"""

import logging

logger = logging.getLogger(__name__)

# モジュールレベルの SQLiteStore 参照。main.py の lifespan で注入される。
_store = None


def set_store(store) -> None:
    """SQLiteStore を注入してアラーム記録を有効化する。

    Args:
        store: SQLiteStore インスタンス（fire_alarm を持つ）。
    """
    global _store
    _store = store


def fire_alarm(
    invariant_id: str,
    *,
    severity: str = "alarm",
    details: dict | None = None,
) -> None:
    """アラームを発火する（store 未設定・失敗時は no-op）。

    Args:
        invariant_id: 発火したインバリアント/検知器の ID。
        severity: "alarm"（調査対象）または "smell"（疑い記録）。
        details: 発火文脈。
    """
    if _store is None:
        return
    try:
        _store.fire_alarm(invariant_id, severity=severity, details=details)
    except Exception:
        logger.exception("アラーム記録に失敗 invariant=%s", invariant_id)
