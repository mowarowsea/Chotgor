"""疲労離席 — 物理（体調圧×没入度の発火式）が終わりを決める終了権。

発火式（docs/planned/aliveness_plan.md §5.2）:

    発火: 体調圧 > θ_base + β × engagement    # 夢中は閾値を持ち上げる
    ただし 体調圧 > θ_hard なら無条件発火      # 限界は限界
    （これがないと出口が再び意志に握られる）

のめりこみは疲労を「忘れさせる」が「消さない」— 疲労成分は裏で溜まり続けるため、
夢中で夜更かしした翌日は体調圧が高い状態から始まる（「後でどっと来る」は
追加実装なしに創発）。

有効化: farewell_config["fatigue"] が dict で存在するキャラだけが対象。
judge（engagement 採点）不在時は engagement=0.5 で縮退する。
発火後は away 状態になり availability ゲートに合流する（以降 LLM を呼ばない）。
"""

import logging
from datetime import datetime, timedelta

from backend.services.pressure import compute_pressures, compute_speech_thresholds

logger = logging.getLogger(__name__)

# 発火式の既定パラメータ（farewell_config["fatigue"] で上書き可能）
# θ_base は既定では**キャラ自身の体調圧分布の p75**を使う（下記 _resolve_theta_base）。
# 絶対値はその定式化のときたまたま噛み合っていた数字にすぎず、体調圧のスケールが
# 変わるたびに「毎ターン離席する」か「一度も離席しない」へ倒れるため。
_DEFAULT_THETA_BASE = 0.75   # 分位点が引けないときの基礎閾値
_DEFAULT_BETA = 0.2          # 没入度による閾値の持ち上げ幅
_DEFAULT_THETA_HARD = 0.95   # 無条件発火の限界閾値
_DEFAULT_REST_HOURS = 4.0    # 離席後の休息時間


def _resolve_theta_base(sqlite, character_id: str, fatigue_cfg: dict, now: datetime) -> float:
    """θ_base を決める。明示設定 > キャラ自身の分位点 > 既定の絶対値。

    θ_base は「その子にとって重い日」を指すべき相対量なので、既定では体調圧の
    分位点（compute_speech_thresholds の MID = p75）を使う。管理UIで theta_base を
    明示設定したキャラはそちらを優先する（守護者の介入枠）。

    **θ_hard はここで相対化しない。** 相対化すると「その子にとっての限界」になり
    いくらでも慣れてしまう（＝出口が再び意志に握られる）。肉体の限界点は物理として
    固定する — 対数中心化した体調圧では 0.95 が「平常の約2倍の負荷」に対応する。
    """
    explicit = fatigue_cfg.get("theta_base")
    if explicit is not None:
        return float(explicit)
    try:
        th = compute_speech_thresholds(sqlite, character_id, now=now).get("body")
    except Exception:
        th = None
    if th is None:
        return _DEFAULT_THETA_BASE
    return th[1]  # (high, mid, good) の mid = p75


def check_fatigue_leave(
    sqlite,
    character_id: str,
    character_name: str,
    session_id: str,
    farewell_config: dict | None,
    engagement: float = 0.5,
    now: datetime | None = None,
) -> bool:
    """疲労離席の発火式を評価し、発火したら away 状態へ移す。

    毎ターン後（farewell judge と同じチェックポイント）に呼ばれる。
    発火時の処理:
        1. characters.away_until / away_reason を設定（availability ゲートに合流）
        2. 退去挨拶のシステムメッセージをセッションへ保存
           （farewell_config["farewell_message"]["fatigue"] があればその文面）
        3. chat.farewell 封筒（payload.farewell_type="fatigue"）を正本に残す

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター。
        character_name: キャラクター名（表示・封筒用）。
        session_id: 対象セッション。
        farewell_config: キャラクターの farewell_config。
            "fatigue" キー（dict）が無ければ疲労離席は無効（False を返す）。
        engagement: 没入度（judge の採点。judge 不在時は 0.5 縮退）。
        now: 基準時刻（テスト注入用）。

    Returns:
        発火したら True。
    """
    fatigue_cfg = (farewell_config or {}).get("fatigue")
    if not isinstance(fatigue_cfg, dict):
        return False
    now = now or datetime.now()

    theta_base = _resolve_theta_base(sqlite, character_id, fatigue_cfg, now)
    beta = float(fatigue_cfg.get("beta", _DEFAULT_BETA))
    theta_hard = float(fatigue_cfg.get("theta_hard", _DEFAULT_THETA_HARD))
    rest_hours = float(fatigue_cfg.get("rest_hours", _DEFAULT_REST_HOURS))

    body = compute_pressures(sqlite, character_id, now=now)["body"]
    threshold = theta_base + beta * max(0.0, min(1.0, engagement))
    if not (body > theta_hard or body > threshold):
        return False

    # 1. away 状態へ（availability ゲートに合流。以降 LLM は呼ばれない）
    away_until = now + timedelta(hours=rest_hours)
    sqlite.update_character(
        character_id,
        away_until=away_until,
        away_reason="疲労のため休息中",
    )

    # 2. 退去挨拶（設定があれば本人の言葉、なければ淡白な掲示）
    farewell_messages = (farewell_config or {}).get("farewell_message") or {}
    goodbye = farewell_messages.get("fatigue") or (
        f"{character_name}は疲れがたまって、少し休みに行きました。"
    )
    import uuid as uuid_mod
    sqlite.create_chat_message(
        message_id=str(uuid_mod.uuid4()),
        session_id=session_id,
        role="character",
        content=goodbye,
        character_name=character_name,
        is_system_message=True,
    )

    # 3. タイムライン封筒（chat.farewell / fatigue）
    sqlite.record_timeline_event(
        character_id=character_id,
        event_type="chat.farewell",
        actor="character",
        counterpart="user",
        origin="real",
        session_id=session_id,
        source_table="chat_sessions",
        source_id=session_id,
        payload={
            "farewell_type": "fatigue",
            "body_pressure": round(body, 3),
            "engagement": round(engagement, 3),
            "away_until": away_until.isoformat(),
        },
    )
    logger.info(
        "疲労離席 発火 char=%s body=%.2f threshold=%.2f engagement=%.2f rest=%.1fh",
        character_name, body, threshold, engagement, rest_hours,
    )
    return True
