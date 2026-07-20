"""発話予約（speak_later）の発火ランナー — 毎分走査・発火判定・生成呼び出し。

docs/planned/speak_later_plan.md §②:
    キャラクター本人が speak_later ツールで仕掛けた「この時刻に、自分からこの会話に
    声をかける」（speech_reservations の pending 行）を毎分走査し、時刻が来たものを
    能動配達（delivery._deliver_session）と同じ 1on1 SSE 等価経路のヘッドレス生成で
    発火させる。発話内容は仕掛け時に固定されておらず、発火時に本人が生成する。

発火判定（毎分・LLM 呼び出しは発火時のみ）:
    1. セッション削除済み・キャラ未解決/estranged・退席済み・トグル OFF
       → status=cancelled ＋記録（トグルは課金ガードなので発火側でも見る）
    2. speak_at + 24h を過ぎても発火できていなければ status=expired ＋記録（declined）。
       黙って消さず記録に残す。
    3. check_availability を再評価。unavailable なら発火せず pending 維持
       （置いたあとに予定が変わった／うつつシーン進行中のケース。復帰後に遅れて発火）。
       毎分続く状態なので決定ログは残さない（ノイズ防止 — 生存確認は heartbeat）。
    4. 日次コストガード: escrow_delivery_daily_cap の予算・カウンタを共有
       （キャラ発の現実接触は経路を問わず1つの予算 — 2026-07-11 裁定の延長）。
       到達日は skipped 記録（日1回）→ 翌日カウンタリセット後に遅延発火。
    5. 発火 = fired マーク＋カウンタ消費を先に確定してから生成する
       （mark_messages_delivered と同順の再送ループ防止 — 生成失敗で毎分 LLM を
       叩き直さない。失敗は決定ログ error に残る）。

合成注釈は DB に保存しない（LLM 渡しのコピーのみ）。発火時にセッションへ未配達の
ユーザメッセージが残っていたら、escrow 配達と同じ手順で併せて配達し、合成注釈を
末尾に添える（_deliver_session の共用 — 別々に2ターン発生させない）。
"""

import logging
from datetime import datetime, timedelta

from backend.character_actions.messenger import delivery_cap_reached
from backend.services.gate.availability import (
    check_availability,
    is_usual_scene_running,
)
from backend.services.gate.delivery import _deliver_session

logger = logging.getLogger(__name__)

# 発火できないまま speak_at からこの時間が過ぎたら expired に倒す
_EXPIRE_HOURS = 24

# 発火が speak_at からこれ以上遅れたら「遅延発火」の合成注釈にする
# （毎分スケジューラの粒度では数分の遅れは定刻扱い）
_DELAYED_THRESHOLD_MINUTES = 10


def build_reservation_annotation(reservation, now: datetime) -> str:
    """発火時に最終ユーザターン相当として渡す合成注釈を組む純関数。

    DB には保存しない — 時間差注釈と同じ「LLM 渡しのコピーのみ」思想。

    Args:
        reservation: SpeechReservation ORM（speak_at / note を持つ）。
        now: 発火時刻。

    Returns:
        定刻/遅延に応じた合成注釈テキスト。
    """
    note = reservation.note
    delayed = (now - reservation.speak_at) >= timedelta(
        minutes=_DELAYED_THRESHOLD_MINUTES
    )
    if delayed:
        planned = reservation.speak_at
        if planned.date() == now.date():
            stamp = f"{planned:%H:%M}"
        else:
            stamp = f"{planned.month}/{planned.day} {planned:%H:%M}"
        return (
            f"（いま {now:%H:%M}。本当は {stamp} に『{note}』をやろうとしていたが、"
            "都合がつかず今になった。ここからはあなたから声をかける番）"
        )
    return (
        f"（いま {now:%H:%M}。あなたはこの時間に『{note}』をやろうとしていた。"
        "ここからはあなたから声をかける番）"
    )


async def run_pending_speech_reservations(state, now: datetime | None = None) -> None:
    """時刻到来した pending 予約を走査し、発火可能なものを発火させる。

    スケジューラ（main.py の _speech_reservation_tick）から毎分呼ばれる。
    1件の失敗が他の予約を止めないよう、個別に握って記録する。

    Args:
        state: FastAPI の app.state（sqlite / chat_service / vector_store /
            uploads_dir を持つ）。
        now: 基準時刻（テスト注入用。省略時は現在時刻）。
    """
    now = now or datetime.now()
    rows = state.sqlite.list_pending_speech_reservations(due_before=now)
    for reservation in rows:
        try:
            await _maybe_fire_reservation(state, reservation, now)
        except Exception:
            logger.exception("発話予約の発火に失敗 reservation=%s", reservation.id)
            state.sqlite.record_scheduler_decision(
                "speech_reservation", "error",
                character_id=reservation.character_id,
                reason="発火処理で例外",
                details={
                    "reservation_id": reservation.id,
                    "session_id": reservation.session_id,
                },
            )


async def _maybe_fire_reservation(state, reservation, now: datetime) -> None:
    """1件分の発火判定 — 世界の状態ゲート・コストガードを通す。

    通過したら fired を確定させて生成する（LLM 呼び出しはここが唯一の入口）。
    """
    sqlite = state.sqlite
    details = {
        "reservation_id": reservation.id,
        "session_id": reservation.session_id,
        "speak_at": reservation.speak_at.isoformat(),
    }

    # --- 1. 発話先の世界が失われていないか（cancelled 系） ---
    char = sqlite.get_character(reservation.character_id)
    session = sqlite.get_chat_session(reservation.session_id)
    cancel_reason = None
    if session is None:
        cancel_reason = "セッション削除済み"
    elif char is None:
        cancel_reason = "キャラクター未解決"
    elif not int(getattr(char, "speak_later_enabled", 0) or 0):
        # トグルは課金ガード — 仕掛け後にユーザが OFF にしたら発火もさせない
        cancel_reason = "機能が無効化された（speak_later_enabled=0）"
    elif getattr(char, "relationship_status", "active") == "estranged":
        cancel_reason = "estranged（別れたキャラは応答しない）"
    else:
        exited = getattr(session, "exited_chars", None) or []
        if any(e.get("char_name") == char.name for e in exited):
            cancel_reason = "このセッションから退席済み"
    if cancel_reason is not None:
        sqlite.set_speech_reservation_status(reservation.id, "cancelled")
        sqlite.record_scheduler_decision(
            "speech_reservation", "skipped",
            character_id=reservation.character_id,
            reason=f"予約を取り消し: {cancel_reason}", details=details,
        )
        return

    # --- 2. 24h 期限切れ（黙って消さず記録に残す） ---
    if now > reservation.speak_at + timedelta(hours=_EXPIRE_HOURS):
        sqlite.set_speech_reservation_status(reservation.id, "expired")
        sqlite.record_scheduler_decision(
            "speech_reservation", "declined",
            character_id=reservation.character_id,
            reason=f"speak_at から{_EXPIRE_HOURS}時間発火できず期限切れ",
            details=details,
        )
        return

    # --- 3. availability 再評価（実行時状態込み — 仕掛け時の無風仮定とは異なる） ---
    availability = check_availability(
        char, now,
        usual_scene_running=is_usual_scene_running(sqlite, char.id, now),
        sqlite=sqlite,
    )
    if not availability.available:
        return  # pending 維持 — 復帰後に遅れて発火（毎分続くため記録しない）

    # --- 4. 日次コストガード（escrow 配達・reach_out と共有の予算） ---
    today_str = now.date().isoformat()
    if delivery_cap_reached(sqlite, now):
        # cap 到達は毎分続くため、決定ログは予約×日付ごとに1回だけ残す
        mark_key = f"speech_reservation_cap_decision_{reservation.id}_{today_str}"
        if not sqlite.get_setting(mark_key, ""):
            sqlite.set_setting(mark_key, "1")
            sqlite.record_scheduler_decision(
                "speech_reservation", "skipped",
                character_id=reservation.character_id,
                reason="日次上限到達（翌日リセット後に遅延発火）", details=details,
            )
        return

    # --- 5. 発火確定 — fired マーク＋カウンタ消費を生成より先に確定する ---
    # （生成失敗で毎分 LLM を叩き直さない。escrow の delivered-before-LLM と同思想）
    sqlite.set_speech_reservation_status(reservation.id, "fired", fired_at=now)
    count_key = f"escrow_delivery_count_{today_str}"
    delivered_today = int(sqlite.get_setting(count_key, "0") or 0)
    sqlite.set_setting(count_key, str(delivered_today + 1))

    annotation = build_reservation_annotation(reservation, now)
    await _deliver_session(
        state, session, char,
        extra_annotation=annotation,
        require_pending=False,
        feature="speech_reservation",
    )
    # 未配達分を併せて配達した場合に備え、escrow 従来経路の復帰観測マーカーを掃除する
    # （stale なマーカーが次回配達のジッターを飛ばさないように）
    sqlite.set_setting(f"escrow_ready_{session.id}", "")

    delayed_min = int((now - reservation.speak_at).total_seconds() // 60)
    sqlite.record_scheduler_decision(
        "speech_reservation", "fired",
        character_id=reservation.character_id,
        reason=(
            f"発話予約を発火（{delayed_min}分遅れ）" if delayed_min
            >= _DELAYED_THRESHOLD_MINUTES else "発話予約を定刻発火"
        ),
        details=details,
    )
    logger.info(
        "発話予約: 発火完了 char=%s session=%s speak_at=%s 遅延=%d分",
        char.name, session.id, reservation.speak_at.isoformat(), delayed_min,
    )
