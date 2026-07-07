"""預かり（escrow）メッセージの能動配達 — availability 復帰時の配達スケジューラ実装。

docs/aliveness_plan.md §5.1 のフォローアップ（v1 簡略化の解消）。
v1 では預かり分の配達は「次のユーザリクエスト時」のみだったが、本モジュールにより
availability が戻った時点でキャラクター本人が預かり分を読み、**自分から返信を書く**。
仕事中に送っても返らないが、昼休みにユーザが何もしなくても返信が届く。

設計:
    - 判定は毎分（main.py の _escrow_delivery_scheduler）。LLM 呼び出しは配達時のみ。
    - 復帰の瞬間ちょうどに返信するのは機械的なので、決定論ジッター
      （0〜_JITTER_MAX_MINUTES 分・乱数は世界に置く — 行動権と同じ思想）を挟む。
      ready マーカー（settings: ``escrow_ready_{session_id}``）が
      「availability 復帰を最初に観測した時刻」を覚え、ジッター経過で配達する。
      配達前に availability が再び失われたらマーカーを破棄し、次の復帰で仕切り直す。
    - 配達マーク（mark_messages_delivered）は LLM 呼び出しの**前**（1on1 経路と同順）。
      LLM が失敗しても再送ループにはならない（メッセージ自体は履歴に残っており、
      次のユーザターンで通常履歴としてキャラに渡る — 時間差注釈だけが失われる）。
    - 応答生成は 1on1 SSE 経路と同じ SceneLoop（OneOnOneRouter/Executor）を
      ヘッドレスで await 回収する（うつつの無人進行と同じパターン）。
    - 日次コストガード: settings ``escrow_delivery_daily_cap``（既定 12 配達/日）。
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta

from backend.lib.debug_logger import logger as debug_logger
from backend.lib.log_context import (
    current_log_session_id,
    current_log_target,
    new_message_id,
)
from backend.services.chat.indexer import get_participant_char_ids, index_message_sync
from backend.services.chat_flow.scene_loop import LoopState, SceneLoop
from backend.services.chat_flow.strategies import OneOnOneExecutor, OneOnOneRouter
from backend.services.gate.availability import (
    check_availability,
    format_escrow_annotation,
    is_usual_scene_running,
)

logger = logging.getLogger(__name__)

# 復帰観測から配達までの決定論ジッターの最大幅（分）
_JITTER_MAX_MINUTES = 10
# 日次コストガードの既定値（配達 = LLM 1呼び出し）
_DEFAULT_DAILY_CAP = 12


def _delivery_jitter_seconds(session_id: str, ready_at: datetime) -> int:
    """配達ジッターを決定論導出する（乱数は世界に置く）。

    同じセッション・同じ復帰観測時刻なら常に同じ待ち時間。「いつ返すか」だけを
    揺らし、「返すかどうか」には関与しない（返すのは確定 — 預かりは必ず届く）。

    Args:
        session_id: 対象セッション ID（シードの一部）。
        ready_at: availability 復帰を最初に観測した時刻。

    Returns:
        0〜_JITTER_MAX_MINUTES 分の秒数。
    """
    rng = random.Random(f"escrow-delivery:{session_id}:{ready_at.isoformat()}")
    return rng.randint(0, _JITTER_MAX_MINUTES * 60)


async def run_pending_escrow_deliveries(state, now: datetime | None = None) -> None:
    """未配達メッセージを持つ全セッションを走査し、配達可能なものを能動配達する。

    スケジューラ（main.py の _escrow_delivery_scheduler）から毎分呼ばれる。
    1セッションの失敗が他セッションの配達を止めないよう、個別に握って記録する。

    Args:
        state: FastAPI の app.state（sqlite / chat_service / vector_store /
            uploads_dir を持つ）。
        now: 基準時刻（テスト注入用。省略時は現在時刻）。
    """
    now = now or datetime.now()
    rows = state.sqlite.list_sessions_with_undelivered()
    for session, _count in rows:
        try:
            await _maybe_deliver_session(state, session, now)
        except Exception:
            logger.exception("能動配達に失敗 session=%s", session.id)


async def _maybe_deliver_session(state, session, now: datetime) -> None:
    """1セッション分の配達判定 — availability・ジッター・コストガードを通す。

    通過したら _deliver_session で実際に配達する（LLM 呼び出しはここが唯一の入口）。

    Args:
        state: FastAPI の app.state。
        session: 未配達メッセージを持つ ChatSession ORM。
        now: 基準時刻。
    """
    sqlite = state.sqlite
    model_id = getattr(session, "model_id", "") or ""
    char_name = model_id.rsplit("@", 1)[0] if "@" in model_id else model_id
    char = sqlite.get_character_by_name(char_name)
    if char is None:
        return  # キャラ未解決（削除済み等）— 配達先がいない
    if getattr(char, "relationship_status", "active") == "estranged":
        return  # 別れたキャラは恒久的に応答しない（1on1 経路と同じ扱い）
    exited = getattr(session, "exited_chars", None) or []
    if any(e.get("char_name") == char_name for e in exited):
        return  # このセッションからは退席済み — 戻って返信はしない

    ready_key = f"escrow_ready_{session.id}"
    availability = check_availability(
        char, now,
        usual_scene_running=is_usual_scene_running(sqlite, char.id, now),
    )
    if not availability.available:
        # 配達前に窓が閉じた（再び unavailable）— 復帰観測を破棄して仕切り直す
        if sqlite.get_setting(ready_key, ""):
            sqlite.set_setting(ready_key, "")
        return

    # 復帰観測時刻（ready マーカー）: 初観測なら今を記録し、ジッター待ちに入る
    raw = sqlite.get_setting(ready_key, "")
    ready_at = None
    if raw and isinstance(raw, str):
        try:
            ready_at = datetime.fromisoformat(raw)
        except ValueError:
            ready_at = None
    if ready_at is None:
        ready_at = now
        sqlite.set_setting(ready_key, now.isoformat())
    due = ready_at + timedelta(seconds=_delivery_jitter_seconds(session.id, ready_at))
    if now < due:
        return  # ジッター待ち（復帰直後の機械的な即レスを避ける）

    # 日次コストガード（配達 = LLM 1呼び出し）。cap=0 は「能動配達を止める」
    # 意味で有効な設定値なので、or フォールバックではなく明示的にパースする
    today_str = now.date().isoformat()
    try:
        cap = int(sqlite.get_setting("escrow_delivery_daily_cap", ""))
    except (TypeError, ValueError):
        cap = _DEFAULT_DAILY_CAP
    count_key = f"escrow_delivery_count_{today_str}"
    delivered_today = int(sqlite.get_setting(count_key, "0") or 0)
    if delivered_today >= cap:
        logger.warning(
            "能動配達: 日次上限 %d 到達。当日はスキップ session=%s", cap, session.id,
        )
        return

    # 配達確定 — マーカーを消費し、カウンタを進めてから配達する
    sqlite.set_setting(ready_key, "")
    sqlite.set_setting(count_key, str(delivered_today + 1))
    await _deliver_session(state, session, char)


async def _deliver_session(state, session, char) -> None:
    """預かり分をキャラ本人へ配達し、返信を生成・保存する。

    1on1 SSE 経路（api/chat.py stream_message）の配達部と同じ手順:
    LLM に渡すコピーにだけ時間差注釈を付け（DB の本文は変更しない）、
    delivered_at と chat.message 封筒を確定させてから応答を生成する。
    生成した返信は通常のキャラクターメッセージとしてセッションに保存され、
    ユーザが次にセッションを開いたときに見える。

    Args:
        state: FastAPI の app.state。
        session: 配達対象の ChatSession ORM。
        char: 配達先の Character ORM。
    """
    # API 層と同等のログ文脈を張る（ログ UI でセッション・キャラに紐づくように）
    log_msg_id = new_message_id()
    current_log_session_id.set(session.id)
    current_log_target.set(char.name)

    sqlite = state.sqlite
    messages = sqlite.list_chat_messages(session.id)
    pending = [
        m for m in messages
        if getattr(m, "delivered_at", None) is None
        and not getattr(m, "is_system_message", None)
    ]
    if not pending:
        return  # 走査後にユーザターンが先に配達したケース（レース）— 何もしない

    # 最後の預かりメッセージを「今回のユーザ発話」、それ以外を履歴として組む
    # （SSE 経路の history / user_content の分割と同じ形）
    last = pending[-1]
    history = [m for m in messages if m.id != last.id]
    for m in pending[:-1]:
        m.content = format_escrow_annotation(m)

    # api 層のヘルパーは lazy import（api → services の逆流 import を起動時に作らない）
    from backend.api.chat import build_1on1_chat_request
    from backend.api.utils import (
        build_message_content,
        format_memories_for_sse,
    )
    from backend.services.memory.format import format_recalled_threads

    user_content = build_message_content(
        format_escrow_annotation(last), last.images or [],
        sqlite, state.uploads_dir,
    )

    # 配達マークは LLM 呼び出しの前（SSE 経路と同順・再送ループ防止）
    sqlite.mark_messages_delivered([m.id for m in pending])
    logger.info(
        "能動配達: %d 件を配達して返信を生成 char=%s session=%s",
        len(pending), char.name, session.id,
    )

    chat_request = await build_1on1_chat_request(state, session, history, user_content)

    # 1on1 SSE 経路と同じ SceneLoop 最小構成をヘッドレスで回収する
    scene_loop = SceneLoop(
        router=OneOnOneRouter(),
        executor=OneOnOneExecutor(state.chat_service),
        max_iterations=1,
    )
    loop_state = LoopState(context={"pending_request": chat_request})
    full_text = ""
    accumulated_reasoning = ""
    anticipation_text = ""
    effective_model_id = session.model_id
    async for event in scene_loop.run(initial_state=loop_state):
        if not (isinstance(event, tuple) and len(event) == 2):
            continue
        chunk_type, content = event
        if chunk_type == "inscribed_memories":
            accumulated_reasoning += format_memories_for_sse(content) or ""
        elif chunk_type == "recall_error":
            accumulated_reasoning += (content or "") + "\n"
        elif chunk_type == "working_memory_threads":
            accumulated_reasoning += format_recalled_threads(content) or ""
        elif chunk_type == "thinking":
            accumulated_reasoning += content or ""
        elif chunk_type == "text":
            full_text += content or ""
        elif chunk_type == "anticipation":
            anticipation_text = content or ""
        elif chunk_type == "angle_switched":
            effective_model_id = content["model_id"]

    if accumulated_reasoning:
        debug_logger.log_reasoning(accumulated_reasoning)
    if not full_text.strip():
        # 空応答は保存しない（配達マークは済んでいるため、次のユーザターンで
        # 通常履歴としてキャラに渡る — 時間差注釈だけが失われる）
        logger.warning(
            "能動配達: 応答が空のため返信を保存しない char=%s session=%s",
            char.name, session.id,
        )
        return

    used_char_name, used_preset_name = (
        effective_model_id.rsplit("@", 1)
        if "@" in (effective_model_id or "")
        else (effective_model_id, None)
    )
    char_msg = sqlite.create_chat_message(
        message_id=str(uuid.uuid4()),
        session_id=session.id,
        role="character",
        content=full_text,
        reasoning=accumulated_reasoning or None,
        character_name=used_char_name,
        preset_name=used_preset_name,
        log_message_id=log_msg_id,
        anticipation=anticipation_text or None,
        face_to_face=int(getattr(char, "face_to_face_mode", 0) or 0),
    )

    # チャット履歴インデックス（SSE 経路と同じ。失敗しても本流は止めない）
    char_ids = get_participant_char_ids(session, sqlite)
    user_name = sqlite.get_setting("user_name", "ユーザ")
    await asyncio.to_thread(
        index_message_sync, char_msg, char_ids, state.vector_store, user_name,
    )

    # 計器 Tier 2: 応答の外形スキャン（誤検知許容の smell 記録）
    from backend.services.instruments.tier2 import record_response_smells
    record_response_smells(
        sqlite, full_text, character_name=used_char_name, feature="escrow_delivery",
    )

    # switch_angle が走った場合はセッションの model_id を追随させる（SSE 経路と同じ）
    if effective_model_id and effective_model_id != session.model_id:
        sqlite.update_chat_session(session.id, model_id=effective_model_id)

    logger.info(
        "能動配達: 返信を保存 char=%s session=%s chars=%d",
        char.name, session.id, len(full_text),
    )
