"""ChatFlow の別れ・疲労離席後処理 — ターン完了後のバックグラウンド判定を担う。

flow.py（経路のオーケストレーション）から分離した「応答が出た後の退席まわり」の層。
judge LLM による別れ検出（run_farewell_detection）と、その起動判定
（launch_farewell_tasks。judge 不在時の疲労離席縮退も含む）を持つ。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from backend.services.chat.models import ChatRequest
    from backend.services.memory.manager import InscribedMemoryManager

from backend.character_actions.farewell_detector import FarewellDetector


async def run_farewell_detection(
    detector: "FarewellDetector",
    character_id: str,
    character_name: str,
    session_id: str,
    preset_id: str,
    farewell_config: dict,
    messages: list[dict],
    settings: dict,
    vector_store=None,
) -> None:
    """FarewellDetectorをバックグラウンドで実行し、退席判定をDBに保存するコルーチン。

    退席確定の場合のみセッションの exited_chars を更新する。
    疎遠化確定時は SQLite に加えてベクトルストアの定義 embedding も "estranged" に更新する。
    SSEストリームは既に終了しているため、イベント送信は行わない。
    次リクエスト時の already_exited チェックで自動検知される。

    Args:
        detector: FarewellDetectorインスタンス。
        character_id: キャラクターID。
        character_name: キャラクター名。
        session_id: 対象セッションID。
        preset_id: judge LLM に使用するプリセットID。
        farewell_config: キャラクターのfarewell_config辞書。
        messages: 判定に使用する会話履歴（最新の応答を含む）。
        settings: グローバル設定辞書。
        vector_store: LanceStore インスタンス（疎遠化時の embedding 更新に使用。None でもよい）。
    """
    try:
        result = await detector.detect(
            character_id=character_id,
            session_id=session_id,
            preset_id=preset_id,
            farewell_config=farewell_config,
            messages=messages,
            settings=settings,
        )
    except Exception:
        _log.exception("FarewellDetector実行エラー char=%s session=%s", character_name, session_id)
        return

    # judge の採点結果を該当ターン（最新キャラ発話）の封筒 payload に残す
    # （Tier 3 サンプリングの材料を兼ねる。めぐり Phase 5）。best-effort。
    if result is not None:
        try:
            detector.sqlite.attach_payload_to_latest_chat_event(
                character_id, session_id,
                {
                    "judge": {
                        "emotions": result.emotions,
                        "engagement": result.engagement,
                    }
                },
            )
        except Exception:
            _log.exception("judge採点の封筒添付に失敗 char=%s", character_name)

    # 疲労離席（めぐり Phase 5）: 物理の発火式が終わりを決める。
    # judge の没入度で閾値が持ち上がる（夢中は疲労を忘れさせるが消さない）。
    # farewell（感情閾値）で退席が確定するターンでは二重離席を避けるためスキップ。
    if result is None or not result.should_exit:
        try:
            from backend.services.gate import check_fatigue_leave
            check_fatigue_leave(
                detector.sqlite,
                character_id=character_id,
                character_name=character_name,
                session_id=session_id,
                farewell_config=farewell_config,
                engagement=result.engagement if result is not None else 0.5,
            )
        except Exception:
            _log.exception("疲労離席チェックに失敗 char=%s", character_name)

    if result is None or not result.should_exit:
        return

    _log.info(
        "別れ検出: セッション退席 char=%s session=%s type=%s emotions=%s",
        character_name, session_id, result.farewell_type, result.emotions,
    )

    # セッションの exited_chars に退席エントリを追記する
    try:
        sqlite = detector.sqlite
        session = sqlite.get_chat_session(session_id)
        if session is None:
            return
        exited_chars: list[dict] = getattr(session, "exited_chars", None) or []
        # 重複チェック: 既に退席済みなら何もしない
        if any(e.get("char_name") == character_name for e in exited_chars):
            return

        # ネガティブ退席時: 累積回数を確認し、閾値超過なら estranged に移行する
        reason = result.reason
        if result.farewell_type == "negative":
            estrangement = farewell_config.get("estrangement", {})
            lookback_days = estrangement.get("lookback_days", 30)
            threshold = estrangement.get("negative_exit_threshold", 5)
            since = datetime.now() - timedelta(days=lookback_days)
            prev_count = sqlite.get_negative_exit_count(character_name, since)
            total_count = prev_count + 1  # 今回の退席を含む合計

            if total_count >= threshold:
                # 閾値到達: relationship_status を estranged に変更する
                sqlite.update_character(character_id, relationship_status="estranged")
                _log.info(
                    "別れ決断: 閾値到達 char=%s total=%d threshold=%d → estranged",
                    character_name, total_count, threshold,
                )
                # ベクトルストアの定義 embedding も estranged に更新する（類似キャラ登録ブロックに必要）
                if vector_store is not None:
                    try:
                        vector_store.mark_definition_estranged(character_id)
                    except Exception:
                        _log.exception("ベクトルストア 疎遠化マーク失敗 char=%s", character_name)
                farewell_messages = farewell_config.get("farewell_message") or {}
                estranged_msg = farewell_messages.get("estranged", "")
                reason = estranged_msg if estranged_msg else reason
            else:
                warning = (
                    f"過去{lookback_days}日間で{total_count}回、"
                    f"{character_name}は嫌がって会話を打ち切りました。\n"
                    f"{lookback_days}日間に{threshold}回これが続けば、"
                    f"{character_name}はあなたとの別れを決断するでしょう。"
                )
                reason = f"{reason}\n{warning}" if reason else warning

        exited_chars = [*exited_chars, {
            "char_name": character_name,
            "reason": reason,
            "farewell_type": result.farewell_type,
        }]
        sqlite.update_chat_session(session_id, exited_chars=exited_chars)
        # タイムライン封筒（chat.farewell）: 退席という出来事を正本に載せる。
        # 中身（理由・種別）は payload（chat_sessions.exited_chars と重複するが、
        # exited_chars は上書き更新される JSON なので封筒側にも凍結して残す）。
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
                "reason": reason,
                "farewell_type": result.farewell_type,
            },
        )
    except Exception:
        _log.exception("退席DB保存エラー char=%s session=%s", character_name, session_id)


def launch_farewell_tasks(
    memory_manager: "InscribedMemoryManager",
    request: "ChatRequest",
    messages: list[dict],
    clean_text: str,
) -> None:
    """ターン完了後の別れ検出／疲労離席をバックグラウンドタスクとして起動する。

    - judge プリセット設定済み（かつ疎遠化前）: run_farewell_detection を起動。
    - judge 未設定でも farewell_config.fatigue があれば疲労離席だけ動かす
      （engagement=0.5 縮退。めぐり Phase 5 — 出口は物理が握る）。
    - session_id が無い経路（シナリオ・バッチ）は何もしない。

    farewell_config / farewell_relationship_status はリクエスト構築時にキャッシュ済みのため
    ここでは get_character() を呼ばない。
    """
    if not request.session_id:
        return
    if (
        request.farewell_config
        and request.judge_preset_id
        and request.farewell_relationship_status != "estranged"
    ):
        farewell_messages = [*messages, {"role": "assistant", "content": clean_text}]
        detector = FarewellDetector(memory_manager.sqlite)
        asyncio.create_task(
            run_farewell_detection(
                detector=detector,
                character_id=request.character_id,
                character_name=request.character_name,
                session_id=request.session_id,
                preset_id=request.judge_preset_id,
                farewell_config=request.farewell_config,
                messages=farewell_messages,
                settings=request.settings,
                vector_store=memory_manager.vector_store,
            )
        )
    elif (
        request.farewell_config
        and isinstance(request.farewell_config.get("fatigue"), dict)
    ):
        def _fatigue_only() -> None:
            """judge 不在時の疲労離席チェック（バックグラウンドスレッド実行）。"""
            try:
                from backend.services.gate import check_fatigue_leave
                check_fatigue_leave(
                    memory_manager.sqlite,
                    character_id=request.character_id,
                    character_name=request.character_name,
                    session_id=request.session_id,
                    farewell_config=request.farewell_config,
                    engagement=0.5,
                )
            except Exception:
                _log.exception("疲労離席チェックに失敗 char=%s", request.character_name)
        asyncio.create_task(asyncio.to_thread(_fatigue_only))
