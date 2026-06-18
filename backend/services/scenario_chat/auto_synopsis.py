"""あらすじ自動更新のトリガー判定と進捗計算。

蒸留本体（synopsis.py の update_auto_synopsis）を呼び出すサービス層。
閾値判定・未蒸留ターンの抽出・進捗レスポンスの整形を担う。
"""

import logging

from backend.services.scenario_chat.context import resolve_history_limits
from backend.services.scenario_chat.serializers import resolve_user_speaker_name
from backend.services.scenario_chat.synopsis import update_auto_synopsis

logger = logging.getLogger(__name__)

# あらすじ自動更新のトリガー閾値。
# 「未蒸留 dropped 群」のターン数または累積文字数が history 上限の半分に達したら
# synopsis_auto を再蒸留する（OR 判定）。
# 上限到達直後に毎レスポンス LLM 呼出が走るのを避けるための間引き閾値で、
# シナリオごとの history_max_turns / history_max_chars に追従させる。
# 手動 regenerate API では無視する（force=True で呼ぶ）。
#
# 用語: ここでの「ターン」は @話者: ブロック単位（scenario_turns 1 行）。
#       GM の 1 レスポンス（=1 LLM 呼出）が複数の話者ブロックを生むため、
#       1 レスポンス ≠ 1 ターンであることに注意。トリガー比は **ターン基準** で
#       history_max_turns（こちらも話者ブロック単位）と整合している。
SYNOPSIS_AUTO_TRIGGER_RATIO = 0.5


async def maybe_update_auto_synopsis(
    sqlite,
    settings: dict,
    scenario,
    history: list,
    session_id: str,
    synopsis_preset_id: str,
    *,
    force: bool = False,
) -> dict | None:
    """前回蒸留以降の新規ターン群（話者ブロック）を `synopsis_auto` へ統合し全体を再蒸留する。

    呼び出しタイミング:
        - 通常チャットフロー: 各レスポンス直前に best-effort で呼ぶ（force=False）。
          前回蒸留からの新規ターン（話者ブロック）が history 上限の SYNOPSIS_AUTO_TRIGGER_RATIO に
          ターン数・文字数のどちらも届いていなければ何もしない。
        - 手動 regenerate API: force=True で呼ぶ。閾値判定をスキップする。

    あらすじは生ログから押し出されたタイミングではなく **last_turn_index 以降の新規ターン**
    全体を対象に発火する。これにより、あらすじは常に生ログより先まで（または同等まで）
    カバーされ、ターンが生ログから押し出された瞬間に未蒸留状態となる「ギャップ」が
    発生しなくなる。生ログとあらすじには直近側にオーバーラップが生じるが、
    UX としては「最近のことを忘却した GM」より「直近を二重で持つ GM」の方が自然。

    既存の synopsis_auto は単純追記せず、新ターンと統合して**全体を再蒸留**する
    （肥大化を防ぐ）。`synopsis_last_turn_index` によって、すでに蒸留へ
    反映済みのターンは再度対象に含めない。`synopsis_manual` には触らない。

    Returns:
        更新した場合は新しい synopsis dict、何もしなかった場合は None。
        失敗時も None（best-effort）。
    """
    try:
        max_turns, max_chars = resolve_history_limits(scenario, settings)
        if not history:
            logger.debug(
                "auto synopsis skip session=%s 理由=履歴空 force=%s",
                session_id, force,
            )
            return None

        synopsis = sqlite.get_scenario_session_synopsis(session_id) or {
            "auto": "",
            "manual": "",
            "last_turn_index": -1,
        }
        last_idx = int(synopsis.get("last_turn_index", -1))

        # turn_index ベースで「前回蒸留以降の新規ターン」を抽出
        # 旧設計と違い、生ログから押し出された (dropped) ターンに限定しない。
        # これによりあらすじが常に生ログ右端まで（または超えて）カバーされる。
        new_turns = [
            t for t in history
            if int(getattr(t, "turn_index", -1)) > last_idx
        ]
        if not new_turns:
            history_max = max(
                (int(getattr(t, "turn_index", -1)) for t in history),
                default=-1,
            )
            # ロールバック前のクランプ漏れが原因のことが多い。WARN で出して気づける状態にする。
            logger.warning(
                "auto synopsis skip session=%s 理由=new_turns空 "
                "(last_turn_index=%d が history最大turn_index=%d 以上。"
                "ターン削除でクランプされていない可能性) force=%s",
                session_id, last_idx, history_max, force,
            )
            return None

        if not force:
            total_chars = sum(len(getattr(t, "content", "") or "") for t in new_turns)
            total_turns = len(new_turns)
            trigger_turns = max(1, int(max_turns * SYNOPSIS_AUTO_TRIGGER_RATIO))
            trigger_chars = max(1, int(max_chars * SYNOPSIS_AUTO_TRIGGER_RATIO))
            if total_turns < trigger_turns and total_chars < trigger_chars:
                logger.debug(
                    "auto synopsis skip session=%s 理由=閾値未満 "
                    "(turns %d<%d かつ chars %d<%d / max_turns=%d max_chars=%d ratio=%.2f)",
                    session_id,
                    total_turns, trigger_turns,
                    total_chars, trigger_chars,
                    max_turns, max_chars, SYNOPSIS_AUTO_TRIGGER_RATIO,
                )
                return None

        def loader(preset_id: str):
            return sqlite.get_model_preset(preset_id)

        logger.info(
            "auto synopsis 蒸留開始 session=%s new_turns=%d 文字数=%d force=%s",
            session_id,
            len(new_turns),
            sum(len(getattr(t, "content", "") or "") for t in new_turns),
            force,
        )
        # ユーザPC名を解決してあらすじ蒸留プロンプトへ渡す（旧 user_alias 廃止）。
        synopsis_session = sqlite.get_scenario_session(session_id)
        user_speaker_name = resolve_user_speaker_name(
            scenario, synopsis_session, sqlite,
        )
        new_auto = await update_auto_synopsis(
            scenario=scenario,
            new_turns=new_turns,
            existing_auto=synopsis.get("auto", ""),
            settings=settings,
            preset_loader=loader,
            synopsis_preset_id=synopsis_preset_id,
            user_speaker_name=user_speaker_name,
        )
        if new_auto is None:
            # update_auto_synopsis 側の WARN で詳細理由は出ているので、ここは要約のみ
            logger.warning(
                "auto synopsis skip session=%s 理由=update_auto_synopsis が None を返却 force=%s",
                session_id, force,
            )
            return None

        latest_idx = max(int(getattr(t, "turn_index", -1)) for t in new_turns)
        logger.info(
            "auto synopsis 蒸留完了 session=%s last_turn_index=%d→%d 出力文字数=%d",
            session_id, last_idx, latest_idx, len(new_auto),
        )
        return sqlite.update_scenario_session_synopsis(
            session_id,
            auto=new_auto,
            last_turn_index=latest_idx,
        )
    except Exception:
        logger.exception("auto synopsis 更新に失敗 session=%s", session_id)
        return None


def compute_synopsis_progress(
    sqlite, settings: dict, scenario, session_id: str
) -> dict | None:
    """前回あらすじ蒸留以降の累積（ターン数・文字数）と history 上限を返す。

    あらすじ作成バーの進捗表示・自動表示判定に使う。LLM 蒸留は一切実行しない。
    前回蒸留（synopsis_last_turn_index）以降の新規ターンを対象に、
    「ターン数」「累積文字数」を数え、それぞれの上限（resolve_history_limits）と
    併せて返す。閾値（50% / 80% など）の判定・色分けは UI 側に委ねる。

    Args:
        sqlite: SQLiteStore。
        settings: グローバル設定辞書（history 上限の解決に使う）。
        scenario: Scenario ORM（history 上限の解決に使う）。
        session_id: 対象セッション ID。

    Returns:
        {"turns", "max_turns", "chars", "max_chars"} の dict。例外時は None。
        履歴空・新規ターン無しでも turns=chars=0 の dict を返す（バー非表示は UI 判定）。
    """
    try:
        history = sqlite.list_scenario_turns(session_id)
        max_turns, max_chars = resolve_history_limits(scenario, settings)
        synopsis = sqlite.get_scenario_session_synopsis(session_id) or {
            "last_turn_index": -1,
        }
        last_idx = int(synopsis.get("last_turn_index", -1))
        new_turns = [
            t for t in history if int(getattr(t, "turn_index", -1)) > last_idx
        ]
        total_chars = sum(len(getattr(t, "content", "") or "") for t in new_turns)
        return {
            "turns": len(new_turns),
            "max_turns": max_turns,
            "chars": total_chars,
            "max_chars": max_chars,
        }
    except Exception:
        logger.exception("compute_synopsis_progress 失敗 session=%s", session_id)
        return None


