"""シナリオチャットサービス — API 層から呼ばれるファサード。

1 ターンのストリーム実行を司る非同期ジェネレータ `run_scenario_turn` と、
API のレスポンス整形に使う dict 変換ヘルパを提供する。

責務:
    - プレイヤー発話を scenario_turns に保存
    - EnsembleEngine を駆動して話者単位の SSE イベントを生成
    - 各発話を scenario_turns に保存
    - raw_response はそのターン内の発話レコードに共通で紐付ける

group_chat.run_group_turn と同じ思想（非同期ジェネレータで SSE 用イベントを yield）
にすることで、API 側の StreamingResponse 実装を一貫させる。
"""

import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from backend.lib.log_context import current_log_feature, current_message_id
from backend.services.scenario_chat.context import resolve_history_limits
from backend.services.scenario_chat.engine import (
    EngineResult,
    EnsembleEngine,
    SceneEngine,
    TurnRecord,
)
from backend.services.scenario_chat.parser import UtteranceDelta
from backend.character_actions.anticipator import extract_anticipation
from backend.services.scenario_chat.synopsis import update_auto_synopsis

logger = logging.getLogger(__name__)

# あらすじ自動更新のトリガー閾値。
# 「未蒸留 dropped 群」のターン数または累積文字数が history 上限の半分に達したら
# synopsis_auto を再蒸留する（OR 判定）。
# 上限到達直後に毎ターン LLM 呼出が走るのを避けるための間引き閾値で、
# シナリオごとの history_max_turns / history_max_chars に追従させる。
# 手動 regenerate API では無視する（force=True で呼ぶ）。
SYNOPSIS_AUTO_TRIGGER_RATIO = 0.5


# ─── dict 変換 ───────────────────────────────────────────────────────────────


def scenario_to_dict(scenario: Any) -> dict:
    """Scenario ORM を JSON 化可能な dict に変換する。"""
    if scenario is None:
        return {}
    return {
        "id": scenario.id,
        "title": scenario.title,
        "scenario": scenario.scenario,
        "intro": scenario.intro,
        "user_alias": scenario.user_alias,
        "history_max_turns": scenario.history_max_turns,
        "history_max_chars": scenario.history_max_chars,
        "custom_system_prompt": scenario.custom_system_prompt,
        "dice_pool_spec": getattr(scenario, "dice_pool_spec", None),
        "pc_slots": getattr(scenario, "pc_slots", None) or [],
        "created_at": scenario.created_at.isoformat() if scenario.created_at else None,
        "updated_at": scenario.updated_at.isoformat() if scenario.updated_at else None,
    }


def scenario_session_to_dict(session: Any) -> dict:
    """ScenarioSession ORM（プレイインスタンス）を JSON 化可能な dict に変換する。

    `gm_preset_id` は同一シナリオから複数セッションを起動した際にそれぞれ別の
    GM モデルで遊ぶための、セッション固有の LLM プリセット ID。
    """
    if session is None:
        return {}
    return {
        "id": session.id,
        "scenario_id": session.scenario_id,
        "title": session.title,
        "engine_type": session.engine_type,
        "status": session.status,
        "gm_preset_id": getattr(session, "gm_preset_id", "") or "",
        "synopsis_preset_id": (
            getattr(session, "synopsis_preset_id", "")
            or getattr(session, "gm_preset_id", "")
            or ""
        ),
        "synopsis_auto": getattr(session, "synopsis_auto", "") or "",
        "synopsis_manual": getattr(session, "synopsis_manual", "") or "",
        "synopsis_last_turn_index": int(getattr(session, "synopsis_last_turn_index", -1) or -1),
        "pc_assignments": getattr(session, "pc_assignments", None) or [],
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


def scenario_npc_to_dict(npc: Any) -> dict:
    """ScenarioNpc ORM を JSON 化可能な dict に変換する。"""
    if npc is None:
        return {}
    return {
        "id": npc.id,
        "scenario_id": npc.scenario_id,
        "name": npc.name,
        "description": npc.description,
        "image_data": npc.image_data,
        "promoted_character_id": npc.promoted_character_id,
        "created_at": npc.created_at.isoformat() if npc.created_at else None,
    }


def scenario_turn_to_dict(turn: Any) -> dict:
    """ScenarioTurn ORM を JSON 化可能な dict に変換する。"""
    if turn is None:
        return {}
    return {
        "id": turn.id,
        "session_id": turn.session_id,
        "turn_index": turn.turn_index,
        "speaker_type": turn.speaker_type,
        "speaker_id": turn.speaker_id,
        "speaker_name": turn.speaker_name,
        "content": turn.content,
        "raw_response": turn.raw_response,
        "log_request_id": getattr(turn, "log_request_id", None),
        "anticipation": getattr(turn, "anticipation", None),
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
    }


# ─── ストリーム実行 ───────────────────────────────────────────────────────────


def _build_default_engine(sqlite) -> SceneEngine:
    """SQLite を preset_loader として束ねる既定 EnsembleEngine を作る。"""
    def loader(preset_id: str):
        return sqlite.get_model_preset(preset_id)
    return EnsembleEngine(preset_loader=loader)


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
    """前回蒸留以降の新規ターン群を `synopsis_auto` へ統合し全体を再蒸留する。

    呼び出しタイミング:
        - 通常チャットフロー: 各ターン直前に best-effort で呼ぶ（force=False）。
          前回蒸留からの新規ターンが history 上限の SYNOPSIS_AUTO_TRIGGER_RATIO に
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
        new_auto = await update_auto_synopsis(
            scenario=scenario,
            new_turns=new_turns,
            existing_auto=synopsis.get("auto", ""),
            settings=settings,
            preset_loader=loader,
            synopsis_preset_id=synopsis_preset_id,
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


def _latest_scenario_anticipation(history: list) -> str:
    """シナリオ履歴から直近の非空 anticipation（GM の前回予想）を返す。無ければ空文字列。

    Args:
        history: ScenarioTurn ORM の時系列昇順リスト。

    Returns:
        直近ターンに保存された予想文字列。無ければ空文字列。
    """
    for turn in reversed(history):
        anticipation = getattr(turn, "anticipation", None)
        if anticipation:
            return anticipation
    return ""


# 1 ユーザターンあたりの最大話者ターン数（GM + PC の合計）。
# 無限連鎖防止用ガード。
_MAX_TURNS_PER_USER_TURN = 10


async def run_scenario_turn(
    session_id: str,
    user_message: str,
    sqlite,
    settings: dict,
    engine: SceneEngine | None = None,
    auto_advance: bool = False,
    chat_service=None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """ユーザ発話を受け取り、シナリオ 1 ターン分の SSE イベントを順次 yield する。

    プレイセッション（scenario_sessions）から元シナリオ（scenarios）を lookup し、
    そのシナリオの NPC・PC枠・GM プリセット等で 1 ターンを進行させる。

    動作モード:
        - engine_type == "ensemble":
            - 従来通り GM 1 ターンのみを実行してユーザターンへ戻す。
        - engine_type == "ensemble_pc":
            - メンション主導の話者ループで動作する:
              1. ユーザ発話末尾のメンションを解析して次話者を決定。
              2. メンション無し / @GM / @Narrator / @NPC → GM ターン。
              3. @<PC枠名> / @<キャラ本名> → 該当 PC ターン（直接ディスパッチ）。
              4. @ALL → 直前話者を除いた PC から random.choice。
              5. 各話者の発話末尾を再度メンション解析して次話者を決める。
              6. 次話者が「ユーザ枠」または「メンション無し」になればループ終了。
              7. 上限 `_MAX_TURNS_PER_USER_TURN` で打ち切る（無限連鎖防止）。

    PC ターン実行のため `chat_service` を渡すこと（None だと PC 行きの分岐は
    スキップされて GM のみ進む）。

    auto_advance=True の場合:
        - user_message は無視される（呼び出し側は空文字を渡してよい）
        - user turn は保存されない
        - SSE の "user_saved" イベントも発火しない
        - GM プロンプト末尾に「プレイヤーは無言、場面を進めて」という OOC 指示が入る
    """
    current_log_feature.set("scenario_chat")
    session = sqlite.get_scenario_session(session_id)
    if not session:
        yield ("error", {"message": f"セッション '{session_id}' が見つかりません"})
        return
    if session.status != "active":
        yield ("error", {"message": "セッションは終了しています"})
        return
    scenario = sqlite.get_scenario(session.scenario_id)
    if not scenario:
        yield ("error", {"message": "元シナリオが見つかりません（孤児セッション）"})
        return

    npcs = sqlite.list_scenario_npcs(scenario.id)
    npc_names = {n.name for n in npcs if getattr(n, "name", None)}
    engine_type = getattr(session, "engine_type", "ensemble") or "ensemble"
    is_pc_mode = engine_type == "ensemble_pc"

    from backend.services.scenario_chat.mention import (
        format_pc_summary,
        normalize_pc_assignments,
        normalize_pc_slots,
        find_last_routing_mention,
        pick_at_all_target,
    )
    from backend.services.scenario_chat.engine import generate_dice_pool

    pc_slots = normalize_pc_slots(getattr(scenario, "pc_slots", None)) if is_pc_mode else []
    pcs = normalize_pc_assignments(
        getattr(session, "pc_assignments", None), pc_slots, sqlite,
    ) if is_pc_mode else []
    pc_summary_text = format_pc_summary(pcs) if is_pc_mode else ""

    # ユーザの @タグ名: user スロットがあればその name、無ければ scenario.user_alias を使う
    user_speaker_name = scenario.user_alias
    if is_pc_mode:
        user_pc = next((p for p in pcs if p.is_user), None)
        if user_pc:
            user_speaker_name = user_pc.name

    # GM が代弁してはならない名前: user 枠 + 全 PC枠名 + 全 AI キャラ本名
    suppress_names: set[str] | None = None
    if is_pc_mode:
        suppress_names = {user_speaker_name}
        for pc in pcs:
            suppress_names.add(pc.name)
            if pc.is_character and pc.character_name:
                suppress_names.add(pc.character_name)

    previous_anticipation = _latest_scenario_anticipation(sqlite.list_scenario_turns(session_id))
    gm_preset_id = getattr(session, "gm_preset_id", "") or ""
    current_synopsis = sqlite.get_scenario_session_synopsis(session_id) or {
        "auto": "",
        "manual": "",
        "last_turn_index": -1,
    }

    if not auto_advance:
        user_turn = _save_turn(
            sqlite=sqlite,
            session_id=session_id,
            speaker_type="user",
            speaker_name=user_speaker_name,
            content=user_message,
        )
        yield ("user_saved", {"turn": scenario_turn_to_dict(user_turn)})

    if engine is None:
        engine = _build_default_engine(sqlite)

    saved_turn_ids: list[str] = []

    # 初動ルーティング決定
    next_kind: str = "gm"
    next_target: str | None = None
    if is_pc_mode and not auto_advance:
        next_kind, next_target = find_last_routing_mention(
            user_message, pcs, npc_names,
        )
        if next_kind == "none":
            next_kind = "gm"

    last_speaker_name: str | None = user_speaker_name if not auto_advance else None
    fired_turns = 0

    try:
        while fired_turns < _MAX_TURNS_PER_USER_TURN:
            if next_kind == "gm":
                # GM ターン実行
                gm_last_raw, gm_last_name = "", None
                async for ev, _meta in _run_gm_turn(
                    engine=engine,
                    scenario=scenario,
                    npcs=npcs,
                    history=sqlite.list_scenario_turns(session_id),
                    user_message=user_message if fired_turns == 0 else "",
                    settings=settings,
                    gm_preset_id=gm_preset_id,
                    auto_advance=auto_advance if fired_turns == 0 else True,
                    synopsis_auto=current_synopsis.get("auto", ""),
                    synopsis_manual=current_synopsis.get("manual", ""),
                    previous_anticipation=previous_anticipation,
                    pc_summary=pc_summary_text,
                    dice_pool=(
                        generate_dice_pool(getattr(scenario, "dice_pool_spec", None))
                        if is_pc_mode else ""
                    ),
                    suppress_names=suppress_names,
                    sqlite=sqlite,
                    session_id=session_id,
                    saved_turn_ids=saved_turn_ids,
                ):
                    yield ev

                # GM の最終 raw_response を直近保存ターンから取り直す
                latest = sqlite.list_scenario_turns(session_id)
                for t in reversed(latest):
                    if getattr(t, "speaker_type", "") in {"narrator", "npc"}:
                        gm_last_raw = getattr(t, "raw_response", "") or ""
                        gm_last_name = getattr(t, "speaker_name", "") or last_speaker_name
                        break
                if gm_last_name:
                    last_speaker_name = gm_last_name
                fired_turns += 1

                if is_pc_mode and gm_last_raw:
                    next_kind, next_target = find_last_routing_mention(
                        gm_last_raw, pcs, npc_names,
                    )
                    # GM 直後の特例: 明示的に @<PC> / @ALL のいずれかが
                    # 指定されていない限り、必ず @ALL にフォールバックする。
                    # find_last_routing_mention は GM ラベル / NPC名 / Narrator も
                    # kind="gm" として返すが、GM が NPC を呼び合うだけのループを
                    # 防ぐため、ここでは pc/all 以外を ALL に格上げする。
                    # PC が 1 人もいなければ ALL は無意味なので従来どおり終了。
                    if next_kind not in {"pc", "all"}:
                        if pcs:
                            next_kind = "all"
                            next_target = None
                            continue
                        break
                    continue
                # 通常モード（ensemble）または PC モードでもメンション無し → 終了
                break

            if next_kind == "all":
                pc = pick_at_all_target(pcs, last_speaker_name=last_speaker_name)
                if pc is None or pc.is_user:
                    break
                next_kind = "pc"
                next_target = pc.name
                # フォールスルー: 同じイテレーションで PC 実行へ

            if next_kind == "pc":
                pc = next((p for p in pcs if p.name == next_target), None)
                if pc is None:
                    break
                if pc.is_user:
                    # ユーザ枠が指名された → ユーザターンへ
                    break
                if chat_service is None:
                    logger.warning(
                        "PC ターン実行不可: chat_service が None session=%s pc=%s",
                        session_id, pc.name,
                    )
                    break

                preset_id = _resolve_pc_preset_id(pc, sqlite)
                if not preset_id:
                    logger.warning(
                        "PC 用プリセット未解決 session=%s slot=%s character_id=%s",
                        session_id, pc.slot_id, pc.character_id,
                    )
                    break

                latest_history = sqlite.list_scenario_turns(session_id)

                yield ("pc_start", {
                    "character": pc.name,
                    "character_id": pc.character_id,
                })
                full_text = ""
                anticipation_text: str | None = None
                try:
                    from backend.services.scenario_chat.pc_runner import stream_pc_response
                    async for ev_type, payload in stream_pc_response(
                        pc=pc,
                        scenario_title=scenario.title,
                        user_alias=user_speaker_name,
                        history=latest_history,
                        preset_id=preset_id,
                        sqlite=sqlite,
                        settings=settings,
                        chat_service=chat_service,
                        scenario_session_id=session_id,
                    ):
                        if ev_type == "pc_done":
                            full_text = payload["full_text"]
                            anticipation_text = payload.get("anticipation")
                        yield (ev_type, payload)
                except Exception as e:
                    logger.exception(
                        "PC ターン実行エラー session=%s pc=%s",
                        session_id, pc.name,
                    )
                    yield ("pc_error", {
                        "character": pc.name,
                        "character_id": pc.character_id,
                        "message": str(e),
                    })
                    break

                if full_text.strip():
                    saved = _save_turn(
                        sqlite=sqlite,
                        session_id=session_id,
                        speaker_type="pc",
                        speaker_name=pc.name,
                        content=full_text,
                        speaker_id=pc.character_id,
                        attach_log_request_id=True,
                        anticipation=anticipation_text,
                    )
                    saved_turn_ids.append(saved.id)
                    yield ("speaker_end", {"turn": scenario_turn_to_dict(saved)})

                last_speaker_name = pc.name
                fired_turns += 1

                # PC 発話末尾のメンションで次話者を決める
                next_kind, next_target = find_last_routing_mention(
                    full_text, pcs, npc_names,
                )
                if next_kind == "none":
                    break
                continue

            # 未知の kind: 防御的に break
            break
    except Exception as e:
        logger.exception("シナリオターン実行エラー session=%s", session_id)
        yield ("error", {"message": str(e)})
        return

    if fired_turns >= _MAX_TURNS_PER_USER_TURN:
        logger.info(
            "シナリオターン上限到達 session=%s fired=%d cap=%d",
            session_id, fired_turns, _MAX_TURNS_PER_USER_TURN,
        )

    sqlite.update_scenario_session(session_id, status=session.status)

    yield ("turn_complete", {"turn_ids": saved_turn_ids})

    progress = compute_synopsis_progress(sqlite, settings, scenario, session_id)
    if progress is not None:
        yield ("synopsis_progress", progress)


async def _run_gm_turn(
    engine,
    scenario,
    npcs: list,
    history: list,
    user_message: str,
    settings: dict,
    gm_preset_id: str,
    auto_advance: bool,
    synopsis_auto: str,
    synopsis_manual: str,
    previous_anticipation: str,
    pc_summary: str,
    dice_pool: str,
    suppress_names: set[str] | None,
    sqlite,
    session_id: str,
    saved_turn_ids: list[str],
) -> AsyncGenerator[tuple[Any, Any], None]:
    """GM 1 ターン分を engine 経由で実行し、SSE イベントを yield しつつ
    scenario_turns へ保存する内部ヘルパ。

    yield 値はメンション主導ループ側で扱いやすいよう (event_tuple, None) の
    2-tuple にしている（将来 metadata を後付けする余地）。
    """
    raw_response = ""
    turn_records_pending: list[TurnRecord] = []

    async for item in engine.generate_stream(
        scenario=scenario,
        npcs=npcs,
        history=history,
        user_message=user_message,
        settings=settings,
        gm_preset_id=gm_preset_id,
        auto_advance=auto_advance,
        synopsis_auto=synopsis_auto,
        synopsis_manual=synopsis_manual,
        previous_anticipation=previous_anticipation,
        pc_summary=pc_summary,
        dice_pool=dice_pool,
        suppress_names=suppress_names,
    ):
        if isinstance(item, UtteranceDelta):
            if item.is_speaker_change:
                yield ((
                    "speaker_start",
                    {
                        "speaker_type": item.speaker_type,
                        "speaker_id": item.speaker_id,
                        "speaker_name": item.speaker_name,
                        "is_known": item.is_known,
                    },
                ), None)
            yield (("content_delta", {"text": item.content_delta}), None)
        elif isinstance(item, TurnRecord):
            turn_records_pending.append(item)
        elif isinstance(item, EngineResult):
            raw_response = item.raw_response

    _, turn_anticipation = extract_anticipation(raw_response)
    last_index = len(turn_records_pending) - 1
    for i, rec in enumerate(turn_records_pending):
        rec_content, _ = extract_anticipation(rec.content)
        saved = _save_turn(
            sqlite=sqlite,
            session_id=session_id,
            speaker_type=rec.speaker_type,
            speaker_name=rec.speaker_name,
            content=rec_content,
            speaker_id=rec.speaker_id,
            raw_response=raw_response,
            attach_log_request_id=True,
            anticipation=turn_anticipation if i == last_index else None,
        )
        saved_turn_ids.append(saved.id)
        yield (("speaker_end", {"turn": scenario_turn_to_dict(saved)}), None)


def _resolve_pc_preset_id(pc, sqlite) -> str:
    """PC キャラに使う LLMModelPreset の ID を返す。

    解決順:
        1. pc.preset_id（セッション側 pc_assignments で指定された preset）
        2. キャラの enabled_providers の任意の1エントリ

    Args:
        pc: PcAssignment（player_type="character"）。
        sqlite: SQLiteStore。

    Returns:
        プリセット ID。解決不能なら空文字列。
    """
    if getattr(pc, "preset_id", None):
        preset_id = pc.preset_id
        if sqlite.get_model_preset(preset_id):
            return preset_id
    char = sqlite.get_character(pc.character_id)
    if not char:
        return ""
    enabled = getattr(char, "enabled_providers", None) or {}
    for preset_id in enabled.keys():
        if sqlite.get_model_preset(preset_id):
            return preset_id
    return ""


def parse_intro_to_turns(
    intro_text: str,
    user_alias: str,
    known_npc_names: dict,
    narrator_name: str = "Narrator",
) -> list[dict]:
    """導入部テキストを `@キャラ: 本文` 記法でパースしてターン辞書のリストを返す。

    GM 出力パーサ（ScenarioChatParser）と違い、`@user_alias:` ブロックも捨てずに
    user 発話として保存する。`@narrator:` は narrator 発話、既知 NPC 名なら npc、
    それ以外は ephemeral NPC として扱う。
    `@` で始まらない冒頭の地の文は Narrator に吸収する。

    Args:
        intro_text: 導入部の生テキスト（複数行可。`@名前:` ブロックを順に並べる）。
        user_alias: ユーザ表示名（@タグで一致する場合 user として扱う）。
        known_npc_names: {NPC名: NPC.id} の辞書。
        narrator_name: ナレーター表示名（@narrator は大小無視）。

    Returns:
        [{speaker_type, speaker_id, speaker_name, content}, ...] のリスト。
        本文が空のブロックはスキップする。
    """
    if not intro_text or not intro_text.strip():
        return []

    def resolve_speaker(raw_name: str) -> tuple[str, str | None, str]:
        name = (raw_name or "").strip()
        if not name:
            return ("narrator", None, narrator_name)
        if name.lower() == user_alias.lower():
            return ("user", None, user_alias)
        if name.lower() == narrator_name.lower():
            return ("narrator", None, narrator_name)
        if name in known_npc_names:
            return ("npc", known_npc_names[name], name)
        return ("npc", None, name)

    blocks: list[dict] = []
    cur_type: str = "narrator"
    cur_id: str | None = None
    cur_name: str = narrator_name
    cur_buffer: list[str] = []

    def flush_block():
        body = "".join(cur_buffer).rstrip()
        if body:
            blocks.append({
                "speaker_type": cur_type,
                "speaker_id": cur_id,
                "speaker_name": cur_name,
                "content": body,
            })

    for raw_line in intro_text.splitlines():
        line = raw_line
        if line.startswith("@"):
            colon = line.find(":", 1)
            if colon > 1:
                flush_block()
                cur_buffer = []
                cur_type, cur_id, cur_name = resolve_speaker(line[1:colon])
                rest = line[colon + 1 :]
                if rest.startswith(" "):
                    rest = rest[1:]
                if rest:
                    cur_buffer.append(rest + "\n")
                continue
        cur_buffer.append(line + "\n")

    flush_block()
    return blocks


def seed_intro_turns(sqlite, session_id: str, scenario) -> int:
    """シナリオ設定の intro をパースして当該セッションの先頭ターンとして保存する。

    `start_session` 直後に呼ぶ想定。すでに intro 由来のターンが存在しないか
    呼出側で保証すること（重複防止）。

    Args:
        sqlite: SQLiteStore インスタンス。
        session_id: 対象セッション ID。
        scenario: Scenario ORM。intro を持つ。

    Returns:
        実際に保存したターン数。
    """
    intro_text = getattr(scenario, "intro", None)
    if not intro_text or not intro_text.strip():
        return 0
    npcs = sqlite.list_scenario_npcs(scenario.id)
    known = {n.name: n.id for n in npcs if getattr(n, "name", None)}
    blocks = parse_intro_to_turns(
        intro_text=intro_text,
        user_alias=scenario.user_alias,
        known_npc_names=known,
    )
    saved = 0
    for b in blocks:
        _save_turn(
            sqlite=sqlite,
            session_id=session_id,
            speaker_type=b["speaker_type"],
            speaker_name=b["speaker_name"],
            content=b["content"],
            speaker_id=b["speaker_id"],
        )
        saved += 1
    return saved


def _save_turn(
    sqlite,
    session_id: str,
    speaker_type: str,
    speaker_name: str,
    content: str,
    speaker_id: str | None = None,
    raw_response: str | None = None,
    attach_log_request_id: bool = False,
    anticipation: str | None = None,
):
    """ターンを次の turn_index で保存して返す共通ヘルパ。

    attach_log_request_id=True のとき、現在の current_message_id を log_request_id として保存する。
    GM ターン保存時のみ True にする（ユーザーターン・intro はログとの紐付け不要）。
    anticipation は GM がターン末尾に書いた予想（期待）。ターンに1つなので、最後の発話行にのみ渡す。
    """
    turn_id = str(uuid.uuid4())
    next_index = sqlite.get_next_scenario_turn_index(session_id)
    log_req_id = current_message_id.get() if attach_log_request_id else None
    return sqlite.create_scenario_turn(
        turn_id=turn_id,
        session_id=session_id,
        turn_index=next_index,
        speaker_type=speaker_type,
        speaker_name=speaker_name,
        content=content,
        speaker_id=speaker_id,
        raw_response=raw_response,
        log_request_id=log_req_id,
        anticipation=anticipation,
    )
