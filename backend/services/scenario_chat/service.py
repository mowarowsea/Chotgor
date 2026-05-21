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
from typing import Any, AsyncGenerator, Optional

from backend.lib.log_context import current_log_feature
from backend.services.scenario_chat.context import (
    dropped_history,
    resolve_history_limits,
)
from backend.services.scenario_chat.engine import (
    EngineResult,
    EnsembleEngine,
    SceneEngine,
    TurnRecord,
)
from backend.services.scenario_chat.parser import UtteranceDelta
from backend.services.scenario_chat.synopsis import update_auto_synopsis

logger = logging.getLogger(__name__)

# あらすじ自動更新のトリガー閾値。
# dropped ターン群の累積文字数がこれを超えたら synopsis_auto への追記を試みる。
# 上限到達直後に毎ターン LLM 呼出が走るのを避けるための閾値。
# 手動 regenerate API では無視する（force=True で呼ぶ）。
SYNOPSIS_AUTO_TRIGGER_CHARS = 1500


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
        "gm_preset_id": scenario.gm_preset_id,
        "history_max_turns": scenario.history_max_turns,
        "history_max_chars": scenario.history_max_chars,
        "created_at": scenario.created_at.isoformat() if scenario.created_at else None,
        "updated_at": scenario.updated_at.isoformat() if scenario.updated_at else None,
    }


def scenario_session_to_dict(session: Any) -> dict:
    """ScenarioSession ORM（プレイインスタンス）を JSON 化可能な dict に変換する。"""
    if session is None:
        return {}
    return {
        "id": session.id,
        "scenario_id": session.scenario_id,
        "title": session.title,
        "engine_type": session.engine_type,
        "status": session.status,
        "synopsis_auto": getattr(session, "synopsis_auto", "") or "",
        "synopsis_manual": getattr(session, "synopsis_manual", "") or "",
        "synopsis_last_turn_index": int(getattr(session, "synopsis_last_turn_index", -1) or -1),
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
    *,
    force: bool = False,
) -> Optional[dict]:
    """履歴のうち今回 LLM に渡らない古いターン群を `synopsis_auto` へ蒸留統合する。

    呼び出しタイミング:
        - 通常チャットフロー: 各ターン直前に best-effort で呼ぶ（force=False）。
          dropped ターンの累積文字数が SYNOPSIS_AUTO_TRIGGER_CHARS 未満なら何もしない。
        - 手動 regenerate API: force=True で呼ぶ。閾値判定をスキップする。

    既存の synopsis_auto は単純追記せず、新ターンと統合して**全体を再蒸留**する
    （肥大化を防ぐ）。`synopsis_last_turn_index` によって、すでに蒸留へ
    反映済みのターンは再度対象に含めない。`synopsis_manual` には触らない。

    Returns:
        更新した場合は新しい synopsis dict、何もしなかった場合は None。
        失敗時も None（best-effort）。
    """
    try:
        max_turns, max_chars = resolve_history_limits(scenario, settings)
        dropped = dropped_history(history, max_turns, max_chars)
        if not dropped:
            logger.debug(
                "auto synopsis skip session=%s 理由=dropped空 (履歴上限内 max_turns=%d max_chars=%d turns=%d force=%s)",
                session_id, max_turns, max_chars, len(history), force,
            )
            return None

        synopsis = sqlite.get_scenario_session_synopsis(session_id) or {
            "auto": "",
            "manual": "",
            "last_turn_index": -1,
        }
        last_idx = int(synopsis.get("last_turn_index", -1))

        # turn_index ベースで「まだ要約していない dropped ターン」を抽出
        new_dropped = [
            t for t in dropped
            if int(getattr(t, "turn_index", -1)) > last_idx
        ]
        if not new_dropped:
            dropped_max = max(
                (int(getattr(t, "turn_index", -1)) for t in dropped),
                default=-1,
            )
            # ロールバック前のクランプ漏れが原因のことが多い。WARN で出して気づける状態にする。
            logger.warning(
                "auto synopsis skip session=%s 理由=new_dropped空 "
                "(last_turn_index=%d が dropped最大turn_index=%d 以上。"
                "ターン削除でクランプされていない可能性) force=%s",
                session_id, last_idx, dropped_max, force,
            )
            return None

        if not force:
            total_chars = sum(len(getattr(t, "content", "") or "") for t in new_dropped)
            if total_chars < SYNOPSIS_AUTO_TRIGGER_CHARS:
                logger.debug(
                    "auto synopsis skip session=%s 理由=閾値未満 "
                    "(累積%d < 閾値%d new_dropped=%d)",
                    session_id, total_chars, SYNOPSIS_AUTO_TRIGGER_CHARS, len(new_dropped),
                )
                return None

        def loader(preset_id: str):
            return sqlite.get_model_preset(preset_id)

        logger.info(
            "auto synopsis 蒸留開始 session=%s new_dropped=%d 文字数=%d force=%s",
            session_id,
            len(new_dropped),
            sum(len(getattr(t, "content", "") or "") for t in new_dropped),
            force,
        )
        new_auto = await update_auto_synopsis(
            scenario=scenario,
            dropped_turns=new_dropped,
            existing_auto=synopsis.get("auto", ""),
            settings=settings,
            preset_loader=loader,
        )
        if new_auto is None:
            # update_auto_synopsis 側の WARN で詳細理由は出ているので、ここは要約のみ
            logger.warning(
                "auto synopsis skip session=%s 理由=update_auto_synopsis が None を返却 force=%s",
                session_id, force,
            )
            return None

        latest_idx = max(int(getattr(t, "turn_index", -1)) for t in new_dropped)
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


async def run_scenario_turn(
    session_id: str,
    user_message: str,
    sqlite,
    settings: dict,
    engine: Optional[SceneEngine] = None,
    auto_advance: bool = False,
) -> AsyncGenerator[tuple[str, Any], None]:
    """ユーザ発話を受け取り、シナリオ 1 ターン分の SSE イベントを順次 yield する。

    プレイセッション（scenario_sessions）から元シナリオ（scenarios）を lookup し、
    そのシナリオの NPC・user_alias・GM プリセット等で 1 ターンを進行させる。

    auto_advance=True の場合:
        - user_message は無視される（呼び出し側は空文字を渡してよい）
        - user turn は保存されない（履歴に「無言で促した」痕跡は残らない）
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

    # 1. プレイヤー発話を保存（auto_advance 時はスキップして痕跡を残さない）
    if not auto_advance:
        user_turn = _save_turn(
            sqlite=sqlite,
            session_id=session_id,
            speaker_type="user",
            speaker_name=scenario.user_alias,
            content=user_message,
        )
        yield ("user_saved", {"turn": scenario_turn_to_dict(user_turn)})

    # 2. 履歴を取得（auto_advance 時はプレイヤー発話を含まない最新ターンまで）
    history = sqlite.list_scenario_turns(session_id)

    # 2.5. あらすじ自動更新（best-effort）
    #      履歴上限を超えるターン群があれば、それを LLM で要約して synopsis_auto に追記する。
    #      閾値未満の場合・失敗時は何もしない（チャット本体は継続）。
    #      ここで取得した synopsis を engine に渡し、GM プロンプトに含める。
    updated_synopsis = await maybe_update_auto_synopsis(
        sqlite=sqlite,
        settings=settings,
        scenario=scenario,
        history=history,
        session_id=session_id,
    )
    if updated_synopsis is not None:
        # クライアントへ更新を通知（UI 側のあらすじパネルを最新化させるため）。
        yield ("synopsis_updated", {"synopsis": updated_synopsis})
        current_synopsis = updated_synopsis
    else:
        current_synopsis = sqlite.get_scenario_session_synopsis(session_id) or {
            "auto": "",
            "manual": "",
            "last_turn_index": -1,
        }

    # 3. エンジン実行
    if engine is None:
        engine = _build_default_engine(sqlite)

    saved_turn_ids: list[str] = []
    raw_response: str = ""
    turn_records_pending: list[TurnRecord] = []
    try:
        async for item in engine.generate_stream(
            scenario=scenario,
            npcs=npcs,
            history=history,
            user_message=user_message,
            settings=settings,
            auto_advance=auto_advance,
            synopsis_auto=current_synopsis.get("auto", ""),
            synopsis_manual=current_synopsis.get("manual", ""),
        ):
            if isinstance(item, UtteranceDelta):
                if item.is_speaker_change:
                    speaker_active = True
                    yield (
                        "speaker_start",
                        {
                            "speaker_type": item.speaker_type,
                            "speaker_id": item.speaker_id,
                            "speaker_name": item.speaker_name,
                            "is_known": item.is_known,
                        },
                    )
                yield ("content_delta", {"text": item.content_delta})
            elif isinstance(item, TurnRecord):
                # 確定した発話を後で raw_response 取得後に保存する
                turn_records_pending.append(item)
            elif isinstance(item, EngineResult):
                raw_response = item.raw_response
    except Exception as e:
        logger.exception("シナリオターン実行エラー session=%s", session_id)
        yield ("error", {"message": str(e)})
        return

    # 4. TurnRecord を保存（raw_response はターン内で共通の値を入れる）
    for rec in turn_records_pending:
        saved = _save_turn(
            sqlite=sqlite,
            session_id=session_id,
            speaker_type=rec.speaker_type,
            speaker_name=rec.speaker_name,
            content=rec.content,
            speaker_id=rec.speaker_id,
            raw_response=raw_response,
        )
        saved_turn_ids.append(saved.id)
        yield ("speaker_end", {"turn": scenario_turn_to_dict(saved)})

    # セッションの updated_at を最新化（タイトルは触らない）
    sqlite.update_scenario_session(session_id, status=session.status)

    yield ("turn_complete", {"turn_ids": saved_turn_ids})


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

    # speaker_type 解決ヘルパ：行頭 @ で抽出した名前を 4 種類に分類する。
    def resolve_speaker(raw_name: str) -> tuple[str, Optional[str], str]:
        name = (raw_name or "").strip()
        if not name:
            return ("narrator", None, narrator_name)
        # @user_alias / @narrator は大小無視で判定（GM 出力と揃える）
        if name.lower() == user_alias.lower():
            return ("user", None, user_alias)
        if name.lower() == narrator_name.lower():
            return ("narrator", None, narrator_name)
        if name in known_npc_names:
            return ("npc", known_npc_names[name], name)
        # 未知 → ephemeral NPC として扱う
        return ("npc", None, name)

    blocks: list[dict] = []
    cur_type: str = "narrator"
    cur_id: Optional[str] = None
    cur_name: str = narrator_name
    cur_buffer: list[str] = []

    def flush_block():
        body = "".join(cur_buffer).rstrip()
        # 話者切替直後の先頭スペース 1 文字を除去する（`@名前: 本文` 慣用）
        if body:
            blocks.append({
                "speaker_type": cur_type,
                "speaker_id": cur_id,
                "speaker_name": cur_name,
                "content": body,
            })

    for raw_line in intro_text.splitlines():
        line = raw_line
        # 行頭 @名前: マッチ判定（半角 `:` のみ識別子終端）。
        if line.startswith("@"):
            colon = line.find(":", 1)
            if colon > 1:
                # 直前話者の蓄積をフラッシュ
                flush_block()
                cur_buffer = []
                cur_type, cur_id, cur_name = resolve_speaker(line[1:colon])
                rest = line[colon + 1 :]
                if rest.startswith(" "):
                    rest = rest[1:]
                if rest:
                    cur_buffer.append(rest + "\n")
                continue
        # 通常本文行（@ なしまたは `:` なし）。現在話者にぶら下げる。
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
    speaker_id: Optional[str] = None,
    raw_response: Optional[str] = None,
):
    """ターンを次の turn_index で保存して返す共通ヘルパ。"""
    turn_id = str(uuid.uuid4())
    next_index = sqlite.get_next_scenario_turn_index(session_id)
    return sqlite.create_scenario_turn(
        turn_id=turn_id,
        session_id=session_id,
        turn_index=next_index,
        speaker_type=speaker_type,
        speaker_name=speaker_name,
        content=content,
        speaker_id=speaker_id,
        raw_response=raw_response,
    )
