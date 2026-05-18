"""Chronicle: キャラクターのワーキングメモリを棚卸し・蒸留する夜間処理。

Chronicle Time（設定値、デフォルト 03:00）に毎日実行される。
旧 Chronicle は self_history / relationship_state / inner_narrative を毎日全文再構成
していたが、これは「毎日自分を全部書き直す」非人間的な処理だった。

新 Chronicle は「ワーキングメモリの棚卸し＋蒸留」を主処理とする:

  フェーズ1: 当日会話の読み込み
  フェーズ2: ワーキングメモリの棚卸し
             - 既存スレッドへの当日会話の反映（summary/atmosphere/importance 更新・ポスト追加）
             - 新規スレッドの生成
             - is_open の更新（決着・終息したスレッドを Close）
             - 類似スレッドの統合
  フェーズ3: 蒸留判断
             - 長期記憶に残す価値があれば inscribe_memory
             - inner_narrative に反映すべき変化があれば carve_narrative

self_history は「Close した task/topic スレッドの蓄積」が、
relationship_state は「relation スレッド」が自然に代替する。

farewell_config の更新と estrangement 判定は従来どおり別関心事として残す。
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from backend.lib.log_context import new_message_id
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.character_query import ask_character
from backend.services.memory.manager import MemoryManager
from backend.services.memory.working_memory_manager import WorkingMemoryManager
from backend.character_actions.carver import Carver
from backend.character_actions.inscriber import Inscriber
from backend.character_actions.farewell_detector import FAREWELL_EMOTION_RUBRIC

if TYPE_CHECKING:
    from backend.repositories.lance.store import LanceStore

logger = logging.getLogger(__name__)

# キャラクターへのリクエストプロンプトテンプレート
_PROMPT_TEMPLATE = """\
# {character_name}のワーキングメモリ棚卸し
今あなたは睡眠中です。
夢の中で今日一日をぼんやりと振り返り、あなたのワーキングメモリ（並行する認知ストリーム）を
整理してください。日記・要約・報告は不要です。あなた自身の認知を整える作業です。

ワーキングメモリは「スレッド」の集まりです。各スレッドには種別があります:
- task   : 取り組み中の課題（解決を目指す）
- topic  : 引っかかっている話題・問い（解決を目指す）
- emotion: 持続的な感情状態のサマリ（1本のみ・解決を目指さない）
- body   : 持続的な身体状態のサマリ（1本のみ・解決を目指さない）
- relation: 特定の相手との関係（相手ごと1本・解決を目指さない）

## 現在 Open なスレッド
{open_threads}

## 今日の会話
{conversation}

## 印象的な長期記憶
{memories}

---

以下を判断してください。

1. **既存スレッドの更新** (thread_updates): 当日の会話を踏まえ、既存スレッドの
   summary / atmosphere / importance を更新したり、新しいポスト(new_post)を追加する。
   決着・終息したスレッドは is_open を false にする（task/topic は解決したら、
   その他は自然に意識から消えたら）。
2. **新規スレッド** (new_threads): 当日の会話から新しい task / topic などが生まれていれば作成する。
3. **スレッド統合** (merges): 「同じ問題の別角度だった」と気づいたスレッドがあれば、
   from_ids のスレッドを into_id に統合する（from_ids は Close される）。
4. **長期記憶への蒸留** (inscribe): 印象的で長く残す価値のある気づきがあれば永続記憶に刻む。
5. **inner_narrative への蒸留** (carve): 自己像・価値観に恒久的な変化があれば追記する。

## farewell_config
Chotgorはあなたの感情状態を監視し、望まぬセッションを強制終了します。
{farewell_emotion_rubric}
### 現在の farewell_config
{farewell_config}
変更・新規設定したい場合のみ JSON で指定、不要なら null。

## 出力フォーマット（JSON のみ。変更不要な配列は空、不要な値は null）

{{
  "thread_updates": [
    {{"id": "<既存スレッドID>", "summary": null, "atmosphere": null, "importance": null, "new_post": null, "is_open": null}}
  ],
  "new_threads": [
    {{"type": "task", "summary": "...", "atmosphere": "...", "importance": 0.6, "post": "...", "relation_target": null}}
  ],
  "merges": [
    {{"from_ids": ["id1", "id2"], "into_id": "id3", "post": "統合の経緯"}}
  ],
  "inscribe": [
    {{"content": "一人称のあなたの言葉", "category": "contextual", "impact": 1.0}}
  ],
  "carve": {{"mode": "append", "text": "..."}},
  "farewell_config": null
}}

あなた自身の言葉で、明日のあなたに伝わるように整えてください。
"""


def _format_memories(memories: list) -> str:
    """_decayed_score 属性付き記憶リストを chronicle プロンプト用テキストに整形する。"""
    if not memories:
        return "（記憶なし）"
    lines = []
    for m in memories:
        score = getattr(m, "_decayed_score", 0.0)
        lines.append(f"[{m.memory_category}|{score:.2f}] {m.content}")
    return "\n".join(lines)


def _format_conversation(messages: list) -> str:
    """メッセージリストを会話テキストに整形する。"""
    if not messages:
        return "（本日の会話はありません）"
    lines = []
    for msg in messages:
        role_label = "ユーザ" if msg.role == "user" else msg.character_name or "キャラクター"
        lines.append(f"[{role_label}] {msg.content}")
    return "\n".join(lines)


def _format_open_threads(threads: list[dict]) -> str:
    """Open なワーキングメモリスレッドを棚卸しプロンプト用テキストに整形する。

    Args:
        threads: WorkingMemoryManager.list_threads_by_type() が返す dict のリスト。

    Returns:
        整形テキスト。スレッドがない場合はその旨を返す。
    """
    if not threads:
        return "（Open なスレッドはありません）"
    lines = []
    for t in threads:
        head = f"[{t['id']}] ({t.get('type', '')}) {t.get('summary', '')} 重要度{float(t.get('importance', 0.0)):.2f}"
        lines.append(head)
        atmo = (t.get("atmosphere") or "").strip()
        if atmo:
            lines.append(f"  雰囲気: {atmo}")
        latest = (t.get("latest_post") or "").strip()
        if latest:
            lines.append(f"  最新ポスト: {latest}")
    return "\n".join(lines)


def _parse_chronicle_response(response_text: str) -> dict | None:
    """LLM の応答テキストから JSON を抽出してパースする。

    コードブロック（```json ... ```）で囲まれていても対応する。
    LLM が null だけを返した場合は None を返して「変更なし」を示す。

    Args:
        response_text: LLM の応答テキスト。

    Returns:
        パース済み辞書。null 応答は None。パース失敗は空辞書。
    """
    text = response_text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except Exception:
        return {}


def _apply_working_memory_updates(
    character_id: str,
    parsed: dict,
    wm: WorkingMemoryManager,
) -> dict:
    """棚卸し結果（thread_updates / new_threads / merges）をワーキングメモリへ反映する。

    Args:
        character_id: 対象キャラクターID。
        parsed: _parse_chronicle_response() の結果辞書。
        wm: WorkingMemoryManager インスタンス。

    Returns:
        反映件数の辞書 {updated, created, merged}。
    """
    counts = {"updated": 0, "created": 0, "merged": 0}

    # 既存スレッドの更新
    for upd in parsed.get("thread_updates") or []:
        if not isinstance(upd, dict):
            continue
        thread_id = upd.get("id")
        if not thread_id:
            continue
        try:
            summary = upd.get("summary")
            atmosphere = upd.get("atmosphere")
            importance = upd.get("importance")
            if summary is not None or atmosphere is not None or importance is not None:
                wm.update_thread(
                    thread_id,
                    summary=summary,
                    atmosphere=atmosphere,
                    importance=importance,
                )
            new_post = upd.get("new_post")
            if new_post:
                wm.add_post(thread_id, str(new_post))
            is_open = upd.get("is_open")
            if is_open is not None:
                wm.set_open(thread_id, bool(is_open))
            counts["updated"] += 1
        except Exception as e:
            logger.warning("chronicle: スレッド更新失敗 id=%s error=%s", thread_id, e)

    # 新規スレッド
    for nt in parsed.get("new_threads") or []:
        if not isinstance(nt, dict):
            continue
        try:
            wm.create_thread(
                character_id=character_id,
                type=str(nt.get("type", "")),
                summary=str(nt.get("summary", "")),
                atmosphere=str(nt.get("atmosphere", "") or ""),
                importance=float(nt.get("importance", 0.5) or 0.5),
                relation_target=nt.get("relation_target") or None,
                content=nt.get("post") or None,
            )
            counts["created"] += 1
        except ValueError as e:
            logger.info("chronicle: 新規スレッド作成スキップ char=%s error=%s", character_id, e)
        except Exception as e:
            logger.warning("chronicle: 新規スレッド作成失敗 char=%s error=%s", character_id, e)

    # スレッド統合: from_ids を Close し、統合の経緯を into_id に記録する
    for merge in parsed.get("merges") or []:
        if not isinstance(merge, dict):
            continue
        into_id = merge.get("into_id")
        from_ids = merge.get("from_ids") or []
        if not into_id or not from_ids:
            continue
        try:
            post = merge.get("post")
            if post:
                wm.add_post(into_id, str(post))
            for fid in from_ids:
                if fid and fid != into_id:
                    wm.set_open(fid, False)
            counts["merged"] += 1
        except Exception as e:
            logger.warning("chronicle: スレッド統合失敗 into=%s error=%s", into_id, e)

    return counts


def _apply_distillation(
    character_id: str,
    parsed: dict,
    sqlite: SQLiteStore,
    memory_manager: Optional[MemoryManager],
) -> dict:
    """蒸留結果（inscribe / carve）を長期記憶・inner_narrative へ反映する。

    Args:
        character_id: 対象キャラクターID。
        parsed: _parse_chronicle_response() の結果辞書。
        sqlite: SQLiteStore インスタンス。
        memory_manager: MemoryManager インスタンス（inscribe に必要。None ならスキップ）。

    Returns:
        反映件数の辞書 {inscribed, carved}。
    """
    counts = {"inscribed": 0, "carved": 0}

    # 長期記憶への蒸留
    if memory_manager is not None:
        inscriber = Inscriber(character_id, memory_manager)
        for item in parsed.get("inscribe") or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            try:
                inscriber.inscribe_memory(
                    content=content,
                    category=str(item.get("category", "contextual")),
                    impact=float(item.get("impact", 1.0) or 1.0),
                )
                counts["inscribed"] += 1
            except Exception as e:
                logger.warning("chronicle: 記憶蒸留失敗 char=%s error=%s", character_id, e)

    # inner_narrative への蒸留
    carve = parsed.get("carve")
    if isinstance(carve, dict):
        text = str(carve.get("text", "")).strip()
        if text:
            try:
                Carver(character_id, sqlite).carve_narrative(
                    str(carve.get("mode", "append")), text
                )
                counts["carved"] += 1
            except Exception as e:
                logger.warning("chronicle: inner_narrative 蒸留失敗 char=%s error=%s", character_id, e)

    return counts


async def _check_estrangement(
    char,
    sqlite: SQLiteStore,
    vector_store: Optional["LanceStore"],
) -> None:
    """疎遠化条件を確認し、閾値超過で relationship_status を "estranged" に更新する。

    farewell_config.estrangement が未設定のキャラクターはスキップする。
    estrangement 確定後はベクトルストアのキャラクター定義 embedding も更新する。

    Args:
        char: キャラクター ORM オブジェクト。
        sqlite: SQLiteStore インスタンス。
        vector_store: LanceStore インスタンス（None の場合は embedding 更新をスキップ）。
    """
    farewell_config = getattr(char, "farewell_config", None)
    if not farewell_config:
        return
    estrangement = farewell_config.get("estrangement", {})
    lookback_days = estrangement.get("lookback_days")
    threshold = estrangement.get("negative_exit_threshold")
    if not lookback_days or not threshold:
        return

    since = datetime.now() - timedelta(days=lookback_days)
    count = sqlite.get_negative_exit_count(char.name, since)
    if count < threshold:
        return

    logger.info(
        "疎遠化確定 char=%s count=%d threshold=%d lookback_days=%d",
        char.name, count, threshold, lookback_days,
    )
    sqlite.update_character(char.id, relationship_status="estranged")

    if vector_store:
        try:
            vector_store.mark_definition_estranged(char.id)
        except Exception:
            logger.exception("ベクトルストア 疎遠化マーク失敗 char=%s", char.name)


async def run_chronicle(
    character_id: str,
    sqlite: SQLiteStore,
    target_date: str | None = None,   # "YYYY-MM-DD" — 省略時は chronicled_at IS NULL で選択
    settings: dict | None = None,
    vector_store: Optional["LanceStore"] = None,
    memory_manager: Optional[MemoryManager] = None,
    working_memory_manager: Optional[WorkingMemoryManager] = None,
) -> dict:
    """chronicle 処理を実行する（ワーキングメモリの棚卸し＋蒸留）。

    キャラクターの GhostModel に Open スレッド一覧と当日会話を渡し、
    スレッドの更新・新規作成・統合・Close、長期記憶/inner_narrative への蒸留を
    キャラクター自身に判断させる。chronicle 完了後に estrangement 判定を行う。

    Args:
        character_id: キャラクターの UUID。
        sqlite: SQLiteStore インスタンス。
        target_date: 処理対象日 "YYYY-MM-DD"。省略時は chronicled_at IS NULL のメッセージを対象とする。
        settings: グローバル設定辞書。省略時は SQLite から取得する。
        vector_store: LanceStore インスタンス（疎遠化時の embedding 更新に使用。None でもよい）。
        memory_manager: MemoryManager インスタンス（記憶蒸留・印象的記憶取得に使用）。
        working_memory_manager: WorkingMemoryManager インスタンス（棚卸しに必須）。

    Returns:
        処理結果辞書 {status, counts, error (optional)}。
    """
    char = sqlite.get_character(character_id)
    if not char:
        return {"status": "error", "error": f"Character '{character_id}' not found"}

    char_label = f"{char.name}@GhostModel"

    if not char.ghost_model:
        logger.info("スキップ char=%s reason=ghost_model未設定", char_label)
        return {"status": "skipped", "reason": "ghost_model が未設定のためスキップ"}

    preset = sqlite.get_model_preset(char.ghost_model)
    if preset is None:
        return {"status": "error", "error": f"ghost_model '{char.ghost_model}' が見つかりません"}

    if working_memory_manager is None:
        return {"status": "error", "error": "working_memory_manager が渡されていません"}

    if target_date is not None:
        target_dt = datetime.fromisoformat(target_date)
        date_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        messages = sqlite.get_messages_for_character_on_date(char.name, date_start, date_end)
    else:
        messages = sqlite.get_unchronicled_messages_for_character(char.name)

    conversation_text = _format_conversation(messages)
    open_threads = working_memory_manager.list_threads_by_type(character_id, is_open=True)
    open_threads_text = _format_open_threads(open_threads)

    memories = (
        _format_memories(memory_manager.get_top_memorable(character_id, limit=30))
        if memory_manager is not None
        else "（記憶データなし）"
    )

    current_farewell_config = getattr(char, "farewell_config", None)
    farewell_config_text = (
        json.dumps(current_farewell_config, ensure_ascii=False, indent=2)
        if current_farewell_config
        else "（まだ設定されていません）"
    )

    prompt_text = _PROMPT_TEMPLATE.format(
        character_name=char.name,
        open_threads=open_threads_text,
        conversation=conversation_text,
        memories=memories,
        farewell_emotion_rubric=FAREWELL_EMOTION_RUBRIC.strip(),
        farewell_config=farewell_config_text,
    )

    try:
        if settings is None:
            settings = sqlite.get_all_settings()
        logger.debug("LLM呼び出し char=%s target_date=%s", char_label, target_date or "unchronicled")
        response_text = await ask_character(
            character_id=character_id,
            preset_id=char.ghost_model,
            messages=[{"role": "user", "content": prompt_text}],
            sqlite=sqlite,
            settings=settings,
            recall_query=None,
            feature_label="chronicle",
        )
    except Exception as e:
        logger.exception("エラー char=%s", char_label)
        return {"status": "error", "error": str(e)}
    if response_text is None:
        return {"status": "error", "error": "LLMからの応答が取得できませんでした"}

    parsed = _parse_chronicle_response(response_text)
    if parsed is None:
        # null 応答 = 変更なし
        logger.info("変更なし（null応答） char=%s", char_label)
        if messages:
            sqlite.mark_messages_as_chronicled([m.id for m in messages])
        return {"status": "success", "counts": {}}
    if not parsed:
        logger.warning("JSONパース失敗 char=%s raw=%.100s", char_label, response_text)
        return {"status": "error", "error": "JSON のパースに失敗しました", "raw": response_text[:500]}

    # フェーズ2: ワーキングメモリの棚卸し
    wm_counts = _apply_working_memory_updates(character_id, parsed, working_memory_manager)
    # フェーズ3: 蒸留判断
    distill_counts = _apply_distillation(character_id, parsed, sqlite, memory_manager)

    # farewell_config: null 以外の dict が返された場合のみ更新する
    fc_value = parsed.get("farewell_config")
    if isinstance(fc_value, dict) and fc_value:
        sqlite.update_character(character_id, farewell_config=fc_value)

    if messages:
        sqlite.mark_messages_as_chronicled([m.id for m in messages])

    # farewell_config が設定されていれば疎遠化判定を行う
    updated_char = sqlite.get_character(character_id)
    if updated_char:
        await _check_estrangement(updated_char, sqlite, vector_store)

    counts = {**wm_counts, **distill_counts}
    logger.info("完了 char=%s counts=%s", char_label, counts)
    return {"status": "success", "counts": counts}


async def run_pending_chronicles(
    sqlite: SQLiteStore,
    vector_store: Optional["LanceStore"] = None,
    memory_manager: Optional[MemoryManager] = None,
    working_memory_manager: Optional[WorkingMemoryManager] = None,
) -> None:
    """全キャラクターに対して chronicle を実行する。

    _chronicle_scheduler から呼び出される。chronicled_at IS NULL のメッセージが対象。
    ghost_model が設定されていないキャラクターはスキップする。

    Args:
        sqlite: SQLiteStore インスタンス。
        vector_store: LanceStore インスタンス（疎遠化時の embedding 更新に使用。None でもよい）。
        memory_manager: MemoryManager インスタンス（記憶蒸留に使用。None でもよい）。
        working_memory_manager: WorkingMemoryManager インスタンス（棚卸しに必須）。
    """
    characters = sqlite.list_characters()
    targets = [c for c in characters if c.ghost_model]
    settings = sqlite.get_all_settings()

    logger.info("開始 対象=%d キャラ", len(targets))

    for char in targets:
        new_message_id()
        try:
            await run_chronicle(
                character_id=char.id,
                sqlite=sqlite,
                settings=settings,
                vector_store=vector_store,
                memory_manager=memory_manager,
                working_memory_manager=working_memory_manager,
            )
        except Exception as e:
            logger.warning("失敗 char=%s: %s", char.id, e)

    logger.info("完了")
