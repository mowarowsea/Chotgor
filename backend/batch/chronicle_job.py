"""Chronicle: キャラクターのワーキングメモリを棚卸し・蒸留する夜間処理。

Chronicle Time（設定値、デフォルト 03:00）に毎日実行される。
「ワーキングメモリの棚卸し＋蒸留」を主処理とする:

  フェーズ1: 当日会話の読み込み
  フェーズ2: ワーキングメモリの棚卸し
             - 既存スレッドへの当日会話の反映（summary/atmosphere_tag/importance 更新・ポスト追加）
             - 新規スレッドの生成
             - is_open の更新（決着・終息したスレッドを Close）
             - 類似スレッドの統合
  フェーズ3: 長期記憶への昇格判断
             - WM スレッドのうち定着したものを長期記憶へ昇格（inscribe_memory）

farewell_config の更新と estrangement 判定も併せて実施する。

--- 三段階の蒸留パイプライン上の位置づけ ---
記憶は WorkingMemory → InscribedMemory → InnerNarrative の三段階で蒸留される。
Chronicle はこのうち第1段（WM 整理）と第2段（WM → 長期記憶への昇格 inscribe）を担う。
第3段（長期記憶 → inner_narrative への昇華 carve）は Forget バッチへ移譲したため、
Chronicle では inner_narrative の更新（carve）は行わない。
（キャラクターが会話中に自発的に carve するリアルタイム経路は別途維持される。）

--- システムプロンプト設計（1on1 チャット基準に統一） ---
Chronicle は通常チャットと同じ ask_character() / build_system_prompt() を使う。
システムプロンプトは 1on1 チャット基準に統一する方針:

  - working_memory_manager を ask_character() に渡し、全スレッド一覧（Block 6・
    Open/Close 問わず）と emotion/body/relation 固定注入（Block 7）を
    1on1 と同じ形でシステムプロンプトへ入れる。
  - 加えて Chronicle はユーザメッセージ本文（_PROMPT_TEMPLATE）に「Open スレッド」
    「Close スレッド」を ID 付きの棚卸し専用フォーマットで埋め込む。これは LLM が
    棚卸し JSON でスレッドを操作するための作業データである。
  - したがってスレッド情報はシステムプロンプト（自己認識用）とユーザメッセージ
    （操作対象データ用）の両方に現れるが、役割が異なるため重複は許容する。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from backend.lib.log_context import new_message_id, current_log_feature, current_log_target
from backend.lib.tool_event_recorder import result_looks_like_error
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.character_query import ask_character
from backend.services.memory.manager import InscribedMemoryManager
from backend.services.memory.working_memory_manager import WorkingMemoryManager
from backend.character_actions.executor import ToolExecutor
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

## 最近 Close したスレッド（参照用。再燃していれば thread_updates で is_open を true に戻せる）
{closed_threads}

## 今日の会話
{conversation}

## 印象的な長期記憶
{memories}

---

以下を判断してください。

1. **既存スレッドの更新** (thread_updates): 当日の会話を踏まえ、既存スレッドの
   summary / atmosphere_tag / importance を更新したり、新しいポスト(new_post)を追加する。
   決着・終息したスレッドは is_open を false にする（task/topic は解決したら、
   その他は自然に意識から消えたら）。
2. **新規スレッド** (new_threads): 当日の会話から新しい task / topic などが生まれていれば作成する。
3. **スレッド統合** (merges): 「同じ問題の別角度だった」と気づいたスレッドがあれば、
   from_ids のスレッドを into_id に統合する（from_ids は Close される）。
4. **長期記憶への昇格** (inscribe): ワーキングメモリのスレッドのうち、もう短期に置いておく
   段階を越えて、あなたに確かに**定着した**気づきだけを、長期記憶へ昇格させる。
   日々の記録をすべて移すのではなく、本当に魂に残ったものを選び抜くこと。何も昇格しない日も自然。

## 記憶の源泉 (origin)
new_threads と inscribe の各項目には origin を付けてください。
- "real" : ユーザと共有した日常の記憶。
- "usual": ユーザの関与しない、あなただけの生活（うつつ）の体験から生まれた記憶。
今日の会話で「うつつ｜」が冠された出来事に由来するものは "usual" にしてください
（ユーザはあなたの "usual" の記憶を知りません）。源泉が混ざる場合や迷う場合は "real"。

## farewell_config
Chotgorはあなたの感情状態を監視し、望まぬセッションを強制終了します。
{farewell_emotion_rubric}
### 現在の farewell_config
{farewell_config}
変更・新規設定したい場合のみ JSON で指定、不要なら null。

## 出力フォーマット（JSON のみ。変更不要な配列は空、不要な値は null）

{{
  "thread_updates": [
    {{"id": "<既存スレッドID>", "summary": null, "atmosphere_tag": null, "importance": null, "new_post": null, "is_open": null}}
  ],
  "new_threads": [
    {{"type": "task", "summary": "...", "atmosphere_tag": "...", "importance": 0.6, "post": "...", "relation_target": null, "origin": "real"}}
  ],
  "merges": [
    {{"from_ids": ["id1", "id2"], "into_id": "id3", "post": "統合の経緯"}}
  ],
  "inscribe": [
    {{"content": "一人称のあなたの言葉", "category": "contextual", "impact": 1.0, "origin": "real"}}
  ],
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
    """メッセージリストを会話テキストに整形する。

    うつつ（usual_days）由来のターン（_UsualTurnView, is_usual=True）には「うつつ｜」を
    冠し、ユーザ不在の自分だけの生活体験だと明示する。ユーザ会話とうつつが同じ日に
    混在しうるため、凡例を先頭に添えてキャラ本人が記憶の源泉を取り違えないようにする。
    """
    if not messages:
        return "（本日の会話はありません）"
    lines = []
    if any(getattr(m, "is_usual", False) for m in messages):
        lines.append(
            "※「うつつ｜」で始まる行は、ユーザの関与しない、あなただけの生活（うつつ）の"
            "出来事です。ユーザはこれらを知りません。\n"
        )
    for msg in messages:
        role_label = "ユーザ" if msg.role == "user" else msg.character_name or "キャラクター"
        if getattr(msg, "is_usual", False):
            lines.append(f"[うつつ｜{role_label}] {msg.content}")
        else:
            lines.append(f"[{role_label}] {msg.content}")
    return "\n".join(lines)


class _UsualTurnView:
    """うつつ（usual_days）の ScenarioTurn を _format_conversation 互換に見せる薄いアダプタ。

    ScenarioTurn は ChatMessage と属性名が異なる（role/character_name ではなく
    speaker_type/speaker_name）。Chronicle の当日会話整形は ChatMessage 前提なので、
    id/role/character_name/content/created_at を持つビューに正規化する。
    うつつ無人ループにユーザ発話は無いが、念のため speaker_type=="user" のみ role="user"
    とし、それ以外（narrator/npc/character/pc）は speaker_name 表示へ倒す。
    """

    def __init__(self, turn) -> None:
        """ScenarioTurn を受け取り、ChatMessage 互換の属性へ写し替える。"""
        self.id = turn.id
        self.role = "user" if getattr(turn, "speaker_type", "") == "user" else "character"
        self.character_name = getattr(turn, "speaker_name", "") or "キャラクター"
        self.content = getattr(turn, "content", "") or ""
        self.created_at = getattr(turn, "created_at", None)
        # うつつ由来であることのフラグ。_format_conversation がラベル付けに使う。
        self.is_usual = True


def _merge_conversation(messages: list, usual_turns: list) -> list:
    """1on1/グループの ChatMessage とうつつ ScenarioTurn を時系列で1本に合流する。

    うつつターンは _UsualTurnView で ChatMessage 互換に正規化してから created_at で
    マージソートする。created_at が None のものは先頭側へ寄せる（実運用では発生しない）。

    Args:
        messages: ChatMessage のリスト（時系列昇順）。
        usual_turns: うつつ ScenarioTurn のリスト（時系列昇順）。

    Returns:
        created_at 昇順に並べた、_format_conversation へ渡せるビューのリスト。
    """
    items = list(messages) + [_UsualTurnView(t) for t in usual_turns]
    items.sort(key=lambda m: getattr(m, "created_at", None) or datetime.min)
    return items


# Chronicle 出力 JSON の origin フィールドが取りうる値。キャラ本人が記憶の源泉を選ぶ。
_VALID_ORIGINS = {"real", "usual", "interlude"}


def _normalize_origin(value) -> str:
    """Chronicle 出力の origin 値を正規化する（不正・未指定は "real" にフォールバック）。

    "real"=ユーザと共有した日常、"usual"=ユーザ未共有のうつつ生活、
    "interlude"=シナリオPCモードの幕間。後方互換のため未指定は安全側の "real" を返す。

    Args:
        value: LLM 出力中の origin 値（文字列以外・None もありうる）。

    Returns:
        正規化済みの origin（"real" / "usual" / "interlude" のいずれか）。
    """
    v = str(value or "").strip().lower()
    return v if v in _VALID_ORIGINS else "real"


def _safe_float(value, default: float | None) -> float | None:
    """value を float に変換する。変換できない場合は default を返す。

    Chronicle の数値フィールド（importance / impact）は LLM 出力由来のため、
    "high" のような非数値が混ざりうる。float() を直接呼ぶと棚卸し全体が中断するため、
    ここで安全に丸めて1件のスキップに留める。

    Args:
        value: 変換対象（None・非数値文字列もありうる）。
        default: 変換に失敗したときの戻り値（None を許す）。

    Returns:
        変換後の float、または変換不能時の default。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# 棚卸しプロンプトに載せる Close 済みスレッドの最大件数（updated_at 降順で新しい順）。
_CLOSED_THREADS_LIMIT = 30


def _format_threads(threads: list[dict], empty_label: str = "（スレッドはありません）") -> str:
    """ワーキングメモリスレッドのリストを棚卸しプロンプト用テキストに整形する。

    Open / Close いずれのスレッドにも使う共通フォーマッタ。

    Args:
        threads: WorkingMemoryManager.list_threads_by_type() が返す dict のリスト。
        empty_label: スレッドが空のときに返す文言。

    Returns:
        整形テキスト。スレッドがない場合は empty_label を返す。
    """
    if not threads:
        return empty_label
    lines = []
    for t in threads:
        head = f"[{t['id']}] ({t.get('type', '')}) {t.get('summary', '')} 重要度{float(t.get('importance', 0.0)):.2f}"
        lines.append(head)
        atmo = (t.get("atmosphere_tag") or "").strip()
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
    executor: ToolExecutor,
) -> dict:
    """棚卸し結果（thread_updates / new_threads / merges）をワーキングメモリへ反映する。

    Chronicle の JSON 棚卸し結果は最終的に MCP ツール群と同じ操作（post / close / reopen / merge）に
    分解できる。それらを ToolExecutor.execute() 経由で実行することで、タグ方式・tool-use 方式と
    同じ source of truth（tool_call_events）に Chronicle の操作も記録される。これにより
    Logs 画面でも Chronicle の効果がツール使用として可視化される。

    Args:
        character_id: 対象キャラクターID。
        parsed: _parse_chronicle_response() の結果辞書。
        executor: Chronicle 用に作られた ToolExecutor インスタンス。origin は
            execute(origin=...) で1呼び出しごとに渡す（executor のインスタンス状態は汚さない）。

    Returns:
        反映件数の辞書 {updated, created, merged}。
    """
    counts = {"updated": 0, "created": 0, "merged": 0}

    def _ran_ok(tool_name: str, args: dict, *, origin: str | None = None) -> bool:
        """execute(source="chronicle") を1回実行し、成功（エラー文字列でない）なら True を返す。

        ToolExecutor.execute() はツール失敗を例外ではなくエラー文字列で返すため、
        例外捕捉だけでなく result_looks_like_error も確認する。これにより
        「見つからないスレッドへの更新」等の失敗を成功として計上することを防ぐ。
        """
        try:
            result = executor.execute(tool_name, args, source="chronicle", origin=origin)
        except Exception as e:
            logger.warning("chronicle: %s 失敗 args=%s error=%s", tool_name, args, e)
            return False
        if result_looks_like_error(result):
            logger.info("chronicle: %s スキップ args=%s detail=%s", tool_name, args, result)
            return False
        return True

    # 既存スレッドの更新
    for upd in parsed.get("thread_updates") or []:
        if not isinstance(upd, dict):
            continue
        thread_id = upd.get("id")
        if not thread_id:
            continue
        any_change = False

        # 内容（summary / atmosphere_tag / importance / new_post）の更新は
        # post_working_memory_thread の1呼び出しに集約できる。
        summary = upd.get("summary")
        atmosphere_tag = upd.get("atmosphere_tag")
        # importance は LLM 由来で非数値がありうる。安全に丸め、変換不能なら更新対象から外す。
        raw_importance = upd.get("importance")
        importance = _safe_float(raw_importance, None) if raw_importance is not None else None
        new_post = upd.get("new_post")
        if summary is not None or atmosphere_tag is not None or importance is not None or new_post:
            args = {"thread_id": str(thread_id)}
            if summary is not None:
                args["summary"] = str(summary)
            if atmosphere_tag is not None:
                args["atmosphere_tag"] = str(atmosphere_tag)
            if importance is not None:
                args["importance"] = importance
            if new_post:
                args["content"] = str(new_post)
            if _ran_ok("post_working_memory_thread", args):
                any_change = True

        # is_open フラグは close / reopen の独立ツールへ振り分ける。
        # None=変更なし。それ以外は bool() で真偽を取り（JSON の 0/1・文字列も拾う）、
        # 真→reopen / 偽→close。
        is_open = upd.get("is_open")
        if is_open is not None:
            tool = "reopen_working_memory_thread" if bool(is_open) else "close_working_memory_thread"
            if _ran_ok(tool, {"thread_id": str(thread_id)}):
                any_change = True

        if any_change:
            counts["updated"] += 1

    # 新規スレッド: post_working_memory_thread(thread_id 省略) で作成する。
    # origin は item ごとに違うため、execute(origin=...) で1呼び出しごとに渡す。
    for nt in parsed.get("new_threads") or []:
        if not isinstance(nt, dict):
            continue
        args = {
            "type": str(nt.get("type", "")),
            "summary": str(nt.get("summary", "")),
            "atmosphere_tag": str(nt.get("atmosphere_tag", "") or ""),
            "importance": _safe_float(nt.get("importance"), 0.5),
        }
        if nt.get("relation_target"):
            args["relation_target"] = str(nt["relation_target"])
        if nt.get("post"):
            args["content"] = str(nt["post"])
        if _ran_ok(
            "post_working_memory_thread", args,
            origin=_normalize_origin(nt.get("origin")),
        ):
            counts["created"] += 1

    # スレッド統合: from_ids を Close し、統合の経緯を into_id に追記する。
    for merge in parsed.get("merges") or []:
        if not isinstance(merge, dict):
            continue
        into_id = merge.get("into_id")
        from_ids = merge.get("from_ids") or []
        if not into_id or not from_ids:
            continue
        args = {
            "from_ids": [str(fid) for fid in from_ids],
            "into_id": str(into_id),
        }
        if merge.get("post"):
            args["post"] = str(merge["post"])
        if _ran_ok("merge_working_memory_threads", args):
            counts["merged"] += 1

    return counts


def _apply_distillation(
    character_id: str,
    parsed: dict,
    executor: ToolExecutor,
) -> dict:
    """昇格結果（inscribe）を長期記憶へ反映する。

    三段階の蒸留パイプライン上、Chronicle は第2段（WM → 長期記憶への昇格）のみを担う。
    第3段の inner_narrative への昇華（carve）は Forget バッチへ移譲したため、ここでは行わない。

    実行は ToolExecutor.execute("inscribe_memory", ..., source="chronicle") 経由。
    タグ・JSON・tool-use 方式 のいずれと同じ source of truth（tool_call_events）に記録される。
    各 item の origin は item ごとに違いうるため、execute(origin=...) で1呼び出しごとに渡す。

    Args:
        character_id: 対象キャラクターID。
        parsed: _parse_chronicle_response() の結果辞書。
        executor: Chronicle 用に作られた ToolExecutor インスタンス。

    Returns:
        反映件数の辞書 {inscribed}。
    """
    counts = {"inscribed": 0}

    for item in parsed.get("inscribe") or []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        args = {
            "content": content,
            "category": str(item.get("category", "contextual")),
            # impact は LLM 由来で非数値がありうる。安全に丸める（変換不能なら 1.0）。
            "impact": _safe_float(item.get("impact"), 1.0),
        }
        try:
            result = executor.execute(
                "inscribe_memory", args, source="chronicle",
                origin=_normalize_origin(item.get("origin")),
            )
        except Exception as e:
            logger.warning("chronicle: 長期記憶への昇格失敗 char=%s error=%s", character_id, e)
            continue
        if not result_looks_like_error(result):
            counts["inscribed"] += 1

    return counts


async def _check_estrangement(
    char,
    sqlite: SQLiteStore,
    vector_store: "LanceStore" | None,
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
    vector_store: "LanceStore" | None = None,
    memory_manager: InscribedMemoryManager | None = None,
    working_memory_manager: WorkingMemoryManager | None = None,
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
        memory_manager: InscribedMemoryManager インスタンス（記憶蒸留・印象的記憶取得に使用）。
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
        usual_turns = sqlite.get_usual_turns_for_character_on_date(character_id, date_start, date_end)
    else:
        messages = sqlite.get_unchronicled_messages_for_character(char.name)
        usual_turns = sqlite.get_unchronicled_usual_turns_for_character(character_id)

    # 1on1/グループの会話と、うつつ（usual_days）世界の出来事を時系列で1本に合流する。
    # うつつのやり取りは ScenarioTurn 別経路に保存されるため、合流しないと当日会話から漏れる。
    conversation_text = _format_conversation(_merge_conversation(messages, usual_turns))

    def _mark_chronicled() -> None:
        """当日会話として読んだ ChatMessage とうつつ ScenarioTurn を処理済みにする。

        chronicled_at をセットして二重処理（同じやり取りを翌日も当日会話に含める）を防ぐ。
        ChatMessage 側とうつつ ScenarioTurn 側の双方を同じタイミングでマークする。
        """
        if messages:
            sqlite.mark_messages_as_chronicled([m.id for m in messages])
        if usual_turns:
            sqlite.mark_scenario_turns_as_chronicled([t.id for t in usual_turns])

    # Open スレッドは棚卸しの操作対象。Close 済みは参照用（再燃時の再オープン判断・merges 判断）。
    open_threads = working_memory_manager.list_threads_by_type(character_id, is_open=True)
    open_threads_text = _format_threads(open_threads, "（Open なスレッドはありません）")
    closed_threads = working_memory_manager.list_threads_by_type(character_id, is_open=False)
    closed_threads_text = _format_threads(
        closed_threads[:_CLOSED_THREADS_LIMIT], "（Close したスレッドはまだありません）"
    )

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
        closed_threads=closed_threads_text,
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
            working_memory_manager=working_memory_manager,
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
        _mark_chronicled()
        return {"status": "success", "counts": {}}
    if not parsed:
        logger.warning("JSONパース失敗 char=%s raw=%.100s", char_label, response_text)
        return {"status": "error", "error": "JSON のパースに失敗しました", "raw": response_text[:500]}

    # フェーズ2/3 を実行する ToolExecutor を作る。タグ方式・tool-use 方式と同じ
    # 関門（execute）を通すことで、Chronicle の各操作も tool_call_events に source="chronicle"
    # として記録され、Logs 画面でツール使用として可視化される。
    executor = ToolExecutor(
        character_id=character_id,
        session_id=None,
        memory_manager=memory_manager,
        working_memory_manager=working_memory_manager,
        # default_origin は item ごとに書き換えるが、初期値は "real" に置く。
        default_origin="real",
        source_preset_id=char.ghost_model or "",
    )

    # フェーズ2: ワーキングメモリの棚卸し
    wm_counts = _apply_working_memory_updates(character_id, parsed, executor)
    # フェーズ3: 長期記憶への昇格（inscribe）。inner_narrative への昇華は Forget へ移譲済み
    # memory_manager が None なら inscribe は呼び出し時に NoneType エラーになるためスキップ。
    if memory_manager is not None:
        distill_counts = _apply_distillation(character_id, parsed, executor)
    else:
        distill_counts = {"inscribed": 0}

    # farewell_config: null 以外の dict が返された場合のみ更新する
    fc_value = parsed.get("farewell_config")
    if isinstance(fc_value, dict) and fc_value:
        sqlite.update_character(character_id, farewell_config=fc_value)

    _mark_chronicled()

    # farewell_config が設定されていれば疎遠化判定を行う
    updated_char = sqlite.get_character(character_id)
    if updated_char:
        await _check_estrangement(updated_char, sqlite, vector_store)

    counts = {**wm_counts, **distill_counts}
    logger.info("完了 char=%s counts=%s", char_label, counts)
    return {"status": "success", "counts": counts}


async def run_pending_chronicles(
    sqlite: SQLiteStore,
    vector_store: "LanceStore" | None = None,
    memory_manager: InscribedMemoryManager | None = None,
    working_memory_manager: WorkingMemoryManager | None = None,
) -> None:
    """全キャラクターに対して chronicle を実行する。

    _chronicle_scheduler から呼び出される。chronicled_at IS NULL のメッセージが対象。
    ghost_model が設定されていないキャラクターはスキップする。

    Args:
        sqlite: SQLiteStore インスタンス。
        vector_store: LanceStore インスタンス（疎遠化時の embedding 更新に使用。None でもよい）。
        memory_manager: InscribedMemoryManager インスタンス（記憶蒸留に使用。None でもよい）。
        working_memory_manager: WorkingMemoryManager インスタンス（棚卸しに必須）。
    """
    characters = sqlite.list_characters()
    targets = [c for c in characters if c.ghost_model]
    settings = sqlite.get_all_settings()

    logger.info("開始 対象=%d キャラ", len(targets))

    for char in targets:
        new_message_id()
        current_log_feature.set("chronicle")
        current_log_target.set(char.name)
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
