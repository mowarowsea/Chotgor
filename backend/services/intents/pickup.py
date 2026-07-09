"""意図の拾い上げ — 本人への問いかけで意図を発見・裁定する。

拾い上げは2点（docs/planned/aliveness_plan.md §4.3）:
    - 夜間 Chronicle 同乗（batch/chronicle_job.py の末尾から呼ばれる）
    - うつつシーン完走後（services/scenario_chat/service.run_usual_days_scene の末尾）

1on1 直後のフックは作らない — 会話で芽生えた意図は WM に残り、
**夢の中で発見される**（欲求は事後に遡って発見される、の実装）。

問いの原則:
    - 「あとに残りそうな『〜したい』はある？　なければないでいい。」（捏造の遮断）
    - 重複気味の意図は機械でマージせず、既存 active 一覧を設問に添えて本人に束ねさせる
    - 失効・不満化は機械が候補を挙げ、本人が裁く
"""

import logging
import re
from datetime import datetime

from backend.services.character_query import ask_character
from backend.services.intents.lifecycle import (
    expired_candidates,
    soured_candidates,
)
from backend.services.pressure import compute_pressures

logger = logging.getLogger(__name__)

# 返答から意図操作を拾うタグ。書式は本人向け設問文と対で維持すること。
_NEW_RE = re.compile(r"\[INTENT_NEW:\s*([^|\]]+?)\s*(?:\|\s*([^\]]*?)\s*)?\]")
_RELEASE_RE = re.compile(r"\[INTENT_RELEASE:\s*([0-9a-fA-F-]+)\s*\]")
_SOURED_RE = re.compile(r"\[INTENT_SOURED:\s*([0-9a-fA-F-]+)\s*\|\s*([^\]]+?)\s*\]")

# 新規意図の source_kind 推定に使う閾値（現在圧が最も高い源を当てる。全部低ければ none）
_SOURCE_KIND_MIN = 0.5

# 一晩に拾い上げる新規意図の上限（暴走ガード）
_MAX_NEW_INTENTS = 3


def _guess_source_kind(pressures: dict) -> str:
    """新規意図の source_kind を現在圧から推定する。

    本人に圧の名前を聞くのは過機械化なので、拾い上げ時点で最も高い圧を
    源とみなす（全部低ければ "none"）。誤推定のコストは意図圧の増減カーブが
    少しずれるだけで、意味層（description）は本人の言葉のまま保たれる。
    """
    best_kind, best_value = "none", _SOURCE_KIND_MIN
    for kind in ("social", "boredom", "body"):
        value = float(pressures.get(kind, 0.0))
        if value >= best_value:
            best_kind, best_value = kind, value
    return best_kind


def build_pickup_question(
    active_intents: list,
    expired: list,
    soured: list,
) -> str:
    """拾い上げの設問文を組み立てる。

    Args:
        active_intents: 既存の active 意図（束ね・重複判断の材料として添える）。
        expired: 失効候補（「まだ心にある？」を問う対象）。
        soured: 不満化候補（「出口がなかった？」を問う対象）。

    Returns:
        ask_character に渡す設問テキスト。
    """
    lines = [
        "一日の振り返りのついでに、ひとつだけ。",
        "",
        "あとに残りそうな「〜したい」はある？　**なければないでいい。**",
        "無理にひねり出さないで。本当に残っているものだけを、あなたの言葉のままで。",
    ]
    if active_intents:
        lines += [
            "",
            "いまあなたの中にあるもの（すでに書き留めてある分）:",
        ]
        for intent in active_intents:
            target = f"（相手: {intent.target}）" if intent.target else ""
            lines.append(f"- [{intent.id}] {intent.description}{target}")
        lines += [
            "新しく挙げるものが上のどれかと同じ・ほぼ同じなら、挙げ直さなくていい。",
        ]
    if expired:
        lines += [
            "",
            "それと、ずいぶん前に書き留めたまま静かになっているものがある。"
            "これ、まだ心にある？　もう手放していいなら、そう言って:",
        ]
        for intent in expired:
            lines.append(f"- [{intent.id}] {intent.description}")
    if soured:
        lines += [
            "",
            "こっちは、ずっと心にあるのに叶っていないみたいだ。"
            "もし不満・もどかしさになっているなら、その気持ちを言葉にしてほしい:",
        ]
        for intent in soured:
            lines.append(f"- [{intent.id}] {intent.description}")
    lines += [
        "",
        "答え方（当てはまるものだけ。1〜3個まで）:",
        "- 新しく残したい: `[INTENT_NEW: あなたの言葉のまま | 相手]`"
        "（相手は user / npc:名前 / self のどれか。省略可）",
        "- 手放す: `[INTENT_RELEASE: id]`",
        "- 不満になっている: `[INTENT_SOURED: id | 不満の言葉]`",
        "- 何もなければ、タグは書かなくていい。",
    ]
    return "\n".join(lines)


def parse_pickup_response(text: str) -> dict:
    """本人の返答から意図操作を抽出する。

    Args:
        text: 本人の返答テキスト。

    Returns:
        {"new": [{"description": str, "target": str|None}],
         "release": [id, ...],
         "soured": [{"id": str, "words": str}]} の辞書。
    """
    text = text or ""
    new: list[dict] = []
    for m in _NEW_RE.finditer(text):
        description = m.group(1).strip()
        target = (m.group(2) or "").strip() or None
        if target and target not in ("user", "self") and not target.startswith("npc:"):
            # 書式外の相手表記は npc 扱いに正規化する（本人の言葉は description 側にある）
            target = f"npc:{target}"
        if description:
            new.append({"description": description, "target": target})
    release = [m.group(1).strip() for m in _RELEASE_RE.finditer(text)]
    soured = [
        {"id": m.group(1).strip(), "words": m.group(2).strip()}
        for m in _SOURED_RE.finditer(text)
    ]
    return {"new": new[:_MAX_NEW_INTENTS], "release": release, "soured": soured}


async def run_intent_pickup(
    character_id: str,
    sqlite,
    settings: dict,
    *,
    born_from: str,
    memory_manager=None,
    working_memory_manager=None,
    now: datetime | None = None,
) -> dict:
    """意図の拾い上げを1回実行する（Chronicle 同乗／うつつシーン完走後の共通実装）。

    1. active 意図と現在圧から失効・不満化の候補を挙げる
    2. ask_character（1on1 同等のシステムプロンプト）で本人に問う
    3. 返答のタグを適用する:
       - INTENT_NEW → create_intent（intent.created 封筒）
       - INTENT_RELEASE → resolve_intent(expired)（intent.expired 封筒）
       - INTENT_SOURED → resolve_intent(soured)＋不満の言葉を記憶へ刻む
         （不満化＝利害と合流）

    Args:
        character_id: 対象キャラクター。
        sqlite: SQLiteStore。
        settings: グローバル設定辞書。
        born_from: "night_chronicle" / "usual_scene"（拾い上げ地点の記録）。
        memory_manager: InscribedMemoryManager（soured の言葉の刻み込みに使う。省略可）。
        working_memory_manager: WM マネージャー（1on1 同等ブロック注入用。省略可）。
        now: 基準時刻（テスト注入用）。

    Returns:
        {"status": ..., "created": int, "expired": int, "soured": int} の集計 dict。
    """
    char = sqlite.get_character(character_id)
    if char is None:
        return {"status": "error", "error": f"キャラクターが見つかりません: {character_id}"}
    ghost_model = getattr(char, "ghost_model", None)
    if not ghost_model:
        return {"status": "skipped", "reason": "ghost_model 未設定"}

    active = sqlite.list_intents(character_id, status="active")
    pressures = compute_pressures(sqlite, character_id, now=now)
    expired = expired_candidates(active, pressures, now=now)
    soured = soured_candidates(active, pressures, now=now)
    # 不満化候補が失効候補と重なることは閾値上ないが、念のため排他にする
    expired = [i for i in expired if i not in soured]

    question = build_pickup_question(active, expired, soured)
    response = await ask_character(
        character_id=character_id,
        preset_id=ghost_model,
        messages=[{"role": "user", "content": question}],
        sqlite=sqlite,
        settings=settings,
        recall_query=None,
        feature_label="intent_pickup",
        working_memory_manager=working_memory_manager,
    )
    if not response:
        return {"status": "error", "error": "本人からの返答が取得できませんでした"}

    parsed = parse_pickup_response(response)
    valid_ids = {i.id for i in active}
    created = 0
    for item in parsed["new"]:
        sqlite.create_intent(
            character_id,
            item["description"],
            target=item["target"],
            source_kind=_guess_source_kind(pressures),
            born_from=born_from,
        )
        created += 1
    expired_count = 0
    for intent_id in parsed["release"]:
        if intent_id in valid_ids and sqlite.resolve_intent(intent_id, "expired"):
            expired_count += 1
    soured_count = 0
    for item in parsed["soured"]:
        if item["id"] not in valid_ids:
            continue
        resolved = sqlite.resolve_intent(item["id"], "soured", words=item["words"])
        if resolved is None:
            continue
        soured_count += 1
        # 不満の言葉を記憶へ刻む（不満化＝利害と合流）。embedding 障害等で失敗しても
        # 遷移自体は成立させる（言葉は intent.payload に凍結済み）。
        if memory_manager is not None and item["words"]:
            try:
                memory_manager.write_inscribed_memory(
                    character_id=character_id,
                    content=f"叶わなかった「{resolved.description}」への不満: {item['words']}",
                    category="contextual",
                    contextual_importance=0.7,
                    user_importance=0.5,
                    source_preset_id=ghost_model,
                )
            except Exception:
                logger.exception("不満の刻み込みに失敗 char=%s intent=%s", char.name, item["id"])

    logger.info(
        "意図の拾い上げ完了 char=%s born_from=%s created=%d expired=%d soured=%d",
        char.name, born_from, created, expired_count, soured_count,
    )
    return {
        "status": "success",
        "created": created,
        "expired": expired_count,
        "soured": soured_count,
    }
