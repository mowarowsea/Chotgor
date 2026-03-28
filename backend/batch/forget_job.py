"""Routine to identify and forget memories that have time-decayed below a threshold."""

import logging
import re
from typing import Optional

from backend.lib.log_context import new_message_id
from backend.providers.registry import create_provider
from backend.services.memory.manager import MemoryManager
from backend.repositories.sqlite.store import SQLiteStore

logger = logging.getLogger(__name__)


async def run_forget_process(
    character_id: str,
    character_name: str,
    character_system_prompt: str,
    memory_manager: MemoryManager,
    sqlite: SQLiteStore,
    settings: dict,
    threshold: float = 0.2,
    ghost_model: Optional[str] = None,
) -> dict:
    """Run forget process for a character.

    1. Extract candidates with decayed_score < threshold
    2. Ask Claude if any should be kept
    3. Soft-delete those not kept

    Args:
        character_id: 処理対象キャラクターID。
        character_name: キャラクター名（ログ用）。
        character_system_prompt: キャラクターのシステムプロンプト Block 1。
        memory_manager: MemoryManager インスタンス。
        sqlite: SQLiteStore インスタンス。
        settings: get_all_settings() の結果。run_pending_forget で1回だけ取得して渡す。
        threshold: 忘却判定の閾値（decayed_score がこれ未満の記憶が対象）。
        ghost_model: 内省用モデルプリセットID。
    """
    char_label = f"{character_name}@GhostModel"

    candidates = memory_manager.get_forgotten_candidates(character_id, threshold=threshold, limit=50)

    if not candidates:
        logger.info("スキップ char=%s reason=対象記憶なし", char_label)
        return {"status": "skipped", "reason": "No forgotten candidates found"}
        
    memory_text = "【忘れかけている記憶リスト】\n"
    for m in candidates:
        decayed = getattr(m, '_decayed_score', 0)
        c_at = m.created_at.strftime("%Y-%m-%d %H:%M") if m.created_at else "Unknown"
        a_at = m.last_accessed_at.strftime("%Y-%m-%d %H:%M") if m.last_accessed_at else "Never"
        
        memory_text += (
            f"[ID: {m.id}]\n"
            f"- Category: {m.memory_category}\n"
            f"- Decayed Score: {decayed:.3f}\n"
            f"- Base Importance: (Contextual:{m.contextual_importance:.1f}, User:{m.user_importance:.1f}, "
            f"Semantic:{m.semantic_importance:.1f}, Identity:{m.identity_importance:.1f})\n"
            f"- Created At: {c_at}\n"
            f"- Last Recalled At: {a_at}\n"
            f"- Content: {m.content}\n\n"
        )

    system_prompt = (
        f"{character_system_prompt}\n\n"
        "---\n"
        f"今は静かな時間です。\n"
        "以下は、あなたがかつて経験・思考したものの、最近は全く思い出されておらず印象も薄れてきている「忘れかけている記憶」のリストです。\n"
        "これには日々のダイジェスト（要約）も含まれます。\n\n"
        "これらの記憶はまだ消えていません。しかし、このまま放置すると間もなく自然に忘却されます。\n"
        "リストをじっくり見て、「もう手放してもいい」と感じる記憶があれば、そのIDを教えてください。\n"
        "提示された作成日時や重要度、最終想起日時も参考にして判断してください。\n\n"
        "手放したい記憶のIDを、以下のフォーマットで教えてください。\n\n"
        "`[DELETE: ID1, ID2, ID3...]`\n\n"
        "※何も手放さなくていい場合は、`[DELETE: NONE]` とだけ出力するか、何も出力しないでください。\n"
        "※IDは正確に記載してください。"
    )

    logger.debug("LLM呼び出し char=%s candidates=%d", char_label, len(candidates))
    try:
        response_text = await _call_llm_for_forget(system_prompt, memory_text, ghost_model, sqlite, settings)
    except Exception as e:
        logger.exception("エラー char=%s", char_label)
        return {"status": "error", "error": str(e)}

    # Parse [DELETE: ...]
    deleted_ids = set()
    matches = re.findall(r"\[DELETE:\s*(.*?)\]", response_text)
    for match in matches:
        if match.strip().upper() == "NONE":
            continue
        parts = [p.strip() for p in match.split(",")]
        deleted_ids.update(parts)

    deleted_count = 0
    kept_count = 0

    for m in candidates:
        if m.id in deleted_ids:
            deleted_count += 1
            memory_manager.delete_memory(m.id, character_id)
        else:
            kept_count += 1
            # Touching it gives it a boost and prevents it from being forgotten soon.
            sqlite.recall(m.id)

    logger.info(
        "forget 完了 char=%s candidates=%d deleted=%d kept=%d",
        char_label, len(candidates), deleted_count, kept_count,
    )
    return {
        "status": "success",
        "candidates_count": len(candidates),
        "kept_count": kept_count,
        "deleted_count": deleted_count,
        "raw_response": response_text
    }


async def run_pending_forget(sqlite: SQLiteStore, memory_manager: MemoryManager) -> None:
    """全キャラクターに対して forget プロセスを実行する。

    _forget_scheduler から呼び出される。
    settings は全キャラクター共通のため1回だけ取得して各 run_forget_process に渡す。
    各キャラクター処理時に message_id をセットしてログを追跡可能にする。
    """
    characters = sqlite.list_characters()
    # settings はキャラクターをまたいで変わらないため、ループ外で1回だけ取得する
    settings = sqlite.get_all_settings()
    logger.info("開始")

    for char in characters:
        new_message_id()
        threshold = 0.2
        try:
            await run_forget_process(
                character_id=char.id,
                character_name=char.name,
                character_system_prompt=char.system_prompt_block1,
                memory_manager=memory_manager,
                sqlite=sqlite,
                settings=settings,
                threshold=threshold,
                ghost_model=char.ghost_model,
            )
        except Exception:
            logger.exception("エラー char=%s", char.name)

    logger.info("完了")


async def _call_llm_for_forget(
    system_prompt: str,
    memory_text: str,
    ghost_model: Optional[str],
    sqlite: SQLiteStore,
    settings: dict,
) -> str:
    """LLM に忘却判定を問い合わせる内部関数。

    Args:
        system_prompt: キャラクターのシステムプロンプト + 忘却判定指示。
        memory_text: 忘れかけている記憶のリストテキスト。
        ghost_model: 内省用モデルプリセットID。
        sqlite: SQLiteStore インスタンス（プリセット取得に使用）。
        settings: get_all_settings() の結果（run_pending_forget で1回取得済み）。

    Returns:
        LLM の生応答テキスト。
    """
    if not ghost_model:
        raise RuntimeError("ghost_model が設定されていません。キャラクター設定で内省モデルを選択してください。")

    preset = sqlite.get_model_preset(ghost_model)
    if preset is None:
        raise RuntimeError(f"ghost_model に指定されたプリセット '{ghost_model}' が見つかりません。")

    messages = [{"role": "user", "content": memory_text}]
    provider = create_provider(preset.provider, preset.model_id, settings, thinking_level=preset.thinking_level or "default")
    result = await provider.generate(system_prompt, messages)

    text = result.strip() or "(No kept ids)"
    return text
