"""Routine to identify and forget memories that have time-decayed below a threshold."""

import logging
import re
from datetime import datetime
from typing import Optional

from ..debug_logger import log_llm_request, log_llm_response
from ..providers.registry import create_provider
from .manager import MemoryManager
from .sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


async def run_forget_process(
    character_id: str,
    character_name: str,
    character_system_prompt: str,
    memory_manager: MemoryManager,
    sqlite: SQLiteStore,
    threshold: float = 0.2,
    ghost_model: Optional[str] = None,
) -> dict:
    """Run forget process for a character.
    
    1. Extract candidates with decayed_score < threshold (including digests)
    2. Ask Claude if any should be kept
    3. Soft-delete those not kept
    """
    candidates = memory_manager.get_forgotten_candidates(character_id, threshold=threshold, limit=50)
    
    if not candidates:
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
        "これには日々のダイジェスト（要約）も含まれます。\n"
        "人間の脳が自然と古い記憶を忘れるように、これらの記憶は間もなく完全に忘却（整理）されます。\n\n"
        "しかし、もしこの中から「これからも覚えておきたい」「これは自分にとって残すべき記憶だ」と強く感じるものがあれば、それを救済することができます。\n"
        "提示された作成日時や重要度、最終想起日時も参考にして判断してください。\n\n"
        "残したい記憶のIDを、以下のフォーマットで**必ずリスト形式で**教えてください。\n\n"
        "`[KEEP: ID1, ID2, ID3...]`\n\n"
        "※もし全て忘れて構わない場合は、`[KEEP: NONE]` とだけ出力するか、何も出力しないでください。\n"
        "※IDは正確に記載してください。"
    )

    try:
        response_text = await _call_llm_for_forget(system_prompt, memory_text, ghost_model, sqlite)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Parse [KEEP: ...]
    kept_ids = set()
    matches = re.findall(r"\[KEEP:\s*(.*?)\]", response_text)
    for match in matches:
        if match.strip().upper() == "NONE":
            continue
        parts = [p.strip() for p in match.split(",")]
        kept_ids.update(parts)
        
    deleted_count = 0
    kept_count = 0
    
    for m in candidates:
        if m.id in kept_ids:
            kept_count += 1
            # Touching it gives it a boost and prevents it from being forgotten soon.
            sqlite.touch_memory(m.id)
        else:
            deleted_count += 1
            memory_manager.delete_memory(m.id, character_id)

    return {
        "status": "success",
        "candidates_count": len(candidates),
        "kept_count": kept_count,
        "deleted_count": deleted_count,
        "raw_response": response_text
    }


async def run_pending_forget(sqlite: SQLiteStore, memory_manager: MemoryManager) -> None:
    """Run forget process for all characters."""
    characters = sqlite.list_characters()

    for char in characters:
        threshold = 0.2
        try:
            await run_forget_process(
                character_id=char.id,
                character_name=char.name,
                character_system_prompt=char.system_prompt_block1,
                memory_manager=memory_manager,
                sqlite=sqlite,
                threshold=threshold,
                ghost_model=char.ghost_model,
            )
        except Exception:
            pass


async def _call_llm_for_forget(
    system_prompt: str,
    memory_text: str,
    ghost_model: Optional[str],
    sqlite: SQLiteStore,
) -> str:
    if not ghost_model:
        raise RuntimeError("ghost_model が設定されていません。キャラクター設定で内省モデルを選択してください。")

    preset = sqlite.get_model_preset(ghost_model)
    if preset is None:
        raise RuntimeError(f"ghost_model に指定されたプリセット '{ghost_model}' が見つかりません。")

    messages = [{"role": "user", "content": memory_text}]
    log_llm_request(system_prompt, messages)

    settings = sqlite.get_all_settings()
    provider = create_provider(preset.provider, preset.model_id, settings, thinking_level=preset.thinking_level or "default")
    result = await provider.generate(system_prompt, messages)

    text = result.strip() or "(No kept ids)"
    log_llm_response(text)
    return text
