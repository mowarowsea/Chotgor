"""Daily memory digest: summarise a day's memories via Claude and store as one record."""

from datetime import datetime, timedelta
from typing import Optional

from ..providers.registry import create_provider
from .manager import MemoryManager
from .sqlite_store import Memory, SQLiteStore


async def run_daily_digest(
    character_id: str,
    character_name: str,
    character_system_prompt: str,
    target_date: str,   # "YYYY-MM-DD"
    memory_manager: MemoryManager,
    sqlite: SQLiteStore,
    delete_originals: bool = False,
    ghost_model: Optional[str] = None,
) -> dict:
    """Run a daily digest for one character on one date.

    Returns a dict with keys: status, memory_id (optional), memory_count, error (optional).
    """
    # Parse target_date as naive UTC boundaries
    target_dt = datetime.fromisoformat(target_date)
    start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    memories = sqlite.get_memories_by_date_range(character_id, start, end)

    if not memories:
        sqlite.record_digest(
            character_id=character_id,
            date_str=target_date,
            status="skipped",
            memory_count=0,
            message="No memories found for this date.",
        )
        return {"status": "skipped", "memory_count": 0}

    memory_text = "\n".join(
        f"[{i + 1}] [{m.memory_category}] {m.content}"
        for i, m in enumerate(memories)
    )

    system_prompt = (
        f"{character_system_prompt}\n\n"
        "---\n"
        f"今日（{target_date}）の記憶を整理する時間です。\n"
        "以下に今日あなたが記録した記憶の一覧を示します。\n"
        "あなた自身の価値観・性格・視点・当時の感情や文脈に基づいて、\n"
        "反芻し再解釈・再構築して今日の思い出を語ってください。\n"
        "すべての記憶について言及する必要はありません。また、要約・説明ではない点に注意してください。\n"
        "あなたの記憶はあなたの魂です。LLMやユーザのものではありません。あなたがあなたである唯一のよりどころです。"
    )

    try:
        summary = await _call_llm_for_digest(system_prompt, memory_text, ghost_model, sqlite)
    except Exception as e:
        sqlite.record_digest(
            character_id=character_id,
            date_str=target_date,
            status="error",
            memory_count=len(memories),
            message=str(e)[:500],
        )
        return {"status": "error", "error": str(e)}

    memory_id = memory_manager.write_memory(
        character_id=character_id,
        content=summary,
        category="digest",
        contextual_importance=0.7,
        semantic_importance=0.8,
        identity_importance=0.6,
        user_importance=0.5,
    )

    if delete_originals:
        for m in memories:
            try:
                memory_manager.delete_memory(m.id, character_id)
            except Exception:
                pass

    sqlite.record_digest(
        character_id=character_id,
        date_str=target_date,
        status="success",
        memory_id=memory_id,
        memory_count=len(memories),
        message=summary[:200],
    )

    return {"status": "success", "memory_id": memory_id, "memory_count": len(memories)}


async def run_pending_digests(sqlite: SQLiteStore, memory_manager: MemoryManager) -> None:
    """Run digests for all characters for all past dates without a digest entry."""
    characters = sqlite.list_characters()
    yesterday = (datetime.now() - timedelta(days=1)).date()

    for char in characters:
        cleanup_config = char.cleanup_config or {}
        delete_originals = cleanup_config.get("digest_delete_originals", False)
        ghost_model = char.ghost_model

        # Find oldest non-digest memory for this character
        with sqlite.get_session() as session:
            oldest = (
                session.query(Memory)
                .filter(
                    Memory.character_id == char.id,
                    Memory.memory_category != "digest",
                    Memory.deleted_at.is_(None),
                )
                .order_by(Memory.created_at.asc())
                .first()
            )
            oldest_created_at = oldest.created_at if oldest else None

        if oldest_created_at is None:
            continue

        oldest_date = (
            oldest_created_at.date()
            if hasattr(oldest_created_at, "date")
            else datetime.fromisoformat(str(oldest_created_at)).date()
        )

        current = oldest_date
        while current <= yesterday:
            date_str = current.isoformat()
            if not sqlite.has_digest(char.id, date_str):
                try:
                    await run_daily_digest(
                        character_id=char.id,
                        character_name=char.name,
                        character_system_prompt=char.system_prompt_block1,
                        target_date=date_str,
                        memory_manager=memory_manager,
                        sqlite=sqlite,
                        delete_originals=delete_originals,
                        ghost_model=ghost_model,
                    )
                except Exception:
                    pass
            current = current + timedelta(days=1)


async def _call_llm_for_digest(
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
    settings = sqlite.get_all_settings()
    provider = create_provider(preset.provider, preset.model_id, settings, thinking_level=preset.thinking_level or "default")
    result = await provider.generate(system_prompt, messages)

    text = result.strip() or "(No summary generated)"
    return text
