"""Memory manager: coordinates SQLite and ChromaDB, handles write/recall/cleanup."""

import uuid
from typing import Optional

from .chroma_store import ChromaStore
from .sqlite_store import SQLiteStore


class MemoryManager:
    def __init__(self, sqlite: SQLiteStore, chroma: ChromaStore):
        self.sqlite = sqlite
        self.chroma = chroma

    def write_memory(
        self,
        character_id: str,
        content: str,
        category: str = "general",
        contextual_importance: float = 0.5,
        semantic_importance: float = 0.5,
        identity_importance: float = 0.5,
        user_importance: float = 0.5,
    ) -> str:
        """Write a memory to both SQLite and ChromaDB. Returns memory_id."""
        memory_id = str(uuid.uuid4())

        self.sqlite.create_memory(
            memory_id=memory_id,
            character_id=character_id,
            content=content,
            memory_category=category,
            contextual_importance=contextual_importance,
            semantic_importance=semantic_importance,
            identity_importance=identity_importance,
            user_importance=user_importance,
        )

        self.chroma.add_memory(
            memory_id=memory_id,
            content=content,
            character_id=character_id,
            metadata={
                "category": category,
                "contextual_importance": contextual_importance,
                "semantic_importance": semantic_importance,
                "identity_importance": identity_importance,
                "user_importance": user_importance,
            },
        )

        return memory_id

    def recall_memory(
        self,
        character_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Recall memories by semantic similarity, updating access metadata."""
        results = self.chroma.recall_memory(query, character_id, top_k)

        # Update access tracking in SQLite and inject created_at
        for mem in results:
            mem_id = mem.get("id")
            if mem_id:
                self.sqlite.touch_memory(mem_id)
                try:
                    m = self.sqlite.get_memory(mem_id)
                    if m and m.created_at:
                        mem.setdefault("metadata", {})["created_at"] = m.created_at.isoformat(timespec="seconds")
                except Exception:
                    pass

        return results

    def delete_memory(self, memory_id: str, character_id: str) -> bool:
        """Soft delete from SQLite, hard delete from ChromaDB."""
        ok = self.sqlite.soft_delete_memory(memory_id)
        if ok:
            self.chroma.delete_memory(memory_id, character_id)
        return ok

    def list_memories(
        self,
        character_id: str,
        category: Optional[str] = None,
        include_deleted: bool = False,
    ) -> list[dict]:
        """List memories from SQLite with full metadata."""
        mems = self.sqlite.list_memories(character_id, category, include_deleted)
        return [
            {
                "id": m.id,
                "character_id": m.character_id,
                "content": m.content,
                "category": m.memory_category,
                "contextual_importance": m.contextual_importance,
                "semantic_importance": m.semantic_importance,
                "identity_importance": m.identity_importance,
                "user_importance": m.user_importance,
                "access_count": m.access_count,
                "last_accessed_at": (
                    m.last_accessed_at.isoformat() if m.last_accessed_at else None
                ),
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "deleted_at": m.deleted_at.isoformat() if m.deleted_at else None,
            }
            for m in mems
        ]
