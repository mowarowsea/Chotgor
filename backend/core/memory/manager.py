"""Memory manager: coordinates SQLite and ChromaDB, handles write/recall/cleanup."""

import uuid
from typing import Optional
from datetime import datetime

from .chroma_store import ChromaStore
from .sqlite_store import SQLiteStore


class MemoryManager:
    def __init__(self, sqlite: SQLiteStore, chroma: ChromaStore):
        self.sqlite = sqlite
        self.chroma = chroma

    def calculate_decayed_score(self, memory, now: Optional[datetime] = None) -> float:
        """Calculate the time-decayed importance score for a memory.
        
        Importance logic:
        - contextual: High weight (1.0), Fast decay (half-life ~7 days)
        - user: Medium weight (0.8), Medium decay (half-life ~30 days)
        - semantic: Medium weight (0.6), Slow decay (half-life ~90 days)
        - identity: Low weight (0.3), No decay (half-life infinity)
        """
        import math
        
        if now is None:
            now = datetime.now()
            
        base_time = memory.last_accessed_at or memory.created_at
        hours_passed = (now - base_time).total_seconds() / 3600.0
        days_passed = hours_passed / 24.0
        if days_passed < 0:
            days_passed = 0.0

        # Math: e^(-lambda * t) where lambda = ln(2) / half_life
        ln2 = 0.69314718
        
        decay_contextual = memory.contextual_importance * math.exp(-(ln2 / 7.0) * days_passed)
        decay_user = memory.user_importance * math.exp(-(ln2 / 30.0) * days_passed)
        decay_semantic = memory.semantic_importance * math.exp(-(ln2 / 90.0) * days_passed)
        decay_identity = memory.identity_importance * 1.0  # No decay

        # Weighted sum
        score = (
            (decay_contextual * 1.0) +
            (decay_user * 0.8) +
            (decay_semantic * 0.6) +
            (decay_identity * 0.3)
        )
        return score

    def get_forgotten_candidates(
        self, character_id: str, threshold: float = 0.3, limit: int = 50
    ) -> list:
        """Return memories whose decayed score falls below the threshold."""
        now = datetime.now()
        candidates = []
        
        memories = self.sqlite.get_all_active_memories(character_id)
        
        for m in memories:
            score = self.calculate_decayed_score(m, now)
            if score < threshold:
                setattr(m, '_decayed_score', score)
                candidates.append(m)
                
        # Sort by score ascending (lowest score first)
        candidates.sort(key=lambda x: getattr(x, '_decayed_score', 0))
        return candidates[:limit]

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
        """Recall memories by semantic similarity, updating access metadata.
        
        Uses Time Decay reranking:
        1. Fetch top_k * 2 from ChromaDB (Semantic Search)
        2. Calculate Decayed Score for each using SQLite (Time-decayed Importance)
        3. Rerank by hybrid score and return top_k
        """
        # Fetch more candidates for reranking
        fetch_k = top_k * 2
        results = self.chroma.recall_memory(query, character_id, fetch_k)

        now = datetime.now()
        
        # Rerank and inject metadata
        reranked = []
        for mem in results:
            mem_id = mem.get("id")
            if not mem_id:
                continue
                
            try:
                m = self.sqlite.get_memory(mem_id)
                if not m:
                    continue
                    
                # Calculate True Decayed Score
                decayed_score = self.calculate_decayed_score(m, now)
                
                # ChromaDB distance is lower-is-better (cosine distance: 0=identical, 2=opposite)
                # Convert distance to similarity (higher is better)
                semantic_similarity = max(0.0, 1.0 - (mem.get("distance", 1.0) / 2.0))
                
                # Hybrid score: Semantic Similarity + Decayed Score
                hybrid_score = (semantic_similarity * 0.5) + (decayed_score * 0.5)
                
                mem["hybrid_score"] = hybrid_score
                mem["decayed_score"] = decayed_score
                mem["semantic_similarity"] = semantic_similarity
                
                if m.created_at:
                    mem.setdefault("metadata", {})["created_at"] = m.created_at.isoformat(timespec="seconds")
                    
                reranked.append(mem)
            except Exception:
                pass
                
        # Sort by hybrid_score descending
        reranked.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        final_results = reranked[:top_k]

        # Update access tracking in SQLite ONLY for the final recalled memories
        for mem in final_results:
            mem_id = mem.get("id")
            if mem_id:
                self.sqlite.touch_memory(mem_id)

        return final_results

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
