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
        - contextual: High weight (1.0), Fast decay (half-life ~4 days)
        - user: Medium weight (0.8), Medium decay (half-life ~10 days)
        - semantic: Medium weight (0.6), Slow decay (half-life ~20 days)
        - identity: Low weight (0.3), Very slow decay (half-life ~90 days)
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

        decay_contextual = memory.contextual_importance * math.exp(-(ln2 / 4.0) * days_passed)
        decay_user = memory.user_importance * math.exp(-(ln2 / 10.0) * days_passed)
        decay_semantic = memory.semantic_importance * math.exp(-(ln2 / 20.0) * days_passed)
        decay_identity = memory.identity_importance * math.exp(-(ln2 / 90.0) * days_passed)

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
        source_preset_id: Optional[str] = None,
    ) -> str:
        """記憶をSQLiteとChromaDBに書き込む。類似記憶があれば更新、なければ新規作成する。

        同一キャラクター・カテゴリ内でコサイン距離 < 0.2 の記憶が既存する場合は
        新規作成せずに上書き更新する（重複排除）。
        更新時は access_count と created_at を引き継ぎ、content・各importance を上書きする。

        Returns:
            書き込んだ記憶のmemory_id（更新の場合は既存ID、新規の場合は新規UUID）。
        """
        # カテゴリ別の更新判定閾値
        # identity は自己定義に関わる記憶のため、ほぼ同文（距離 < 0.05）のみ上書きする
        similarity_threshold = 0.05 if category == "identity" else 0.15

        # 同一カテゴリ内で類似記憶を検索する
        existing_id = self.chroma.find_similar_in_category(
            content=content,
            character_id=character_id,
            category=category,
            threshold=similarity_threshold,
        )

        if existing_id:
            # 類似記憶が見つかった → 旧記憶をsoft-delete＋ChromaDB削除し、新規レコードとして挿入
            self.sqlite.soft_delete_memory(existing_id)
            self.chroma.delete_memory(existing_id, character_id)
            # 以降は新規作成フローに合流する

        # 類似記憶なし → 新規作成
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
            source_preset_id=source_preset_id,
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
        where: Optional[dict] = None,
    ) -> list[dict]:
        """類似度検索＋時間減衰リランクで記憶を想起する。

        処理フロー:
        1. ChromaDB から top_k * 2 件取得（セマンティック検索）
        2. SQLite で各記憶の時間減衰スコアを計算
        3. ハイブリッドスコアでリランクして top_k 件を返す

        Args:
            character_id: キャラクターID。
            query: 検索クエリテキスト。
            top_k: 返す最大件数。
            where: ChromaDB の where フィルタ。recall_with_identity から使用。
        """
        # Fetch more candidates for reranking
        fetch_k = top_k * 2
        results = self.chroma.recall_memory(query, character_id, fetch_k, where=where)

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

        # 想起回数をインクリメント（last_accessed_at は更新しない — decay タイマー保持のため）
        for mem in final_results:
            mem_id = mem.get("id")
            if mem_id:
                self.sqlite.remember(mem_id)

        return final_results

    def recall_with_identity(
        self,
        character_id: str,
        query: str,
        identity_top_k: int = 5,
        other_top_k: int = 5,
    ) -> tuple[list[dict], list[dict]]:
        """identity カテゴリとそれ以外を別枠で想起して返す。

        identity 記憶はキャラクターの自己定義に関わるため、スコアに関係なく常時注入する。
        各枠は独立して recall_memory を呼び出し、リランク済みの結果をそのまま返す。

        Args:
            character_id: キャラクターID。
            query: 検索クエリテキスト。
            identity_top_k: identity カテゴリから返す最大件数。
            other_top_k: identity 以外のカテゴリから返す最大件数。

        Returns:
            (identity_memories, other_memories) のタプル。
        """
        identity_memories = self.recall_memory(
            character_id=character_id,
            query=query,
            top_k=identity_top_k,
            where={"category": "identity"},
        )
        other_memories = self.recall_memory(
            character_id=character_id,
            query=query,
            top_k=other_top_k,
            where={"category": {"$ne": "identity"}},
        )
        return identity_memories, other_memories

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
        sort_by: str = "created_at",
    ) -> list[dict]:
        """キャラクターの記憶一覧をdictリストで返す。"""
        mems = self.sqlite.list_memories(character_id, category, include_deleted, sort_by)
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
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                "deleted_at": m.deleted_at.isoformat() if m.deleted_at else None,
                "source_preset_id": m.source_preset_id,
            }
            for m in mems
        ]
