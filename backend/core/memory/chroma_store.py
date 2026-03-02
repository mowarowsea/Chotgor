"""ChromaDB integration for semantic memory retrieval."""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, db_path: str):
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

    def _get_collection(self, character_id: str):
        """Get or create a collection per character."""
        collection_name = f"char_{character_id.replace('-', '_')}"
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_memory(
        self,
        memory_id: str,
        content: str,
        character_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add or update a memory embedding in ChromaDB."""
        collection = self._get_collection(character_id)
        meta = {"character_id": character_id}
        if metadata:
            # ChromaDB metadata values must be str/int/float/bool
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
        collection.upsert(
            ids=[memory_id],
            documents=[content],
            metadatas=[meta],
        )

    def recall_memory(
        self,
        query: str,
        character_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Semantic similarity search for memories.

        Returns list of dicts with keys: id, content, distance, metadata.
        """
        collection = self._get_collection(character_id)
        count = collection.count()
        if count == 0:
            return []

        n = min(top_k, count)
        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                memories.append(
                    {
                        "id": mem_id,
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i],
                    }
                )
        return memories

    def delete_memory(self, memory_id: str, character_id: str) -> None:
        """Remove a memory from ChromaDB."""
        collection = self._get_collection(character_id)
        collection.delete(ids=[memory_id])

    def delete_all_memories(self, character_id: str) -> None:
        """Remove the entire collection for a character."""
        collection_name = f"char_{character_id.replace('-', '_')}"
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
