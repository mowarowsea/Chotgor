"""保存記憶（inscribed_memories テーブル）の CRUD・類似検索。

LanceStore に合成されるミックスイン。self.db / self._embed_query 等の
基盤属性・メソッドは base.LanceStoreBase が提供する前提。
"""

import logging

from backend.repositories.lance.base import (
    _TABLE_INSCRIBED_MEMORIES,
    _quote_id,
    _where_dict_to_sql,
)

logger = logging.getLogger(__name__)


class InscribedMemoryOpsMixin:
    """保存記憶（inscribed_memories テーブル）の CRUD・類似検索。"""

    def add_inscribed_memory(
        self,
        memory_id: str,
        content: str,
        character_id: str,
        metadata: dict | None = None,
    ) -> None:
        """保存記憶を ``inscribed_memories`` テーブルに upsert する（merge_insert）。

        同一 ID が存在すれば更新、なければ挿入。

        Args:
            memory_id: 記憶の一意 ID（SQLite の InscribedMemory.id と同一）。
            content: embedding するテキスト本文。
            character_id: キャラクター ID。
            metadata: 追加メタデータ。``category`` / ``*_importance`` を読み取る。
        """
        meta = metadata or {}
        with self._write_lock:
            tbl = self._ensure_table(_TABLE_INSCRIBED_MEMORIES)
            vec = self._embed_documents([content])[0]
            row = {
                "id": memory_id,
                "character_id": character_id,
                "content": content,
                "vector": vec,
                "category": str(meta.get("category", "")),
                "contextual_importance": float(meta.get("contextual_importance", 0.0)),
                "semantic_importance": float(meta.get("semantic_importance", 0.0)),
                "identity_importance": float(meta.get("identity_importance", 0.0)),
                "user_importance": float(meta.get("user_importance", 0.0)),
            }
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )

    def recall_inscribed_memory(
        self,
        query: str,
        character_id: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """類似度検索で保存記憶を取得する。

        戻り値は ``id`` / ``content`` / ``distance`` / ``metadata`` の dict リスト。
        距離は cosine（0=同一、2=対極）。

        Args:
            query: 検索クエリテキスト。
            character_id: キャラクター ID（必ず character_id でフィルタする）。
            top_k: 取得する最大件数。
            where: 追加 where 辞書（カテゴリ絞り込み等）。

        Returns:
            id / content / distance / metadata のリスト。
        """
        # テーブルが存在しないなら結果は空（embed すら呼ばない）
        if _TABLE_INSCRIBED_MEMORIES not in self._list_table_names():
            return []
        tbl = self._db.open_table(_TABLE_INSCRIBED_MEMORIES)
        if tbl.count_rows() == 0:
            return []

        vec = self._embed_query(query)
        sql_clauses = [f"character_id = {_quote_id(character_id)}"]
        if where:
            extra = _where_dict_to_sql(where)
            if extra:
                sql_clauses.append(f"({extra})")
        full_where = " AND ".join(sql_clauses)

        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where(full_where, prefilter=True)
                .limit(top_k)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning(
                "LanceStore.recall_inscribed_memory 失敗 char=%s where=%s error=%s",
                character_id, where, e,
            )
            return []

        memories = []
        for r in results:
            memories.append({
                "id": r["id"],
                "content": r["content"],
                "distance": float(r.get("_distance", 0.0)),
                "metadata": {
                    "character_id": r["character_id"],
                    "category": r.get("category", ""),
                    "contextual_importance": r.get("contextual_importance"),
                    "semantic_importance": r.get("semantic_importance"),
                    "identity_importance": r.get("identity_importance"),
                    "user_importance": r.get("user_importance"),
                },
            })
        return memories

    def find_similar_in_category(
        self,
        content: str,
        character_id: str,
        category: str,
        threshold: float = 0.15,
    ) -> str | None:
        """同一キャラクター・カテゴリ内で類似する記憶 ID を返す（重複排除用）。

        find_similar 系は「文書同士の比較」なので、検索クエリも文書プレフィックスで embed する。

        Args:
            content: 検索クエリとなる新しい記憶テキスト。
            character_id: キャラクター ID。
            category: 検索対象のカテゴリ。
            threshold: 更新判定のコサイン距離しきい値。

        Returns:
            類似記憶が見つかった場合はその memory_id、見つからなければ None。
        """
        if _TABLE_INSCRIBED_MEMORIES not in self._list_table_names():
            return None
        tbl = self._db.open_table(_TABLE_INSCRIBED_MEMORIES)
        if tbl.count_rows() == 0:
            return None

        # 文書対文書比較なので _embed_documents を使う
        vec = self._embed_documents([content])[0]
        full_where = f"character_id = {_quote_id(character_id)} AND category = {_quote_id(category)}"
        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where(full_where, prefilter=True)
                .limit(1)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning(
                "LanceStore.find_similar_in_category 失敗 char=%s category=%s error=%s",
                character_id, category, e,
            )
            return None

        if not results:
            return None
        if float(results[0].get("_distance", 1.0)) < threshold:
            return results[0]["id"]
        return None

    def delete_inscribed_memory(self, memory_id: str, character_id: str) -> None:
        """指定 ID の保存記憶を物理削除する。

        character_id は呼び出し側互換のため受け取るが、単一テーブルなので
        id だけで一意に特定できる。

        Args:
            memory_id: 削除する記憶 ID。
            character_id: キャラクター ID（互換シグネチャ用）。
        """
        if _TABLE_INSCRIBED_MEMORIES not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_INSCRIBED_MEMORIES)
            tbl.delete(f"id = {_quote_id(memory_id)}")

    def delete_all_inscribed_memories(self, character_id: str) -> None:
        """指定キャラクターの全保存記憶を削除する（キャラクター削除時に使用）。

        LanceStore は単一テーブル方式なので ``character_id`` でフィルタした行を削除する。
        テーブル自体は維持する。
        """
        if _TABLE_INSCRIBED_MEMORIES not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_INSCRIBED_MEMORIES)
            tbl.delete(f"character_id = {_quote_id(character_id)}")

    # ─── チャット履歴コレクション ────────────────────────────────────

