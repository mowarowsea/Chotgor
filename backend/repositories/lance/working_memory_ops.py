"""ワーキングメモリスレッド（working_memory_threads テーブル）の upsert・想起・削除。

LanceStore に合成されるミックスイン。self.db / self._embed_query 等の
基盤属性・メソッドは base.LanceStoreBase が提供する前提。
"""

import logging

from backend.repositories.lance.base import (
    _TABLE_WORKING_MEMORY_THREADS,
    _quote_id,
    _where_dict_to_sql,
)

logger = logging.getLogger(__name__)


class WorkingMemoryThreadOpsMixin:
    """ワーキングメモリスレッド（working_memory_threads テーブル）の upsert・想起・削除。"""

    def upsert_working_memory_thread(
        self,
        thread_id: str,
        index_text: str,
        character_id: str,
        metadata: dict | None = None,
    ) -> None:
        """ワーキングメモリスレッドを ``working_memory_threads`` テーブルに upsert する。

        ``index_text`` は embedding の素材。スレッドの summary 単独では中身に
        強く関連するが summary に出てこない語の想起精度が落ちるため、
        ``summary + 最新ポスト本文`` を結合したテキストを渡すこと。
        summary 更新時・ポスト追加時のどちらでも呼ばれる。

        Args:
            thread_id: スレッドの一意 ID（SQLite の WorkingMemoryThread.id と同一）。
            index_text: embedding する素材テキスト（summary + 最新ポスト本文）。
            character_id: キャラクター ID。
            metadata: ``type`` / ``importance`` / ``is_open`` を読み取る。
        """
        meta = metadata or {}
        with self._write_lock:
            tbl = self._ensure_table(_TABLE_WORKING_MEMORY_THREADS)
            vec = self._embed_documents([index_text])[0]
            row = {
                "id": thread_id,
                "character_id": character_id,
                "content": index_text,
                "vector": vec,
                "type": str(meta.get("type", "")),
                "importance": float(meta.get("importance", 0.5)),
                "is_open": int(meta.get("is_open", 1)),
            }
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )

    def recall_working_memory_threads(
        self,
        query: str,
        character_id: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """類似度検索でワーキングメモリスレッドを取得する（heat 想起用）。

        戻り値の ``distance`` は cosine 距離。呼び出し側（WorkingMemoryManager）が
        ``relevance`` に変換し、importance × 時間減衰と乗じて heat を算出する。

        Args:
            query: 検索クエリテキスト（直近のユーザー発言など）。
            character_id: キャラクター ID。
            top_k: 取得する最大件数。
            where: 追加 where 辞書（``{"type": {"$in": [...]}, "is_open": 1}`` 等）。

        Returns:
            id / content / distance / metadata（type / importance / is_open）のリスト。
        """
        if _TABLE_WORKING_MEMORY_THREADS not in self._list_table_names():
            return []
        tbl = self._db.open_table(_TABLE_WORKING_MEMORY_THREADS)
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
                "LanceStore.recall_working_memory_threads 失敗 char=%s where=%s error=%s",
                character_id, where, e,
            )
            return []

        threads = []
        for r in results:
            threads.append({
                "id": r["id"],
                "content": r["content"],
                "distance": float(r.get("_distance", 0.0)),
                "metadata": {
                    "character_id": r["character_id"],
                    "type": r.get("type", ""),
                    "importance": r.get("importance"),
                    "is_open": r.get("is_open"),
                },
            })
        return threads

    def delete_working_memory_thread(self, thread_id: str) -> None:
        """指定 ID のワーキングメモリスレッドを物理削除する（スレッド削除時に使用）。"""
        if _TABLE_WORKING_MEMORY_THREADS not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_WORKING_MEMORY_THREADS)
            tbl.delete(f"id = {_quote_id(thread_id)}")

    def delete_all_working_memory_threads(self, character_id: str) -> None:
        """指定キャラクターの全スレッドを削除する（キャラクター削除時に使用）。"""
        if _TABLE_WORKING_MEMORY_THREADS not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_WORKING_MEMORY_THREADS)
            tbl.delete(f"character_id = {_quote_id(character_id)}")

    # ─── 全件再インデックス（embedding model 変更時） ─────────────

