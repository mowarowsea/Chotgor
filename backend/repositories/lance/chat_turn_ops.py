"""チャット履歴（chat_turns テーブル）の追加・想起・削除。

LanceStore に合成されるミックスイン。self.db / self._embed_query 等の
基盤属性・メソッドは base.LanceStoreBase が提供する前提。
"""

import logging

from backend.repositories.lance.base import _TABLE_CHAT_TURNS, _quote_id

logger = logging.getLogger(__name__)


class ChatTurnOpsMixin:
    """チャット履歴（chat_turns テーブル）の追加・想起・削除。"""

    def add_chat_turn(
        self,
        message_id: str,
        content: str,
        character_id: str,
        metadata: dict | None = None,
    ) -> None:
        """チャット履歴ターンを ``chat_turns`` テーブルに upsert する。

        Args:
            message_id: SQLite の chat_messages.id と同一。
            content: ``"{speaker_name}: {content}"`` 形式のテキスト。
            character_id: 発言を所属させるキャラクター ID。
            metadata: ``session_id`` / ``role`` / ``speaker_name`` / ``created_at`` を読む。
        """
        meta = metadata or {}
        with self._write_lock:
            tbl = self._ensure_table(_TABLE_CHAT_TURNS)
            vec = self._embed_documents([content])[0]
            row = {
                "id": message_id,
                "character_id": character_id,
                "content": content,
                "vector": vec,
                "session_id": str(meta.get("session_id", "")),
                "role": str(meta.get("role", "")),
                "speaker_name": str(meta.get("speaker_name", "")),
                "created_at": str(meta.get("created_at", "")),
            }
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )

    def recall_chat_turns(
        self,
        query: str,
        character_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """類似度検索でチャット履歴ターンを取得する（PowerRecall 用）。

        戻り値は id / content / distance / metadata の dict リスト。

        Args:
            query: 検索クエリテキスト。
            character_id: キャラクター ID。
            top_k: 取得する最大件数。

        Returns:
            id / content / distance / metadata のリスト。
        """
        if _TABLE_CHAT_TURNS not in self._list_table_names():
            return []
        tbl = self._db.open_table(_TABLE_CHAT_TURNS)
        if tbl.count_rows() == 0:
            return []

        vec = self._embed_query(query)
        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where(f"character_id = {_quote_id(character_id)}", prefilter=True)
                .limit(top_k)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning("LanceStore.recall_chat_turns 失敗 char=%s error=%s", character_id, e)
            return []

        turns = []
        for r in results:
            turns.append({
                "id": r["id"],
                "content": r["content"],
                "distance": float(r.get("_distance", 0.0)),
                "metadata": {
                    "character_id": r["character_id"],
                    "session_id": r.get("session_id", ""),
                    "role": r.get("role", ""),
                    "speaker_name": r.get("speaker_name", ""),
                    "created_at": r.get("created_at", ""),
                },
            })
        return turns

    def delete_all_chat_turns(self, character_id: str) -> None:
        """指定キャラクターの全チャット履歴ターンを削除する（キャラクター削除時に使用）。"""
        if _TABLE_CHAT_TURNS not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_CHAT_TURNS)
            tbl.delete(f"character_id = {_quote_id(character_id)}")

    # ─── キャラクター定義 ────────────────────────────────────────────

