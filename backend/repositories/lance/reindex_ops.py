"""全テーブルの再インデックス（embedding モデル変更時の drop → 再 embed → insert）。

LanceStore に合成されるミックスイン。self.db / self._embed_query 等の
基盤属性・メソッドは base.LanceStoreBase が提供する前提。
"""

import logging

from backend.repositories.lance.base import (
    _REINDEX_BATCH_SIZE,
    _TABLE_CHAT_TURNS,
    _TABLE_DEFINITIONS,
    _TABLE_INSCRIBED_MEMORIES,
    _TABLE_WORKING_MEMORY_THREADS,
)

logger = logging.getLogger(__name__)


class ReindexOpsMixin:
    """全テーブルの再インデックス（embedding モデル変更時の drop → 再 embed → insert）。"""

    def reindex_all(
        self,
        new_embedding_fn,
        sqlite,
    ) -> dict:
        """embedding model 変更時に全テーブルを drop → 新 embedding で再構築する。

        SQLite を source of truth として全データを読み直し、
        新 embedding で再 embed して LanceStore に流し込む。

        - inscribed_memories     : SQLite ``InscribedMemory`` テーブルから全アクティブ記憶を取得して再 embed
        - chat_turns             : SQLite ``chat_messages`` から全メッセージを取得して再 embed
        - definitions            : SQLite ``characters`` から system_prompt_block1 を取得して再 embed
        - working_memory_threads : SQLite ``working_memory_threads`` から全スレッドを取得し summary + 最新ポストで再 embed

        パフォーマンス: 行データをテーブルごとに一旦メモリに集約し、
        ``_REINDEX_BATCH_SIZE`` 件単位でまとめて embedding サーバへ送る。
        1件ずつ呼び出す ``add_*`` 系と比べ HTTP 往復回数が件数 / バッチサイズ に減るため、
        infinity 等のローカル embedding でも全体時間が大幅に短縮される。

        Args:
            new_embedding_fn: 新しい embedding function（``__call__`` を持つ）。
            sqlite: SQLiteStore インスタンス。

        Returns:
            ``{"inscribed_memories": int, "chat_turns": int, "definitions": int, "working_memory_threads": int}`` の件数辞書。
        """
        from backend.services.chat.indexer import build_chat_doc_and_metadata, get_participant_char_ids

        with self._write_lock:
            # 1. 既存テーブルを全 drop（vector 次元が変わる可能性があるため）
            for name in (
                _TABLE_INSCRIBED_MEMORIES,
                _TABLE_CHAT_TURNS,
                _TABLE_DEFINITIONS,
                _TABLE_WORKING_MEMORY_THREADS,
            ):
                if name in self._list_table_names():
                    self._db.drop_table(name)
                    logger.info("LanceStore.reindex_all: drop %s", name)

            # 2. embedding fn 差し替え + 次元再決定
            self._embedding_fn = new_embedding_fn
            self._vector_dim = None  # 次元を再決定させる

            counts = {
                "inscribed_memories": 0,
                "chat_turns": 0,
                "definitions": 0,
                "working_memory_threads": 0,
            }

            characters = sqlite.list_characters()

            # 3. inscribed_memories: 行データを集約してバッチ embed
            im_rows: list[dict] = []
            im_texts: list[str] = []
            for char in characters:
                memories = sqlite.get_all_active_inscribed_memories(char.id)
                for m in memories:
                    im_rows.append({
                        "id": m.id,
                        "character_id": char.id,
                        "content": m.content,
                        "category": m.memory_category or "",
                        "contextual_importance": float(m.contextual_importance),
                        "semantic_importance": float(m.semantic_importance),
                        "identity_importance": float(m.identity_importance),
                        "user_importance": float(m.user_importance),
                    })
                    im_texts.append(m.content)
            counts["inscribed_memories"] = self._bulk_embed_and_upsert(
                _TABLE_INSCRIBED_MEMORIES, im_rows, im_texts,
            )

            # 4. chat_turns: 行データを集約してバッチ embed
            ct_rows: list[dict] = []
            ct_texts: list[str] = []
            sessions = sqlite.list_chat_sessions(limit=1_000_000)
            for session in sessions:
                participant_ids = get_participant_char_ids(session, sqlite)
                if not participant_ids:
                    continue
                msgs = sqlite.list_chat_messages(session.id)
                for msg in msgs:
                    built = build_chat_doc_and_metadata(msg)
                    if built is None:
                        continue
                    doc, meta = built
                    for char_id in participant_ids:
                        ct_rows.append({
                            "id": msg.id,
                            "character_id": char_id,
                            "content": doc,
                            "session_id": str(meta.get("session_id", "")),
                            "role": str(meta.get("role", "")),
                            "speaker_name": str(meta.get("speaker_name", "")),
                            "created_at": str(meta.get("created_at", "")),
                        })
                        ct_texts.append(doc)
            counts["chat_turns"] = self._bulk_embed_and_upsert(
                _TABLE_CHAT_TURNS, ct_rows, ct_texts,
            )

            # 5. definitions: 行データを集約してバッチ embed
            def_rows: list[dict] = []
            def_texts: list[str] = []
            for char in characters:
                text = (getattr(char, "system_prompt_block1", "") or "").strip()
                if not text:
                    continue
                status = getattr(char, "relationship_status", "active") or "active"
                def_rows.append({
                    "id": char.id,
                    "character_id": char.id,
                    "content": text,
                    "status": status,
                })
                def_texts.append(text)
            counts["definitions"] = self._bulk_embed_and_upsert(
                _TABLE_DEFINITIONS, def_rows, def_texts,
            )

            # 6. working_memory_threads: summary + 最新ポスト本文を素材にバッチ embed
            wm_rows: list[dict] = []
            wm_texts: list[str] = []
            for char in characters:
                threads = sqlite.list_working_memory_threads(char.id)
                for t in threads:
                    latest = sqlite.get_latest_working_memory_post(t.id)
                    index_text = (t.summary or "").strip()
                    if latest and latest.content:
                        index_text = (index_text + "\n" + latest.content).strip()
                    if not index_text:
                        continue
                    wm_rows.append({
                        "id": t.id,
                        "character_id": char.id,
                        "content": index_text,
                        "type": t.type,
                        "importance": float(t.importance),
                        "is_open": int(t.is_open),
                    })
                    wm_texts.append(index_text)
            counts["working_memory_threads"] = self._bulk_embed_and_upsert(
                _TABLE_WORKING_MEMORY_THREADS, wm_rows, wm_texts,
            )

            logger.info("LanceStore.reindex_all 完了 counts=%s", counts)
            return counts

    def _bulk_embed_and_upsert(
        self,
        table_name: str,
        rows: list[dict],
        texts: list[str],
    ) -> int:
        """texts を ``_REINDEX_BATCH_SIZE`` 件ずつまとめて embed し、rows に vector を埋めて merge_insert する。

        reindex_all 専用ヘルパ。embedding サーバへの HTTP 往復を件数 / バッチサイズに削減し、
        ローカル infinity でも全体時間が大幅に短縮される。

        Args:
            table_name: 投入先テーブル名。
            rows: ``vector`` フィールド以外を埋めた行 dict のリスト。長さは texts と一致させる。
            texts: 各行に対応する embedding 素材テキスト。

        Returns:
            投入した行数（= len(rows)）。
        """
        if not rows:
            return 0
        tbl = self._ensure_table(table_name)
        total = len(rows)
        inserted = 0
        for i in range(0, total, _REINDEX_BATCH_SIZE):
            chunk_rows = rows[i:i + _REINDEX_BATCH_SIZE]
            chunk_texts = texts[i:i + _REINDEX_BATCH_SIZE]
            vecs = self._embed_documents(chunk_texts)
            for r, v in zip(chunk_rows, vecs):
                r["vector"] = v
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(chunk_rows)
            )
            inserted += len(chunk_rows)
            logger.info(
                "LanceStore.reindex_all: %s 進捗 %d/%d",
                table_name, inserted, total,
            )
        return inserted
