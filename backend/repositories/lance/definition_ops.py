"""キャラクター定義（definitions テーブル）の upsert・類似検索・削除。

LanceStore に合成されるミックスイン。self.db / self._embed_query 等の
基盤属性・メソッドは base.LanceStoreBase が提供する前提。
"""

import logging

from backend.repositories.lance.base import _TABLE_DEFINITIONS, _quote_id

logger = logging.getLogger(__name__)


class DefinitionOpsMixin:
    """キャラクター定義（definitions テーブル）の upsert・類似検索・削除。"""

    def upsert_character_definition(
        self,
        character_id: str,
        definition_text: str,
        status: str = "active",
    ) -> str:
        """キャラクター定義を ``definitions`` テーブルに upsert する。

        Args:
            character_id: キャラクター ID（doc ID として使用）。
            definition_text: キャラクター定義テキスト（system_prompt_block1）。
            status: ``"active"`` または ``"estranged"``。

        Returns:
            upsert した doc ID（character_id と同一）。
        """
        with self._write_lock:
            tbl = self._ensure_table(_TABLE_DEFINITIONS)
            vec = self._embed_documents([definition_text])[0]
            row = {
                "id": character_id,
                "character_id": character_id,
                "content": definition_text,
                "vector": vec,
                "status": status,
            }
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )
        return character_id

    def find_similar_definition(
        self,
        definition_text: str,
        exclude_character_id: str = "",
        threshold: float = 0.1,
    ) -> list[dict]:
        """estranged キャラクターの定義と類似するものを返す。

        キャラクター再作成による「なかったことにする」防止のため、status='estranged' に
        絞り込んで類似検索する。

        Args:
            definition_text: 比較対象の定義テキスト。
            exclude_character_id: 除外するキャラクター ID（更新時に自分自身を除外）。
            threshold: コサイン距離しきい値。

        Returns:
            類似する estranged キャラクターの情報リスト（character_id, distance）。
        """
        if _TABLE_DEFINITIONS not in self._list_table_names():
            return []
        tbl = self._db.open_table(_TABLE_DEFINITIONS)
        if tbl.count_rows() == 0:
            return []

        vec = self._embed_documents([definition_text])[0]
        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where("status = 'estranged'", prefilter=True)
                .limit(5)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning("LanceStore.find_similar_definition 失敗 error=%s", e)
            return []

        similar = []
        for r in results:
            if r["character_id"] == exclude_character_id:
                continue
            dist = float(r.get("_distance", 1.0))
            if dist < threshold:
                similar.append({
                    "character_id": r["character_id"],
                    "distance": dist,
                })
        return similar

    def mark_definition_estranged(self, character_id: str) -> None:
        """キャラクター定義の status を ``estranged`` に更新する。"""
        if _TABLE_DEFINITIONS not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_DEFINITIONS)
            try:
                tbl.update(
                    where=f"character_id = {_quote_id(character_id)}",
                    values={"status": "estranged"},
                )
            except Exception as e:
                logger.warning(
                    "LanceStore.mark_definition_estranged 更新失敗 char=%s error=%s",
                    character_id, e,
                )

    def delete_definition(self, character_id: str) -> None:
        """指定キャラクターの定義ベクトルを削除する（キャラクター削除時に使用）。"""
        if _TABLE_DEFINITIONS not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_DEFINITIONS)
            tbl.delete(f"character_id = {_quote_id(character_id)}")

    # ─── ワーキングメモリスレッド ────────────────────────────────────

