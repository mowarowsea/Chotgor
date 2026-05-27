"""保存記憶（InscribedMemory）レコード CRUD — SQLiteStore Mixin。

`inscribe_memory` ツールでキャラクター本人が長期に残すと決めた記憶を扱う層。
短期記憶（WorkingMemoryThread/Post）とは別系統で、忘却バッチ・想起検索の対象。
"""

from datetime import datetime

class InscribedMemoryStoreMixin:
    """保存記憶レコードの作成・取得・更新・削除を担う Mixin。"""

    def create_inscribed_memory(
        self,
        memory_id: str,
        character_id: str,
        content: str,
        memory_category: str = "general",
        contextual_importance: float = 0.5,
        semantic_importance: float = 0.5,
        identity_importance: float = 0.5,
        user_importance: float = 0.5,
        source_preset_id: str | None = None,
    ):
        """保存記憶レコードを新規作成する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            mem = InscribedMemory(
                id=memory_id,
                character_id=character_id,
                content=content,
                memory_category=memory_category,
                contextual_importance=contextual_importance,
                semantic_importance=semantic_importance,
                identity_importance=identity_importance,
                user_importance=user_importance,
                source_preset_id=source_preset_id,
            )
            session.add(mem)
            session.commit()
            session.refresh(mem)
            return mem

    def get_inscribed_memory(self, memory_id: str):
        """ID で保存記憶レコードを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            return session.get(InscribedMemory, memory_id)

    def list_inscribed_memories(
        self,
        character_id: str,
        category: str | None = None,
        include_deleted: bool = False,
        sort_by: str = "created_at",
    ) -> list:
        """キャラクターの保存記憶一覧を指定順で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            q = session.query(InscribedMemory).filter(
                InscribedMemory.character_id == character_id
            )
            if not include_deleted:
                q = q.filter(InscribedMemory.deleted_at.is_(None))
            if category:
                q = q.filter(InscribedMemory.memory_category == category)
            if sort_by == "updated_at":
                from sqlalchemy import func
                q = q.order_by(
                    func.coalesce(InscribedMemory.updated_at, InscribedMemory.created_at).desc()
                )
            else:
                q = q.order_by(InscribedMemory.created_at.desc())
            return q.all()


    def recall_inscribed_memory(self, memory_id: str) -> None:
        """last_accessed_at を更新し access_count をインクリメントする。

        忘却バッチで「残す」と判断した時・上書き時に使用。
        単純な想起では使わないこと（参照日付更新による decay リセット防止のため）。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            mem = session.get(InscribedMemory, memory_id)
            if mem:
                mem.last_accessed_at = datetime.now()
                mem.access_count = (mem.access_count or 0) + 1
                session.commit()

    def remember_inscribed_memory(self, memory_id: str) -> None:
        """access_count をインクリメントする（last_accessed_at は更新しない）。

        システムによる自動想起時に使用。参照日付を更新しないことで、decay タイマーを保持する。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            mem = session.get(InscribedMemory, memory_id)
            if mem:
                mem.access_count = (mem.access_count or 0) + 1
                session.commit()

    def update_inscribed_memory_for_overwrite(
        self,
        memory_id: str,
        content: str,
        memory_category: str,
        contextual_importance: float,
        semantic_importance: float,
        identity_importance: float,
        user_importance: float,
    ) -> bool:
        """既存保存記憶を in-place 上書き更新する（重複排除時に使用）。

        write 経路で類似既存記憶が見つかった際、既存 ID をそのまま再利用して
        content / category / importances を上書きすることで、ベクトル DB 側も
        単一 upsert（merge_insert）で完結させる。

        access_count / created_at / last_accessed_at は維持し、updated_at のみ更新する。
        deleted_at が設定されている記憶は更新対象外（False を返す）。

        Args:
            memory_id: 更新対象の記憶ID。
            content: 新しい本文。
            memory_category: 新しいカテゴリ。
            contextual_importance: 新しい contextual_importance。
            semantic_importance: 新しい semantic_importance。
            identity_importance: 新しい identity_importance。
            user_importance: 新しい user_importance。

        Returns:
            更新できたら True、対象が存在しない／soft-delete 済みなら False。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            mem = session.get(InscribedMemory, memory_id)
            if not mem or mem.deleted_at is not None:
                return False
            mem.content = content
            mem.memory_category = memory_category
            mem.contextual_importance = contextual_importance
            mem.semantic_importance = semantic_importance
            mem.identity_importance = identity_importance
            mem.user_importance = user_importance
            mem.updated_at = datetime.now()
            session.commit()
            return True

    def soft_delete_inscribed_memory(self, memory_id: str) -> bool:
        """保存記憶をソフト削除する（deleted_at をセット）。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            mem = session.get(InscribedMemory, memory_id)
            if not mem:
                return False
            mem.deleted_at = datetime.now()
            session.commit()
            return True

    def restore_inscribed_memory(self, memory_id: str) -> bool:
        """ソフト削除された保存記憶を復元する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            mem = session.get(InscribedMemory, memory_id)
            if not mem:
                return False
            mem.deleted_at = None
            session.commit()
            return True

    def get_all_active_inscribed_memories(self, character_id: str) -> list:
        """キャラクターの全アクティブ保存記憶（削除済みを除く）を返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import InscribedMemory
            return (
                session.query(InscribedMemory)
                .filter(
                    InscribedMemory.character_id == character_id,
                    InscribedMemory.deleted_at.is_(None),
                )
                .all()
            )
