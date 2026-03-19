"""記憶レコード CRUD — SQLiteStore Mixin。"""

from datetime import datetime
from typing import Optional


class MemoryStoreMixin:
    """記憶レコードの作成・取得・更新・削除を担う Mixin。"""

    def create_memory(
        self,
        memory_id: str,
        character_id: str,
        content: str,
        memory_category: str = "general",
        contextual_importance: float = 0.5,
        semantic_importance: float = 0.5,
        identity_importance: float = 0.5,
        user_importance: float = 0.5,
        source_preset_id: Optional[str] = None,
    ):
        """記憶レコードを新規作成する。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            mem = Memory(
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

    def get_memory(self, memory_id: str):
        """IDで記憶レコードを取得する。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            return session.get(Memory, memory_id)

    def list_memories(
        self,
        character_id: str,
        category: Optional[str] = None,
        include_deleted: bool = False,
        sort_by: str = "created_at",
    ) -> list:
        """キャラクターの記憶一覧を指定順で返す。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            q = session.query(Memory).filter(Memory.character_id == character_id)
            if not include_deleted:
                q = q.filter(Memory.deleted_at.is_(None))
            if category:
                q = q.filter(Memory.memory_category == category)
            if sort_by == "updated_at":
                from sqlalchemy import func
                q = q.order_by(
                    func.coalesce(Memory.updated_at, Memory.created_at).desc()
                )
            else:
                q = q.order_by(Memory.created_at.desc())
            return q.all()

    def update_memory_content(
        self,
        memory_id: str,
        content: str,
        contextual_importance: float,
        semantic_importance: float,
        identity_importance: float,
        user_importance: float,
    ) -> bool:
        """記憶のcontentと各importanceを上書き更新する。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            mem = session.get(Memory, memory_id)
            if not mem:
                return False
            mem.content = content
            mem.contextual_importance = contextual_importance
            mem.semantic_importance = semantic_importance
            mem.identity_importance = identity_importance
            mem.user_importance = user_importance
            mem.last_accessed_at = datetime.now()
            mem.updated_at = datetime.now()
            session.commit()
            return True

    def touch_memory(self, memory_id: str) -> None:
        """last_accessed_at を更新し access_count をインクリメントする。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            mem = session.get(Memory, memory_id)
            if mem:
                mem.last_accessed_at = datetime.now()
                mem.access_count = (mem.access_count or 0) + 1
                session.commit()

    def soft_delete_memory(self, memory_id: str) -> bool:
        """記憶をソフト削除する（deleted_at をセット）。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            mem = session.get(Memory, memory_id)
            if not mem:
                return False
            mem.deleted_at = datetime.now()
            session.commit()
            return True

    def restore_memory(self, memory_id: str) -> bool:
        """ソフト削除された記憶を復元する。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            mem = session.get(Memory, memory_id)
            if not mem:
                return False
            mem.deleted_at = None
            session.commit()
            return True

    def get_memories_by_date_range(self, character_id: str, start: datetime, end: datetime) -> list:
        """指定期間内に作成された、削除済みでないダイジスト以外の記憶を返す。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            return (
                session.query(Memory)
                .filter(
                    Memory.character_id == character_id,
                    Memory.deleted_at.is_(None),
                    Memory.memory_category != "digest",
                    Memory.created_at >= start,
                    Memory.created_at < end,
                )
                .order_by(Memory.created_at.asc())
                .all()
            )

    def get_all_active_memories(self, character_id: str) -> list:
        """キャラクターの全アクティブ記憶（削除済みを除く）を返す。"""
        with self.get_session() as session:
            from ..sqlite_store import Memory
            return (
                session.query(Memory)
                .filter(
                    Memory.character_id == character_id,
                    Memory.deleted_at.is_(None),
                )
                .all()
            )
