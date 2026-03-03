"""SQLite store for settings, character metadata, and memory records."""

import json
import os
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker



class Base(DeclarativeBase):
    pass


class GlobalSetting(Base):
    __tablename__ = "global_settings"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=True)


class Character(Base):
    __tablename__ = "characters"

    id = Column(String, primary_key=True)  # UUID
    name = Column(String, nullable=False)
    system_prompt_block1 = Column(Text, nullable=False, default="")
    meta_instructions = Column(Text, nullable=False, default="")
    cleanup_config = Column(JSON, nullable=False, default=dict)
    enabled_providers = Column(JSON, nullable=False, default=dict)
    image_data = Column(Text, nullable=True)  # base64 data URI
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )


class Memory(Base):
    __tablename__ = "memories"

    id = Column(String, primary_key=True)  # UUID
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    content = Column(Text, nullable=False)
    memory_category = Column(String, nullable=False, default="general")
    # Importance scores (0.0 - 1.0)
    contextual_importance = Column(Float, default=0.5)
    semantic_importance = Column(Float, default=0.5)
    identity_importance = Column(Float, default=0.5)
    user_importance = Column(Float, default=0.5)
    # Access tracking
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now())
    deleted_at = Column(DateTime, nullable=True)  # Soft delete


class DigestLog(Base):
    __tablename__ = "digest_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    digest_date = Column(String, nullable=False)   # "2026-03-01"
    status = Column(String, nullable=False)         # "success" | "error" | "skipped"
    memory_id = Column(String, nullable=True)       # Created digest memory ID
    memory_count = Column(Integer, default=0)       # Number of source memories
    message = Column(Text, nullable=True)           # Error message or summary excerpt
    created_at = Column(DateTime, default=lambda: datetime.now())


class SQLiteStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self):
        """Add new columns to existing tables (idempotent)."""
        with self.engine.connect() as conn:
            for stmt in [
                "ALTER TABLE characters ADD COLUMN enabled_providers TEXT NOT NULL DEFAULT '{}'",
                "ALTER TABLE characters ADD COLUMN image_data TEXT",
            ]:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass

    def get_session(self) -> Session:
        return self.SessionLocal()

    # --- Global Settings ---

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.get_session() as session:
            row = session.get(GlobalSetting, key)
            if row is None:
                return default
            try:
                return json.loads(row.value)
            except (json.JSONDecodeError, TypeError):
                return row.value

    def set_setting(self, key: str, value: Any) -> None:
        with self.get_session() as session:
            serialized = json.dumps(value) if not isinstance(value, str) else value
            row = session.get(GlobalSetting, key)
            if row:
                row.value = serialized
            else:
                session.add(GlobalSetting(key=key, value=serialized))
            session.commit()

    def get_all_settings(self) -> dict[str, Any]:
        with self.get_session() as session:
            rows = session.query(GlobalSetting).all()
            result = {}
            for row in rows:
                try:
                    result[row.key] = json.loads(row.value)
                except (json.JSONDecodeError, TypeError):
                    result[row.key] = row.value
            return result

    # --- Characters ---

    def create_character(
        self,
        character_id: str,
        name: str,
        system_prompt_block1: str = "",
        meta_instructions: str = "",
        cleanup_config: Optional[dict] = None,
        enabled_providers: Optional[dict] = None,
        image_data: Optional[str] = None,
    ) -> Character:
        with self.get_session() as session:
            char = Character(
                id=character_id,
                name=name,
                system_prompt_block1=system_prompt_block1,
                meta_instructions=meta_instructions,
                cleanup_config=cleanup_config or {},
                enabled_providers=enabled_providers or {},
                image_data=image_data,
            )
            session.add(char)
            session.commit()
            session.refresh(char)
            return char

    def get_character(self, character_id: str) -> Optional[Character]:
        with self.get_session() as session:
            return session.get(Character, character_id)

    def list_characters(self) -> list[Character]:
        with self.get_session() as session:
            return session.query(Character).all()

    def update_character(self, character_id: str, **kwargs) -> Optional[Character]:
        with self.get_session() as session:
            char = session.get(Character, character_id)
            if not char:
                return None
            for k, v in kwargs.items():
                if hasattr(char, k):
                    setattr(char, k, v)
            char.updated_at = datetime.now()
            session.commit()
            session.refresh(char)
            return char

    def delete_character(self, character_id: str) -> bool:
        with self.get_session() as session:
            char = session.get(Character, character_id)
            if not char:
                return False
            session.delete(char)
            session.commit()
            return True

    # --- Memories ---

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
    ) -> Memory:
        with self.get_session() as session:
            mem = Memory(
                id=memory_id,
                character_id=character_id,
                content=content,
                memory_category=memory_category,
                contextual_importance=contextual_importance,
                semantic_importance=semantic_importance,
                identity_importance=identity_importance,
                user_importance=user_importance,
            )
            session.add(mem)
            session.commit()
            session.refresh(mem)
            return mem

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        with self.get_session() as session:
            return session.get(Memory, memory_id)

    def list_memories(
        self,
        character_id: str,
        category: Optional[str] = None,
        include_deleted: bool = False,
    ) -> list[Memory]:
        with self.get_session() as session:
            q = session.query(Memory).filter(Memory.character_id == character_id)
            if not include_deleted:
                q = q.filter(Memory.deleted_at.is_(None))
            if category:
                q = q.filter(Memory.memory_category == category)
            return q.order_by(Memory.created_at.desc()).all()

    def touch_memory(self, memory_id: str) -> None:
        """Update last_accessed_at and increment access_count."""
        with self.get_session() as session:
            mem = session.get(Memory, memory_id)
            if mem:
                mem.last_accessed_at = datetime.now()
                mem.access_count = (mem.access_count or 0) + 1
                session.commit()

    def soft_delete_memory(self, memory_id: str) -> bool:
        with self.get_session() as session:
            mem = session.get(Memory, memory_id)
            if not mem:
                return False
            mem.deleted_at = datetime.now()
            session.commit()
            return True

    def restore_memory(self, memory_id: str) -> bool:
        with self.get_session() as session:
            mem = session.get(Memory, memory_id)
            if not mem:
                return False
            mem.deleted_at = None
            session.commit()
            return True

    def get_memories_by_date_range(
        self, character_id: str, start: datetime, end: datetime
    ) -> list[Memory]:
        """Return non-deleted, non-digest memories created in [start, end)."""
        with self.get_session() as session:
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

    def has_digest(self, character_id: str, date_str: str) -> bool:
        """Return True if any DigestLog entry exists for this character + date."""
        with self.get_session() as session:
            count = (
                session.query(DigestLog)
                .filter(
                    DigestLog.character_id == character_id,
                    DigestLog.digest_date == date_str,
                )
                .count()
            )
            return count > 0

    def record_digest(
        self,
        character_id: str,
        date_str: str,
        status: str,
        memory_id: Optional[str] = None,
        memory_count: int = 0,
        message: Optional[str] = None,
    ) -> None:
        """Append a DigestLog row."""
        with self.get_session() as session:
            log = DigestLog(
                character_id=character_id,
                digest_date=date_str,
                status=status,
                memory_id=memory_id,
                memory_count=memory_count,
                message=message,
            )
            session.add(log)
            session.commit()

    def get_digest_logs(self, character_id: str, limit: int = 50) -> list[DigestLog]:
        """Return recent DigestLog rows for a character, newest first."""
        with self.get_session() as session:
            return (
                session.query(DigestLog)
                .filter(DigestLog.character_id == character_id)
                .order_by(DigestLog.created_at.desc())
                .limit(limit)
                .all()
            )
