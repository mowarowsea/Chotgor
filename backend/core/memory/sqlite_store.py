"""SQLite store — 設定・キャラクターメタデータ・記憶レコードの永続化層。"""

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
    """SQLAlchemy 宣言ベースクラス。"""

    pass


class GlobalSetting(Base):
    """グローバル設定 — キー/バリュー形式の設定テーブル。"""

    __tablename__ = "global_settings"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=True)


class ChatSession(Base):
    """チャットセッション — 1on1またはグループチャットの会話スレッド。"""

    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)   # 1on1: "{char_name}@{preset_name}", グループ: "group"
    title = Column(String, nullable=False, default="新しいチャット")
    session_type = Column(String, nullable=False, default="1on1")   # "1on1" | "group"
    group_config = Column(Text, nullable=True)  # グループチャット設定JSON（session_type="group"時のみ）
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(DateTime, default=lambda: datetime.now(), onupdate=lambda: datetime.now())


class ChatMessage(Base):
    """チャットメッセージ — セッション内の1発言。"""

    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)           # "user" | "character"
    content = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)         # 思考ブロック・想起記憶テキスト
    images = Column(JSON, nullable=True)            # [image_id, ...] 添付画像IDリスト
    character_name = Column(String, nullable=True)  # グループチャット時の発言キャラクター名
    preset_name = Column(String, nullable=True)     # メッセージ送信時に使用したプリセット名
    created_at = Column(DateTime, default=lambda: datetime.now())


class ChatImage(Base):
    """チャット添付画像 — セッションに紐づく画像ファイルのメタデータ。

    実ファイルは uploads_dir/{id} に保存される。
    セッション削除時にファイルとレコードをまとめて削除する。
    """

    __tablename__ = "chat_images"

    id = Column(String, primary_key=True)       # UUID（ファイル名としても使用）
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    message_id = Column(String, nullable=True)  # メッセージ保存後に設定
    mime_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now())


class Character(Base):
    """キャラクター — 人格・システムプロンプト・プロバイダー設定を保持するテーブル。"""

    __tablename__ = "characters"

    id = Column(String, primary_key=True)  # UUID
    name = Column(String, nullable=False)
    system_prompt_block1 = Column(Text, nullable=False, default="")
    meta_instructions = Column(Text, nullable=False, default="")
    cleanup_config = Column(JSON, nullable=False, default=dict)
    enabled_providers = Column(JSON, nullable=False, default=dict)
    ghost_model = Column(String, nullable=True)  # digest/forget に使うプリセットID
    image_data = Column(Text, nullable=True)  # base64 data URI
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )


class Memory(Base):
    """記憶レコード — キャラクターの記憶をカテゴリ・重要度スコアとともに保持する。"""

    __tablename__ = "memories"

    id = Column(String, primary_key=True)  # UUID
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    content = Column(Text, nullable=False)
    memory_category = Column(String, nullable=False, default="general")
    # 重要度スコア (0.0 - 1.0)
    contextual_importance = Column(Float, default=0.5)
    semantic_importance = Column(Float, default=0.5)
    identity_importance = Column(Float, default=0.5)
    user_importance = Column(Float, default=0.5)
    # アクセス追跡
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now())
    deleted_at = Column(DateTime, nullable=True)  # ソフト削除


class DigestLog(Base):
    """ダイジェストログ — 記憶整理処理の実行履歴を記録する。"""

    __tablename__ = "digest_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    digest_date = Column(String, nullable=False)   # "2026-03-01"
    status = Column(String, nullable=False)         # "success" | "error" | "skipped"
    memory_id = Column(String, nullable=True)       # 作成されたダイジスト記憶のID
    memory_count = Column(Integer, default=0)       # 処理元記憶の件数
    message = Column(Text, nullable=True)           # エラーメッセージまたはサマリー抜粋
    created_at = Column(DateTime, default=lambda: datetime.now())


class SessionDrift(Base):
    """SELF_DRIFT — キャラクターがチャット内で自分自身に課した一時的な行動指針。"""

    __tablename__ = "session_drifts"

    id = Column(String, primary_key=True)           # UUID
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    character_id = Column(String, nullable=False)   # キャラクターID（参照のみ）
    content = Column(Text, nullable=False)           # drift内容テキスト
    enabled = Column(Integer, nullable=False, default=1)  # 1=ON, 0=OFF
    created_at = Column(DateTime, default=lambda: datetime.now())


class LLMModelPreset(Base):
    """LLMモデルプリセット — プロバイダー・モデルIDの設定を保持する。"""

    __tablename__ = "llm_model_presets"

    id = Column(String, primary_key=True)          # UUID
    name = Column(String, nullable=False)           # "Google-Gemini3Flash"
    provider = Column(String, nullable=False)       # "google"
    model_id = Column(String, nullable=False, default="")  # "gemini-2.0-flash"
    thinking_level = Column(String, nullable=False, default="default")  # default/low/medium/high
    created_at = Column(DateTime, default=lambda: datetime.now())


class SQLiteStore:
    """SQLite永続化ストア — 全テーブルへのCRUD操作を提供する。"""

    def __init__(self, db_path: str):
        """データベースを初期化し、マイグレーションを実行する。"""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self):
        """既存テーブルへの新カラム追加とテーブル新設を冪等に実行する。"""
        with self.engine.connect() as conn:
            # 既存テーブルへのカラム追加（失敗は無視）
            for stmt in [
                "ALTER TABLE characters ADD COLUMN enabled_providers TEXT NOT NULL DEFAULT '{}'",
                "ALTER TABLE characters ADD COLUMN image_data TEXT",
                "ALTER TABLE characters ADD COLUMN ghost_model TEXT",
                "ALTER TABLE llm_model_presets ADD COLUMN thinking_level TEXT NOT NULL DEFAULT 'default'",
            ]:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass

            # chat_messages テーブルへの新カラム追加（冪等）
            for stmt in [
                "ALTER TABLE chat_messages ADD COLUMN reasoning TEXT",
                "ALTER TABLE chat_messages ADD COLUMN images TEXT",
                "ALTER TABLE chat_messages ADD COLUMN character_name TEXT",
                "ALTER TABLE chat_messages ADD COLUMN preset_name TEXT",
                "ALTER TABLE chat_sessions ADD COLUMN session_type TEXT NOT NULL DEFAULT '1on1'",
                "ALTER TABLE chat_sessions ADD COLUMN group_config TEXT",
            ]:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass

            # チャット関連テーブルの新設（IF NOT EXISTS で冪等）
            for stmt in [
                (
                    "CREATE TABLE IF NOT EXISTS chat_sessions "
                    "(id TEXT PRIMARY KEY, model_id TEXT NOT NULL, "
                    "title TEXT NOT NULL DEFAULT '新しいチャット', "
                    "created_at TIMESTAMP, updated_at TIMESTAMP)"
                ),
                (
                    "CREATE TABLE IF NOT EXISTS chat_messages "
                    "(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                    "role TEXT NOT NULL, content TEXT NOT NULL, reasoning TEXT, images TEXT, "
                    "created_at TIMESTAMP, "
                    "FOREIGN KEY (session_id) REFERENCES chat_sessions(id))"
                ),
                (
                    "CREATE TABLE IF NOT EXISTS chat_images "
                    "(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                    "message_id TEXT, mime_type TEXT NOT NULL, "
                    "created_at TIMESTAMP, "
                    "FOREIGN KEY (session_id) REFERENCES chat_sessions(id))"
                ),
                (
                    "CREATE TABLE IF NOT EXISTS session_drifts "
                    "(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                    "character_id TEXT NOT NULL, content TEXT NOT NULL, "
                    "enabled INTEGER NOT NULL DEFAULT 1, "
                    "created_at TIMESTAMP, "
                    "FOREIGN KEY (session_id) REFERENCES chat_sessions(id))"
                ),
            ]:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass

    def get_session(self) -> Session:
        """新しいDBセッションを返す。"""
        return self.SessionLocal()

    # --- Global Settings ---

    def get_setting(self, key: str, default: Any = None) -> Any:
        """キーで設定値を取得する。JSON文字列は自動的にパースする。"""
        with self.get_session() as session:
            row = session.get(GlobalSetting, key)
            if row is None:
                return default
            try:
                return json.loads(row.value)
            except (json.JSONDecodeError, TypeError):
                return row.value

    def set_setting(self, key: str, value: Any) -> None:
        """設定値をupsertする。文字列以外はJSONシリアライズする。"""
        with self.get_session() as session:
            serialized = json.dumps(value) if not isinstance(value, str) else value
            row = session.get(GlobalSetting, key)
            if row:
                row.value = serialized
            else:
                session.add(GlobalSetting(key=key, value=serialized))
            session.commit()

    def get_all_settings(self) -> dict[str, Any]:
        """全設定をdict形式で返す。"""
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
        ghost_model: Optional[str] = None,
        image_data: Optional[str] = None,
    ) -> Character:
        """キャラクターを新規作成する。"""
        with self.get_session() as session:
            char = Character(
                id=character_id,
                name=name,
                system_prompt_block1=system_prompt_block1,
                meta_instructions=meta_instructions,
                cleanup_config=cleanup_config or {},
                enabled_providers=enabled_providers or {},
                ghost_model=ghost_model,
                image_data=image_data,
            )
            session.add(char)
            session.commit()
            session.refresh(char)
            return char

    def get_character(self, character_id: str) -> Optional[Character]:
        """IDでキャラクターを取得する。"""
        with self.get_session() as session:
            return session.get(Character, character_id)

    def get_character_by_name(self, name: str) -> Optional[Character]:
        """名前でキャラクターを取得する。"""
        with self.get_session() as session:
            return session.query(Character).filter(Character.name == name).first()

    def list_characters(self) -> list[Character]:
        """全キャラクターを返す。"""
        with self.get_session() as session:
            return session.query(Character).all()

    def update_character(self, character_id: str, **kwargs) -> Optional[Character]:
        """キャラクターの指定フィールドを更新する。"""
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
        """キャラクターを削除する。"""
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
        """記憶レコードを新規作成する。"""
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
        """IDで記憶レコードを取得する。"""
        with self.get_session() as session:
            return session.get(Memory, memory_id)

    def list_memories(
        self,
        character_id: str,
        category: Optional[str] = None,
        include_deleted: bool = False,
    ) -> list[Memory]:
        """キャラクターの記憶一覧を新しい順で返す。"""
        with self.get_session() as session:
            q = session.query(Memory).filter(Memory.character_id == character_id)
            if not include_deleted:
                q = q.filter(Memory.deleted_at.is_(None))
            if category:
                q = q.filter(Memory.memory_category == category)
            return q.order_by(Memory.created_at.desc()).all()

    def update_memory_content(
        self,
        memory_id: str,
        content: str,
        contextual_importance: float,
        semantic_importance: float,
        identity_importance: float,
        user_importance: float,
    ) -> bool:
        """記憶のcontentと各importanceを上書き更新する。

        引き継ぐフィールド: access_count, created_at
        更新するフィールド: content, 各importance, last_accessed_at（現在時刻）

        類似記憶が見つかった際の更新専用メソッド（Issue #50）。

        Args:
            memory_id: 更新対象の記憶ID。
            content: 新しい記憶テキスト。
            contextual_importance: 新しいコンテクスト重要度 (0.0-1.0)。
            semantic_importance: 新しい意味的重要度 (0.0-1.0)。
            identity_importance: 新しいアイデンティティ重要度 (0.0-1.0)。
            user_importance: 新しいユーザー重要度 (0.0-1.0)。

        Returns:
            更新成功した場合True、レコードが存在しない場合False。
        """
        with self.get_session() as session:
            mem = session.get(Memory, memory_id)
            if not mem:
                return False
            mem.content = content
            mem.contextual_importance = contextual_importance
            mem.semantic_importance = semantic_importance
            mem.identity_importance = identity_importance
            mem.user_importance = user_importance
            mem.last_accessed_at = datetime.now()
            # access_count と created_at は引き継ぐ（変更しない）
            session.commit()
            return True

    def touch_memory(self, memory_id: str) -> None:
        """last_accessed_at を更新し access_count をインクリメントする。"""
        with self.get_session() as session:
            mem = session.get(Memory, memory_id)
            if mem:
                mem.last_accessed_at = datetime.now()
                mem.access_count = (mem.access_count or 0) + 1
                session.commit()

    def soft_delete_memory(self, memory_id: str) -> bool:
        """記憶をソフト削除する（deleted_at をセット）。"""
        with self.get_session() as session:
            mem = session.get(Memory, memory_id)
            if not mem:
                return False
            mem.deleted_at = datetime.now()
            session.commit()
            return True

    def restore_memory(self, memory_id: str) -> bool:
        """ソフト削除された記憶を復元する。"""
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
        """指定期間内に作成された、削除済みでないダイジスト以外の記憶を返す。"""
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

    def get_all_active_memories(self, character_id: str) -> list[Memory]:
        """キャラクターの全アクティブ記憶（削除済みを除く）を返す。"""
        with self.get_session() as session:
            return (
                session.query(Memory)
                .filter(
                    Memory.character_id == character_id,
                    Memory.deleted_at.is_(None)
                )
                .all()
            )

    def has_digest(self, character_id: str, date_str: str) -> bool:
        """指定キャラクター・日付のダイジストログが存在するか返す。"""
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
        """ダイジストログを1行追記する。"""
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
        """キャラクターのダイジストログを新しい順で返す。"""
        with self.get_session() as session:
            return (
                session.query(DigestLog)
                .filter(DigestLog.character_id == character_id)
                .order_by(DigestLog.created_at.desc())
                .limit(limit)
                .all()
            )

    # --- LLM Model Presets ---

    def create_model_preset(self, preset_id: str, name: str, provider: str, model_id: str, thinking_level: str = "default") -> LLMModelPreset:
        """LLMモデルプリセットを新規作成する。"""
        with self.get_session() as session:
            preset = LLMModelPreset(id=preset_id, name=name, provider=provider, model_id=model_id, thinking_level=thinking_level)
            session.add(preset)
            session.commit()
            session.refresh(preset)
            return preset

    def list_model_presets(self) -> list[LLMModelPreset]:
        """LLMモデルプリセット一覧を作成日順で返す。"""
        with self.get_session() as session:
            return session.query(LLMModelPreset).order_by(LLMModelPreset.created_at).all()

    def get_model_preset(self, preset_id: str) -> Optional[LLMModelPreset]:
        """IDでLLMモデルプリセットを取得する。"""
        with self.get_session() as session:
            return session.get(LLMModelPreset, preset_id)

    def get_model_preset_by_name(self, name: str) -> Optional[LLMModelPreset]:
        """名前でLLMモデルプリセットを取得する。"""
        with self.get_session() as session:
            return session.query(LLMModelPreset).filter(LLMModelPreset.name == name).first()

    def update_model_preset(self, preset_id: str, **kwargs) -> Optional[LLMModelPreset]:
        """LLMモデルプリセットの指定フィールドを更新する。"""
        with self.get_session() as session:
            preset = session.get(LLMModelPreset, preset_id)
            if not preset:
                return None
            for k, v in kwargs.items():
                if hasattr(preset, k):
                    setattr(preset, k, v)
            session.commit()
            session.refresh(preset)
            return preset

    def delete_model_preset(self, preset_id: str) -> bool:
        """LLMモデルプリセットを削除する。"""
        with self.get_session() as session:
            preset = session.get(LLMModelPreset, preset_id)
            if not preset:
                return False
            session.delete(preset)
            session.commit()
            return True

    # --- Chat Sessions ---

    def create_chat_session(
        self,
        session_id: str,
        model_id: str,
        title: str = "新しいチャット",
        session_type: str = "1on1",
        group_config: Optional[str] = None,
    ) -> "ChatSession":
        """チャットセッションを作成する。

        Args:
            session_id: セッションのUUID。
            model_id: 1on1は "{char_name}@{preset_name}"、グループは "group"。
            title: セッションタイトル。
            session_type: "1on1" または "group"。
            group_config: グループチャット設定のJSONテキスト（session_type="group"時のみ）。
        """
        with self.get_session() as session:
            obj = ChatSession(
                id=session_id,
                model_id=model_id,
                title=title,
                session_type=session_type,
                group_config=group_config,
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj

    def get_chat_session(self, session_id: str) -> Optional["ChatSession"]:
        """IDでチャットセッションを取得する。"""
        with self.get_session() as session:
            return session.get(ChatSession, session_id)

    def list_chat_sessions(self, limit: int = 100) -> list:
        """チャットセッション一覧を新しい順で返す。"""
        with self.get_session() as session:
            return (
                session.query(ChatSession)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
                .all()
            )

    def update_chat_session(self, session_id: str, **kwargs) -> Optional["ChatSession"]:
        """チャットセッションを更新する。"""
        with self.get_session() as session:
            obj = session.get(ChatSession, session_id)
            if not obj:
                return None
            for k, v in kwargs.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            obj.updated_at = datetime.now()
            session.commit()
            session.refresh(obj)
            return obj

    def delete_chat_session(self, session_id: str) -> bool:
        """チャットセッションとそのメッセージ・画像レコードを削除する。

        ディスク上の画像ファイルは呼び出し元（chat.py）で削除すること。
        """
        with self.get_session() as session:
            # 関連レコードを先に削除してから親セッションを削除する
            session.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            session.query(ChatImage).filter(ChatImage.session_id == session_id).delete()
            session.query(SessionDrift).filter(SessionDrift.session_id == session_id).delete()
            obj = session.get(ChatSession, session_id)
            if not obj:
                session.commit()
                return False
            session.delete(obj)
            session.commit()
            return True

    # --- Chat Messages ---

    def create_chat_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        reasoning: Optional[str] = None,
        images: Optional[list] = None,
        character_name: Optional[str] = None,
        preset_name: Optional[str] = None,
    ) -> "ChatMessage":
        """チャットメッセージを作成する。

        Args:
            message_id: メッセージのUUID。
            session_id: 所属セッションのID。
            role: "user" または "character"。
            content: クリーン済みの本文テキスト。
            reasoning: 思考ブロック・想起記憶テキスト（キャラクターメッセージのみ）。
            images: 添付画像IDのリスト（ユーザメッセージのみ）。
            character_name: グループチャット時の発言キャラクター名（1on1では None）。
            preset_name: メッセージ送信時に使用したプリセット名（バブル表示用）。
        """
        with self.get_session() as session:
            msg = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                reasoning=reasoning,
                images=images or None,
                character_name=character_name,
                preset_name=preset_name,
            )
            session.add(msg)
            session.commit()
            session.refresh(msg)
            return msg

    def list_chat_messages(self, session_id: str) -> list:
        """セッション内のメッセージを時系列順で返す。"""
        with self.get_session() as session:
            return (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .all()
            )

    def delete_chat_messages_from(self, session_id: str, message_id: str) -> bool:
        """指定メッセージ以降（自身を含む）をすべて削除する。

        ユーザメッセージ編集・キャラクター応答再生成で使用する。
        対象メッセージの created_at 以降の全メッセージを削除する。

        Args:
            session_id: セッションID。
            message_id: 削除起点メッセージのID。

        Returns:
            削除対象メッセージが存在した場合 True、存在しなかった場合 False。
        """
        with self.get_session() as session:
            target = (
                session.query(ChatMessage)
                .filter(ChatMessage.id == message_id, ChatMessage.session_id == session_id)
                .first()
            )
            if not target:
                return False
            pivot_time = target.created_at
            session.query(ChatMessage).filter(
                ChatMessage.session_id == session_id,
                ChatMessage.created_at >= pivot_time,
            ).delete(synchronize_session=False)
            session.commit()
            return True

    # --- Chat Images ---

    def create_chat_image(
        self,
        image_id: str,
        session_id: str,
        mime_type: str,
        message_id: Optional[str] = None,
    ) -> "ChatImage":
        """添付画像レコードを作成する。

        Args:
            image_id: 画像のUUID（ディスク上のファイル名としても使用）。
            session_id: 所属セッションのID。
            mime_type: 画像のMIMEタイプ（例: "image/jpeg"）。
            message_id: 紐づくメッセージID（メッセージ保存前は None）。
        """
        with self.get_session() as session:
            img = ChatImage(
                id=image_id,
                session_id=session_id,
                message_id=message_id,
                mime_type=mime_type,
            )
            session.add(img)
            session.commit()
            session.refresh(img)
            return img

    def get_chat_image(self, image_id: str) -> Optional["ChatImage"]:
        """画像IDでレコードを取得する。存在しない場合は None を返す。"""
        with self.get_session() as session:
            return session.get(ChatImage, image_id)

    def list_chat_images_by_session(self, session_id: str) -> list:
        """セッションに紐づく全画像レコードを返す。セッション削除時のファイル掃除に使う。"""
        with self.get_session() as session:
            return (
                session.query(ChatImage)
                .filter(ChatImage.session_id == session_id)
                .all()
            )

    # --- Session Drifts (SELF_DRIFT) ---

    def add_session_drift(
        self, session_id: str, character_id: str, content: str
    ) -> "SessionDrift":
        """SELF_DRIFT指針を追加する。同キャラの上限3件を超えた場合は最古を削除してから追加する。

        キャラクターごとに独立管理するため、session_id と character_id の両方でフィルタする。

        Args:
            session_id: 所属セッションのID。
            character_id: キャラクターID。
            content: drift内容テキスト。

        Returns:
            作成された SessionDrift レコード。
        """
        import uuid as _uuid
        with self.get_session() as session:
            # 同キャラの既存driftを作成日時順で取得し、上限3件を超えていれば古いものを削除する
            existing = (
                session.query(SessionDrift)
                .filter(
                    SessionDrift.session_id == session_id,
                    SessionDrift.character_id == character_id,
                )
                .order_by(SessionDrift.created_at.asc())
                .all()
            )
            while len(existing) >= 3:
                oldest = existing.pop(0)
                session.delete(oldest)
            drift = SessionDrift(
                id=str(_uuid.uuid4()),
                session_id=session_id,
                character_id=character_id,
                content=content,
            )
            session.add(drift)
            session.commit()
            session.refresh(drift)
            return drift

    def list_session_drifts(self, session_id: str) -> list:
        """セッションの全キャラのdrift一覧を作成日時順で返す（UI表示用）。"""
        with self.get_session() as session:
            return (
                session.query(SessionDrift)
                .filter(SessionDrift.session_id == session_id)
                .order_by(SessionDrift.created_at.asc())
                .all()
            )

    def list_active_session_drifts(self, session_id: str, character_id: str) -> list[str]:
        """指定キャラの有効（enabled=1）なdrift内容テキスト一覧を返す（システムプロンプト注入用）。"""
        with self.get_session() as session:
            rows = (
                session.query(SessionDrift)
                .filter(
                    SessionDrift.session_id == session_id,
                    SessionDrift.character_id == character_id,
                    SessionDrift.enabled == 1,
                )
                .order_by(SessionDrift.created_at.asc())
                .all()
            )
            return [r.content for r in rows]

    def toggle_session_drift(self, drift_id: str) -> Optional["SessionDrift"]:
        """drift の enabled フラグを反転する。"""
        with self.get_session() as session:
            drift = session.get(SessionDrift, drift_id)
            if not drift:
                return None
            drift.enabled = 0 if drift.enabled else 1
            session.commit()
            session.refresh(drift)
            return drift

    def reset_session_drifts(self, session_id: str, character_id: str) -> int:
        """指定キャラのdriftを全件物理削除する。[DRIFT_RESET] マーカー処理用。

        他キャラのdriftには影響しない。

        Args:
            session_id: セッションID。
            character_id: リセット対象のキャラクターID。

        Returns:
            削除件数。
        """
        with self.get_session() as session:
            deleted = (
                session.query(SessionDrift)
                .filter(
                    SessionDrift.session_id == session_id,
                    SessionDrift.character_id == character_id,
                )
                .delete()
            )
            session.commit()
            return deleted
