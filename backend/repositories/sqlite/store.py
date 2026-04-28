"""SQLite store — 設定・キャラクターメタデータ・記憶レコードの永続化層。

SQLiteStore はドメイン別 Mixin を多重継承したファサードクラス。
各ドメインの実装は backend/core/memory/stores/ 以下を参照。

  SettingsStoreMixin  — グローバル設定 (key/value)
  CharacterStoreMixin — キャラクター管理
  MemoryStoreMixin    — 記憶レコード
  PresetStoreMixin    — LLMモデルプリセット
  ChatStoreMixin      — セッション・メッセージ・画像
  DriftStoreMixin     — SELF_DRIFT指針
"""

import os
from datetime import datetime

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

from backend.repositories.sqlite.stores.character_store import CharacterStoreMixin
from backend.repositories.sqlite.stores.chat_store import ChatStoreMixin
from backend.repositories.sqlite.stores.drift_store import DriftStoreMixin
from backend.repositories.sqlite.stores.memory_store import MemoryStoreMixin
from backend.repositories.sqlite.stores.preset_store import PresetStoreMixin
from backend.repositories.sqlite.stores.settings_store import SettingsStoreMixin


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
    # Afterglow（感情継続機構）: このセッションが引き継ぐ元セッションID。NULLなら引き継ぎなし。
    afterglow_session_id = Column(String, nullable=True)
    # 退席者リスト: [{"char_name": "Alice", "reason": "理由"}]。NULLなら退席者なし。
    exited_chars = Column(JSON, nullable=True)
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
    # システムメッセージフラグ: 1=退席通知などのシステムメッセージ。NULLまたは0=通常メッセージ。
    is_system_message = Column(Integer, nullable=True)
    # クロニクル処理日時: NULL=未処理、タイムスタンプあり=処理済み
    chronicled_at = Column(DateTime, nullable=True)
    # デバッグログフォルダ名（8桁hex）。CHOTGOR_DEBUG=1 時のみ記録される。
    log_message_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())


class ChatImage(Base):
    """チャット添付画像 — セッションに紐づく画像ファイルのメタデータ。"""

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
    inner_narrative = Column(Text, nullable=False, default="")
    cleanup_config = Column(JSON, nullable=False, default=dict)
    enabled_providers = Column(JSON, nullable=False, default=dict)
    ghost_model = Column(String, nullable=True)  # digest/forget に使うプリセットID
    image_data = Column(Text, nullable=True)  # base64 data URI
    switch_angle_enabled = Column(Integer, nullable=False, default=0)  # 1=ON, 0=OFF
    # Afterglow（感情継続機構）: 新規チャット作成時のデフォルト値。1=ON, 0=OFF
    afterglow_default = Column(Integer, nullable=False, default=0)
    # 自己参照ループ設定
    self_reflection_mode = Column(String, nullable=False, default="disabled")  # disabled/local_trigger/always
    self_reflection_preset_id = Column(String, nullable=True)   # 契機判断モデルプリセットID（local_trigger 時）
    self_reflection_n_turns = Column(Integer, nullable=False, default=5)  # 自己参照に使う直近Nターン数
    # キャラクター自己更新フィールド: chronicle 処理で更新される
    self_history = Column(Text, nullable=False, default="")       # これまでの経緯と現在の状態
    relationship_state = Column(Text, nullable=False, default="") # ユーザ・他キャラとの関係
    # 別れ機能フィールド
    farewell_config = Column(JSON, nullable=True)  # chronicle で更新される感情閾値・退席設定JSON
    relationship_status = Column(String, nullable=False, default="active")  # "active" | "estranged"
    definition_embedding_id = Column(String, nullable=True)  # ChromaDB char_definitions コレクション内の doc ID
    # キャラクターごとの外部ツール許可設定 (web_search/google_calendar/gmail/google_drive)
    allowed_tools = Column(JSON, nullable=False, default=dict)
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
    # 記憶を作成したプリセットID（NULLは旧データまたは不明）
    source_preset_id = Column(String, nullable=True)
    # アクセス追跡
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(DateTime, nullable=True)  # content/importance変更時のみ更新
    deleted_at = Column(DateTime, nullable=True)  # ソフト削除



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


class SQLiteStore(
    SettingsStoreMixin,
    CharacterStoreMixin,
    MemoryStoreMixin,
    PresetStoreMixin,
    ChatStoreMixin,
    DriftStoreMixin,
):
    """SQLite永続化ストア — 全テーブルへのCRUD操作を提供するファサードクラス。

    ドメイン別 Mixin を多重継承し、外部からは従来通り単一クラスとして利用できる。
    """

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
            for stmt in [
                "ALTER TABLE memories ADD COLUMN updated_at TIMESTAMP",
                "ALTER TABLE characters ADD COLUMN enabled_providers TEXT NOT NULL DEFAULT '{}'",
                "ALTER TABLE characters ADD COLUMN image_data TEXT",
                "ALTER TABLE characters ADD COLUMN ghost_model TEXT",
                "ALTER TABLE llm_model_presets ADD COLUMN thinking_level TEXT NOT NULL DEFAULT 'default'",
                "ALTER TABLE characters ADD COLUMN switch_angle_enabled INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE memories ADD COLUMN source_preset_id TEXT",
                "ALTER TABLE characters ADD COLUMN inner_narrative TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE characters ADD COLUMN afterglow_default INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE characters ADD COLUMN self_history TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE characters ADD COLUMN relationship_state TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE characters ADD COLUMN self_reflection_mode TEXT NOT NULL DEFAULT 'disabled'",
                "ALTER TABLE characters ADD COLUMN self_reflection_preset_id TEXT",
                "ALTER TABLE characters ADD COLUMN self_reflection_n_turns INTEGER NOT NULL DEFAULT 5",
                "ALTER TABLE characters ADD COLUMN farewell_config TEXT",
                "ALTER TABLE characters ADD COLUMN relationship_status TEXT NOT NULL DEFAULT 'active'",
                "ALTER TABLE characters ADD COLUMN definition_embedding_id TEXT",
                "ALTER TABLE characters ADD COLUMN allowed_tools TEXT NOT NULL DEFAULT '{}'",
            ]:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass

            # characters テーブルに旧カラム（meta_instructions 等）が残っている場合、
            # テーブルを再作成して現行 ORM スキーマに揃える。
            # meta_instructions は NOT NULL かつ DEFAULT なしのため、
            # ORM 経由の INSERT が失敗する原因になる。
            try:
                result = conn.execute(
                    text("SELECT count(*) FROM pragma_table_info('characters') WHERE name='meta_instructions'")
                )
                if result.fetchone()[0] > 0:
                    # 中途失敗時の残骸テーブルを除去してからリトライ
                    conn.execute(text("DROP TABLE IF EXISTS characters_new"))
                    conn.execute(text("""
                        CREATE TABLE characters_new (
                            id VARCHAR NOT NULL PRIMARY KEY,
                            name VARCHAR NOT NULL,
                            system_prompt_block1 TEXT NOT NULL DEFAULT '',
                            inner_narrative TEXT NOT NULL DEFAULT '',
                            cleanup_config JSON NOT NULL DEFAULT '{}',
                            enabled_providers TEXT NOT NULL DEFAULT '{}',
                            ghost_model TEXT,
                            image_data TEXT,
                            switch_angle_enabled INTEGER NOT NULL DEFAULT 0,
                            afterglow_default INTEGER NOT NULL DEFAULT 0,
                            self_history TEXT NOT NULL DEFAULT '',
                            relationship_state TEXT NOT NULL DEFAULT '',
                            self_reflection_mode TEXT NOT NULL DEFAULT 'disabled',
                            self_reflection_preset_id TEXT,
                            self_reflection_n_turns INTEGER NOT NULL DEFAULT 5,
                            farewell_config TEXT,
                            relationship_status TEXT NOT NULL DEFAULT 'active',
                            definition_embedding_id TEXT,
                            allowed_tools TEXT NOT NULL DEFAULT '{}',
                            created_at DATETIME,
                            updated_at DATETIME
                        )
                    """))
                    conn.execute(text("""
                        INSERT INTO characters_new
                            (id, name, system_prompt_block1, inner_narrative,
                             cleanup_config, enabled_providers, ghost_model,
                             image_data, switch_angle_enabled, afterglow_default,
                             self_history, relationship_state,
                             created_at, updated_at)
                        SELECT
                            id, name, system_prompt_block1,
                            COALESCE(NULLIF(inner_narrative, ''), meta_instructions, ''),
                            COALESCE(cleanup_config, '{}'),
                            COALESCE(enabled_providers, '{}'),
                            ghost_model, image_data,
                            COALESCE(switch_angle_enabled, 0),
                            COALESCE(afterglow_default, 0),
                            '', '',
                            created_at, updated_at
                        FROM characters
                    """))
                    conn.execute(text("ALTER TABLE characters RENAME TO characters_old"))
                    conn.execute(text("ALTER TABLE characters_new RENAME TO characters"))
                    conn.execute(text("DROP TABLE characters_old"))
                    conn.commit()
            except Exception:
                pass

            for stmt in [
                "ALTER TABLE chat_messages ADD COLUMN reasoning TEXT",
                "ALTER TABLE chat_messages ADD COLUMN images TEXT",
                "ALTER TABLE chat_messages ADD COLUMN character_name TEXT",
                "ALTER TABLE chat_messages ADD COLUMN preset_name TEXT",
                "ALTER TABLE chat_sessions ADD COLUMN session_type TEXT NOT NULL DEFAULT '1on1'",
                "ALTER TABLE chat_sessions ADD COLUMN group_config TEXT",
                "ALTER TABLE chat_sessions ADD COLUMN afterglow_session_id TEXT",
                "ALTER TABLE chat_sessions ADD COLUMN exited_chars TEXT",
                "ALTER TABLE chat_messages ADD COLUMN is_system_message INTEGER",
                "ALTER TABLE chat_messages ADD COLUMN log_message_id TEXT",
            ]:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass

            # chronicled_at カラム追加と既存メッセージの一括マークは同一トランザクション内で実行する。
            # ALTER TABLE が失敗（カラム既存）した場合は UPDATE も実行しないことで、
            # 起動のたびに未処理メッセージが上書きされるバグを防ぐ。
            try:
                conn.execute(text("ALTER TABLE chat_messages ADD COLUMN chronicled_at TIMESTAMP"))
                conn.execute(text("UPDATE chat_messages SET chronicled_at = created_at WHERE chronicled_at IS NULL"))
                conn.commit()
            except Exception:
                pass

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
        """新しいDBセッションを返す。Mixin クラスが共通して使用する。"""
        return self.SessionLocal()
