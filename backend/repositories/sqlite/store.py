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
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.repositories.sqlite.stores.character_store import CharacterStoreMixin
from backend.repositories.sqlite.stores.chat_store import ChatStoreMixin
from backend.repositories.sqlite.stores.drift_store import DriftStoreMixin
from backend.repositories.sqlite.stores.memory_store import MemoryStoreMixin
from backend.repositories.sqlite.stores.preset_store import PresetStoreMixin
from backend.repositories.sqlite.stores.scenario_store import ScenarioChatStoreMixin
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


class ZetaScenario(Base):
    """シナリオテンプレート — Zeta モードで何度でも遊べるシナリオ設定本体。

    NPC 構成・GM プリセット・世界観などをテンプレートとして登録しておき、
    そこから ZetaSession（プレイインスタンス）を起動する。
    テンプレート編集中もプレイ中のセッションには影響しないよう、
    セッションはテンプレートの内容を「実行時に lookup する」設計。
    """

    __tablename__ = "zeta_scenarios"

    id = Column(String, primary_key=True)                  # UUID
    title = Column(String, nullable=False)
    scenario = Column(Text, nullable=True)                 # シナリオ概要・世界観テキスト（自由記述：場所・空気感・語り口など全部ここに詰める）
    intro = Column(Text, nullable=True)                    # 導入部（@キャラ:記法）。セッション開始時に固定ターンとして挿入
    user_alias = Column(String, nullable=False)            # ユーザの @タグ用エイリアス
    gm_preset_id = Column(String, nullable=False)          # LLMModelPreset.id（FK 制約は付けない: preset 削除時もテンプレは残す）
    history_max_turns = Column(Integer, nullable=True)     # 送信履歴の最大ターン数。NULL=settings 既定
    history_max_chars = Column(Integer, nullable=True)     # 送信履歴の最大文字数。NULL=settings 既定
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )


class ZetaNpc(Base):
    """シナリオテンプレートに紐づく NPC — 軽量 Character。

    シナリオの編集対象。P2 で promoted_character_id を使って既存キャラへ昇格できる予約あり。
    複数のプレイセッションで共有される（テンプレ編集は新セッションに反映される）。

    description には人物像・口調・話し方のサンプルなどを自由テキストで全部詰め込む。
    （以前は description / speaking_style の 2 つに分けていたが、線引きが曖昧だったため統合）
    """

    __tablename__ = "zeta_npcs"

    id = Column(String, primary_key=True)                          # UUID
    scenario_id = Column(String, ForeignKey("zeta_scenarios.id"), nullable=False)
    name = Column(String, nullable=False)                          # @タグに使う シナリオ内ユニーク名
    description = Column(Text, nullable=True)                      # 人物像・口調・話し方を自由記述
    image_data = Column(Text, nullable=True)                       # アバター画像 (base64 data URI)
    promoted_character_id = Column(String, nullable=True)          # P2 予約。P1 では常に NULL
    created_at = Column(DateTime, default=lambda: datetime.now())

    # 同一シナリオ内では NPC 名（@タグ）はユニーク。@タグの曖昧性を防ぐ。
    __table_args__ = (
        UniqueConstraint("scenario_id", "name", name="uq_zeta_npcs_scenario_name"),
    )


class ZetaSession(Base):
    """シナリオから起動されたプレイインスタンス。

    テンプレ（ZetaScenario）の `scenario_id` を参照し、プレイ独自の
    `title` / `status` / 発話履歴（zeta_turns）を持つ薄いテーブル。
    user_alias や scenario・NPCs などのプレイ用設定はテンプレを lookup して取得する。
    """

    __tablename__ = "zeta_sessions"

    id = Column(String, primary_key=True)                  # UUID
    scenario_id = Column(String, ForeignKey("zeta_scenarios.id"), nullable=False)
    title = Column(String, nullable=False)                 # 起動時はテンプレ title からコピー（編集可）
    engine_type = Column(String, nullable=False, default="ensemble")  # P1 では 'ensemble' 固定
    status = Column(String, nullable=False, default="active")  # active / ended
    # あらすじ機構（記憶捏造対策）
    #   synopsis_auto: LLM が古いターン群を要約して「追記」していく主あらすじ。
    #                  ユーザも UI から自由編集可能（捏造の混入を発見した際に削除・修正できる）。
    #   synopsis_manual: プレイヤーが手で書く補足メモ。自動更新では絶対に触らない。
    #                    GM への補正指示として機能する。
    #   synopsis_last_turn_index: synopsis_auto に「どこまで要約済みか」を記録する境界。
    #                             この値以下の turn_index は既に要約反映済みと見做す。
    synopsis_auto = Column(Text, nullable=False, default="")
    synopsis_manual = Column(Text, nullable=False, default="")
    synopsis_last_turn_index = Column(Integer, nullable=False, default=-1)
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )


class ZetaTurn(Base):
    """シナリオセッションの発話ターン — ユーザ・Narrator・NPC・(将来)既存キャラを多態で格納する。"""

    __tablename__ = "zeta_turns"

    id = Column(String, primary_key=True)                          # UUID
    session_id = Column(String, ForeignKey("zeta_sessions.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)                   # セッション内連番（0始まり）
    speaker_type = Column(String, nullable=False)                  # user / narrator / npc / character
    speaker_id = Column(String, nullable=True)                     # npc: zeta_npcs.id / character: characters.id / 他NULL
    speaker_name = Column(String, nullable=False)                  # 表示・履歴整形用のスナップショット名（NPC 名変更後も履歴保護）
    content = Column(Text, nullable=False)
    raw_response = Column(Text, nullable=True)                     # GM の単一呼出で得たターン全体の生出力（デバッグ用）
    created_at = Column(DateTime, default=lambda: datetime.now())


class SQLiteStore(
    SettingsStoreMixin,
    CharacterStoreMixin,
    MemoryStoreMixin,
    PresetStoreMixin,
    ChatStoreMixin,
    DriftStoreMixin,
    ScenarioChatStoreMixin,
):
    """SQLite永続化ストア — 全テーブルへのCRUD操作を提供するファサードクラス。

    ドメイン別 Mixin を多重継承し、外部からは従来通り単一クラスとして利用できる。
    """

    def __init__(self, db_path: str):
        """データベースを初期化し、マイグレーションを実行する。"""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._predrop_legacy_zeta_tables()
        Base.metadata.create_all(self.engine)
        self._migrate()

    def _predrop_legacy_zeta_tables(self) -> None:
        """zeta_* の旧スキーマを検出して破壊的に DROP する。

        スキーマ進化に合わせて、以下のいずれかが検出されたら 4 テーブルを一括作り直す:
            - zeta_npcs.session_id 残存（最初の 1 層構造）
            - zeta_npcs.speaking_style 残存（旧 NPC スキーマ）
            - zeta_npcs.display_order 残存（display_order 廃止前）
            - zeta_npcs.image_data 未追加（旧 NPC スキーマ）
            - zeta_scenarios.location 残存（location/scene_summary 等の細分フィールド廃止前）

        zeta_* は開発段階の試行データのみのため、旧スキーマ検出 = 一括 DROP で問題ない。
        """
        with self.engine.connect() as conn:
            try:
                npc_cols = {
                    row[0]
                    for row in conn.execute(
                        text("SELECT name FROM pragma_table_info('zeta_npcs')")
                    ).fetchall()
                }
                scen_cols = {
                    row[0]
                    for row in conn.execute(
                        text("SELECT name FROM pragma_table_info('zeta_scenarios')")
                    ).fetchall()
                }
                if not npc_cols and not scen_cols:
                    return  # テーブル未作成 → create_all に任せる
                needs_rebuild = (
                    "session_id" in npc_cols
                    or "speaking_style" in npc_cols
                    or "display_order" in npc_cols
                    or (bool(npc_cols) and "image_data" not in npc_cols)
                    or "location" in scen_cols
                )
                if needs_rebuild:
                    for tbl in (
                        "zeta_turns",
                        "zeta_npcs",
                        "zeta_sessions",
                        "zeta_scenarios",
                    ):
                        conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
                    conn.commit()
            except Exception:
                # pragma 失敗等は無視（テーブル未作成時など）
                pass

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
                "ALTER TABLE zeta_scenarios ADD COLUMN intro TEXT",
                # zeta_sessions のあらすじ機構（記憶捏造対策）
                "ALTER TABLE zeta_sessions ADD COLUMN synopsis_auto TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE zeta_sessions ADD COLUMN synopsis_manual TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE zeta_sessions ADD COLUMN synopsis_last_turn_index INTEGER NOT NULL DEFAULT -1",
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
