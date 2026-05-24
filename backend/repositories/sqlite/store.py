"""SQLite store — 設定・キャラクターメタデータ・保存記憶レコードの永続化層。

SQLiteStore はドメイン別 Mixin を多重継承したファサードクラス。
各ドメインの実装は backend/core/memory/stores/ 以下を参照。

  SettingsStoreMixin                — グローバル設定 (key/value)
  CharacterStoreMixin               — キャラクター管理
  InscribedMemoryStoreMixin         — 保存記憶レコード
  PresetStoreMixin                  — LLMモデルプリセット
  ChatStoreMixin                    — セッション・メッセージ・画像
  DriftStoreMixin                   — SELF_DRIFT指針
  WorkingMemoryStoreMixin           — ワーキングメモリ（短期記憶スレッド・ポスト）
  ScenarioChatStoreMixin            — シナリオチャット（テンプレ・セッション・NPC・ターン）
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
from backend.repositories.sqlite.stores.inscribed_memory_store import (
    InscribedMemoryStoreMixin,
)
from backend.repositories.sqlite.stores.preset_store import PresetStoreMixin
from backend.repositories.sqlite.stores.scenario_store import ScenarioChatStoreMixin
from backend.repositories.sqlite.stores.settings_store import SettingsStoreMixin
from backend.repositories.sqlite.stores.working_memory_store import (
    WorkingMemoryStoreMixin,
)


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
    inner_narrative = Column(Text, nullable=False, default="")  # 内的叙述（キャラクター自身の自己物語テキスト）
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
    definition_embedding_id = Column(String, nullable=True)  # LanceDB definitions テーブル内の doc ID
    # キャラクターごとの外部ツール許可設定 (web_search/google_calendar/gmail/google_drive)
    allowed_tools = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )


class InscribedMemory(Base):
    """保存記憶レコード — キャラクターが `inscribe_memory` で残した記憶を、

    カテゴリ・重要度スコアとともに保持する永続テーブル。短期記憶（ワーキングメモリ・
    `WorkingMemoryThread`）と対になる「長期に残す記憶」を表す。
    """

    __tablename__ = "inscribed_memories"

    id = Column(String, primary_key=True)  # UUID
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    content = Column(Text, nullable=False)
    memory_category = Column(String, nullable=False, default="general")
    # 重要度スコア (0.0 - 1.0)
    contextual_importance = Column(Float, default=0.5)
    semantic_importance = Column(Float, default=0.5)
    identity_importance = Column(Float, default=0.5)
    user_importance = Column(Float, default=0.5)
    # 記憶を作成したプリセットID
    source_preset_id = Column(String, nullable=True)
    # アクセス追跡
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(DateTime, nullable=True)  # content/importance変更時のみ更新
    deleted_at = Column(DateTime, nullable=True)  # ソフト削除



class WorkingMemoryThread(Base):
    """ワーキングメモリスレッド — キャラクターの並行する短期記憶ストリームの1本。

    BBS/チケット管理に似た「スレッド方式」で、task/topic/emotion/body/relation の
    並走する短期記憶ストリームを表現する。各スレッドは時系列に連なる WorkingMemoryPost を持つ。
    キャラクターの短期・中期記憶（ワーキングメモリ）を担う層。

    type 別の制約（emotion/body は1本のみ、relation は相手ごとに1本など）は
    DB ではなくアプリ層（WorkingMemoryManager）で担保する。
    """

    __tablename__ = "working_memory_threads"

    id = Column(String, primary_key=True)  # UUID
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    type = Column(String, nullable=False)  # emotion / body / task / topic / relation
    summary = Column(Text, nullable=False, default="")      # タイトル相当。embedding index の素材
    atmosphere_tag = Column(Text, nullable=False, default="")  # 質感の短いタグ（Active時:温度感／Close時:終わり方）
    importance = Column(Float, nullable=False, default=0.5)  # 0.0 - 1.0
    is_open = Column(Integer, nullable=False, default=1)     # 1=Open（運用中・一覧表示対象）, 0=Archived
    # relation 型のみ使用: 関係相手の識別子（ユーザ名・他キャラ名など）。重複作成防止のキーにもなる。
    relation_target = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )
    # heat の時間減衰の起点。Post 追加・スレッド更新時に明示的に更新する。
    # updated_at は onupdate で勝手に動くため、decay 起点としては流用せず別カラムで管理する。
    last_touched_at = Column(DateTime, nullable=True)


class WorkingMemoryPost(Base):
    """ワーキングメモリポスト — スレッド内に時系列順で連なる1書き込み。"""

    __tablename__ = "working_memory_posts"

    id = Column(String, primary_key=True)  # UUID
    thread_id = Column(String, ForeignKey("working_memory_threads.id"), nullable=False)
    content = Column(Text, nullable=False)
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


class Scenario(Base):
    """シナリオテンプレート — シナリオチャットで何度でも遊べるシナリオ設定本体。

    NPC 構成・世界観などをテンプレートとして登録しておき、
    そこから ScenarioSession（プレイインスタンス）を起動する。
    テンプレート編集中もプレイ中のセッションには影響しないよう、
    セッションはテンプレートの内容を「実行時に lookup する」設計。

    GM プリセット（LLM 設定）はテンプレート単位ではなくセッション単位
    （ScenarioSession.gm_preset_id）で保持する。同一シナリオから複数
    セッションを起動した際、それぞれ異なる GM モデルで遊べるようにするため。
    """

    __tablename__ = "scenarios"

    id = Column(String, primary_key=True)                  # UUID
    title = Column(String, nullable=False)
    scenario = Column(Text, nullable=True)                 # シナリオ概要・世界観テキスト（自由記述：場所・空気感・語り口など全部ここに詰める）
    intro = Column(Text, nullable=True)                    # 導入部（@キャラ:記法）。セッション開始時に固定ターンとして挿入
    user_alias = Column(String, nullable=False)            # ユーザの @タグ用エイリアス
    history_max_turns = Column(Integer, nullable=True)     # 送信履歴の最大ターン数。NULL=settings 既定
    history_max_chars = Column(Integer, nullable=True)     # 送信履歴の最大文字数。NULL=settings 既定
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )


class ScenarioNpc(Base):
    """シナリオテンプレートに紐づく NPC — 軽量 Character。

    シナリオの編集対象。P2 で promoted_character_id を使って既存キャラへ昇格できる予約あり。
    複数のプレイセッションで共有される（テンプレ編集は新セッションに反映される）。

    description には人物像・口調・話し方のサンプルなどを自由テキストで全部詰め込む。
    （以前は description / speaking_style の 2 つに分けていたが、線引きが曖昧だったため統合）
    """

    __tablename__ = "scenario_npcs"

    id = Column(String, primary_key=True)                          # UUID
    scenario_id = Column(String, ForeignKey("scenarios.id"), nullable=False)
    name = Column(String, nullable=False)                          # @タグに使う シナリオ内ユニーク名
    description = Column(Text, nullable=True)                      # 人物像・口調・話し方を自由記述
    image_data = Column(Text, nullable=True)                       # アバター画像 (base64 data URI)
    promoted_character_id = Column(String, nullable=True)          # P2 予約。P1 では常に NULL
    created_at = Column(DateTime, default=lambda: datetime.now())

    # 同一シナリオ内では NPC 名（@タグ）はユニーク。@タグの曖昧性を防ぐ。
    __table_args__ = (
        UniqueConstraint("scenario_id", "name", name="uq_scenario_npcs_scenario_name"),
    )


class ScenarioSession(Base):
    """シナリオから起動されたプレイインスタンス。

    テンプレ（Scenario）の `scenario_id` を参照し、プレイ独自の
    `title` / `status` / 発話履歴（scenario_turns）を持つ薄いテーブル。
    user_alias や scenario・NPCs などのプレイ用設定はテンプレを lookup して取得する。
    """

    __tablename__ = "scenario_sessions"

    id = Column(String, primary_key=True)                  # UUID
    scenario_id = Column(String, ForeignKey("scenarios.id"), nullable=False)
    title = Column(String, nullable=False)                 # 起動時はテンプレ title からコピー（編集可）
    engine_type = Column(String, nullable=False, default="ensemble")  # P1 では 'ensemble' 固定
    status = Column(String, nullable=False, default="active")  # active / ended
    # GM が使う LLM プリセット ID（LLMModelPreset.id）。
    # FK 制約は付けない: preset 削除時もセッション履歴は残したいため。
    # セッション開始時に必ず指定する（フロントの「新しい会話」モーダルでユーザが選ぶ）。
    # チャット中も左上ヘッダーから変更可能（1on1 のモデル切替と同様）。
    gm_preset_id = Column(String, nullable=False, default="")
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


class ScenarioTurn(Base):
    """シナリオセッションの発話ターン — ユーザ・Narrator・NPC・(将来)既存キャラを多態で格納する。"""

    __tablename__ = "scenario_turns"

    id = Column(String, primary_key=True)                          # UUID
    session_id = Column(String, ForeignKey("scenario_sessions.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)                   # セッション内連番（0始まり）
    speaker_type = Column(String, nullable=False)                  # user / narrator / npc / character
    speaker_id = Column(String, nullable=True)                     # npc: scenario_npcs.id / character: characters.id / 他NULL
    speaker_name = Column(String, nullable=False)                  # 表示・履歴整形用のスナップショット名（NPC 名変更後も履歴保護）
    content = Column(Text, nullable=False)
    raw_response = Column(Text, nullable=True)                     # GM の単一呼出で得たターン全体の生出力（デバッグ用）
    created_at = Column(DateTime, default=lambda: datetime.now())


class SQLiteStore(
    SettingsStoreMixin,
    CharacterStoreMixin,
    InscribedMemoryStoreMixin,
    PresetStoreMixin,
    ChatStoreMixin,
    DriftStoreMixin,
    WorkingMemoryStoreMixin,
    ScenarioChatStoreMixin,
):
    """SQLite永続化ストア — 全テーブルへのCRUD操作を提供するファサードクラス。

    ドメイン別 Mixin を多重継承し、外部からは従来通り単一クラスとして利用できる。
    """

    def __init__(self, db_path: str):
        """データベースを初期化する。

        スキーマは ORM 定義（このモジュール上部の各 Base クラス）が唯一の正。
        ``Base.metadata.create_all`` が新規 DB に全テーブルを定義通りに作成する。
        テーブル名・カラム追加等の変更は、必要なら別途使い捨ての移行スクリプトで対応する。
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self._migrate_gm_preset_id_to_session()

    def _migrate_gm_preset_id_to_session(self) -> None:
        """`gm_preset_id` を scenarios → scenario_sessions に移行する。

        旧スキーマ: scenarios.gm_preset_id（NOT NULL）にシナリオ単位で保持
        新スキーマ: scenario_sessions.gm_preset_id（NOT NULL）にセッション単位で保持

        移行手順:
            1. scenario_sessions に gm_preset_id 列が無ければ ADD COLUMN
            2. scenarios.gm_preset_id を JOIN で各セッションへバックフィル
            3. scenarios.gm_preset_id を DROP COLUMN（SQLite 3.35+ が必要）

        新規 DB（既に新スキーマ）では何もしない。冪等。
        """
        def _columns_of(conn, table: str) -> set[str]:
            rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            return {r[1] for r in rows}

        with self.engine.begin() as conn:
            # 両テーブルが存在するか（最初の起動時は scenarios すら無いこともある）
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "scenarios" not in tables or "scenario_sessions" not in tables:
                return

            session_cols = _columns_of(conn, "scenario_sessions")
            scenario_cols = _columns_of(conn, "scenarios")

            # 1. scenario_sessions に列を追加（既に新スキーマなら skip）
            if "gm_preset_id" not in session_cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenario_sessions "
                    "ADD COLUMN gm_preset_id TEXT NOT NULL DEFAULT ''"
                )

            # 2. 旧 scenarios.gm_preset_id が残っていればバックフィル
            if "gm_preset_id" in scenario_cols:
                conn.exec_driver_sql(
                    "UPDATE scenario_sessions "
                    "SET gm_preset_id = COALESCE(("
                    "  SELECT s.gm_preset_id FROM scenarios s "
                    "  WHERE s.id = scenario_sessions.scenario_id"
                    "), gm_preset_id) "
                    "WHERE (gm_preset_id IS NULL OR gm_preset_id = '')"
                )
                # 3. 旧列を削除（SQLite 3.35+）
                conn.exec_driver_sql(
                    "ALTER TABLE scenarios DROP COLUMN gm_preset_id"
                )

    def get_session(self) -> Session:
        """新しい DB セッションを返す。Mixin クラスが共通して使用する。"""
        return self.SessionLocal()

