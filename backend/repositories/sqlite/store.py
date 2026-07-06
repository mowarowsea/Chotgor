"""SQLite store — 設定・キャラクターメタデータ・保存記憶レコードの永続化層。

SQLiteStore はドメイン別 Mixin を多重継承したファサードクラス。
各ドメインの実装は backend/repositories/sqlite/stores/ 以下を参照。

  SettingsStoreMixin                — グローバル設定 (key/value)
  CharacterStoreMixin               — キャラクター管理
  InscribedMemoryStoreMixin         — 保存記憶レコード
  PresetStoreMixin                  — LLMモデルプリセット
  ChatStoreMixin                    — セッション・メッセージ・画像
  WorkingMemoryStoreMixin           — ワーキングメモリ（短期記憶スレッド・ポスト）
  ScenarioChatStoreMixin            — シナリオチャット（テンプレ・セッション・NPC・ターン）
  DebugLogStoreMixin                — デバッグログエントリ
  UsageStoreMixin                   — LLM 使用量イベント
  ToolEventStoreMixin               — ツール実行イベント（Logs 画面のツール使用表示）
  TimelineStoreMixin                — タイムライン封筒（めぐり / Aliveness の正本）
  InstrumentStoreMixin              — 計器（アラーム・メータースナップショット）
  IntentStoreMixin                  — 意図（動機経済の「〜したい」レコード）
  SQLiteMigrationsMixin             — 冪等マイグレーション（migrations.py）

ORM モデル定義は models.py にあり、後方互換のため本モジュールから再エクスポートする。
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# 後方互換の再エクスポート: 既存コードは `from backend.repositories.sqlite.store import Character`
# のように本モジュール経由でモデルを import しているため、この一覧は削除しないこと。
from backend.repositories.sqlite.models import (  # noqa: F401
    Alarm,
    Base,
    Character,
    ChatImage,
    ChatMessage,
    ChatSession,
    DebugLogEntry,
    GlobalSetting,
    InscribedMemory,
    Intent,
    LlmUsageEvent,
    LLMModelPreset,
    MeterSnapshot,
    Scenario,
    ScenarioNpc,
    ScenarioSession,
    ScenarioTurn,
    TimelineEvent,
    ToolCallEvent,
    WorkingMemoryPost,
    WorkingMemoryThread,
)
from backend.repositories.sqlite.migrations import SQLiteMigrationsMixin
from backend.repositories.sqlite.stores.character_store import CharacterStoreMixin
from backend.repositories.sqlite.stores.instrument_store import InstrumentStoreMixin
from backend.repositories.sqlite.stores.intent_store import IntentStoreMixin
from backend.repositories.sqlite.stores.chat_store import ChatStoreMixin
from backend.repositories.sqlite.stores.debug_log_store import DebugLogStoreMixin
from backend.repositories.sqlite.stores.inscribed_memory_store import (
    InscribedMemoryStoreMixin,
)
from backend.repositories.sqlite.stores.preset_store import PresetStoreMixin
from backend.repositories.sqlite.stores.scenario_store import ScenarioChatStoreMixin
from backend.repositories.sqlite.stores.settings_store import SettingsStoreMixin
from backend.repositories.sqlite.stores.timeline_store import TimelineStoreMixin
from backend.repositories.sqlite.stores.tool_event_store import ToolEventStoreMixin
from backend.repositories.sqlite.stores.usage_store import UsageStoreMixin
from backend.repositories.sqlite.stores.working_memory_store import (
    WorkingMemoryStoreMixin,
)


class SQLiteStore(
    SettingsStoreMixin,
    CharacterStoreMixin,
    InscribedMemoryStoreMixin,
    PresetStoreMixin,
    ChatStoreMixin,
    WorkingMemoryStoreMixin,
    ScenarioChatStoreMixin,
    DebugLogStoreMixin,
    UsageStoreMixin,
    ToolEventStoreMixin,
    TimelineStoreMixin,
    InstrumentStoreMixin,
    IntentStoreMixin,
    SQLiteMigrationsMixin,
):
    """SQLite永続化ストア — 全テーブルへのCRUD操作を提供するファサードクラス。

    ドメイン別 Mixin を多重継承し、外部からは従来通り単一クラスとして利用できる。
    """

    def __init__(self, db_path: str):
        """データベースを初期化する。

        スキーマは ORM 定義（models.py）が唯一の正。
        ``Base.metadata.create_all`` が新規 DB に全テーブルを定義通りに作成する。
        既存 DB のスキーマ追従は migrations.py の冪等マイグレーション群が行う。
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self._migrate_gm_preset_id_to_session()
        self._migrate_add_preset_timeout_seconds()
        self._migrate_add_synopsis_preset_id()
        self._migrate_add_debug_log_entries()
        self._migrate_add_scenario_turn_log_request_id()
        self._migrate_add_scenario_custom_system_prompt()
        self._migrate_drop_afterglow_columns()
        self._migrate_add_chat_message_anticipation()
        self._migrate_add_scenario_turn_anticipation()
        self._migrate_add_scenario_turn_chronicled_at()
        self._migrate_add_scenario_banner_data()
        self._migrate_add_memory_origin()
        self._migrate_add_scenario_pc_mode()
        self._migrate_drop_session_drifts()
        self._migrate_unify_user_alias_to_pc_slot()
        self._migrate_add_usual_days()
        self._migrate_extract_user_fields_to_character()
        self._migrate_add_user_visibility_note()
        self._migrate_drop_group_chat()
        self._migrate_add_face_to_face_columns()
        self._migrate_add_pressure_profile()
        self._migrate_add_gate_columns()
        self._migrate_backfill_timeline_events()

    def get_session(self) -> Session:
        """新しい DB セッションを返す。Mixin クラスが共通して使用する。"""
        return self.SessionLocal()
