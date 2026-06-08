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
from backend.repositories.sqlite.stores.debug_log_store import DebugLogStoreMixin
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
    # ANTICIPATE_RESPONSE: キャラクターが本文末尾に書いた「次の展開への予想（期待）」。
    # 次ターンのシステムプロンプトに「前回のあなたの予想」として注入される。
    anticipation = Column(Text, nullable=True)
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
    # キャラクターごとの外部ツール許可設定 (google_calendar/gmail/google_drive)。
    # 旧 web_search キーは廃止（外部情報取得は Chotgor MCP の web_search ツールに一本化）。
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
    # 記憶のソース識別。"real"=日常体験、"interlude"=シナリオPCモードで演じた幕間体験。
    # 検索時のフィルタやキャラ本人の文脈把握に使う。
    origin = Column(String, nullable=False, default="real")



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
    # スレッドのソース識別。"real"=日常、"interlude"=シナリオPCモードで生じた幕間スレッド。
    # InscribedMemory.origin と対。検索フィルタとキャラ本人の文脈把握に使う。
    origin = Column(String, nullable=False, default="real")


class WorkingMemoryPost(Base):
    """ワーキングメモリポスト — スレッド内に時系列順で連なる1書き込み。"""

    __tablename__ = "working_memory_posts"

    id = Column(String, primary_key=True)  # UUID
    thread_id = Column(String, ForeignKey("working_memory_threads.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now())


class LLMModelPreset(Base):
    """LLMモデルプリセット — プロバイダー・モデルIDの設定を保持する。"""

    __tablename__ = "llm_model_presets"

    id = Column(String, primary_key=True)          # UUID
    name = Column(String, nullable=False)           # "Google-Gemini3Flash"
    provider = Column(String, nullable=False)       # "google"
    model_id = Column(String, nullable=False, default="")  # "gemini-2.0-flash"
    thinking_level = Column(String, nullable=False, default="default")  # default/low/medium/high
    # プロバイダーAPIリクエストのタイムアウト秒数。デフォルトは5分（=300秒）。
    # 現状はOllamaのみ参照する（ローカルモデルは応答が遅いケースが多いため）。
    timeout_seconds = Column(Integer, nullable=False, default=300)
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
    custom_system_prompt = Column(Text, nullable=True)     # GMシステムプロンプトの完全カスタマイズ。NULLなら自動生成ロジック使用
    # ダイスプール仕様（JSON: {"d6": 10, "d100": 5} 等）。NULL/未指定なら engine 側で {"d6": 10} を既定値として使う。
    # ensemble_pc エンジン時のみ参照する。毎ターン engine 内で乱数生成して GM system prompt の {dice_pool} へ注入。
    dice_pool_spec = Column(JSON, nullable=True)
    # PC枠定義（ensemble_pc エンジン時のみ参照）。JSON 配列 [{"slot_id":"pc1","name":"アリス","description":"..."}]。
    # シナリオ作成時にユーザが PC として登場する「枠」（人物像・知っていること）を定義する。
    # 各セッション起動時、`scenario_sessions.pc_assignments` で各 slot_id に「ユーザが演じる/AIキャラが演じる」を割り当てる。
    # NULL/空配列なら ensemble_pc を使えない（API バリデーションで弾く）。
    pc_slots = Column(JSON, nullable=True)
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
    # あらすじ蒸留専用の LLM プリセット ID（LLMModelPreset.id）。
    # シナリオ本編のレートリミット節約を目的に、GM とは別のモデルを指定できる。
    # 起動時は必ず指定する（未指定なら gm_preset_id と同じ値を入れること）。
    # チャット中も同モーダルから変更可能。
    synopsis_preset_id = Column(String, nullable=False, default="")
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
    # PC配役（ensemble_pc エンジン時のみ参照）。JSON 配列。
    # 形式: [{"slot_id":"pc1","player_type":"user"|"character","character_id":"...","preset_id":"..."}]
    #   - player_type="user": そのスロットをユーザ自身が演じる。character_id/preset_id 不要。
    #     ユーザの @タグ名は親シナリオの pc_slots[slot_id].name を使う（user_alias は無視される）。
    #   - player_type="character": Chotgor キャラが PC として演じる。character_id 必須、preset_id 推奨。
    # ensemble エンジン時は NULL/空配列のまま使われない。
    pc_assignments = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )


class DebugLogEntry(Base):
    """デバッグログエントリ — 1LLM呼び出し単位のログレコード。

    1ユーザーリクエスト内に複数の LLM 呼び出し（chat/farewell/trigger 等）が
    あった場合、それぞれが独立した行として記録される。同一ユーザーリクエストの行は
    request_id が共通し、Logs 画面ではまとめて1行として表示される。
    シナリオの再生成は同一 request_id で行を追加することで試行履歴として保持する。
    """

    __tablename__ = "debug_log_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String, nullable=False, index=True)  # 8桁hex、ユーザーリクエスト単位で共有
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now())
    source_type = Column(String, nullable=False)  # 'chat'/'scenario'/'farewell'/'trigger'/'batch' 等
    session_id = Column(String, nullable=True)     # chat_sessions.id または scenario_sessions.id
    turn_sequence = Column(Integer, nullable=True) # セッション内ターン番号
    target = Column(String, nullable=True)         # キャラ名/シナリオ名/バッチ対象名
    preset = Column(String, nullable=True)         # 使用プリセット名
    user_message = Column(Text, nullable=True)     # ユーザー発言本文（chat/scenario のみ）
    response = Column(Text, nullable=True)         # 応答テキスト本文
    reasoning = Column(Text, nullable=True)        # 思考ブロック等の推論テキスト
    mcp_calls_json = Column(Text, nullable=True)   # MCPツール呼び出しのJSON配列
    has_error = Column(Integer, nullable=False, default=0)  # 1=エラーあり
    warn_reason = Column(Text, nullable=True)      # 警告・エラーの人間可読な理由文
    raw_dir = Column(String, nullable=True)        # 生ファイルフォルダのパス


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
    log_request_id = Column(String, nullable=True)                 # debug_log_entries.request_id との紐付け（再生成で引き継ぐ）
    # ANTICIPATE_RESPONSE: GM がターン末尾に書いた「次の展開への予想（期待）」。
    # ターンに1つ。次ターンの GM システムプロンプトに「前回の予想」として注入される。
    anticipation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())


class SQLiteStore(
    SettingsStoreMixin,
    CharacterStoreMixin,
    InscribedMemoryStoreMixin,
    PresetStoreMixin,
    ChatStoreMixin,
    WorkingMemoryStoreMixin,
    ScenarioChatStoreMixin,
    DebugLogStoreMixin,
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
        self._migrate_add_preset_timeout_seconds()
        self._migrate_add_synopsis_preset_id()
        self._migrate_add_debug_log_entries()
        self._migrate_add_scenario_turn_log_request_id()
        self._migrate_add_scenario_custom_system_prompt()
        self._migrate_drop_afterglow_columns()
        self._migrate_add_chat_message_anticipation()
        self._migrate_add_scenario_turn_anticipation()
        self._migrate_add_memory_origin()
        self._migrate_add_scenario_pc_mode()
        self._migrate_drop_session_drifts()

    def _migrate_drop_session_drifts(self) -> None:
        """SELF_DRIFT 機能撤去に伴い session_drifts テーブルを削除する。

        ドリフト機能（drift_manager.py / chat_drifts.py / drift ツール）は撤去済みのため、
        既存 DB にのみ残る session_drifts テーブルを物理削除する。
        新規 DB には ORM 定義が無く作成されないため、本マイグレーションは冪等。
        """
        with self.engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE IF EXISTS session_drifts")

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

    def _migrate_add_synopsis_preset_id(self) -> None:
        """`scenario_sessions` に `synopsis_preset_id` 列を追加する。

        旧スキーマ: あらすじ蒸留はセッションの `gm_preset_id` を使い回す
        新スキーマ: あらすじ蒸留専用の `synopsis_preset_id` を持つ

        移行手順:
            1. scenario_sessions に列が無ければ ADD COLUMN
            2. 既存行は `gm_preset_id` を初期値として埋める（従来挙動を維持）

        新規 DB（既に新スキーマ）では何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "scenario_sessions" not in tables:
                return
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenario_sessions)"
                ).fetchall()
            }
            if "synopsis_preset_id" in cols:
                return
            conn.exec_driver_sql(
                "ALTER TABLE scenario_sessions "
                "ADD COLUMN synopsis_preset_id TEXT NOT NULL DEFAULT ''"
            )
            # 既存行は gm_preset_id をそのままコピー（従来挙動を維持）
            conn.exec_driver_sql(
                "UPDATE scenario_sessions "
                "SET synopsis_preset_id = gm_preset_id "
                "WHERE (synopsis_preset_id IS NULL OR synopsis_preset_id = '')"
            )

    def _migrate_add_preset_timeout_seconds(self) -> None:
        """`llm_model_presets` テーブルに `timeout_seconds` 列を追加する。

        プロバイダーAPIリクエストのタイムアウトをプリセット単位で指定するための列。
        既存DBでは列が存在しないため ALTER TABLE で追加し、デフォルト300秒（5分）を入れる。
        新規DB（既に新スキーマ）では何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "llm_model_presets" not in tables:
                return
            cols = {
                r[1]
                for r in conn.exec_driver_sql("PRAGMA table_info(llm_model_presets)").fetchall()
            }
            if "timeout_seconds" in cols:
                return
            conn.exec_driver_sql(
                "ALTER TABLE llm_model_presets "
                "ADD COLUMN timeout_seconds INTEGER NOT NULL DEFAULT 300"
            )

    def _migrate_add_debug_log_entries(self) -> None:
        """`debug_log_entries` テーブルが存在しない既存 DB への互換マイグレーション。

        `Base.metadata.create_all` が新規テーブルを作るが、既存 DB には
        インデックスが追加されない場合があるため、インデックスだけ別途作成する。
        新規 DB（既に新スキーマ）では何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "debug_log_entries" not in tables:
                # テーブルがない場合は create_all で作成済みのはずだが念のため
                return
            # request_id インデックスが存在しなければ作成（冪等）
            indexes = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND tbl_name='debug_log_entries'"
                ).fetchall()
            }
            if "ix_debug_log_entries_request_id" not in indexes:
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_debug_log_entries_request_id "
                    "ON debug_log_entries (request_id)"
                )

    def _migrate_add_scenario_turn_log_request_id(self) -> None:
        """`scenario_turns` に `log_request_id` 列を追加する。

        再生成時に同一 request_id を引き継ぐためのカラム。
        既存 DB には列がないため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenario_turns)"
                ).fetchall()
            }
            if "log_request_id" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenario_turns ADD COLUMN log_request_id TEXT"
                )

    def _migrate_add_chat_message_anticipation(self) -> None:
        """`chat_messages` に `anticipation` 列を追加する。

        キャラクターが本文末尾に書いた予想（期待）タグの抽出結果を保存する列。
        既存 DB には列がないため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(chat_messages)"
                ).fetchall()
            }
            if "anticipation" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_messages ADD COLUMN anticipation TEXT"
                )

    def _migrate_add_scenario_turn_anticipation(self) -> None:
        """`scenario_turns` に `anticipation` 列を追加する。

        GM がターン末尾に書いた予想（期待）タグの抽出結果を保存する列。
        既存 DB には列がないため ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenario_turns)"
                ).fetchall()
            }
            if "anticipation" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenario_turns ADD COLUMN anticipation TEXT"
                )

    def _migrate_add_memory_origin(self) -> None:
        """`inscribed_memories` と `working_memory_threads` に `origin` 列を追加する。

        Scenario PC モード（TRPG的にキャラがPCを演じるモード）で生じた記憶を
        `origin='interlude'` で識別するための列。日常体験は `origin='real'`（既定）。
        検索フィルタとキャラ本人の文脈把握に使う。
        既存DBに列がなければ ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "inscribed_memories" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(inscribed_memories)"
                    ).fetchall()
                }
                if "origin" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE inscribed_memories "
                        "ADD COLUMN origin TEXT NOT NULL DEFAULT 'real'"
                    )
            if "working_memory_threads" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(working_memory_threads)"
                    ).fetchall()
                }
                if "origin" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE working_memory_threads "
                        "ADD COLUMN origin TEXT NOT NULL DEFAULT 'real'"
                    )

    def _migrate_add_scenario_pc_mode(self) -> None:
        """Scenario PC モード関連カラムを追加する。

        - `scenarios.dice_pool_spec` (JSON, NULL可): ダイスプール仕様。
        - `scenarios.pc_slots` (JSON, NULL可): PC枠定義。
        - `scenario_sessions.pc_assignments` (JSON, NULL可): PC配役一覧。

        いずれも `engine_type='ensemble_pc'` のセッション専用フィールド。
        既存DBに列がなければ ALTER TABLE で追加する。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "scenarios" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(scenarios)"
                    ).fetchall()
                }
                if "dice_pool_spec" not in cols:
                    # SQLite の JSON 型は TEXT として保存される。
                    conn.exec_driver_sql(
                        "ALTER TABLE scenarios ADD COLUMN dice_pool_spec TEXT"
                    )
                if "pc_slots" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE scenarios ADD COLUMN pc_slots TEXT"
                    )
            if "scenario_sessions" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(scenario_sessions)"
                    ).fetchall()
                }
                if "pc_assignments" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE scenario_sessions ADD COLUMN pc_assignments TEXT"
                    )

    def _migrate_drop_afterglow_columns(self) -> None:
        """Afterglow（感情継続機構）廃止に伴い関連カラムを削除する。

        - `chat_sessions.afterglow_session_id`
        - `characters.afterglow_default`

        SQLite 3.35+ の DROP COLUMN を使う。新規DBには列が無いため何もしない。冪等。
        """
        with self.engine.begin() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "chat_sessions" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(chat_sessions)"
                    ).fetchall()
                }
                if "afterglow_session_id" in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE chat_sessions DROP COLUMN afterglow_session_id"
                    )
            if "characters" in tables:
                cols = {
                    r[1]
                    for r in conn.exec_driver_sql(
                        "PRAGMA table_info(characters)"
                    ).fetchall()
                }
                if "afterglow_default" in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE characters DROP COLUMN afterglow_default"
                    )

    def _migrate_add_scenario_custom_system_prompt(self) -> None:
        """`scenarios` に `custom_system_prompt` 列を追加し、既存シナリオに規定プロンプトを設定する。

        GMシステムプロンプトをシナリオ単位でカスタマイズするためのカラム。
        既存シナリオには DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE を設定する。冪等。
        """
        with self.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenarios)"
                ).fetchall()
            }
            if "custom_system_prompt" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenarios ADD COLUMN custom_system_prompt TEXT"
                )
                # カラム追加後、既存シナリオにデフォルトプロンプトを設定
                from backend.services.scenario_chat.prompt_builder import (
                    DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE,
                )

                conn.exec_driver_sql(
                    "UPDATE scenarios SET custom_system_prompt = ? WHERE custom_system_prompt IS NULL",
                    (DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE,),
                )

    def get_session(self) -> Session:
        """新しい DB セッションを返す。Mixin クラスが共通して使用する。"""
        return self.SessionLocal()

