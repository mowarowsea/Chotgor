"""SQLite ORM モデル定義。

スキーマはこの ORM 定義が唯一の正。
``Base.metadata.create_all`` が新規 DB に全テーブルを定義通りに作成する。
テーブル名・カラム追加等の変更は `migrations.py` の冪等マイグレーションで追従する。
後方互換のため、各モデルは `backend.repositories.sqlite.store` からも import できる。
"""

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
)
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """SQLAlchemy 宣言ベースクラス。"""

    pass


class GlobalSetting(Base):
    """グローバル設定 — キー/バリュー形式の設定テーブル。"""

    __tablename__ = "global_settings"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=True)


class ChatSession(Base):
    """チャットセッション — 1on1 の会話スレッド（旧グループチャット撤去済み）。"""

    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)   # "{char_name}@{preset_name}"
    title = Column(String, nullable=False, default="新しいチャット")
    # session_type は将来の拡張余地として残置（現状は "1on1" のみ）。
    session_type = Column(String, nullable=False, default="1on1")
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
    character_name = Column(String, nullable=True)  # シナリオPC・うつつ発話時のキャラクター名
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
    # このメッセージが交わされた時点のチャットモード（0=テキスト / 1=対面）。
    # キャラクタースコープの face_to_face_mode を送信時に焼き付ける。後からハレ履歴を
    # うつつへ流し込む際、対面とテキストでラベルを切り替えるために使う。
    face_to_face = Column(Integer, nullable=False, default=0)
    # メッセージ預かり（escrow・めぐり §5.1）: NULL = 預かり中（キャラ未読・LLM 未到達）、
    # タイムスタンプ = キャラに渡った時刻。chronicled_at と同パターン。
    # unavailable 中のユーザ発言は NULL で保存だけされ、availability が戻った時点で
    # まとめて時間差注釈付きで渡される。通常のメッセージは作成時に即セットされる。
    delivered_at = Column(DateTime, nullable=True)
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
    # このキャラが対話する「ユーザ」の人物像。1on1 チャット・全バッチ処理（chronicle/forget/
    # self_reflection/うつつ headless）すべてのシステムプロンプトに「あなたが対話する相手」
    # ブロックとして注入される。
    # - user_label: キャラがユーザを呼ぶ呼称。空なら Settings の user_name にフォールバック。
    # - user_position: ユーザの位置づけ（役職・関係・接触の仕方など短文）。空なら呼称のみ注入。
    # うつつでは pc_slots[slot_id="user"] にもこの値が同期保存され、GM の「不在 PC を代弁しない」保護に使われる。
    user_label = Column(String, nullable=False, default="")
    user_position = Column(Text, nullable=False, default="")
    # キャラ本人が「周囲（NPC・知人）にユーザのことをどこまで・どんなふうに伝えているか」を
    # 自分の言葉で書き下ろした自由記述。うつつ世界の GM システムプロンプトに
    # 「不在の関係者」ブロックとして流し込まれ、NPC の自発的な言及度を本人の流儀で
    # コントロールする。空なら「完全に秘匿（NPC は触れない）」を意味する。
    # 想定運用: キャラ編集画面の「キャラに聞く」ボタンで本人に問い、返答を書き留める。
    user_visibility_note = Column(Text, nullable=False, default="")
    # 対面モード状態（0=テキスト / 1=対面）。キャラクタースコープで保持し、1on1チャットの
    # トグルで切り替える。対面中は ChatView で背景画像を出し、system prompt に対面ブロックを
    # 差し込み、うつつスケジューラはこのキャラのスロットをスキップする。
    face_to_face_mode = Column(Integer, nullable=False, default=0)
    # 対面モード時に ChatView 背景へ表示する画像（base64 data URI）。image_data と同じ
    # 保存形式。未設定 = 背景表示なし（モード ON でも背景なしで動作する）。
    face_to_face_bg_image = Column(Text, nullable=True)
    # ユーザダイヤル（覗き窓・めぐり / Aliveness §2.4）。user_ui 観測者の開示段階:
    #   0 = 全開（開発期・全 content）
    #   1 = 生活の秘匿（scene.*（usual）→ envelope）
    #   2 = 内面の秘匿（＋ memory.* / intent.* / night.* → hidden）
    #   3 = 最終形（チャット応答のみ。計器はダイヤル非依存で残る）
    # v1 は手動。計器の静音期間（無事故N日）が「窓を閉じてよい」確信を支える。
    timeline_dial = Column(Integer, nullable=False, default=0)
    # 会話外行動メニュー（めぐり / Aliveness §5.3）。JSON:
    #   {"push": bool, "research": bool, "impromptu_scene": bool}
    # 個別 ON/OFF トグル（キャラ設定画面）。NULL / 空 = 全部 OFF（オプトイン）。
    #   push            = 新規セッションを立ててキャラ発メッセージ
    #   research        = web_search で興味 intent を消費（調べもの）
    #   impromptu_scene = スロット外の臨時うつつシーン（うつつ有効が前提条件）
    action_menu = Column(JSON, nullable=True)
    # 生活時間割（めぐり / Aliveness §5.1）。JSON:
    #   {"mon": [{"from": "09:00", "to": "18:00", "label": "仕事"}], "tue": [...], ...}
    # 曜日キー（mon〜sun）ごとの **応答できない時間帯** のリスト。
    # NULL / 空 = 常時応答可能。キャラクター設計者（ユーザ）が管理UIで設定する。
    availability_schedule = Column(JSON, nullable=True)
    # away 状態（動的な不在。疲労離席・take_leave が設定する）。
    # away_until が未来の間は availability ゲートが unavailable(away_reason) を返す。
    # NULL = away でない。時刻が過ぎれば自動的に解除扱い（行の掃除は不要）。
    away_until = Column(DateTime, nullable=True)
    away_reason = Column(String, nullable=True)
    # 圧力の体質プロファイル（めぐり / Aliveness §4.2）。JSON:
    #   {"version": 1,
    #    "social": {"tau_days": float, "sharpness": float},
    #    "boredom": {"sensitivity": float},
    #    "body": {"fatigue_sensitivity": float},
    #    "interview": {"answers": {...}, "raw": str, "asked_at": str}}
    # NULL = 標準プロファイル（services/pressure の DEFAULT_PROFILE）。
    # 初期化は本人インタビュー（体験の質問 → 固定ルーブリックが係数へ決定論写像）。
    # 本人からの更新経路は作らない（圧力は物理 — 非制御性の担保）。
    # ユーザは管理UIから編集できる（守護者の介入枠）。
    pressure_profile = Column(JSON, nullable=True)
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
    # 記憶のソース識別（3値）。"real"=日常体験（ユーザと共有）、
    # "usual"=うつつ（ユーザ未共有の自分の生活体験）、"interlude"=シナリオPCモードで演じた幕間体験。
    # 検索時のフィルタやキャラ本人の文脈把握に使う（由来タグであり、想起・蒸留・忘却では同次元に扱う）。
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
    # スレッドのソース識別（3値）。"real"=日常、"usual"=うつつ（ユーザ未共有の自分の生活体験）、
    # "interlude"=シナリオPCモードで生じた幕間スレッド。
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
    # （旧 user_alias 列は廃止。ユーザPCも pc_slots の 1 枠として表現し、
    #   セッションの pc_assignments で player_type="user" を割り当てる。）
    history_max_turns = Column(Integer, nullable=True)     # 送信履歴の最大ターン数。NULL=settings 既定
    history_max_chars = Column(Integer, nullable=True)     # 送信履歴の最大文字数。NULL=settings 既定
    custom_system_prompt = Column(Text, nullable=True)     # GMシステムプロンプトの完全カスタマイズ。NULLなら自動生成ロジック使用
    # ダイスプール仕様（JSON: {"d6": 10, "d100": 5} 等）。NULL/未指定なら engine 側で {"d6": 10} を既定値として使う。
    # ensemble_pc エンジン時のみ参照する。毎ターン engine 内で乱数生成して GM system prompt の {dice_pool} へ注入。
    dice_pool_spec = Column(JSON, nullable=True)
    # PC枠定義（ensemble_pc エンジン時のみ参照）。JSON 配列
    # [{"slot_id":"pc1","name":"アリス","description":"...","image_data":"data:image/..."}]。
    # シナリオ作成時にユーザが PC として登場する「枠」（人物像・知っていること）を定義する。
    # image_data は任意のアバター（base64 data URI）。プロンプトには載らない
    # （normalize_pc_slots が既知キーのみ抽出するため表示専用）。
    # 各セッション起動時、`scenario_sessions.pc_assignments` で各 slot_id に「ユーザが演じる/AIキャラが演じる」を割り当てる。
    # NULL/空配列なら ensemble_pc を使えない（API バリデーションで弾く）。
    pc_slots = Column(JSON, nullable=True)
    banner_data = Column(Text, nullable=True)              # バナー画像 (base64 data URI)。一覧・編集画面の見栄え用
    # うつつ（Usual Days）所有者キャラ ID。
    #   NULL  = 通常の汎用シナリオ（汎用シナリオ一覧に出る）。
    #   値あり = そのキャラの「うつつ（生活世界）」専用シナリオ。汎用一覧からは除外し、
    #            1 キャラ 1 世界として扱う。除外判定のキーになる。
    owner_character_id = Column(String, nullable=True)
    # うつつ運用設定（JSON, NULL可）。owner_character_id が NULL のときは未使用。
    #   {"enabled": bool, "slots": ["10:00","13:00","17:00"],
    #    "time_grid": {曜日×時間帯→ラベル}, "event_categories": {...},
    #    "event_probability": float, "max_responses_per_scene": int,
    #    "gm_preset_id": str, "pc_preset_id": str}
    # SQLite では TEXT として保存される。
    # 後方互換: 旧キー "max_turns_per_scene" は service 側の読み出しでフォールバック。
    usual_config = Column(JSON, nullable=True)
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
    # 実行エンジン種別。'ensemble'（GMのみ）/ 'ensemble_pc'（PC配役あり）/
    # 'usual_days'（うつつ無人ループ。GM部分は ensemble_pc と共有し、無人ループ制御だけ service 側で分岐）。
    engine_type = Column(String, nullable=False, default="ensemble")
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


class LlmUsageEvent(Base):
    """LLM 使用量イベント — 1 API 呼び出し単位のリクエスト数・トークン数の記録。

    レスポンスから使用量が判明するプロバイダー（claude_cli / google）が
    usage_recorder 経由で1呼び出しごとに1行を追加する。tool-use ループで
    複数回 API を叩いた場合はその回数だけ行が増える（リクエストごとの粒度）。
    ダッシュボード（/ui/）の日次・週次集計の元データ。
    """

    __tablename__ = "llm_usage_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(), index=True)
    provider = Column(String, nullable=False)       # claude_cli / google 等
    model = Column(String, nullable=True)           # 実際に使われたモデルID（レスポンス由来があれば優先）
    preset_name = Column(String, nullable=True)     # 使用プリセット名
    target = Column(String, nullable=True)          # キャラ名/シナリオ名/バッチ対象名（log_context 由来）
    feature = Column(String, nullable=True)         # chat / scenario / chronicle / trigger 等
    request_id = Column(String, nullable=True, index=True)  # debug_log_entries.request_id と同じ8桁hex
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cache_read_input_tokens = Column(Integer, nullable=False, default=0)      # claude_cli のみ
    cache_creation_input_tokens = Column(Integer, nullable=False, default=0)  # claude_cli のみ
    total_cost_usd = Column(Float, nullable=True)   # claude_cli の result イベント由来（参考値）


class ToolCallEvent(Base):
    """ツール実行イベント — キャラクターのツール使用1回 = 1行の実行時記録。

    Logs 画面のツール使用表示は、かつて debug/ の生ログ（JSON / stream-json /
    テキストタグの5形式）を表示時に逆解析していたが、プロバイダーやログ書式の
    変更のたびに壊れる構造だった。本テーブルは「実行時に確定した事実」を
    tool_event_recorder（backend/lib/tool_event_recorder.py）経由でその場で記録し、
    表示時の解析を不要にする（llm_usage_events と同じ思想）。

    記録経路:
        - tool-use 方式:  ToolExecutor.execute()（全プロバイダー・MCP プロキシ・バッチ共通の関門）
        - タグ方式:       inscriber / carver / switcher の *_from_text と
                          ChatService の power_recall 実行箇所
        - 予想タグ:       ANTICIPATE_RESPONSE の採用箇所（保存・次ターン注入される値のみ）
    """

    __tablename__ = "tool_call_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(), index=True)
    request_id = Column(String, nullable=True, index=True)  # debug_log_entries.request_id と同じ8桁hex（再生成で共有）
    dir_id = Column(String, nullable=True, index=True)      # debug フォルダ名と同じ8桁hex（試行ごとに fresh）
    target = Column(String, nullable=True)                  # キャラ名/シナリオ名/バッチ対象名（log_context 由来）
    feature = Column(String, nullable=True)                 # chat / scenario / chronicle / forget 等
    source = Column(String, nullable=False, default="tool_use")  # tool_use / tag / anticipation（記録経路の区別）
    tool_name = Column(String, nullable=False)              # inscribe_memory / carve_narrative / anticipate_response 等
    arguments_json = Column(Text, nullable=True)            # ツール引数 dict の JSON（表示は tool_tags が担う）
    status = Column(String, nullable=False, default="ok")   # ok / error
    error_message = Column(Text, nullable=True)             # status=error 時の詳細（結果文字列 or 例外メッセージ）


class Intent(Base):
    """意図 — キャラクターの「〜したい」の経済層レコード（めぐり / Aliveness §4.3）。

    欲求は行動に先行し、事後に遡って発見できねばならない — 意図は夜のChronicleと
    うつつシーン完走後の「拾い上げ」で本人の言葉のまま記録される（機械が要約・
    丸めをしない）。

    圧力カラムは持たない: 意図圧は g(経過日数, source_kind の現在圧) の読み取り時
    計算（services/intents/lifecycle.py）。タイムライン封筒には遷移
    （intent.created / fulfilled / expired / soured）だけが載る。
    """

    __tablename__ = "intents"

    id = Column(String, primary_key=True)  # UUID
    character_id = Column(String, ForeignKey("characters.id"), nullable=False, index=True)
    description = Column(Text, nullable=False)   # 本人の言葉のまま（丸めない）
    target = Column(String, nullable=True)       # user / npc:<名前> / self / NULL
    # 唯一の可変状態: active / fulfilled / expired / soured
    status = Column(String, nullable=False, default="active")
    # 意図の源になった圧: social / boredom / body / none
    source_kind = Column(String, nullable=False, default="none")
    # 拾い上げ地点: night_chronicle / usual_scene
    born_from = Column(String, nullable=False, default="night_chronicle")
    payload = Column(JSON, nullable=True)        # 不満の言葉・裁定文脈など
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now())
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
    )
    resolved_at = Column(DateTime, nullable=True)  # active 以外へ遷移した時刻


class Alarm(Base):
    """計器アラーム — 幻想の穴が開いた証拠の追記型記録（めぐり / Aliveness §3）。

    アラーム（severity="alarm"）＝発火したら調査すべき異常。
    スメル（severity="smell"）＝Tier 2 検知器の疑い記録（誤検知許容・傾向観測）。
    静音期間（無事故N日）の計算対象は severity="alarm" のみ。

    計器は観測者ではなく監査者 — キャラクターの世界には一切現れず、
    ユーザダイヤルにも依存せず常時稼働する。
    """

    __tablename__ = "alarms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    invariant_id = Column(String, nullable=False, index=True)  # fabrication_backstop / usual_scene_error 等
    severity = Column(String, nullable=False, default="alarm")  # alarm | smell
    occurred_at = Column(DateTime, nullable=False, default=lambda: datetime.now(), index=True)
    details = Column(JSON, nullable=True)          # 発火文脈（キャラ名・対象ID・検知内容など）
    acknowledged_at = Column(DateTime, nullable=True)  # ユーザが確認済みにした時刻


class MeterSnapshot(Base):
    """計器メーター — 傾向観測の日次スナップショット（発火概念なし）。

    肥大メーター（inner_narrative 長・WMスレッド数・記憶件数など）と
    圧力の日次スナップショット（Phase 3）の記録先。
    アラームと違い「異常」ではなく「傾向」— グラフで眺める素材。
    """

    __tablename__ = "meter_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meter_id = Column(String, nullable=False, index=True)   # inner_narrative_len / wm_thread_count / pressure_social 等
    character_id = Column(String, nullable=True, index=True)  # 対象キャラ（全体系メーターは NULL）
    value = Column(Float, nullable=False)
    occurred_at = Column(DateTime, nullable=False, default=lambda: datetime.now(), index=True)
    details = Column(JSON, nullable=True)


class TimelineEvent(Base):
    """タイムライン封筒 — キャラクターの身に起きた出来事の「存在・順序・相手・時刻」の正本。

    めぐり（巡り / Aliveness）の中核テーブル（docs/planned/aliveness_plan.md §2）。
    中身（発言本文・記憶本文など）は既存テーブルに残し、source_table / source_id で
    JOIN する（封筒 dual-write 方式。中身の複製はしない）。
    payload 完結型イベント（night.* / scene.closed / memory.carved 等）は
    source_table = NULL で payload に可変属性を持つ。

    不可逆性の担保: 履歴の巻き戻し（再生成・編集による削除）では封筒を削除せず
    retracted_at をマークする。retracted なイベントは全観測者から hidden（存在層ごと）
    だがデータは残る。

    可視性（self / world_frame / user_ui の開示レベル）は読み取り時ポリシーで判定し、
    この行には焼き付けない。
    """

    __tablename__ = "timeline_events"

    id = Column(String, primary_key=True)                      # UUID
    character_id = Column(String, nullable=False, index=True)  # 誰のタイムラインか
    event_type = Column(String, nullable=False)                # ドット記法カタログ（chat.message / scene.turn / memory.inscribed ...）
    occurred_at = Column(DateTime, nullable=False, index=True) # 出来事の時刻（バックフィルは源の created_at）
    actor = Column(String, nullable=True)                      # user / character / narrator / npc:<名前> / system
    counterpart = Column(String, nullable=True)                # 封筒の「相手」: user / npc:<名前> / NULL
    origin = Column(String, nullable=False, default="real")    # real / usual / interlude（既存3値と同次元）
    modality = Column(String, nullable=True)                   # text / face（chat.message のみ）
    session_id = Column(String, nullable=True, index=True)     # 投影の封筒集約キー（chat/scenario セッション）
    source_table = Column(String, nullable=True)               # 中身への参照（payload 完結型は NULL）
    source_id = Column(String, nullable=True, index=True)      # 中身への参照（payload 完結型は NULL）
    intent_id = Column(String, nullable=True)                  # intent.* / action.* が張る FK（intents.id）
    payload = Column(JSON, nullable=True)                      # 型ごとの可変属性（判定スコア等もここ）
    retracted_at = Column(DateTime, nullable=True)             # 巻き戻しマーク（削除の代わり）
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now())  # 記録された時刻（occurred_at と分離）


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
    # クロニクル処理日時: NULL=未処理、タイムスタンプあり=処理済み。
    # うつつ（usual_days）のやり取りを Chronicle 対象に合流させるための列
    # （ChatMessage.chronicled_at と同じ役割）。通常シナリオのターンでは使われない。
    chronicled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now())


