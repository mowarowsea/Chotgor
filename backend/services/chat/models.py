"""フロントエンド非依存のドメインモデル。"""

from dataclasses import dataclass, field

@dataclass
class Message:
    role: str                          # "user" | "system" | "assistant"（※ API 仕様上の呼称。内部ではキャラクターターンとして扱う）
    content: str | list | None


@dataclass
class ChatRequest:
    character_id: str
    character_name: str                # プロバイダーの XML タグ用
    provider: str
    model: str
    messages: list[Message]
    character_system_prompt: str = ""
    self_history: str = ""
    relationship_state: str = ""
    inner_narrative: str = ""
    provider_additional_instructions: str = ""
    thinking_level: str = "default"
    settings: dict = field(default_factory=dict)
    enable_time_awareness: bool = False
    current_time_str: str = ""
    time_since_last_interaction: str = ""
    # SELF_DRIFT: セッションIDと現在有効なdrift指針テキスト一覧。
    # session_id が空の場合はdrift処理をスキップする。
    session_id: str = ""
    active_drifts: list = field(default_factory=list)
    # switch_angle: このキャラクターが切り替え可能なプリセット一覧。
    # 各エントリは {preset_id, preset_name, provider, model_id, additional_instructions,
    #               thinking_level, when_to_switch} を持つ dict。
    # 空の場合は switch_angle ツール・タグを使用不可とする。
    available_presets: list[dict] = field(default_factory=list)
    # 現在使用中のプリセット名（システムプロンプトの「現在のアングル」表示に使用）。
    current_preset_name: str = ""
    # 現在使用中のプリセットID（記憶作成時の出所記録に使用）。
    current_preset_id: str = ""
    # PowerRecall: キャラクターが能動的に検索した記憶・会話履歴。
    # {"inscribed_memories": [...], "chat_turns": [...]} の辞書。
    # 再呼び出し時に Chotgor ユーザーターン（messages）として注入される。
    # 非空時はループ防止フラグとしても機能する。
    power_recalled: dict = field(default_factory=dict)
    # 自己参照ループ設定（キャラクターごとに保持）
    self_reflection_mode: str = "disabled"       # disabled/local_trigger/always
    self_reflection_preset_id: str = ""          # 契機判断モデルプリセットID
    self_reflection_n_turns: int = 5             # 自己参照に使う直近ターン数
    # 外部ツール許可設定（ClaudeCliProvider の --tools フラグに反映される）
    allowed_tools: dict = field(default_factory=dict)
    # プロバイダーAPIリクエストのタイムアウト秒数。プリセット単位で設定可能。
    # 現状は OllamaProvider のみが参照する。0 以下は無効として扱われ、各プロバイダー側でデフォルト値が使われる。
    timeout_seconds: int = 300
    # 別れ検出用: _build_chat_request でキャラクターから一度だけ取得してサービス層に渡す。
    # ストリーム完了後の get_character() 再クエリを省略するためのキャッシュ。
    farewell_config: dict | None = None
    farewell_relationship_status: str = "active"
    # ANTICIPATE_RESPONSE: 前ターンでキャラクター自身が本文末尾に書いた「次の展開への予想（期待）」。
    # 次ターンのシステムプロンプトに「前回のあなたの予想」として注入される。
    previous_anticipation: str = ""
    # inscribe_memory / post_working_memory_thread で保存される記憶/スレッドの origin（3値）。
    # 1on1 通常経路では "real"、シナリオ PC モードからの呼び出しでは "interlude"、
    # うつつ（Usual Days）無人経路では "usual"。
    # ToolExecutor.default_origin に流れ、inscribe_memory / post_working_memory_thread の保存時に付与される。
    default_origin: str = "real"
    # うつつ（Usual Days）有効キャラかどうか。True なら 1on1 システムプロンプトに
    # 「ユーザの知らない日常生活と記憶がある」注釈ブロックを出す（request_builder）。
    usual_days_enabled: bool = False
    # このキャラがユーザを呼ぶ呼称（character_query._resolve_user_info で解決済み）。
    # キャラ別 user_label > Settings の user_name > 空 の優先順位を request_factory で適用。
    # request_builder が「あなたが対話する相手」ブロックを構築するのに使う。
    user_label: str = ""
    # ユーザの位置づけ短文（キャラ別の characters.user_position のみ）。空なら呼称のみ注入。
    user_position: str = ""
    # 対面モードフラグ。True なら 1on1 システムプロンプトに「いまは対面でユーザと向き合っている」
    # 注釈ブロックを差し込む。キャラスコープの characters.face_to_face_mode をリクエスト時に
    # 反映する（送信時値の焼き付けは ChatMessage 側で別途行う）。
    face_to_face: bool = False
