"""フロントエンド非依存のドメインモデル。"""

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Message:
    role: str                          # "user" | "assistant" | "system"
    content: Union[str, list, None]


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
    # グループチャット用: 記憶想起クエリの上書き。
    # 空文字列の場合は messages の最後のユーザーメッセージを使う。
    recall_query_override: str = ""
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
    # {"memories": [...], "chat_turns": [...]} の辞書。
    # 再呼び出し時に Chotgor ユーザーターン（messages）として注入される。
    # 非空時はループ防止フラグとしても機能する。
    power_recalled: dict = field(default_factory=dict)
