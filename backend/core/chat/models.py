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
    meta_instructions: str = ""
    provider_additional_instructions: str = ""
    thinking_level: str = "default"
    settings: dict = field(default_factory=dict)
    enable_time_awareness: bool = False
    current_time_str: str = ""
    time_since_last_interaction: str = ""
