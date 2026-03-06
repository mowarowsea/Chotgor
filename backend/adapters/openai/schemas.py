"""OpenAI ワイヤーフォーマット専用の Pydantic モデル。"""

from typing import Optional, Union

from pydantic import BaseModel


class OAIChatMessage(BaseModel):
    role: str
    content: Union[str, list, None] = None


class OAIChatRequest(BaseModel):
    model: str  # format: "{character_id}@{provider}"
    messages: list[OAIChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = None
