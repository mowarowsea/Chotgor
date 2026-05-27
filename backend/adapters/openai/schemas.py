"""OpenAI ワイヤーフォーマット専用の Pydantic モデル。"""

from pydantic import BaseModel


class OAIChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class OAIChatRequest(BaseModel):
    model: str  # format: "{character_id}@{provider}"
    messages: list[OAIChatMessage]
    stream: bool = False
    max_tokens: int | None = 4096
    temperature: float | None = None
