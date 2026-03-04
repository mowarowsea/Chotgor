"""Pydantic schemas for the REST API."""

from typing import Optional

from pydantic import BaseModel


# --- Character API Schemas ---

class CharacterCreate(BaseModel):
    name: str
    system_prompt_block1: str = ""
    meta_instructions: str = ""
    cleanup_config: dict = {}


class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt_block1: Optional[str] = None
    meta_instructions: Optional[str] = None
    cleanup_config: Optional[dict] = None


# --- OpenAI-compatible API Schemas ---

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str  # format: "{character_id}@{provider}"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = None
