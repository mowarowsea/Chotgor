"""Pydantic schemas for the REST API."""

from typing import Optional

from pydantic import BaseModel


# --- Character API Schemas ---

class CharacterCreate(BaseModel):
    name: str
    system_prompt_block1: str = ""
    inner_narrative: str = ""
    cleanup_config: dict = {}
    ghost_model: Optional[str] = None


class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt_block1: Optional[str] = None
    inner_narrative: Optional[str] = None
    self_history: Optional[str] = None
    relationship_state: Optional[str] = None
    cleanup_config: Optional[dict] = None
    ghost_model: Optional[str] = None
