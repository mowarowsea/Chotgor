"""Pydantic schemas for the REST API."""

from typing import Optional

from pydantic import BaseModel


# --- Character API Schemas ---

class CharacterCreate(BaseModel):
    """キャラクター作成リクエストスキーマ。"""

    name: str
    system_prompt_block1: str = ""
    inner_narrative: str = ""
    cleanup_config: dict = {}
    ghost_model: Optional[str] = None
    allowed_tools: dict = {}


class CharacterUpdate(BaseModel):
    """キャラクター更新リクエストスキーマ。

    farewell_config: Chronicle でキャラクター自身が設定する感情閾値・退席メッセージ・疎遠化条件。
        フォーマット: {"thresholds": {"anger": 0.8, ...}, "farewell_message": {"negative": "...", ...},
                       "estrangement": {"lookback_days": 30, "negative_exit_threshold": 5}}
    relationship_status: "active" または "estranged"。
        "estranged" になるとそのキャラクターへの全リクエストが恒久的に拒否される。
    """

    name: Optional[str] = None
    system_prompt_block1: Optional[str] = None
    inner_narrative: Optional[str] = None
    self_history: Optional[str] = None
    relationship_state: Optional[str] = None
    cleanup_config: Optional[dict] = None
    ghost_model: Optional[str] = None
    farewell_config: Optional[dict] = None
    relationship_status: Optional[str] = None
    allowed_tools: Optional[dict] = None
