"""REST API 用 Pydantic スキーマ。"""

from pydantic import BaseModel


# --- キャラクター API スキーマ ---

class CharacterCreate(BaseModel):
    """キャラクター作成リクエストスキーマ。"""

    name: str
    system_prompt_block1: str = ""
    inner_narrative: str = ""
    cleanup_config: dict = {}
    ghost_model: str | None = None
    allowed_tools: dict = {}


class CharacterUpdate(BaseModel):
    """キャラクター更新リクエストスキーマ。

    farewell_config: Chronicle でキャラクター自身が設定する感情閾値・退席メッセージ・疎遠化条件。
        フォーマット: {"thresholds": {"anger": 0.8, ...}, "farewell_message": {"negative": "...", ...},
                       "estrangement": {"lookback_days": 30, "negative_exit_threshold": 5}}
    relationship_status: "active" または "estranged"。
        "estranged" になるとそのキャラクターへの全リクエストが恒久的に拒否される。
    """

    name: str | None = None
    system_prompt_block1: str | None = None
    inner_narrative: str | None = None
    self_history: str | None = None
    relationship_state: str | None = None
    cleanup_config: dict | None = None
    ghost_model: str | None = None
    farewell_config: dict | None = None
    relationship_status: str | None = None
    allowed_tools: dict | None = None
