"""API 共通ユーティリティ。

session_to_dict / message_to_dict / char_to_dict を一元管理し、各エンドポイントの重複を排除する。
画像付きメッセージの vision 形式変換は core/chat/content.py で定義し、ここで再エクスポートする。
"""

from datetime import datetime
from typing import Optional

from ..core.chat.content import build_1on1_history, build_message_content
from ..core.memory.format import format_recalled_memories

__all__ = [
    "build_message_content",
    "build_1on1_history",
    "session_to_dict",
    "message_to_dict",
    "char_to_dict",
    "format_memories_for_sse",
    "fmt_dt",
]


def fmt_dt(dt: Optional[datetime]) -> Optional[str]:
    """datetime を ISO 形式文字列に変換する。None の場合は None を返す。

    21箇所に散在していた `dt.isoformat() if dt else None` パターンを集約。
    """
    return dt.isoformat() if dt else None


def char_to_dict(char) -> dict:
    """Character ORM オブジェクトを API レスポンス用 dict に変換する。

    image_data / enabled_providers は含まない（サイズ・センシティビティのため）。
    """
    return {
        "id": char.id,
        "name": char.name,
        "system_prompt_block1": char.system_prompt_block1,
        "meta_instructions": char.meta_instructions,
        "cleanup_config": char.cleanup_config,
        "ghost_model": char.ghost_model,
        "created_at": fmt_dt(char.created_at),
        "updated_at": fmt_dt(char.updated_at),
    }


def session_to_dict(s) -> dict:
    """ChatSession ORMオブジェクトを辞書に変換する。

    group_config が存在する場合のみレスポンスに含める。
    """
    result = {
        "id": s.id,
        "model_id": s.model_id,
        "title": s.title,
        "session_type": getattr(s, "session_type", "1on1") or "1on1",
        "created_at": fmt_dt(s.created_at),
        "updated_at": fmt_dt(s.updated_at),
    }
    group_config = getattr(s, "group_config", None)
    if group_config:
        result["group_config"] = group_config
    return result


def message_to_dict(m) -> dict:
    """ChatMessage ORMオブジェクトを辞書に変換する。

    reasoning / images / character_name は None の場合は省略してレスポンスサイズを削減する。
    """
    result = {
        "id": m.id,
        "session_id": m.session_id,
        "role": m.role,
        "content": m.content,
        "created_at": fmt_dt(m.created_at),
    }
    if getattr(m, "reasoning", None):
        result["reasoning"] = m.reasoning
    if getattr(m, "images", None):
        result["images"] = m.images
    if getattr(m, "character_name", None):
        result["character_name"] = m.character_name
    if getattr(m, "preset_name", None):
        result["preset_name"] = m.preset_name
    return result


def format_memories_for_sse(recalled: list) -> str:
    """想起した記憶リストをSSE送信用のテキストにフォーマットする。"""
    return format_recalled_memories(recalled)
