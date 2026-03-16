"""API 共通ユーティリティ。

session_to_dict / message_to_dict を一元管理し、chat.py / group_chat.py の重複を排除する。
画像付きメッセージの vision 形式変換は core/chat/content.py で定義し、ここで再エクスポートする。
"""

from ..core.chat.content import build_1on1_history, build_message_content
from ..core.memory.format import format_recalled_memories

__all__ = [
    "build_message_content",
    "build_1on1_history",
    "session_to_dict",
    "message_to_dict",
    "format_memories_for_sse",
]


def session_to_dict(s) -> dict:
    """ChatSession ORMオブジェクトを辞書に変換する。

    group_config が存在する場合のみレスポンスに含める。
    """
    result = {
        "id": s.id,
        "model_id": s.model_id,
        "title": s.title,
        "session_type": getattr(s, "session_type", "1on1") or "1on1",
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
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
        "created_at": m.created_at.isoformat() if m.created_at else None,
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
