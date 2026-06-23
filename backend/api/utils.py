"""API 共通ユーティリティ。

session_to_dict / message_to_dict / char_to_dict を一元管理し、各エンドポイントの重複を排除する。
画像付きメッセージの vision 形式変換は core/chat/content.py で定義し、ここで再エクスポートする。
"""

from datetime import datetime
from backend.services.chat.content import build_1on1_history, build_message_content
from backend.services.chat.request_factory import build_available_presets
from backend.services.memory.format import format_recalled_memories

__all__ = [
    "build_message_content",
    "build_1on1_history",
    "session_to_dict",
    "message_to_dict",
    "char_to_dict",
    "format_memories_for_sse",
    "fmt_dt",
    "build_available_presets",
]


def fmt_dt(dt: datetime | None) -> str | None:
    """datetime を ISO 形式文字列に変換する。None の場合は None を返す。

    21箇所に散在していた `dt.isoformat() if dt else None` パターンを集約。
    """
    return dt.isoformat() if dt else None


def char_to_dict(char) -> dict:
    """Character ORM オブジェクトを API レスポンス用 dict に変換する。

    image_data / face_to_face_bg_image / enabled_providers は含まない
    （サイズ・センシティビティのため。背景画像は別エンドポイント経由でバイナリ取得）。
    """
    return {
        "id": char.id,
        "name": char.name,
        "system_prompt_block1": char.system_prompt_block1,
        "inner_narrative": char.inner_narrative,
        "self_history": char.self_history,
        "relationship_state": char.relationship_state,
        "cleanup_config": char.cleanup_config,
        "ghost_model": char.ghost_model,
        "allowed_tools": getattr(char, "allowed_tools", None) or {},
        "face_to_face_mode": int(getattr(char, "face_to_face_mode", 0) or 0),
        "has_face_to_face_bg_image": bool(getattr(char, "face_to_face_bg_image", None)),
        "created_at": fmt_dt(char.created_at),
        "updated_at": fmt_dt(char.updated_at),
    }


def session_to_dict(s) -> dict:
    """ChatSession ORMオブジェクトを辞書に変換する。

    group_config / exited_chars は存在する場合のみレスポンスに含める。
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
    exited_chars = getattr(s, "exited_chars", None)
    if exited_chars:
        result["exited_chars"] = exited_chars
    return result


def message_to_dict(m) -> dict:
    """ChatMessage ORMオブジェクトを辞書に変換する。

    reasoning / images / character_name / is_system_message は None の場合は省略してレスポンスサイズを削減する。
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
    if getattr(m, "is_system_message", None):
        result["is_system_message"] = True
    if getattr(m, "log_message_id", None):
        result["log_message_id"] = m.log_message_id
    if getattr(m, "anticipation", None):
        result["anticipation"] = m.anticipation
    if getattr(m, "face_to_face", 0):
        result["face_to_face"] = True
    return result


def format_memories_for_sse(recalled: list) -> str:
    """想起した記憶リストをSSE送信用のテキストにフォーマットする。"""
    return format_recalled_memories(recalled)


