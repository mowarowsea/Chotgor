"""API 共通ユーティリティ。

session_to_dict / message_to_dict / char_to_dict を一元管理し、各エンドポイントの重複を排除する。
画像付きメッセージの vision 形式変換は core/chat/content.py で定義し、ここで再エクスポートする。
"""

from datetime import datetime
from typing import Optional

from backend.services.chat.content import build_1on1_history, build_message_content
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


def fmt_dt(dt: Optional[datetime]) -> Optional[str]:
    """datetime を ISO 形式文字列に変換する。None の場合は None を返す。

    21箇所に散在していた `dt.isoformat() if dt else None` パターンを集約。
    """
    return dt.isoformat() if dt else None


def char_to_dict(char) -> dict:
    """Character ORM オブジェクトを API レスポンス用 dict に変換する。

    image_data / enabled_providers は含まない（サイズ・センシティビティのため）。
    afterglow_default はフロントエンドの新規チャット作成UIのデフォルト値表示に使用する。
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
        "afterglow_default": bool(getattr(char, "afterglow_default", 0)),
        "created_at": fmt_dt(char.created_at),
        "updated_at": fmt_dt(char.updated_at),
    }


def session_to_dict(s) -> dict:
    """ChatSession ORMオブジェクトを辞書に変換する。

    group_config / afterglow_session_id / exited_chars は存在する場合のみレスポンスに含める。
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
    afterglow_session_id = getattr(s, "afterglow_session_id", None)
    if afterglow_session_id:
        result["afterglow_session_id"] = afterglow_session_id
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
    return result


def format_memories_for_sse(recalled: list) -> str:
    """想起した記憶リストをSSE送信用のテキストにフォーマットする。"""
    return format_recalled_memories(recalled)


def build_available_presets(character, current_preset, sqlite) -> list[dict]:
    """switch_angle 用の切り替え候補プリセット一覧を構築する。

    switch_angle_enabled が OFF または有効プロバイダーが1件以下の場合は空リストを返す。
    current_preset 自身はリストから除外する。

    Args:
        character: Character ORM オブジェクト。
        current_preset: 現在使用中の LLMModelPreset ORM オブジェクト。
        sqlite: SQLiteStore インスタンス。

    Returns:
        各プリセットの設定を格納した dict のリスト。
    """
    enabled_providers = character.enabled_providers or {}
    if not getattr(character, "switch_angle_enabled", 0) or len(enabled_providers) <= 1:
        return []
    result = []
    for p in sqlite.list_model_presets():
        if p.id == current_preset.id:
            continue
        cfg = enabled_providers.get(p.id)
        if cfg is None:
            continue
        result.append({
            "preset_id": p.id,
            "preset_name": p.name,
            "provider": p.provider,
            "model_id": p.model_id,
            "additional_instructions": cfg.get("additional_instructions", ""),
            "thinking_level": p.thinking_level or "default",
            "when_to_switch": cfg.get("when_to_switch", ""),
        })
    return result
