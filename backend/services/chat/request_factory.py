"""ChatRequest 共通ファクトリ。

1on1 / グループチャットなど、キャラクターへの LLM 呼び出しが発生するすべての文脈で
ChatRequest を一貫して構築するためのファクトリ関数を提供する。
文脈固有のフィールド（available_presets など）は **overrides で上乗せする。
"""

from datetime import datetime

from backend.lib.time_awareness import compute_time_awareness
from backend.services.chat.models import ChatRequest, Message


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
            "timeout_seconds": p.timeout_seconds or 300,
        })
    return result


def build_character_request(
    char,
    preset,
    messages: list[Message],
    session_id: str,
    settings: dict,
    sqlite,
    **overrides,
) -> ChatRequest:
    """キャラクター + プリセット + メッセージリストから ChatRequest を構築する共通ファクトリ。

    time_awareness の計算・last_interaction の更新・全フィールドのマッピングを担う。
    文脈固有のフィールド（available_presets / 1on1 専用フィールドなど）は
    **overrides で上乗せすることで、呼び出し側が自由に拡張できる。

    Args:
        char: キャラクター ORM オブジェクト。
        preset: プリセット ORM オブジェクト。
        messages: LLM に送信するメッセージリスト。
        session_id: 対象セッション ID（空文字 = セッションなし）。
        settings: グローバル設定辞書（SQLiteStore.get_all_settings() の戻り値）。
        sqlite: SQLiteStore インスタンス（last_interaction 更新に使用）。
        **overrides: ChatRequest の任意フィールドを上書きするキーワード引数。
    """
    model_config = (char.enabled_providers or {}).get(preset.id, {})
    now = datetime.now()
    ta = compute_time_awareness(settings, char.id, sqlite, now)
    sqlite.set_setting(f"last_interaction_{char.id}", now.isoformat())

    return ChatRequest(
        character_id=char.id,
        character_name=char.name,
        provider=preset.provider,
        model=preset.model_id,
        messages=messages,
        character_system_prompt=char.system_prompt_block1,
        self_history=char.self_history,
        relationship_state=char.relationship_state,
        inner_narrative=char.inner_narrative,
        provider_additional_instructions=model_config.get("additional_instructions", ""),
        thinking_level=preset.thinking_level or "default",
        settings=settings,
        enable_time_awareness=ta.enabled,
        current_time_str=ta.current_time_str,
        time_since_last_interaction=ta.time_since_last_interaction,
        session_id=session_id,
        current_preset_name=preset.name,
        current_preset_id=preset.id,
        allowed_tools=getattr(char, "allowed_tools", None) or {},
        timeout_seconds=preset.timeout_seconds,
        farewell_config=getattr(char, "farewell_config", None),
        farewell_relationship_status=getattr(char, "relationship_status", "active"),
        self_reflection_mode=getattr(char, "self_reflection_mode", "disabled"),
        self_reflection_preset_id=getattr(char, "self_reflection_preset_id", None) or "",
        self_reflection_n_turns=getattr(char, "self_reflection_n_turns", 5),
        **overrides,
    )
