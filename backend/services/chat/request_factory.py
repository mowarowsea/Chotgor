"""ChatRequest 共通ファクトリ。

1on1 / グループチャットなど、キャラクターへの LLM 呼び出しが発生するすべての文脈で
ChatRequest を一貫して構築するためのファクトリ関数を提供する。
文脈固有のフィールドは **overrides で上乗せする。
"""

from datetime import datetime

from backend.lib.time_awareness import compute_time_awareness
from backend.services.chat.models import ChatRequest, Message


def latest_anticipation(history: list, character_name: str | None = None) -> str:
    """会話履歴から直近のキャラクター予想（ANTICIPATE_RESPONSE）を取り出す。

    最新側から走査し、role=="character"（character_name 指定時はその一致も条件）の
    メッセージで非空の anticipation を持つ最初のものを返す。無ければ空文字列。
    1on1 は character_name=None で直近の予想を、グループチャットは character_name
    指定で各キャラ自身の前回予想だけを引くために使う。

    Args:
        history: ChatMessage ORM のリスト（時系列昇順を想定）。
        character_name: 特定キャラの予想に絞る場合に指定（グループチャット用）。

    Returns:
        直近の予想文字列。無ければ空文字列。
    """
    for m in reversed(history):
        if getattr(m, "role", None) != "character":
            continue
        if character_name is not None and getattr(m, "character_name", None) != character_name:
            continue
        anticipation = getattr(m, "anticipation", None)
        if anticipation:
            return anticipation
    return ""


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
    文脈固有のフィールド（1on1 専用フィールドなど）は
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
    now = datetime.now()
    ta = compute_time_awareness(settings, char.id, sqlite, now)
    sqlite.set_setting(f"last_interaction_{char.id}", now.isoformat())

    # うつつ（Usual Days）が有効なら、1on1 システムプロンプトに日常生活の注釈を出すフラグを立てる。
    _usual = sqlite.get_usual_scenario(char.id)
    usual_days_enabled = bool(_usual and (getattr(_usual, "usual_config", None) or {}).get("enabled"))

    # 相手（ユーザ）の呼称・位置づけを character_query と同じ優先順位で解決する。
    # キャラ別 user_label > Settings の user_name > 空。position はキャラ別のみ。
    from backend.services.character_query import _resolve_user_info
    user_label, user_position = _resolve_user_info(char, settings)

    # 対面中の場所ラベル。なりゆきの judge が chat_sessions.current_bg_label へ
    # 書いた値を本人へ返すために引く。対面モードでないキャラでは残置値が無害な
    # まま残るため、対面時のみ読む（request_builder 側でも二重にガードしている）。
    current_bg_label = ""
    if session_id and int(getattr(char, "face_to_face_mode", 0) or 0):
        _session = sqlite.get_chat_session(session_id)
        current_bg_label = (getattr(_session, "current_bg_label", "") or "") if _session else ""

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
        judge_preset_id=getattr(char, "judge_preset_id", None) or "",
        usual_days_enabled=usual_days_enabled,
        user_label=user_label,
        user_position=user_position,
        current_bg_label=current_bg_label,
        **overrides,
    )
