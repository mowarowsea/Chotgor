"""コンテキスト別ツール出し分け — どの文脈でどの追加ツールをキャラへ渡すかの単一判定点。

基本ツール（executor.ANTHROPIC_TOOLS / api/mcp_tools の基本リスト）は全文脈共通だが、
以下のツールは文脈によって露出が変わる（2026-07-11 要件 ①②④）:

    - reach_out         : うつつ（origin=="usual"）専用。相手へ現実のメッセージを送る。
                          日次上限（escrow_delivery_daily_cap 共有）到達日は露出自体を消す。
    - visit_user        : 1on1（origin=="real" かつ session_id あり）専用。対面モードON。
                          すでに対面中なら露出しない。
    - override_schedule : 1on1 専用かつ生活カレンダー有効（living_schedule_enabled=1）のみ。

出し分けの消費者は3系統あり、すべてこのモジュールを参照する（判定の重複を作らない）:
    1. in-process tool-use プロバイダー（anthropic/openai/google）
       — chat_flow/flow.py が resolve_context_tools() の結果を provider.extra_tools へ渡す。
    2. Claude CLI の MCP 経路
       — mcp_server.py が tools/list 時にクエリで文脈を伝え、api/mcp_tools.py が
         resolve_context_tools() で追加分を合成する。
    3. システムプロンプトの操作ガイド
       — flow.py が resolve_context_tool_hints() の結果を request_builder へ渡す。

実行側のガード（うつつ以外から reach_out を叩かれた等）は各ツール実装
（messenger.py / rescheduler.py）が持つ。ここは「見せるかどうか」だけを決める。
"""

from datetime import datetime

from backend.character_actions.messenger import (
    REACH_OUT_SCHEMA,
    REACH_OUT_TOOL_DESCRIPTION,
    REACH_OUT_TOOLS_HINT,
    VISIT_USER_SCHEMA,
    VISIT_USER_TOOL_DESCRIPTION,
    VISIT_USER_TOOLS_HINT,
    delivery_cap_reached,
)
from backend.character_actions.rescheduler import (
    OVERRIDE_SCHEDULE_SCHEMA,
    OVERRIDE_SCHEDULE_TOOL_DESCRIPTION,
    OVERRIDE_SCHEDULE_TOOLS_HINT,
)

# ツール名 → (Anthropic 形式定義, プロンプトヒント) の対応表
_CONTEXT_TOOL_DEFS: dict[str, tuple[dict, str]] = {
    "reach_out": (
        {
            "name": "reach_out",
            "description": REACH_OUT_TOOL_DESCRIPTION,
            "input_schema": REACH_OUT_SCHEMA,
        },
        REACH_OUT_TOOLS_HINT,
    ),
    "visit_user": (
        {
            "name": "visit_user",
            "description": VISIT_USER_TOOL_DESCRIPTION,
            "input_schema": VISIT_USER_SCHEMA,
        },
        VISIT_USER_TOOLS_HINT,
    ),
    "override_schedule": (
        {
            "name": "override_schedule",
            "description": OVERRIDE_SCHEDULE_TOOL_DESCRIPTION,
            "input_schema": OVERRIDE_SCHEDULE_SCHEMA,
        },
        OVERRIDE_SCHEDULE_TOOLS_HINT,
    ),
}

# コンテキストツール名の全集合（テスト・許可設定の照合用）
CONTEXT_TOOL_NAMES: frozenset[str] = frozenset(_CONTEXT_TOOL_DEFS)


def resolve_context_tool_names(
    sqlite,
    character_id: str,
    *,
    origin: str = "real",
    session_id: str | None = None,
    now: datetime | None = None,
) -> list[str]:
    """この文脈でキャラへ露出する追加ツール名のリストを返す（判定の本体）。

    Args:
        sqlite: SQLiteStore（None なら追加ツールなし — 判定材料を読めないため安全側）。
        character_id: 対象キャラクター ID。
        origin: 呼び出し文脈の origin（"real" / "usual" / "interlude"）。
        session_id: 1on1 チャットセッション ID（バッチ・シナリオ経路は None/空）。
        now: 基準時刻（テスト注入用）。

    Returns:
        露出するツール名のリスト（露出順）。該当なしなら空リスト。
    """
    if sqlite is None or not character_id:
        return []
    char = sqlite.get_character(character_id)
    if char is None:
        return []

    names: list[str] = []
    if origin == "usual":
        # うつつ専用: 現実へのメッセージ送信。上限到達日は「見せない」で使用も止まる
        if not delivery_cap_reached(sqlite, now):
            names.append("reach_out")
    elif origin == "real" and session_id:
        # 1on1 専用: 対面切替（すでに対面中なら不要）
        if not int(getattr(char, "face_to_face_mode", 0) or 0):
            names.append("visit_user")
        # 1on1 専用: 予定の一時上書き（生活カレンダー有効キャラのみ）
        if int(getattr(char, "living_schedule_enabled", 0) or 0):
            names.append("override_schedule")
    return names


def resolve_context_tools(
    sqlite,
    character_id: str,
    *,
    origin: str = "real",
    session_id: str | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """この文脈で追加するツール定義（Anthropic 形式）のリストを返す。

    OpenAI 形式が必要な消費者（openai/google プロバイダー）は
    executor.to_openai_tools() で変換する。
    """
    return [
        _CONTEXT_TOOL_DEFS[name][0]
        for name in resolve_context_tool_names(
            sqlite, character_id, origin=origin, session_id=session_id, now=now,
        )
    ]


def resolve_context_tool_hints(
    sqlite,
    character_id: str,
    *,
    origin: str = "real",
    session_id: str | None = None,
    now: datetime | None = None,
) -> list[str]:
    """この文脈でシステムプロンプトへ注入する操作ガイド（ヒント）のリストを返す。"""
    return [
        _CONTEXT_TOOL_DEFS[name][1]
        for name in resolve_context_tool_names(
            sqlite, character_id, origin=origin, session_id=session_id, now=now,
        )
    ]
