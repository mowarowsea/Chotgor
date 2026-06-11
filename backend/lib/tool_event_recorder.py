"""ツール実行イベントレコーダー — キャラクターのツール使用を実行時に記録する。

Logs 画面のツール使用表示は、かつて debug/ の生ログ（Gemini JSON / Anthropic JSON /
OpenAI JSON / Claude CLI stream-json / テキストタグの5形式）を表示時に逆解析していた。
本モジュールは usage_recorder と同じ思想で「実行時に確定した事実」をその場で
tool_call_events テーブルへ記録し、表示時の解析を不要にする。

記録箇所:
    - ToolExecutor.execute()（tool-use 方式・MCP プロキシ・バッチの共通関門）
    - タグ方式の inscriber / carver / switcher / ChatService の power_recall 実行箇所
    - ANTICIPATE_RESPONSE の採用箇所（chat / scenario / pc_runner）

request_id / dir_id / feature / target は log_context の ContextVar から自動補完するため、
呼び出し側はツール名と引数だけ渡せばよい。Claude CLI の MCP プロキシ経路は
プロセス越境で ContextVar が届かないため、env→HTTP リレーで運ばれた値を
api/mcp_tools.py が ContextVar に復元してから ToolExecutor を呼ぶ（リレーの詳細は
mcp_server.py のモジュール docstring を参照）。

記録失敗はチャット本流を絶対に妨げない（握り潰して WARNING ログのみ）。
"""

import json
import logging
import re

from backend.lib.log_context import (
    current_log_dir_id,
    current_log_feature,
    current_log_target,
    current_message_id,
)

_log = logging.getLogger(__name__)

# SQLiteStore インスタンス（main.py 起動時に set_store() で注入される）
_store = None

# ContextVar 未設定時のデフォルト値（log_context.py のデフォルトと一致させる）
_UNSET_ID = "--------"

# ツール結果文字列のエラー規約: "[inscribe_memory error: ...]" / "[Error: ...]" /
# "[Unknown tool: ...]" のように、角括弧開始でエラーを示す既存の慣習に合わせた判定。
_ERROR_RESULT_RE = re.compile(r"^\s*\[(?:[^\]]*\berror\b|Unknown tool)", re.IGNORECASE)


def set_store(sqlite) -> None:
    """SQLiteStore を注入する。main.py の起動時に呼ぶ。"""
    global _store
    _store = sqlite


def result_looks_like_error(result_text: str) -> bool:
    """ツール実行結果の文字列がエラー規約に一致するか判定する。

    ToolExecutor の各ツールは例外を握り潰して "[<tool> error: ...]" 形式の
    文字列を返す設計のため、status の判定はこの文字列規約に依存する。

    Args:
        result_text: ToolExecutor.execute() が返した結果テキスト。

    Returns:
        エラー規約に一致すれば True。
    """
    return bool(_ERROR_RESULT_RE.match(result_text or ""))


def _ctx_or_none(value: str | None) -> str | None:
    """ContextVar の値を返す。未設定デフォルト（"--------" / 空文字）は None に正規化する。"""
    if not value or value == _UNSET_ID:
        return None
    return value


def record_tool_event(
    tool_name: str,
    arguments: dict | None = None,
    status: str = "ok",
    error_message: str | None = None,
    source: str = "tool_use",
) -> None:
    """ツール実行イベントを1行記録する。

    Args:
        tool_name: ツール名（"inscribe_memory" / "anticipate_response" 等）。
            MCP プレフィックス（"mcp__<server>__"）は付けない素の名前を渡す。
        arguments: ツール引数 dict。表示変換（tool_tags）が解釈できる
            tool-use 形式のキー名（content / category / impact 等）で渡す。
        status: 実行結果（"ok" / "error"）。
        error_message: status="error" 時の詳細（結果文字列 or 例外メッセージ）。
        source: 記録経路（"tool_use"=ToolExecutor / "tag"=テキストタグ方式 /
            "anticipation"=予想タグ採用）。
    """
    if _store is None:
        return
    try:
        arguments_json = (
            json.dumps(arguments, ensure_ascii=False, default=str)
            if arguments is not None
            else None
        )
        _store.add_tool_call_event(
            tool_name=tool_name,
            arguments_json=arguments_json,
            status=status,
            error_message=error_message,
            source=source,
            request_id=_ctx_or_none(current_message_id.get()),
            dir_id=_ctx_or_none(current_log_dir_id.get()),
            target=_ctx_or_none(current_log_target.get()),
            feature=_ctx_or_none(current_log_feature.get()),
        )
    except Exception as e:
        _log.warning("ツール実行イベントの記録に失敗: tool=%s error=%s", tool_name, e)
