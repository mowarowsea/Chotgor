"""Chotgor MCP stdio サーバー。

Claude Code CLI がスポーンする MCP サーバーとして機能する。
環境変数からキャラクターコンテキストを受け取り、
inscribe_memory / drift / drift_reset / carve_narrative / power_recall
の 5 ツールを公開する。

登録方法（一度だけ実行）:
    claude mcp add chotgor -s user -- python C:/Users/seamo/Chotgor/backend/mcp_server.py

環境変数:
    CHOTGOR_CHARACTER_ID : 操作対象キャラクターID（必須）
    CHOTGOR_SESSION_ID   : セッションID（SELF_DRIFT 操作に使用）
    SQLITE_DB_PATH       : SQLite ファイルパス（省略時はプロジェクト内 data/chotgor.db）
    CHROMA_DB_PATH       : ChromaDB ディレクトリパス（省略時はプロジェクト内 data/chroma）
"""

import io
import json
import os
import sys

# UTF-8 強制（Windows 環境での文字化け対策）
sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

# プロジェクトルートを sys.path に追加（backend.* モジュールを import するため）
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

CHARACTER_ID: str = os.environ.get("CHOTGOR_CHARACTER_ID", "")
SESSION_ID: str = os.environ.get("CHOTGOR_SESSION_ID", "")
SQLITE_DB_PATH: str = os.environ.get(
    "SQLITE_DB_PATH", os.path.join(_PROJECT_ROOT, "data", "chotgor.db")
)
CHROMA_DB_PATH: str = os.environ.get(
    "CHROMA_DB_PATH", os.path.join(_PROJECT_ROOT, "data", "chroma")
)


def _log(msg: str) -> None:
    """デバッグログを stderr に書き出す（Claude CLI には表示されない）。"""
    sys.stderr.write(f"[MCP] {msg}\n")
    sys.stderr.flush()


def _write(obj: dict) -> None:
    """JSON-RPC レスポンスを stdout に書き出す。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _init_executor():
    """DB・MemoryManager・ToolExecutor を初期化して返す。起動時に一度だけ呼ぶ。"""
    from backend.repositories.sqlite.store import SQLiteStore
    from backend.repositories.chroma.store import ChromaStore
    from backend.services.memory.manager import MemoryManager
    from backend.services.memory.drift_manager import DriftManager
    from backend.character_actions.executor import ToolExecutor

    sqlite = SQLiteStore(SQLITE_DB_PATH)
    all_settings = sqlite.get_all_settings()
    chroma = ChromaStore(
        CHROMA_DB_PATH,
        embedding_provider=all_settings.get("embedding_provider", "default"),
        embedding_model=all_settings.get("embedding_model", ""),
        api_key=all_settings.get("google_api_key", ""),
        base_url=all_settings.get("infinity_base_url", "http://localhost:7997"),
    )
    memory_manager = MemoryManager(sqlite=sqlite, chroma=chroma)
    drift_manager = DriftManager(sqlite=sqlite)

    return ToolExecutor(
        character_id=CHARACTER_ID,
        session_id=SESSION_ID or None,
        memory_manager=memory_manager,
        drift_manager=drift_manager,
    )


def _build_tool_list() -> list[dict]:
    """公開するMCPツール定義リストを構築する。"""
    from backend.character_actions.inscriber import INSCRIBE_MEMORY_SCHEMA, INSCRIBE_MEMORY_TOOL_DESCRIPTION
    from backend.character_actions.drifter import DRIFT_SCHEMA, DRIFT_RESET_SCHEMA, DRIFT_TOOL_DESCRIPTION, DRIFT_RESET_TOOL_DESCRIPTION
    from backend.character_actions.carver import CARVE_NARRATIVE_SCHEMA, CARVE_NARRATIVE_TOOL_DESCRIPTION
    from backend.character_actions.recaller import POWER_RECALL_SCHEMA, POWER_RECALL_TOOL_DESCRIPTION

    return [
        {"name": "inscribe_memory",  "description": INSCRIBE_MEMORY_TOOL_DESCRIPTION,  "inputSchema": INSCRIBE_MEMORY_SCHEMA},
        {"name": "drift",            "description": DRIFT_TOOL_DESCRIPTION,            "inputSchema": DRIFT_SCHEMA},
        {"name": "drift_reset",      "description": DRIFT_RESET_TOOL_DESCRIPTION,      "inputSchema": DRIFT_RESET_SCHEMA},
        {"name": "carve_narrative",  "description": CARVE_NARRATIVE_TOOL_DESCRIPTION,  "inputSchema": CARVE_NARRATIVE_SCHEMA},
        {"name": "power_recall",     "description": POWER_RECALL_TOOL_DESCRIPTION,     "inputSchema": POWER_RECALL_SCHEMA},
    ]


def main() -> None:
    """stdin から JSON-RPC リクエストを行単位で読み、レスポンスを返すメインループ。"""
    if not CHARACTER_ID:
        _log("WARNING: CHOTGOR_CHARACTER_ID が未設定。ツール呼び出しはエラーになります。")

    _log(f"起動 character_id={CHARACTER_ID!r} session_id={SESSION_ID!r}")
    _log(f"  SQLITE={SQLITE_DB_PATH}")
    _log(f"  CHROMA={CHROMA_DB_PATH}")

    executor = None
    tools: list[dict] = []

    try:
        executor = _init_executor()
        tools = _build_tool_list()
        _log("初期化完了")
    except Exception as exc:
        _log(f"初期化失敗: {exc}")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError:
            _log(f"JSON decode error: {raw_line[:200]}")
            continue

        method: str = req.get("method", "")
        req_id = req.get("id")
        _log(f"受信 method={method} id={req_id}")

        # 通知（id なし）はレスポンス不要
        if req_id is None:
            continue

        if method == "initialize":
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "chotgor", "version": "1.0.0"},
                },
            })

        elif method == "tools/list":
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tools},
            })

        elif method == "tools/call":
            params = req.get("params", {})
            tool_name: str = params.get("name", "")
            arguments: dict = params.get("arguments", {})
            _log(f"ツール呼び出し: {tool_name} args={arguments}")

            if executor is None:
                result_text = "[Error: MCP サーバーの初期化に失敗しています]"
                is_error = True
            elif not CHARACTER_ID:
                result_text = "[Error: CHOTGOR_CHARACTER_ID が設定されていません]"
                is_error = True
            else:
                try:
                    result_text = executor.execute(tool_name, arguments)
                    is_error = False
                    _log(f"完了: {tool_name}")
                except Exception as exc:
                    result_text = f"[Error: {type(exc).__name__}: {exc}]"
                    is_error = True
                    _log(f"エラー: {tool_name} → {exc}")

            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": is_error,
                },
            })

        else:
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


if __name__ == "__main__":
    main()
