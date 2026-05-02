"""Chotgor MCP stdio サーバー。

Claude Code CLI がスポーンする MCP サーバーとして機能する。
環境変数からキャラクターコンテキストを受け取り、
inscribe_memory / drift / drift_reset / carve_narrative / power_recall
の 5 ツールを公開する。

このプロセスは **Chotgor backend (uvicorn) の薄いプロキシ** として動作する。
ChromaStore / MemoryManager / DriftManager は backend が保持する単一インスタンスを
利用するため、本プロセスは ChromaDB を一切開かない（構造的欠陥 A 対策）。

登録方法（一度だけ実行）:
    claude mcp add chotgor -s user -- python C:/Users/seamo/Chotgor/backend/mcp_server.py

環境変数:
    CHOTGOR_CHARACTER_ID  : 操作対象キャラクターID（必須）
    CHOTGOR_SESSION_ID    : セッションID（SELF_DRIFT 操作に使用）
    CHOTGOR_BACKEND_URL   : Chotgor backend のベース URL（省略時は http://127.0.0.1:8000）
"""

import io
import json
import os
import sys
from urllib import error as urlerror
from urllib import request as urlrequest

# UTF-8 強制（Windows 環境での文字化け対策）
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

CHARACTER_ID: str = os.environ.get("CHOTGOR_CHARACTER_ID", "")
SESSION_ID: str = os.environ.get("CHOTGOR_SESSION_ID", "")
BACKEND_URL: str = os.environ.get("CHOTGOR_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# HTTP タイムアウト秒。inscribe や power_recall は embedding 計算で数秒かかることがある。
_HTTP_TIMEOUT_SEC = 30.0


def _log(msg: str) -> None:
    """デバッグログを stderr に書き出す（Claude CLI には表示されない）。"""
    sys.stderr.write(f"[MCP] {msg}\n")
    sys.stderr.flush()


def _write(obj: dict) -> None:
    """JSON-RPC レスポンスを stdout に書き出す。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _http_post_json(path: str, body: dict) -> tuple[bool, dict | str]:
    """backend の JSON エンドポイントに POST する。

    Args:
        path: backend からの相対パス（先頭の "/" 含む）。
        body: リクエストボディ。

    Returns:
        (成功かどうか, 成功時はレスポンス dict / 失敗時はエラーメッセージ文字列)。
    """
    url = BACKEND_URL + path
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
            return True, json.loads(raw)
    except urlerror.URLError as e:
        return False, f"backend 接続失敗 ({url}): {e.reason if hasattr(e, 'reason') else e}"
    except Exception as e:
        return False, f"backend 呼び出し失敗 ({url}): {type(e).__name__}: {e}"


def _http_get_json(path: str) -> tuple[bool, dict | str]:
    """backend の JSON エンドポイントに GET する。"""
    url = BACKEND_URL + path
    try:
        with urlrequest.urlopen(url, timeout=_HTTP_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
            return True, json.loads(raw)
    except urlerror.URLError as e:
        return False, f"backend 接続失敗 ({url}): {e.reason if hasattr(e, 'reason') else e}"
    except Exception as e:
        return False, f"backend 呼び出し失敗 ({url}): {type(e).__name__}: {e}"


def _fetch_tool_list() -> list[dict]:
    """backend から MCP ツール定義リストを取得する。失敗時は空リスト。"""
    ok, payload = _http_get_json("/api/mcp/tools")
    if not ok:
        _log(f"tools/list 取得失敗: {payload}")
        return []
    if not isinstance(payload, dict):
        return []
    tools = payload.get("tools") or []
    return [t for t in tools if isinstance(t, dict)]


def main() -> None:
    """stdin から JSON-RPC リクエストを行単位で読み、レスポンスを返すメインループ。"""
    if not CHARACTER_ID:
        _log("WARNING: CHOTGOR_CHARACTER_ID が未設定。ツール呼び出しはエラーになります。")

    _log(f"起動 character_id={CHARACTER_ID!r} session_id={SESSION_ID!r}")
    _log(f"  BACKEND_URL={BACKEND_URL}")

    # ツール定義は backend から取得。backend が落ちていたら空リストで起動するが、
    # tools/call が呼ばれた時点でも改めて backend を叩くため、後から復旧してもよい。
    tools = _fetch_tool_list()
    if tools:
        _log(f"初期化完了 (ツール数={len(tools)})")
    else:
        _log("初期化時 backend に接続できず。tools/list は空で応答する。")

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
                    "serverInfo": {"name": "chotgor", "version": "2.0.0"},
                },
            })

        elif method == "tools/list":
            # 都度 backend に問い合わせる（起動時取得失敗からの復旧を可能にする）
            current_tools = _fetch_tool_list() or tools
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": current_tools},
            })

        elif method == "tools/call":
            params = req.get("params", {})
            tool_name: str = params.get("name", "")
            arguments: dict = params.get("arguments", {})
            _log(f"ツール呼び出し: {tool_name} args={arguments}")

            if not CHARACTER_ID:
                result_text = "[Error: CHOTGOR_CHARACTER_ID が設定されていません]"
                is_error = True
            else:
                ok, payload = _http_post_json(
                    "/api/mcp/tools/call",
                    {
                        "character_id": CHARACTER_ID,
                        "session_id": SESSION_ID or None,
                        "name": tool_name,
                        "arguments": arguments,
                    },
                )
                if not ok:
                    result_text = f"[Error: {payload}]"
                    is_error = True
                    _log(f"エラー: {tool_name} → {payload}")
                else:
                    assert isinstance(payload, dict)
                    result_text = str(payload.get("result", ""))
                    is_error = bool(payload.get("is_error", False))
                    _log(f"完了: {tool_name} is_error={is_error}")

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
