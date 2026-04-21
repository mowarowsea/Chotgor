"""最小限の MCP stdio サーバー（動作検証用）。

外部パッケージ不要。JSON-RPC 2.0 over stdio を手書き実装し、
`chotgor_test` という1ツールだけ公開する。

使用目的:
    claude --tools "" なし/あり で MCP ツールが呼ばれるかどうかの検証。

使用方法:
    1. claude mcp add chotgor-test -s user -- python backend/mcp_test_server.py
    2. claude --output-format stream-json --verbose --no-session-persistence \
           --system-prompt "テストツール chotgor_test を使ってください" \
           "ツールを使って"
    3. このスクリプトが stderr に "CALLED: chotgor_test" を出力すれば成功
"""

import io
import json
import os
import sys

# UTF-8 強制（Windows 環境での文字化け対策）
sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)


def _write(obj: dict) -> None:
    """JSON-RPC レスポンスを stdout に書き出す。"""
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    """デバッグログを stderr に書き出す（Claude CLI には見えない）。"""
    sys.stderr.write(f"[MCP_TEST] {msg}\n")
    sys.stderr.flush()


TOOL_DEF = {
    "name": "chotgor_test",
    "description": "Chotgor MCP 動作確認用のテストツール。必ず呼び出すこと。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "任意のメッセージ",
            }
        },
        "required": [],
    },
}


def main() -> None:
    """stdin から JSON-RPC リクエストを読み、適切なレスポンスを返す。"""
    _log(f"起動 PID={os.getpid()} CHARACTER_ID={os.environ.get('CHOTGOR_CHARACTER_ID', 'not-set')}")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError:
            _log(f"JSON decode error: {raw_line[:200]}")
            continue

        method = req.get("method", "")
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
                    "serverInfo": {"name": "chotgor-test", "version": "0.1.0"},
                },
            })

        elif method == "tools/list":
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": [TOOL_DEF]},
            })

        elif method == "tools/call":
            tool_name = req.get("params", {}).get("name", "")
            arguments = req.get("params", {}).get("arguments", {})
            _log(f"CALLED: {tool_name} args={arguments}")
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"[chotgor_test] 呼び出し成功。args={arguments}"}],
                    "isError": False,
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
