"""MCP 接続レースガードのテスト。

Claude CLI 起動直後に chotgor MCP サーバーの接続が間に合わないまま
リクエストが走ると、そのターンはツールが一切提供されず、キャラクターが
「ツール使用の演技」（擬似構文のテキスト出力）に流れる事故が起きる
（debug/cfd5bf43、実測で約2割が pending 起動）。

このガードは stream-json 先頭の init イベントで `mcp_servers[].status` を
検査し、未接続なら CLI プロセスを kill して作り直す（最大3回起動、
上限到達時は未接続のまま続行）。ここでは以下を検証する:

- `_mcp_pending_in_init`: init イベントの判定ロジック単体
- `_spawn_cli_mcp_guarded`: 再試行・上限到達・即成功・非JSON行の各分岐と、
  破棄した試行の init 行が先読みリストに残ること（debug ログの痕跡用）
- `_cli_output_lines`: 先読み行→残り stdout の順序と空行スキップ
"""

import json
import subprocess
from unittest.mock import patch

import pytest

from backend.providers.claude_cli_provider import (
    _MCP_GUARD_MAX_ATTEMPTS,
    _cli_output_lines,
    _mcp_pending_in_init,
    _spawn_cli_mcp_guarded,
)


def _init_line(status: str) -> str:
    """指定 status の MCP サーバーを1つ持つ init イベント行を作る。"""
    return json.dumps({
        "type": "system",
        "subtype": "init",
        "mcp_servers": [{"name": "chotgor", "status": status}],
    })


class _FakePipe:
    """readline / read / write / close だけを持つ最小のパイプ代替。"""

    def __init__(self, lines: list[bytes] | None = None):
        self._lines = list(lines or [])
        self.closed = False
        self.written = b""

    def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    def read(self) -> bytes:
        rest = b"".join(self._lines)
        self._lines = []
        return rest

    def write(self, data: bytes):
        self.written += data

    def close(self):
        self.closed = True


class _FakeProc:
    """subprocess.Popen の代替。stdout に与えた行を返し、kill/wait を記録する。"""

    def __init__(self, stdout_lines: list[bytes]):
        self.stdin = _FakePipe()
        self.stdout = _FakePipe(stdout_lines)
        self.stderr = _FakePipe()
        self.killed = False
        self.returncode = 0
        self.args = []

    def kill(self):
        self.killed = True

    def wait(self):
        return self.returncode


def _patch_popen(procs: list[_FakeProc]):
    """subprocess.Popen を順番に procs を返すモックへ差し替える patcher を返す。"""
    it = iter(procs)
    return patch(
        "backend.providers.claude_cli_provider.subprocess.Popen",
        side_effect=lambda *a, **kw: next(it),
    )


# ---------------------------------------------------------------------------
# _mcp_pending_in_init
# ---------------------------------------------------------------------------

class TestMcpPendingInInit:
    """init イベント判定の単体テスト。

    「pending を含む init イベントのときだけ True」であること、および
    init 以外のイベント・MCP サーバー未設定の場合に誤検知しないことを
    検証する。誤検知すると全リクエストが無限に再起動されかねないため、
    False 側の網羅が重要。
    """

    def test_pending_returns_true(self):
        """status=pending の init は True（再起動対象）。"""
        assert _mcp_pending_in_init(json.loads(_init_line("pending"))) is True

    def test_connected_returns_false(self):
        """status=connected の init は False（そのまま続行）。"""
        assert _mcp_pending_in_init(json.loads(_init_line("connected"))) is False

    def test_failed_status_returns_true(self):
        """connected 以外（failed 等）は未接続扱いで True。"""
        assert _mcp_pending_in_init(json.loads(_init_line("failed"))) is True

    def test_no_mcp_servers_returns_false(self):
        """MCP サーバー未設定（空リスト）の init は False。"""
        event = {"type": "system", "subtype": "init", "mcp_servers": []}
        assert _mcp_pending_in_init(event) is False

    def test_missing_mcp_servers_key_returns_false(self):
        """mcp_servers キー自体が無い init も False。"""
        assert _mcp_pending_in_init({"type": "system", "subtype": "init"}) is False

    def test_non_init_event_returns_false(self):
        """init 以外のイベント（assistant 等）は常に False。"""
        assert _mcp_pending_in_init({"type": "assistant", "message": {}}) is False
        assert _mcp_pending_in_init({"type": "system", "subtype": "status"}) is False


# ---------------------------------------------------------------------------
# _spawn_cli_mcp_guarded
# ---------------------------------------------------------------------------

class TestSpawnCliMcpGuarded:
    """起動ガードの再試行分岐テスト。

    Popen をモックに差し替え、init 行の内容によって
    「即成功」「1回再試行後成功」「上限到達で未接続のまま続行」
    「非JSON先頭行はそのまま続行」の4分岐を検証する。
    併せて、kill された試行の init 行も先読みリストへ残ること
    （debug ログで再試行の痕跡を追える）と、stdin へ入力が
    書き込まれることを確認する。
    """

    def test_connected_first_try_no_retry(self):
        """初回から connected なら再起動せずそのプロセスを返す。"""
        proc = _FakeProc([(_init_line("connected") + "\n").encode()])
        with _patch_popen([proc]):
            got, pre = _spawn_cli_mcp_guarded(["claude"], b"hi", {})
        assert got is proc
        assert not proc.killed
        assert len(pre) == 1
        assert proc.stdin.written == b"hi"
        assert proc.stdin.closed

    def test_pending_then_connected_retries_once(self):
        """pending → kill → 再起動 → connected の流れ。

        破棄した1回目の init 行も先読みリストに含まれる。
        """
        p1 = _FakeProc([(_init_line("pending") + "\n").encode()])
        p2 = _FakeProc([(_init_line("connected") + "\n").encode()])
        with _patch_popen([p1, p2]):
            got, pre = _spawn_cli_mcp_guarded(["claude"], b"hi", {})
        assert got is p2
        assert p1.killed
        assert not p2.killed
        assert len(pre) == 2  # 破棄した init + 採用した init
        # 両プロセスに stdin が書き込まれている（再試行でも入力を渡し直す）
        assert p1.stdin.written == b"hi"
        assert p2.stdin.written == b"hi"

    def test_all_pending_gives_up_and_proceeds(self):
        """全試行 pending なら上限到達後、最後のプロセスで未接続のまま続行する。

        ターン全損よりツール無しで進む方がましという設計判断の検証。
        """
        procs = [
            _FakeProc([(_init_line("pending") + "\n").encode()])
            for _ in range(_MCP_GUARD_MAX_ATTEMPTS)
        ]
        with _patch_popen(procs):
            got, pre = _spawn_cli_mcp_guarded(["claude"], b"hi", {})
        assert got is procs[-1]
        assert all(p.killed for p in procs[:-1])
        assert not procs[-1].killed
        assert len(pre) == _MCP_GUARD_MAX_ATTEMPTS

    def test_non_json_first_line_proceeds(self):
        """先頭行が JSON でなくても再起動せず続行する（判定不能は素通し）。"""
        proc = _FakeProc([b"not-json\n"])
        with _patch_popen([proc]):
            got, pre = _spawn_cli_mcp_guarded(["claude"], b"hi", {})
        assert got is proc
        assert not proc.killed
        assert pre == ["not-json"]

    def test_empty_stdout_proceeds(self):
        """プロセスが即死して stdout が空でも例外を出さず続行する。

        呼び出し側の returncode 判定でエラー処理される前提。
        """
        proc = _FakeProc([])
        with _patch_popen([proc]):
            got, pre = _spawn_cli_mcp_guarded(["claude"], b"hi", {})
        assert got is proc
        assert pre == []


# ---------------------------------------------------------------------------
# _cli_output_lines
# ---------------------------------------------------------------------------

class TestCliOutputLines:
    """先読み行と残り stdout の結合イテレータのテスト。

    ガードが先読みした行が欠落せず、続く stdout の行と正しい順序で
    連結されること、空行がスキップされることを検証する。
    """

    def test_pre_lines_then_stdout(self):
        """先読み行 → stdout 行の順で yield される。"""
        proc = _FakeProc([b'{"a":1}\n', b'{"b":2}\n'])
        lines = list(_cli_output_lines(proc, ["pre1", "pre2"]))
        assert lines == ["pre1", "pre2", '{"a":1}', '{"b":2}']

    def test_empty_lines_skipped(self):
        """stdout 中の空行は yield されない。"""
        proc = _FakeProc([b"\n", b'{"a":1}\n', b"  \n"])
        lines = list(_cli_output_lines(proc, []))
        assert lines == ['{"a":1}']
