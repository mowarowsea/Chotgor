"""``backend.mcp_server._load_batch_context`` の単体テスト。

mcp_server.py は Claude CLI に subprocess としてスポーンされる薄いプロキシで、
forget 蒸留などのバッチ処理が指定する ``batch_context``（例: force_insert_memory）を
``CHOTGOR_BATCH_CONTEXT`` 環境変数で受け取り、/api/mcp/tools/call の HTTP ペイロードへ転載する。

本テストはその「環境変数 → dict」変換の最小契約を守る：

  - 未設定／空文字なら None
  - 正しい JSON dict なら同じ dict
  - 不正な JSON / dict 以外の JSON はフェイルセーフで None（バッチ専用フラグが
    通常チャットへ漏れて副作用を起こさないため）

import 時の副作用（stdin/stdout の TextIOWrapper 化・モジュール定数 BATCH_CONTEXT 計算）は
すでに行われている前提で、関数だけを呼び出す純粋関数テスト。
"""

import importlib

import pytest

import backend.mcp_server as mcp_server


class TestLoadBatchContext:
    """``_load_batch_context()`` の正常系／異常系をひととおりカバー。"""

    def test_returns_none_when_env_unset(self, monkeypatch):
        """環境変数未設定なら None。通常 1on1 チャット時に余計なフラグが立たないことを保証。"""
        monkeypatch.delenv("CHOTGOR_BATCH_CONTEXT", raising=False)
        assert mcp_server._load_batch_context() is None

    def test_returns_none_when_env_empty_string(self, monkeypatch):
        """空文字／空白のみでも None。""と "  " を None と同義に扱う。"""
        monkeypatch.setenv("CHOTGOR_BATCH_CONTEXT", "   ")
        assert mcp_server._load_batch_context() is None

    def test_returns_dict_for_valid_json_object(self, monkeypatch):
        """forget 蒸留が渡す ``{"force_insert_memory": True}`` がそのまま復元されること。"""
        monkeypatch.setenv("CHOTGOR_BATCH_CONTEXT", '{"force_insert_memory": true}')
        assert mcp_server._load_batch_context() == {"force_insert_memory": True}

    def test_returns_none_for_invalid_json(self, monkeypatch, capsys):
        """JSON parse 失敗時は None を返し、stderr に警告を出す（フェイルセーフ）。

        Claude CLI 起動後に不正な env を渡されても、通常チャット相当に縮退して
        キャラクター側に副作用が漏れないことを保証する。
        """
        monkeypatch.setenv("CHOTGOR_BATCH_CONTEXT", "{not json")
        assert mcp_server._load_batch_context() is None
        captured = capsys.readouterr()
        assert "CHOTGOR_BATCH_CONTEXT" in captured.err

    def test_returns_none_for_non_dict_json(self, monkeypatch, capsys):
        """配列やプリミティブが渡されても None にフェイルセーフされること。

        ``ToolExecutor.batch_context`` は dict 前提のため、list / int / str などを
        そのまま渡すと AttributeError を起こす。型ガードはここで行う。
        """
        monkeypatch.setenv("CHOTGOR_BATCH_CONTEXT", "[1, 2, 3]")
        assert mcp_server._load_batch_context() is None
        captured = capsys.readouterr()
        assert "CHOTGOR_BATCH_CONTEXT" in captured.err
