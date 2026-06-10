"""backend.api.logs_ui の表示系（HTTPエンドポイント・HTML整形・DB行ビュー）のユニットテスト。

対象:
    log_raw_file()              — 生ログファイルをプレーンテキストで返す HTTP エンドポイント
    _render_json_html()         — JSON ログのハイライト付き HTML レンダリング
    _build_attempt_detail()     — DB 1行からの試行詳細構築（skip_files オプション）
    _build_entry_from_db_rows() — 同一 request_id の DB 行からのエントリ構築（skip_files オプション）

テスト方針:
    - HTTP エンドポイントは TestClient 経由でテストする
    - DEBUG_BASE は logs_ui.config のモジュール変数を monkeypatch で差し替える
    - LLM へのアクセスは発生しないため外部 mock は不要
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.logs_ui import config as logs_ui_config
from backend.api.logs_ui.entries import (
    _build_attempt_detail,
    _build_entry_from_db_rows,
)
from backend.api.logs_ui.json_view import _render_json_html
from backend.api.logs_ui.routes import router
from tests._logs_ui_helpers import _make_debug_dir

# ─── log_raw_file HTTP エンドポイント ─────────────────────────────────────────


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """logs_ui ルーターを組み込んだ最小 FastAPI アプリのテストクライアントを返すフィクスチャ。

    DEBUG_BASE を tmp_path 配下に差し替え、外部依存を排除する。
    """
    monkeypatch.setattr(logs_ui_config, "DEBUG_BASE", tmp_path / "debug")
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), tmp_path


class TestLogRawFile:
    """log_raw_file エンドポイントのセキュリティと正常系を検証するテストクラス。"""

    def test_valid_file_returns_content(self, test_client):
        """正しい message_id とファイル名で 200 とファイル内容が返ること。"""
        client, tmp_path = test_client
        debug = tmp_path / "debug"
        folder = _make_debug_dir(debug, "abc12345")
        (folder / "01_FrontInput.log").write_text("テスト内容", encoding="utf-8")

        resp = client.get("/ui/logs/abc12345/raw/01_FrontInput.log")
        assert resp.status_code == 200
        assert "テスト内容" in resp.text

    def test_missing_file_returns_404(self, test_client):
        """存在しないファイルへのリクエストは 404 を返すこと。"""
        client, tmp_path = test_client
        debug = tmp_path / "debug"
        _make_debug_dir(debug, "abc12345")

        resp = client.get("/ui/logs/abc12345/raw/nonexistent.log")
        assert resp.status_code == 404

    def test_filename_with_dotdot_returns_400(self, test_client):
        """ファイル名に '..' が含まれる場合は 400 を返すこと（ディレクトリトラバーサル防止）。"""
        client, _ = test_client
        resp = client.get("/ui/logs/abc12345/raw/../../etc/passwd")
        assert resp.status_code in (400, 404)  # FastAPI がパスを正規化する場合もある

    def test_filename_with_backslash_returns_400(self, test_client):
        """ファイル名に '\\' が含まれる場合は 400 を返すこと。"""
        client, _ = test_client
        resp = client.get("/ui/logs/abc12345/raw/foo\\bar.log")
        assert resp.status_code == 400

    def test_message_id_with_dotdot_returns_400(self, test_client):
        """message_id に '..' が含まれる場合は 400 を返すこと。"""
        client, _ = test_client
        resp = client.get("/ui/logs/../secret/raw/file.log")
        assert resp.status_code in (400, 404)

    def test_content_type_is_plain_text(self, test_client):
        """レスポンスの Content-Type が text/plain であること。"""
        client, tmp_path = test_client
        debug = tmp_path / "debug"
        folder = _make_debug_dir(debug, "abc12345")
        (folder / "test.log").write_text("内容", encoding="utf-8")

        resp = client.get("/ui/logs/abc12345/raw/test.log")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]


# ─── _render_json_html ────────────────────────────────────────────────────────




class TestRenderJsonHtml:
    """_render_json_html() のハイライトとエスケープ展開を検証するテストクラス。

    text/content/thought/system_prompt/system_instruction/conversation/thinking/result
    キーの文字列値がハイライトされること、JSON エスケープ（\\n 等）が展開されること、
    NDJSON 形式でも整形されることを確認する。
    """

    def test_highlight_text_key(self):
        """'text' キーの文字列値が <mark class="jv"> でハイライトされること。"""
        html = _render_json_html('{"text": "hello world"}')
        assert '<mark class="jv">' in html
        assert 'hello world' in html

    def test_highlight_content_key(self):
        """'content' キーがハイライトされること。"""
        html = _render_json_html('{"content": "テスト内容"}')
        assert '<mark class="jv">' in html

    def test_highlight_thought_key(self):
        """'thought' キーがハイライトされること。"""
        html = _render_json_html('{"thought": "内部思考"}')
        assert '<mark class="jv">' in html

    def test_highlight_system_prompt_key(self):
        """'system_prompt' キーがハイライトされること。"""
        html = _render_json_html('{"system_prompt": "プロンプト内容"}')
        assert '<mark class="jv">' in html

    def test_highlight_system_instruction_key(self):
        """'system_instruction' キーがハイライトされること（Gemini 向け）。"""
        html = _render_json_html('{"system_instruction": "指示内容"}')
        assert '<mark class="jv">' in html

    def test_highlight_thinking_key(self):
        """'thinking' キーがハイライトされること（Claude CLI 向け）。"""
        html = _render_json_html('{"thinking": "思考内容"}')
        assert '<mark class="jv">' in html

    def test_highlight_result_key(self):
        """'result' キーがハイライトされること（Claude CLI 向け）。"""
        html = _render_json_html('{"result": "完了"}')
        assert '<mark class="jv">' in html

    def test_no_highlight_for_unknown_key(self):
        """ハイライト対象外のキーには <mark> が生成されないこと。"""
        html = _render_json_html('{"unknown_key": "値"}')
        assert '<mark class="jv">' not in html

    def test_newline_escape_expanded_in_highlight(self):
        """ハイライト値内の \\n が実際の改行として展開されること。"""
        data = {"text": "line1\nline2\nline3"}
        html = _render_json_html(json.dumps(data))
        # <mark> と </mark> の間に実際の改行文字（HTMLエスケープ後の改行）が含まれること
        import re as _re
        m = _re.search(r'<mark class="jv">(.*?)</mark>', html, _re.DOTALL)
        assert m is not None
        inner = m.group(1)
        assert "\n" in inner or "&#10;" in inner or "<br" in inner

    def test_cr_escape_expanded_in_highlight(self):
        """ハイライト値内の \\r が展開されること。"""
        data = {"text": "line1\r\nline2"}
        html = _render_json_html(json.dumps(data))
        assert '<mark class="jv">' in html

    def test_null_value_not_highlighted(self):
        """null 値はハイライトされないこと。"""
        html = _render_json_html('{"text": null}')
        assert '<mark class="jv">' not in html

    def test_jk_span_for_highlighted_key(self):
        """ハイライトされたキーに <span class="jk"> が付くこと。"""
        html = _render_json_html('{"text": "hello"}')
        assert '<span class="jk">' in html

    def test_ndjson_is_rendered(self):
        """NDJSON 形式のファイルが整形・ハイライトされること。"""
        lines = [
            json.dumps({"type": "system", "id": "s1"}),
            json.dumps({"type": "assistant", "text": "応答内容"}),
            json.dumps({"type": "result", "ok": True}),
        ]
        html = _render_json_html("\n".join(lines))
        # NDJSON をリストとして整形した結果に "text" のハイライトが含まれること
        assert '<mark class="jv">' in html

    def test_invalid_json_returns_escaped_plain_text(self):
        """パース不能なテキストは HTML エスケープされたプレーンテキストで返されること。"""
        html = _render_json_html("not json <script>")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ─── _build_attempt_detail: skip_files ───────────────────────────────────────


class TestBuildAttemptDetailSkipFiles:
    """_build_attempt_detail() の skip_files オプションを検証するテストクラス。

    一覧の高速化のためにファイルI/Oをスキップする skip_files=True（デフォルト）動作と、
    詳細エンドポイント用の skip_files=False（ファイル読み込みあり）動作を確認する。
    """

    def _make_row(self, raw_dir=None, **kwargs):
        """テスト用の DB 行辞書を生成するヘルパ。"""
        defaults = {
            "raw_dir": raw_dir,
            "has_error": False,
            "warn_reason": "",
            "preset": "ClaudeCode",
            "response": "応答テキスト",
            "reasoning": "",
            "created_at": None,
        }
        defaults.update(kwargs)
        return defaults

    def test_skip_files_true_returns_empty_tool_calls(self, tmp_path):
        """skip_files=True のとき raw_dir が存在しても tool_calls/warnings/files が空であること。

        ディレクトリが存在していても読み込まないことを確認する（アクセス不要）。
        """
        raw_dir = tmp_path / "abc12345"
        raw_dir.mkdir()
        (raw_dir / "02_chat_Request_ClaudeCode.log").write_text("{}", encoding="utf-8")
        (raw_dir / "03_chat_Response_ClaudeCode.log").write_text("応答", encoding="utf-8")

        result = _build_attempt_detail(self._make_row(raw_dir=str(raw_dir)), 0, {}, skip_files=True)

        assert result["tool_calls"] == []
        assert result["warnings"] == []
        assert result["files"] == []

    def test_skip_files_true_preserves_db_fields(self):
        """skip_files=True のとき DB 由来のフィールド（response/preset/index 等）は正しく返ること。"""
        row = self._make_row(response="キャラの返答", preset="Gemini", has_error=True, warn_reason="エラー")
        result = _build_attempt_detail(row, 2, {}, skip_files=True)

        assert result["index"] == 3  # 0始まり + 1
        assert result["response"] == "キャラの返答"
        assert result["preset"] == "Gemini"
        assert result["has_error"] is True
        assert result["warn_reason"] == "エラー"

    def test_skip_files_true_with_nonexistent_raw_dir_no_error(self):
        """skip_files=True のとき raw_dir が存在しないパスでも例外を送出しないこと。"""
        row = self._make_row(raw_dir="/nonexistent/path/abc12345")
        result = _build_attempt_detail(row, 0, {}, skip_files=True)
        assert result["tool_calls"] == []

    def test_skip_files_false_reads_tool_calls_from_files(self, tmp_path):
        """skip_files=False のとき raw_dir のファイルを読み込んで tool_calls を返すこと。"""
        raw_dir = tmp_path / "abc12345"
        raw_dir.mkdir()
        (raw_dir / "02_chat_Request_ClaudeCode.log").write_text("{}", encoding="utf-8")
        (raw_dir / "03_chat_Response_ClaudeCode.log").write_text("応答テキスト", encoding="utf-8")

        result = _build_attempt_detail(self._make_row(raw_dir=str(raw_dir)), 0, {}, skip_files=False)

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["feature"] == "chat"
        assert result["files"] != []

    def test_skip_files_false_with_none_raw_dir_returns_empty(self):
        """skip_files=False でも raw_dir が None の場合、tool_calls は空であること。"""
        result = _build_attempt_detail(self._make_row(raw_dir=None), 0, {}, skip_files=False)
        assert result["tool_calls"] == []
        assert result["dir_id"] == ""


# ─── _build_entry_from_db_rows: skip_files ───────────────────────────────────


class TestBuildEntryFromDbRowsSkipFiles:
    """_build_entry_from_db_rows() の skip_files オプションを検証するテストクラス。

    一覧では skip_files=True（デフォルト）、詳細エンドポイントでは skip_files=False
    を使い分けることで、一覧の高速化と詳細の情報完全性を両立させることを確認する。
    """

    def _make_row(self, **kwargs):
        """テスト用の DB 行辞書を生成するヘルパ。"""
        from datetime import datetime
        defaults = {
            "request_id": "abc12345",
            "source_type": "chat",
            "created_at": datetime(2026, 5, 1, 12, 0, 0),
            "target": "はる",
            "preset": "ClaudeCode",
            "user_message": "こんにちは",
            "response": "こんにちは！",
            "reasoning": "",
            "has_error": False,
            "warn_reason": "",
            "raw_dir": None,
        }
        defaults.update(kwargs)
        return defaults

    def test_skip_files_true_is_default(self, tmp_path):
        """引数省略時（デフォルト）は skip_files=True として動作すること。

        raw_dir が存在しない場合でも例外を送出せず、tool_calls が空のエントリを返すこと。
        """
        rows = [self._make_row(raw_dir=str(tmp_path / "nonexistent"))]
        entry = _build_entry_from_db_rows(rows)

        assert entry["character"] == "はる"
        assert entry["attempt_count"] == 1
        assert entry["attempts"][0]["tool_calls"] == []

    def test_skip_files_true_builds_entry_from_db_only(self):
        """skip_files=True のとき、サマリー表示に必要なフィールドが DB から正しく構築されること。"""
        rows = [self._make_row(
            response="応答テキスト",
            user_message="質問テキスト",
            has_error=True,
            warn_reason="タイムアウト",
        )]
        entry = _build_entry_from_db_rows(rows, skip_files=True)

        assert entry["character_response"] == "応答テキスト"
        assert entry["user_message"] == "質問テキスト"
        assert entry["has_error"] is True
        assert entry["warn_reason"] == "タイムアウト"

    def test_skip_files_false_reads_file_tool_calls(self, tmp_path):
        """skip_files=False のとき raw_dir からファイルを読み込んで tool_calls を取得すること。"""
        raw_dir = tmp_path / "abc12345"
        raw_dir.mkdir()
        (raw_dir / "02_chat_Request_ClaudeCode.log").write_text("{}", encoding="utf-8")
        (raw_dir / "03_chat_Response_ClaudeCode.log").write_text("応答テキスト", encoding="utf-8")

        rows = [self._make_row(raw_dir=str(raw_dir))]
        entry = _build_entry_from_db_rows(rows, skip_files=False)

        assert len(entry["attempts"][0]["tool_calls"]) == 1

    def test_multiple_attempts_all_skip_files(self, tmp_path):
        """複数試行がある場合、全試行で skip_files=True が適用されること。

        シナリオ再生成で3回試行したケースを想定し、全試行の tool_calls が空であることを確認する。
        """
        from datetime import datetime, timedelta
        rows = [
            self._make_row(
                raw_dir=str(tmp_path / f"attempt{i}"),
                response=f"{i}回目の応答",
                created_at=datetime(2026, 5, 1, 12, 0, i),
            )
            for i in range(3)
        ]
        entry = _build_entry_from_db_rows(rows, skip_files=True)

        assert entry["attempt_count"] == 3
        for attempt in entry["attempts"]:
            assert attempt["tool_calls"] == []
