"""backend.api.logs_ui.entries モジュールのユニットテスト。

ログエントリ構築と一覧ロードを網羅的に検証する。

対象関数:
    _build_char_index() — chotgor.log から {msg_id: char_label} 辞書を構築する
    _parse_entry()      — 1リクエストフォルダを解析してエントリ辞書を返す
    _load_entries()     — debug/ ディレクトリ全体をページネーションして返す

テスト方針:
    - ファイルシステムアクセスは tmp_path (pytest 組み込みフィクスチャ) で隔離する
    - DEBUG_BASE / CHOTGOR_LOG は logs_ui.config のモジュール変数を monkeypatch で差し替える
    - LLM・DB へのアクセスは発生しないため外部 mock は不要
"""

import json
import os
import time
from pathlib import Path

from backend.api.logs_ui import config as logs_ui_config
from backend.api.logs_ui.entries import (
    _build_char_index,
    _load_entries,
    _parse_entry,
)
from tests._logs_ui_helpers import (
    _make_debug_dir,
    _write_front_input,
    _write_front_input_with_newlines,
    _write_response,
)

# ─── _build_char_index ────────────────────────────────────────────────────────


class TestBuildCharIndex:
    """_build_char_index() の動作を検証するテストクラス。"""

    def test_missing_log_returns_empty(self, tmp_path, monkeypatch):
        """chotgor.log が存在しない場合、空辞書を返すこと。"""
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", tmp_path / "not_exist.log")
        assert _build_char_index() == {}

    def test_extracts_char_label(self, tmp_path, monkeypatch):
        """char= パターンを含む行から msg_id と char_label を正しく抽出すること。"""
        log = tmp_path / "chotgor.log"
        log.write_text(
            "2026-04-09 04:01:04 INFO    [a25e00da] services.character_query:ask_character"
            " | ask_character: 完了 char=はる@ClaudeCode feature=forget response_chars=379\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", log)
        idx = _build_char_index()
        assert idx.get("a25e00da") == "はる@ClaudeCode"

    def test_last_match_wins_for_same_msg_id(self, tmp_path, monkeypatch):
        """同一 msg_id で複数行ある場合、最後にマッチした値が採用されること。

        ask_character の完了ログ（char=はる@ClaudeCode）の後に
        forget_job の完了ログ（char=はる@GhostModel）が続く実際のパターンを再現する。
        最後の値が採用されるので GhostModel が残る。
        """
        log = tmp_path / "chotgor.log"
        log.write_text(
            "2026-04-09 04:01:04 INFO    [a25e00da] ... | ask_character: 完了 char=はる@ClaudeCode feature=forget\n"
            "2026-04-09 04:01:04 INFO    [a25e00da] ... | forget 完了 char=はる@GhostModel candidates=4 deleted=3\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", log)
        idx = _build_char_index()
        assert idx.get("a25e00da") == "はる@GhostModel"

    def test_multiple_different_msg_ids(self, tmp_path, monkeypatch):
        """異なる msg_id の行が混在する場合、それぞれ独立して格納されること。"""
        log = tmp_path / "chotgor.log"
        log.write_text(
            "2026-04-09 03:02:02 INFO    [1ebae989] ... | ask_character: 完了 char=Vivier@Gemma4 feature=chronicle\n"
            "2026-04-09 03:02:03 INFO    [5efff325] ... | ask_character: 完了 char=朝倉麻衣@Gemma4 feature=chronicle\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", log)
        idx = _build_char_index()
        assert idx.get("1ebae989") == "Vivier@Gemma4"
        assert idx.get("5efff325") == "朝倉麻衣@Gemma4"

    def test_lines_without_char_are_skipped(self, tmp_path, monkeypatch):
        """char= を含まない行は無視されること。"""
        log = tmp_path / "chotgor.log"
        log.write_text(
            "2026-04-09 INFO    [deadbeef] some.module:func | 普通のログメッセージ\n"
            "2026-04-09 INFO    [deadbeef] other:func | recall_with_identity: identity=0 char_id=uuid-xxx\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", log)
        idx = _build_char_index()
        assert "deadbeef" not in idx

    def test_unreadable_log_returns_empty(self, tmp_path, monkeypatch):
        """ログファイルが読み取れない場合も空辞書を返し例外を送出しないこと。"""
        non_existent = tmp_path / "no_such.log"
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", non_existent)
        assert _build_char_index() == {}


# ─── _parse_entry ─────────────────────────────────────────────────────────────


class TestParseEntry:
    """_parse_entry() の動作を検証するテストクラス。"""

    def test_user_chat_basic(self, tmp_path):
        """FrontInput と FrontOutput を持つ通常ユーザーチャットが正しく解析されること。"""
        folder = _make_debug_dir(tmp_path, "abc12345")
        _write_front_input(folder, "はる@ClaudeCode", "こんにちは")
        _write_response(folder, "02_FrontOutput.log", "こんにちは！")

        entry = _parse_entry("abc12345", folder, {})

        assert entry["message_id"] == "abc12345"
        assert entry["character"] == "はる"
        assert entry["preset"] == "ClaudeCode"
        assert entry["model_id"] == "はる@ClaudeCode"
        assert entry["source"] == "ユーザ"
        assert entry["user_message"] == "こんにちは"
        assert entry["character_response"] == "こんにちは！"

    def test_front_input_with_embedded_newlines(self, tmp_path):
        """debug_logger が生成する「JSON 文字列内に生改行あり」ファイルが strict=False でパースできること。

        debug_logger._unescape_text() により \\n が実際の改行になるため、
        json.loads() のデフォルト（strict=True）では JSONDecodeError になる。
        strict=False で制御文字を許容することを確認する。
        """
        folder = _make_debug_dir(tmp_path, "abc12345")
        _write_front_input_with_newlines(
            folder,
            "朝倉麻衣@ClaudeCode",
            "1行目\n2行目\n3行目",  # 改行を含むメッセージ
        )

        entry = _parse_entry("abc12345", folder, {})

        assert entry["character"] == "朝倉麻衣"
        assert entry["preset"] == "ClaudeCode"
        assert "1行目" in entry["user_message"]
        assert "2行目" in entry["user_message"]

    def test_model_id_without_at(self, tmp_path):
        """model_id に @ が含まれない場合、character に全体が入り preset は空になること。"""
        folder = _make_debug_dir(tmp_path, "abc12345")
        _write_front_input(folder, "テストキャラ", "メッセージ")

        entry = _parse_entry("abc12345", folder, {})

        assert entry["character"] == "テストキャラ"
        assert entry["preset"] == ""

    def test_batch_entry_uses_char_index(self, tmp_path):
        """FrontInput がない batch エントリは char_index からキャラクター名を補完すること。"""
        folder = _make_debug_dir(tmp_path, "batchabc1")
        # chronicle の Request ファイルのみ、FrontInput なし
        _write_response(folder, "01_chronicle_Request_Gemma4.log", "{}")
        _write_response(folder, "02_chronicle_Response_Gemma4.log", "更新なし")

        char_index = {"batchabc1": "Vivier@Gemma4"}
        entry = _parse_entry("batchabc1", folder, char_index)

        assert entry["character"] == "Vivier"
        assert entry["preset"] == "Gemma4"
        assert entry["model_id"] == "Vivier@Gemma4"
        assert entry["source"] == "system"

    def test_batch_entry_not_in_index_stays_empty(self, tmp_path):
        """FrontInput なし・char_index にも存在しない場合、model_id は空のままであること。"""
        folder = _make_debug_dir(tmp_path, "unknown1")
        _write_response(folder, "01_forget_Request_ClaudeCode.log", "{}")

        entry = _parse_entry("unknown1", folder, {})

        assert entry["model_id"] == ""
        assert entry["character"] == ""

    def test_tool_calls_request_response_pair(self, tmp_path):
        """Request/Response ファイルペアが tool_calls に正しくまとめられること。"""
        folder = _make_debug_dir(tmp_path, "tooltest")
        _write_front_input(folder, "はる@ClaudeCode", "質問")
        _write_response(folder, "01_FrontInput.log", "")  # 上書きになるが気にしない
        _write_response(folder, "02_chat_Request_ClaudeCode.log", "{}")
        _write_response(
            folder,
            "03_chat_Response_ClaudeCode.log",
            "回答[INSCRIBE_MEMORY:user|1.0|記憶内容]",
        )
        _write_response(folder, "04_FrontOutput.log", "回答")

        entry = _parse_entry("tooltest", folder, {})

        assert len(entry["tool_calls"]) == 1
        tc = entry["tool_calls"][0]
        assert tc["feature"] == "chat"
        assert tc["preset"] == "ClaudeCode"
        assert tc["request_file"] == "02_chat_Request_ClaudeCode.log"
        assert tc["response_file"] == "03_chat_Response_ClaudeCode.log"
        # Response ファイルのタグも抽出されていること
        assert len(tc["tags"]) == 1
        assert tc["tags"][0]["tag_name"] == "INSCRIBE_MEMORY"

    def test_tool_calls_multiple_features(self, tmp_path):
        """chat + trigger など複数フィーチャーが独立した tool_calls になること。"""
        folder = _make_debug_dir(tmp_path, "multi1")
        _write_front_input(folder, "はる@ClaudeCode", "質問")
        _write_response(folder, "02_chat_Request_ClaudeCode.log", "{}")
        _write_response(folder, "03_chat_Response_ClaudeCode.log", "回答")
        _write_response(folder, "04_trigger_Request_Gemma4.log", "{}")
        _write_response(folder, "05_trigger_Response_Gemma4.log", "トリガー応答")

        entry = _parse_entry("multi1", folder, {})

        features = {tc["feature"] for tc in entry["tool_calls"]}
        assert "chat" in features
        assert "trigger" in features

    def test_has_error_false_on_normal_response(self, tmp_path):
        """正常なレスポンスの場合、has_error は False であること。"""
        folder = _make_debug_dir(tmp_path, "ok1")
        _write_front_input(folder, "はる@ClaudeCode", "質問")
        _write_response(folder, "02_chat_Request_ClaudeCode.log", "{}")
        _write_response(folder, "03_chat_Response_ClaudeCode.log", "正常な返答です")

        entry = _parse_entry("ok1", folder, {})
        assert entry["has_error"] is False

    def test_has_error_true_when_response_missing(self, tmp_path):
        """Request があるのに Response が存在しない場合、has_error が True になること。

        chronicle などで LLM API がエラーを投げ _log_response が呼ばれなかったケース。
        新しいプロバイダーのエラーロギング修正前のレガシーフォルダでも検出できることを確認する。
        """
        folder = _make_debug_dir(tmp_path, "err1")
        _write_response(folder, "01_chronicle_Request_Gemma4.log", "{}")
        # Response ファイルなし

        entry = _parse_entry("err1", folder, {})
        assert entry["has_error"] is True

    def test_has_error_true_when_response_is_error_string(self, tmp_path):
        """Response ファイルがエラー文字列で始まる場合、has_error が True になること。

        プロバイダーの _log_error 修正後に記録されるエラー Response ファイルを想定。
        """
        folder = _make_debug_dir(tmp_path, "err2")
        _write_response(folder, "01_chronicle_Request_Gemma4.log", "{}")
        _write_response(
            folder,
            "02_chronicle_Response_Gemma4.log",
            "[Google API error: 400 INVALID_ARGUMENT. Thinking budget is not supported]",
        )

        entry = _parse_entry("err2", folder, {})
        assert entry["has_error"] is True

    def test_has_error_false_when_response_contains_bracket(self, tmp_path):
        """レスポンスが '[' で始まるが error ではない場合、has_error は False のままであること。

        キャラクターが '[' 始まりのテキスト（例: Markdown リスト）を返すケースで
        誤検知しないことを確認する。
        """
        folder = _make_debug_dir(tmp_path, "ok2")
        _write_response(folder, "01_chat_Request_ClaudeCode.log", "{}")
        _write_response(folder, "02_chat_Response_ClaudeCode.log", "[OK] 正常な返答です")

        entry = _parse_entry("ok2", folder, {})
        assert entry["has_error"] is False

    def test_oserror_on_stat_uses_epoch(self, tmp_path, monkeypatch):
        """フォルダの stat が失敗した場合、dt は epoch (1970-01-01) になること。"""
        folder = _make_debug_dir(tmp_path, "statfail")

        original_stat = Path.stat

        def raise_oserror(self, *a, **kw):
            if self == folder:
                raise OSError("stat failed")
            return original_stat(self)

        monkeypatch.setattr(Path, "stat", raise_oserror)
        entry = _parse_entry("statfail", folder, {})
        assert entry["dt"].year == 1970

    def test_warnings_empty_when_no_warning_files(self, tmp_path):
        """Warning ファイルがない場合、warnings は空リストであること。"""
        folder = _make_debug_dir(tmp_path, "warn0")
        _write_front_input(folder, "はる@ClaudeCode", "こんにちは")

        entry = _parse_entry("warn0", folder, {})

        assert entry["warnings"] == []

    def test_warnings_parsed_from_warning_file(self, tmp_path):
        """Warning_{tag}.log ファイルが存在する場合、warnings リストに内容が入ること。

        log_warning("context_window", "全20件 → 12件に圧縮 ...") が書き出すファイルを再現する。
        """
        folder = _make_debug_dir(tmp_path, "warn1")
        _write_front_input(folder, "はる@ClaudeCode", "こんにちは")
        _write_response(
            folder,
            "03_Warning_context_window.log",
            "全20件 → 12件に圧縮 (chronicle済み上限: 10)",
        )

        entry = _parse_entry("warn1", folder, {})

        assert len(entry["warnings"]) == 1
        w = entry["warnings"][0]
        assert w["tag"] == "context_window"
        assert "全20件" in w["message"]
        assert w["file"] == "03_Warning_context_window.log"

    def test_multiple_warning_files(self, tmp_path):
        """複数の Warning ファイルがある場合、すべて warnings に含まれること。"""
        folder = _make_debug_dir(tmp_path, "warn2")
        _write_response(folder, "02_Warning_context_window.log", "圧縮メッセージA")
        _write_response(folder, "03_Warning_other_tag.log", "別の警告B")

        entry = _parse_entry("warn2", folder, {})

        assert len(entry["warnings"]) == 2
        tags = {w["tag"] for w in entry["warnings"]}
        assert "context_window" in tags
        assert "other_tag" in tags

    def test_warning_file_read_error_skips_gracefully(self, tmp_path, monkeypatch):
        """Warning ファイルの読み取りに失敗した場合、message が空文字で warnings に含まれること。"""
        folder = _make_debug_dir(tmp_path, "warn3")
        warn_file = folder / "02_Warning_context_window.log"
        warn_file.write_text("圧縮メッセージ", encoding="utf-8")

        original_read = Path.read_text

        def raise_on_warning(self, *a, **kw):
            if self == warn_file:
                raise OSError("read error")
            return original_read(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", raise_on_warning)
        entry = _parse_entry("warn3", folder, {})

        assert len(entry["warnings"]) == 1
        assert entry["warnings"][0]["message"] == ""


# ─── _load_entries ────────────────────────────────────────────────────────────


class TestLoadEntries:
    """_load_entries() のページネーションとソートを検証するテストクラス。"""

    def test_missing_debug_dir_returns_empty(self, tmp_path, monkeypatch):
        """debug/ が存在しない場合、([], 0) を返すこと。"""
        monkeypatch.setattr(logs_ui_config, "DEBUG_BASE", tmp_path / "no_debug")
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", tmp_path / "no.log")
        entries, total = _load_entries()
        assert entries == []
        assert total == 0

    def test_total_count_correct(self, tmp_path, monkeypatch):
        """フォルダ数が total に正しく反映されること。"""
        debug = tmp_path / "debug"
        for i in range(5):
            _make_debug_dir(debug, f"msg{i:08x}")
        monkeypatch.setattr(logs_ui_config, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", tmp_path / "no.log")
        _, total = _load_entries()
        assert total == 5

    def test_sorted_newest_first(self, tmp_path, monkeypatch):
        """エントリが更新日時の降順（新しい順）で返されること。"""
        debug = tmp_path / "debug"
        base_time = time.time()
        ids_in_order = ["oldest01", "middle02", "newest03"]
        for i, name in enumerate(ids_in_order):
            folder = _make_debug_dir(debug, name)
            # 意図的に異なる mtime を設定する
            os.utime(folder, (base_time + i, base_time + i))

        monkeypatch.setattr(logs_ui_config, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", tmp_path / "no.log")
        entries, _ = _load_entries()
        returned_ids = [e["message_id"] for e in entries]
        assert returned_ids == ["newest03", "middle02", "oldest01"]

    def test_pagination_page1(self, tmp_path, monkeypatch):
        """ページ1は先頭 per_page 件を返すこと。"""
        debug = tmp_path / "debug"
        for i in range(7):
            _make_debug_dir(debug, f"msg{i:08x}")
        monkeypatch.setattr(logs_ui_config, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", tmp_path / "no.log")
        entries, total = _load_entries(page=1, per_page=3)
        assert total == 7
        assert len(entries) == 3

    def test_pagination_last_page(self, tmp_path, monkeypatch):
        """最終ページは余り件数だけ返すこと。"""
        debug = tmp_path / "debug"
        for i in range(7):
            _make_debug_dir(debug, f"msg{i:08x}")
        monkeypatch.setattr(logs_ui_config, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", tmp_path / "no.log")
        entries, total = _load_entries(page=3, per_page=3)
        assert total == 7
        assert len(entries) == 1  # 7件、3件ずつ → 3ページ目は1件

    def test_trush_folder_excluded(self, tmp_path, monkeypatch):
        """"trush" という名前のフォルダはカウントから除外されること。"""
        debug = tmp_path / "debug"
        _make_debug_dir(debug, "normal01")
        _make_debug_dir(debug, "trush")  # 除外対象
        monkeypatch.setattr(logs_ui_config, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_config, "CHOTGOR_LOG", tmp_path / "no.log")
        _, total = _load_entries()
        assert total == 1


