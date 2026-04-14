"""backend.api.logs_ui モジュールのユニットテスト。

ログ閲覧UIの各関数を網羅的に検証する。

対象関数:
    _build_char_index()                  — chotgor.log から {msg_id: char_label} 辞書を構築する
    _parse_tag_body()                    — タグ本体文字列を表示用辞書に変換する
    _parse_json_safely()                 — debug_logger 生成 JSON をバックスラッシュ+改行前処理込みでパースする
    _extract_function_calls_from_json()  — JSON レスポンスから function_call を抽出してタグ構造体で返す
    _extract_tags_from_file()            — ログファイルからタグを抽出して出現順で返す
    _parse_entry()                       — 1リクエストフォルダを解析してエントリ辞書を返す
    _load_entries()                      — debug/ ディレクトリ全体をページネーションして返す
    log_raw_file()                       — 生ログファイルをプレーンテキストで返す HTTP エンドポイント

テスト方針:
    - ファイルシステムアクセスは tmp_path (pytest 組み込みフィクスチャ) で隔離する
    - DEBUG_BASE / CHOTGOR_LOG はモジュール変数を monkeypatch で差し替える
    - HTTP エンドポイントは httpx の ASGITransport または TestClient 経由でテストする
    - LLM・DB へのアクセスは発生しないため外部 mock は不要
"""

import json
import os
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api.logs_ui as logs_ui_module
from backend.api.logs_ui import (
    _build_char_index,
    _extract_function_calls_from_json,
    _extract_tags_from_file,
    _load_entries,
    _parse_entry,
    _parse_json_safely,
    _parse_tag_body,
    router,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────────────

def _make_debug_dir(base: Path, msg_id: str) -> Path:
    """指定 base 配下に debug フォルダを作成して返す。"""
    folder = base / msg_id
    folder.mkdir(parents=True)
    return folder


def _write_front_input(folder: Path, model_id: str, content: str) -> None:
    """01_FrontInput.log をフォルダに書き込む。"""
    data = {"content": content, "image_ids": None, "model_id": model_id}
    (folder / "01_FrontInput.log").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_front_input_with_newlines(folder: Path, model_id: str, content: str) -> None:
    """debug_logger の _unescape_text() 相当の処理を再現して FrontInput.log を書き込む。

    debug_logger は JSON の文字列内 \\n を実際の改行に変換するため、
    厳密な JSON としては無効なファイルが生成される。
    """
    data = {"content": content, "image_ids": None, "model_id": model_id}
    raw = json.dumps(data, ensure_ascii=False, indent=2)
    # debug_logger._unescape_text() の再現: エスケープ済み \\n → 実際の改行
    raw = raw.replace("\\n", "\n").replace("\\t", "\t")
    (folder / "01_FrontInput.log").write_text(raw, encoding="utf-8")


def _write_response(folder: Path, filename: str, content: str) -> None:
    """指定ファイル名でレスポンスログを書き込む。"""
    (folder / filename).write_text(content, encoding="utf-8")


# ─── _build_char_index ────────────────────────────────────────────────────────


class TestBuildCharIndex:
    """_build_char_index() の動作を検証するテストクラス。"""

    def test_missing_log_returns_empty(self, tmp_path, monkeypatch):
        """chotgor.log が存在しない場合、空辞書を返すこと。"""
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", tmp_path / "not_exist.log")
        assert _build_char_index() == {}

    def test_extracts_char_label(self, tmp_path, monkeypatch):
        """char= パターンを含む行から msg_id と char_label を正しく抽出すること。"""
        log = tmp_path / "chotgor.log"
        log.write_text(
            "2026-04-09 04:01:04 INFO    [a25e00da] services.character_query:ask_character"
            " | ask_character: 完了 char=はる@ClaudeCode feature=forget response_chars=379\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", log)
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
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", log)
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
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", log)
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
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", log)
        idx = _build_char_index()
        assert "deadbeef" not in idx

    def test_unreadable_log_returns_empty(self, tmp_path, monkeypatch):
        """ログファイルが読み取れない場合も空辞書を返し例外を送出しないこと。"""
        non_existent = tmp_path / "no_such.log"
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", non_existent)
        assert _build_char_index() == {}


# ─── _parse_tag_body ──────────────────────────────────────────────────────────


class TestParseTagBody:
    """_parse_tag_body() の各タグ種別変換を検証するテストクラス。"""

    def test_inscribe_memory_full(self):
        """INSCRIBE_MEMORY: category|importance|content が正しく分解されること。"""
        result = _parse_tag_body("INSCRIBE_MEMORY", "user|1.4|ユーザは猫が好き")
        assert result["tag_name"] == "INSCRIBE_MEMORY"
        assert result["fields"]["カテゴリ"] == "user"
        assert result["fields"]["重要度"] == "1.4"
        assert result["fields"]["内容"] == "ユーザは猫が好き"
        assert result["preview"] == "ユーザは猫が好き"
        assert result["meta"]["label"] == "記憶"
        assert result["meta"]["cls"] == "tag-memory"

    def test_inscribe_memory_content_with_pipe(self):
        """INSCRIBE_MEMORY: 内容フィールドに | が含まれる場合、最大2分割で正しく扱われること。"""
        result = _parse_tag_body("INSCRIBE_MEMORY", "semantic|0.9|A|B|C という内容")
        assert result["fields"]["カテゴリ"] == "semantic"
        assert result["fields"]["重要度"] == "0.9"
        assert result["fields"]["内容"] == "A|B|C という内容"

    def test_inscribe_memory_missing_importance(self):
        """INSCRIBE_MEMORY: importance が省略された場合、空文字として扱われること。"""
        result = _parse_tag_body("INSCRIBE_MEMORY", "contextual")
        assert result["fields"]["カテゴリ"] == "contextual"
        assert result["fields"]["重要度"] == ""
        assert result["fields"]["内容"] == ""

    def test_carve_narrative_append(self):
        """CARVE_NARRATIVE: append|content が正しく分解されること。"""
        result = _parse_tag_body("CARVE_NARRATIVE", "append|新しい自己認識が芽生えた")
        assert result["fields"]["モード"] == "append"
        assert result["fields"]["内容"] == "新しい自己認識が芽生えた"
        assert result["preview"] == "新しい自己認識が芽生えた"
        assert result["meta"]["cls"] == "tag-narrative"

    def test_carve_narrative_content_with_pipe(self):
        """CARVE_NARRATIVE: 内容フィールドに | が含まれる場合も正しく扱われること。"""
        result = _parse_tag_body("CARVE_NARRATIVE", "append|A|B")
        assert result["fields"]["モード"] == "append"
        assert result["fields"]["内容"] == "A|B"

    def test_drift_body_is_content(self):
        """DRIFT: body 全体が "内容" フィールドになること。"""
        result = _parse_tag_body("DRIFT", "新しいキャラクター設計の方針")
        assert result["fields"]["内容"] == "新しいキャラクター設計の方針"
        assert result["preview"] == "新しいキャラクター設計の方針"
        assert result["meta"]["cls"] == "tag-drift"

    def test_drift_reset_fixed_marker(self):
        """DRIFT_RESET: 固定マーカーで body が空でも "(リセット)" と表示されること。"""
        result = _parse_tag_body("DRIFT_RESET", "")
        assert result["fields"]["内容"] == "(リセット)"
        assert result["preview"] == "(リセット)"

    def test_switch_angle_preset_and_context(self):
        """SWITCH_ANGLE: preset|context が分解され、preview は context になること（Bug修正確認）。"""
        result = _parse_tag_body("SWITCH_ANGLE", "Gemma4|ミオとして大胆になったシーン")
        assert result["fields"]["プリセット"] == "Gemma4"
        assert result["fields"]["コンテキスト"] == "ミオとして大胆になったシーン"
        # preview は context（第2フィールド）であること。preset|context の body 全体ではない
        assert result["preview"] == "ミオとして大胆になったシーン"
        assert result["meta"]["cls"] == "tag-switch"

    def test_switch_angle_preset_only(self):
        """SWITCH_ANGLE: context 省略時も preset が取れ、preview は空文字にならないこと。"""
        result = _parse_tag_body("SWITCH_ANGLE", "Gemma4")
        assert result["fields"]["プリセット"] == "Gemma4"
        assert result["fields"]["コンテキスト"] == ""
        # コンテキストが空 → フォールバックで body（"Gemma4"）が preview になる
        assert result["preview"] == "Gemma4"

    def test_power_recall(self):
        """POWER_RECALL: body が "内容" フィールドになること。"""
        result = _parse_tag_body("POWER_RECALL", "記憶を全て呼び起こせ")
        assert result["fields"]["内容"] == "記憶を全て呼び起こせ"
        assert result["meta"]["cls"] == "tag-recall"

    def test_end_session_with_body(self):
        """END_SESSION: body が "内容" フィールドになること。"""
        result = _parse_tag_body("END_SESSION", "さようなら")
        assert result["fields"]["内容"] == "さようなら"
        assert result["meta"]["cls"] == "tag-end"

    def test_unknown_tag_uses_fallback_meta(self):
        """未知のタグ名の場合、fallback meta (cls=tag-unknown) が使われること。"""
        result = _parse_tag_body("UNKNOWN_TAG", "body")
        assert result["meta"]["cls"] == "tag-unknown"
        assert result["meta"]["label"] == "UNKNOWN_TAG"


# ─── _extract_tags_from_file ─────────────────────────────────────────────────


class TestExtractTagsFromFile:
    """_extract_tags_from_file() の動作を検証するテストクラス。"""

    def test_single_inscribe_memory_tag(self, tmp_path):
        """単一の INSCRIBE_MEMORY タグが正しく抽出されること。"""
        f = tmp_path / "response.log"
        f.write_text(
            "会話の内容[INSCRIBE_MEMORY:user|1.0|ユーザは猫が好き]以上",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "INSCRIBE_MEMORY"
        assert tags[0]["fields"]["内容"] == "ユーザは猫が好き"

    def test_file_order_preserved(self, tmp_path):
        """複数タグが混在する場合、ファイル内の出現順で返されること（Bug修正確認）。

        ファイル内で DRIFT → INSCRIBE_MEMORY の順で出現した場合、
        タグ種別のリスト順（INSCRIBE_MEMORY→DRIFT）ではなく、
        出現順（DRIFT→INSCRIBE_MEMORY）で返されること。
        """
        f = tmp_path / "response.log"
        f.write_text(
            "テキスト[DRIFT:ドリフト内容]間のテキスト[INSCRIBE_MEMORY:contextual|0.8|記憶内容]終わり",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 2
        assert tags[0]["tag_name"] == "DRIFT"
        assert tags[1]["tag_name"] == "INSCRIBE_MEMORY"

    def test_multiline_drift_tag(self, tmp_path):
        """DRIFT タグが複数行にまたがる場合も正しく抽出されること。"""
        f = tmp_path / "response.log"
        f.write_text(
            "応答テキスト[DRIFT:1行目の内容\n2行目の内容\n3行目]終わり",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "DRIFT"
        assert "1行目の内容" in tags[0]["fields"]["内容"]
        assert "3行目" in tags[0]["fields"]["内容"]

    def test_drift_reset_fixed_marker(self, tmp_path):
        """[DRIFT_RESET] 固定マーカーが抽出されること。"""
        f = tmp_path / "response.log"
        f.write_text("リセットします[DRIFT_RESET]以後よろしく", encoding="utf-8")
        tags = _extract_tags_from_file(f)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "DRIFT_RESET"
        assert tags[0]["preview"] == "(リセット)"

    def test_no_tags_returns_empty(self, tmp_path):
        """タグが存在しないファイルでは空リストを返すこと。"""
        f = tmp_path / "response.log"
        f.write_text("タグなしのテキストです", encoding="utf-8")
        assert _extract_tags_from_file(f) == []

    def test_missing_file_returns_empty(self, tmp_path):
        """ファイルが存在しない場合も例外を送出せず空リストを返すこと。"""
        assert _extract_tags_from_file(tmp_path / "missing.log") == []

    def test_multiple_same_tag_type(self, tmp_path):
        """同一タグ種別が複数ある場合、すべて抽出されること。"""
        f = tmp_path / "response.log"
        f.write_text(
            "[INSCRIBE_MEMORY:user|1.0|記憶A]テキスト[INSCRIBE_MEMORY:semantic|0.5|記憶B]",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 2
        contents = [t["fields"]["内容"] for t in tags]
        assert "記憶A" in contents
        assert "記憶B" in contents


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

    def test_old_format_no_feature_name(self, tmp_path):
        """旧形式（機能名なし）: "02_Request_claude_cli.log" が tool_calls に含まれること。"""
        folder = _make_debug_dir(tmp_path, "oldformat")
        _write_response(folder, "02_Request_claude_cli.log", "{}")
        _write_response(folder, "03_Response_claude_cli.log", "旧形式の応答")

        entry = _parse_entry("oldformat", folder, {})

        assert len(entry["tool_calls"]) == 1
        tc = entry["tool_calls"][0]
        assert tc["feature"] == ""
        assert tc["preset"] == "claude_cli"

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
        monkeypatch.setattr(logs_ui_module, "DEBUG_BASE", tmp_path / "no_debug")
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", tmp_path / "no.log")
        entries, total = _load_entries()
        assert entries == []
        assert total == 0

    def test_total_count_correct(self, tmp_path, monkeypatch):
        """フォルダ数が total に正しく反映されること。"""
        debug = tmp_path / "debug"
        for i in range(5):
            _make_debug_dir(debug, f"msg{i:08x}")
        monkeypatch.setattr(logs_ui_module, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", tmp_path / "no.log")
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

        monkeypatch.setattr(logs_ui_module, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", tmp_path / "no.log")
        entries, _ = _load_entries()
        returned_ids = [e["message_id"] for e in entries]
        assert returned_ids == ["newest03", "middle02", "oldest01"]

    def test_pagination_page1(self, tmp_path, monkeypatch):
        """ページ1は先頭 per_page 件を返すこと。"""
        debug = tmp_path / "debug"
        for i in range(7):
            _make_debug_dir(debug, f"msg{i:08x}")
        monkeypatch.setattr(logs_ui_module, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", tmp_path / "no.log")
        entries, total = _load_entries(page=1, per_page=3)
        assert total == 7
        assert len(entries) == 3

    def test_pagination_last_page(self, tmp_path, monkeypatch):
        """最終ページは余り件数だけ返すこと。"""
        debug = tmp_path / "debug"
        for i in range(7):
            _make_debug_dir(debug, f"msg{i:08x}")
        monkeypatch.setattr(logs_ui_module, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", tmp_path / "no.log")
        entries, total = _load_entries(page=3, per_page=3)
        assert total == 7
        assert len(entries) == 1  # 7件、3件ずつ → 3ページ目は1件

    def test_trush_folder_excluded(self, tmp_path, monkeypatch):
        """"trush" という名前のフォルダはカウントから除外されること。"""
        debug = tmp_path / "debug"
        _make_debug_dir(debug, "normal01")
        _make_debug_dir(debug, "trush")  # 除外対象
        monkeypatch.setattr(logs_ui_module, "DEBUG_BASE", debug)
        monkeypatch.setattr(logs_ui_module, "CHOTGOR_LOG", tmp_path / "no.log")
        _, total = _load_entries()
        assert total == 1


# ─── log_raw_file HTTP エンドポイント ─────────────────────────────────────────


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """logs_ui ルーターを組み込んだ最小 FastAPI アプリのテストクライアントを返すフィクスチャ。

    DEBUG_BASE を tmp_path 配下に差し替え、外部依存を排除する。
    """
    monkeypatch.setattr(logs_ui_module, "DEBUG_BASE", tmp_path / "debug")
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


# ─── _parse_json_safely ───────────────────────────────────────────────────────


class TestParseJsonSafely:
    """_parse_json_safely() の動作を検証するテストクラス。

    debug_logger._unescape_text() により JSON 文字列値内の \\n が実際の改行になることで
    生じる不正エスケープ（バックスラッシュ + 実際の改行）を前処理して json.loads できること、
    および通常の JSON・非 JSON テキストの扱いを確認する。
    """

    def test_normal_json_is_parsed(self):
        """通常の JSON 文字列が正しくパースされること。"""
        text = '{"key": "value", "num": 42}'
        result = _parse_json_safely(text)
        assert result == {"key": "value", "num": 42}

    def test_backslash_newline_in_string_value_is_repaired(self):
        """JSON 文字列値内のバックスラッシュ+改行（不正エスケープ）が前処理で修復されること。

        debug_logger._unescape_text() は JSON 文字列値内の \\n を実際の改行に変換するため、
        バックスラッシュ + 実際の改行（\\<LF>）という不正エスケープが生まれる。
        strict=False は制御文字を許容するが不正エスケープシーケンスは許容しないため、
        素の json.loads でも失敗する。_parse_json_safely はこれを修復する。
        """
        # Python の '...\\\n...' は文字列として「バックスラッシュ + 実際の改行（0x0A）」を含む。
        # JSON パーサは \<LF> を不正エスケープとして拒否する。
        raw = '{"key": "line1\\\nline2\\\nline3"}'

        # 前提確認: 素の json.loads(strict=False) では JSONDecodeError になること
        try:
            json.loads(raw, strict=False)
            raw_fails = False
        except json.JSONDecodeError:
            raw_fails = True
        assert raw_fails, "前提条件: 素の json.loads は \\ + 実改行で失敗するはず"

        # _parse_json_safely は修復してパース成功すること
        result = _parse_json_safely(raw)
        assert "key" in result
        assert "line1" in result["key"]
        assert "line3" in result["key"]

    def test_non_json_raises_value_error(self):
        """JSON でないテキストを渡すと ValueError または JSONDecodeError が送出されること。"""
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_json_safely("これはJSONではありません")

    def test_list_json_is_parsed(self):
        """配列形式の JSON もパースできること。"""
        result = _parse_json_safely('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_embedded_control_chars_are_allowed(self):
        """strict=False により制御文字を含む JSON も許容されること。"""
        # 実際の改行（制御文字）を文字列値に直接含む（strict=True では拒否される）
        raw = '{"msg": "line1\nline2"}'
        result = _parse_json_safely(raw)
        assert "line1" in result["msg"]
        assert "line2" in result["msg"]


# ─── _extract_function_calls_from_json ───────────────────────────────────────


class TestExtractFunctionCallsFromJson:
    """_extract_function_calls_from_json() の動作を検証するテストクラス。

    Gemini / Anthropic / OpenAI 各プロバイダーの function_call フォーマットが
    タグ構造体に正しく変換されること、および非 JSON ファイルは空リストを返すことを確認する。
    """

    def test_gemini_inscribe_memory(self):
        """Gemini形式: candidates[].content.parts[].function_call が INSCRIBE_MEMORY に変換されること。

        id:ff8716f7 の実際のレスポンスログ構造を再現し、
        inscribe_memory の function_call が正しくタグとして抽出されることを検証する。
        """
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "id": "vtmqucqw",
                                    "name": "inscribe_memory",
                                    "args": {
                                        "content": "ユーザはドン引きしていた",
                                        "category": "contextual",
                                        "impact": 1.5,
                                    },
                                },
                                "text": None,
                            }
                        ],
                        "role": "model",
                    }
                }
            ],
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        tag = tags[0]
        assert tag["tag_name"] == "INSCRIBE_MEMORY"
        assert tag["fields"]["カテゴリ"] == "contextual"
        assert "ドン引き" in tag["fields"]["内容"]

    def test_gemini_multiple_parts(self):
        """Gemini形式: 同一 parts 内に複数の function_call があれば両方抽出されること。"""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "inscribe_memory",
                                    "args": {"category": "user_info", "impact": 1.0, "content": "記憶A"},
                                }
                            },
                            {
                                "function_call": {
                                    "name": "drift",
                                    "args": {"content": "ドリフト内容"},
                                }
                            },
                        ],
                        "role": "model",
                    }
                }
            ],
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 2
        tag_names = {t["tag_name"] for t in tags}
        assert "INSCRIBE_MEMORY" in tag_names
        assert "DRIFT" in tag_names

    def test_anthropic_tool_use(self):
        """Anthropic形式: content[].type == "tool_use" が正しく抽出されること。"""
        data = {
            "content": [
                {"type": "text", "text": "応答テキスト"},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "carve_narrative",
                    "input": {"mode": "append", "content": "新しい方針"},
                },
            ]
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        tag = tags[0]
        assert tag["tag_name"] == "CARVE_NARRATIVE"
        assert tag["fields"]["モード"] == "append"
        assert "新しい方針" in tag["fields"]["内容"]

    def test_openai_tool_calls(self):
        """OpenAI形式: choices[].message.tool_calls[].function が正しく抽出されること。

        arguments は JSON 文字列である点に注意。
        """
        args_json = json.dumps({"preset_name": "Gemma4", "self_instruction": "大胆に"})
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "switch_angle",
                                    "arguments": args_json,
                                },
                            }
                        ],
                    }
                }
            ]
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        tag = tags[0]
        assert tag["tag_name"] == "SWITCH_ANGLE"
        assert tag["fields"]["プリセット"] == "Gemma4"

    def test_openai_invalid_arguments_json_gracefully_handled(self):
        """OpenAI形式: arguments が不正 JSON の場合も例外を送出せず処理を続けること。"""
        data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "end_session",
                                    "arguments": "{invalid json",  # 不正 JSON
                                }
                            }
                        ]
                    }
                }
            ]
        }
        text = json.dumps(data)
        # 例外を送出せず、空 args で変換されること
        tags = _extract_function_calls_from_json(text)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "END_SESSION"

    def test_non_json_text_returns_empty_list(self):
        """JSON でないテキスト（テキスト形式タグなど）は空リストを返すこと。"""
        text = "[INSCRIBE_MEMORY:user|1.0|猫が好き]テキスト応答"
        tags = _extract_function_calls_from_json(text)
        assert tags == []

    def test_empty_candidates_returns_empty_list(self):
        """function_call を持たない通常レスポンス JSON は空リストを返すこと。"""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "普通の応答テキスト"}],
                        "role": "model",
                    }
                }
            ]
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)
        assert tags == []

    def test_unknown_tool_name_uses_fallback_meta(self):
        """未知のツール名は tag-unknown メタで変換されること。"""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "mystery_tool",
                                    "args": {"key": "val"},
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        assert tags[0]["tag_name"] == "MYSTERY_TOOL"
        assert tags[0]["meta"]["cls"] == "tag-unknown"

    def test_backslash_newline_in_args_is_parsed(self):
        """JSON 文字列値内のバックスラッシュ+改行（debug_logger 生成パターン）でも正しく抽出できること。

        id:ff8716f7 の実際のレスポンスで発生した、thought_signature フィールドに
        バックスラッシュ+実際改行が含まれるケースを再現する。
        """
        # thought_signature に不正エスケープを含む Gemini レスポンス構造
        raw_json = (
            '{"candidates": [{"content": {"parts": ['
            '{"function_call": {"name": "inscribe_memory",'
            ' "args": {"category": "contextual", "impact": 1.5, "content": "テスト"}},'
            ' "thought_signature": "b\'\\\\x124\\\n2\\\\x01\'"}],'
            ' "role": "model"}}]}'
        )
        tags = _extract_function_calls_from_json(raw_json)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "INSCRIBE_MEMORY"


# ─── _parse_entry: tool-use 複数ラウンドトリップ ─────────────────────────────


class TestParseEntryToolUseRoundTrip:
    """tool-use 複数ラウンドトリップ時の _parse_entry() FIFO ペアリングを検証するテストクラス。

    Gemini 等の tool-use 対応プロバイダーでは以下の 4 ファイルが生成される:
        03_chat_Request_Gemini.log   ← 1 回目リクエスト
        04_chat_Response_Gemini.log  ← function_call を含むレスポンス
        05_chat_Request_Gemini.log   ← ツール結果を含む 2 回目リクエスト
        06_chat_Response_Gemini.log  ← 最終テキスト応答

    旧実装 (dict キー管理) では 03→05、04→06 の上書きが発生し
    04 の function_call が消失していた。新実装 (deque FIFO) で両 Response が独立して保持されること。
    """

    def _make_gemini_function_call_json(self, content: str) -> str:
        """inscribe_memory の function_call を含む Gemini 形式レスポンス JSON を生成するヘルパ。"""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "inscribe_memory",
                                    "args": {
                                        "category": "contextual",
                                        "impact": 1.5,
                                        "content": content,
                                    },
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        return json.dumps(data)

    def test_two_round_trips_create_two_tool_calls(self, tmp_path):
        """Request→Response→Request→Response の 4 ファイルが 2 件の tool_calls になること。

        deque FIFO ペアリングにより:
          - 03_Request → 04_Response （1 回目）
          - 05_Request → 06_Response （2 回目）
        となることを確認する。
        """
        folder = _make_debug_dir(tmp_path, "roundtrip1")
        _write_front_input(folder, "はる@Gemini", "テスト質問")
        _write_response(folder, "03_chat_Request_Gemini.log", "{}")
        _write_response(
            folder,
            "04_chat_Response_Gemini.log",
            self._make_gemini_function_call_json("ドン引き記憶"),
        )
        _write_response(folder, "05_chat_Request_Gemini.log", "{}")
        _write_response(folder, "06_chat_Response_Gemini.log", "最終テキスト応答")
        _write_response(folder, "07_FrontOutput.log", "最終テキスト応答")

        entry = _parse_entry("roundtrip1", folder, {})

        assert len(entry["tool_calls"]) == 2

        tc1 = entry["tool_calls"][0]
        assert tc1["request_file"] == "03_chat_Request_Gemini.log"
        assert tc1["response_file"] == "04_chat_Response_Gemini.log"

        tc2 = entry["tool_calls"][1]
        assert tc2["request_file"] == "05_chat_Request_Gemini.log"
        assert tc2["response_file"] == "06_chat_Response_Gemini.log"

    def test_function_call_json_extracted_as_tags(self, tmp_path):
        """04_Response の JSON function_call が tags として抽出されること。

        deque ペアリングで 04_Response が tc1 に正しく紐付いていること、
        さらに JSON 内の function_call が INSCRIBE_MEMORY タグとして抽出されることを確認する。
        """
        folder = _make_debug_dir(tmp_path, "roundtrip2")
        _write_front_input(folder, "はる@Gemini", "記憶テスト")
        _write_response(folder, "03_chat_Request_Gemini.log", "{}")
        _write_response(
            folder,
            "04_chat_Response_Gemini.log",
            self._make_gemini_function_call_json("テスト用記憶内容"),
        )
        _write_response(folder, "05_chat_Request_Gemini.log", "{}")
        _write_response(folder, "06_chat_Response_Gemini.log", "了解しました")

        entry = _parse_entry("roundtrip2", folder, {})

        # tc1 (04_Response) に INSCRIBE_MEMORY タグが抽出されていること
        tc1 = entry["tool_calls"][0]
        assert len(tc1["tags"]) == 1
        assert tc1["tags"][0]["tag_name"] == "INSCRIBE_MEMORY"
        assert "テスト用記憶内容" in tc1["tags"][0]["fields"]["内容"]

        # tc2 (06_Response) にはタグなし
        tc2 = entry["tool_calls"][1]
        assert tc2["tags"] == []

    def test_second_response_not_lost(self, tmp_path):
        """旧実装のキー衝突バグ再現: 04_Response が 06_Response に上書きされないこと。

        dict キー管理では (chat, Gemini) の dict エントリが 05_Request で上書きされ、
        04_Response への参照が消失していた。FIFO deque ではこれが発生しないことを確認する。
        """
        folder = _make_debug_dir(tmp_path, "roundtrip3")
        _write_front_input(folder, "はる@Gemini", "上書きバグテスト")
        _write_response(folder, "03_chat_Request_Gemini.log", "{}")
        _write_response(
            folder,
            "04_chat_Response_Gemini.log",
            self._make_gemini_function_call_json("消えてはいけない記憶"),
        )
        _write_response(folder, "05_chat_Request_Gemini.log", "{}")
        _write_response(folder, "06_chat_Response_Gemini.log", "最終応答")

        entry = _parse_entry("roundtrip3", folder, {})

        # 04_Response に紐付いた tool_call の tags が空でないこと（消えていないこと）
        tc_with_response_04 = next(
            (tc for tc in entry["tool_calls"] if tc["response_file"] == "04_chat_Response_Gemini.log"),
            None,
        )
        assert tc_with_response_04 is not None, "04_Response に対応する tool_call が存在すること"
        assert len(tc_with_response_04["tags"]) == 1
        assert "消えてはいけない記憶" in tc_with_response_04["tags"][0]["fields"]["内容"]

    def test_orphan_response_without_request(self, tmp_path):
        """対応する Request がない Response（異常ケース）は単独エントリとして追加されること。"""
        folder = _make_debug_dir(tmp_path, "orphan1")
        _write_response(folder, "02_chat_Response_ClaudeCode.log", "孤立したレスポンス")

        entry = _parse_entry("orphan1", folder, {})

        assert len(entry["tool_calls"]) == 1
        tc = entry["tool_calls"][0]
        assert tc["request_file"] is None
        assert tc["response_file"] == "02_chat_Response_ClaudeCode.log"

    def test_three_requests_before_responses_fifo_order(self, tmp_path):
        """3 リクエスト → 3 レスポンスの FIFO 対応が正しく行われること。

        FIFO 順序: req02→resp05, req03→resp06, req04→resp07 となることを確認する。
        """
        folder = _make_debug_dir(tmp_path, "fifo3")
        _write_response(folder, "02_chat_Request_Gemini.log", "{}")
        _write_response(folder, "03_chat_Request_Gemini.log", "{}")
        _write_response(folder, "04_chat_Request_Gemini.log", "{}")
        _write_response(folder, "05_chat_Response_Gemini.log", "応答A")
        _write_response(folder, "06_chat_Response_Gemini.log", "応答B")
        _write_response(folder, "07_chat_Response_Gemini.log", "応答C")

        entry = _parse_entry("fifo3", folder, {})

        assert len(entry["tool_calls"]) == 3
        assert entry["tool_calls"][0]["request_file"] == "02_chat_Request_Gemini.log"
        assert entry["tool_calls"][0]["response_file"] == "05_chat_Response_Gemini.log"
        assert entry["tool_calls"][1]["request_file"] == "03_chat_Request_Gemini.log"
        assert entry["tool_calls"][1]["response_file"] == "06_chat_Response_Gemini.log"
        assert entry["tool_calls"][2]["request_file"] == "04_chat_Request_Gemini.log"
        assert entry["tool_calls"][2]["response_file"] == "07_chat_Response_Gemini.log"
