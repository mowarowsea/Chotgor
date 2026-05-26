"""ログ閲覧UIのルーター。

debug/ ディレクトリ内のリクエスト単位ログフォルダを読み込み、
Web画面で閲覧できるエンドポイントを提供する。
CHOTGOR_DEBUG=1 の時のみ debug/ フォルダが作成されるため、
フォルダが存在しない場合は空のリストを返す。
"""

import html as _html
import json
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from backend.character_actions import tool_tags
from backend.lib.tag_parser import parse_tags

router = APIRouter(prefix="/ui/logs", tags=["logs-ui"])

# debug/ フォルダのベースパス（サーバ起動ディレクトリ基準）
DEBUG_BASE = Path("debug")
# chotgor.log のパス（RotatingFileHandler の primary ファイル）
CHOTGOR_LOG = Path("logs/chotgor.log")

# chotgor.log から char= を抽出するパターン
# 例: "... [a25e00da] ... char=はる@ClaudeCode ..."
_LOG_CHAR_RE = re.compile(r"\[([0-9a-f]{8})\].*?\bchar=(\S+)")

# ファイル名パターン: "02_chat_Request_ClaudeCode.log"
_FILE_PATTERN = re.compile(r"^\d+_(.+?)_(Request|Response)_(.+)\.log$")
# 警告ファイルパターン: "03_Warning_context_window.log"
_FILE_PATTERN_WARNING = re.compile(r"^\d+_Warning_(.+)\.log$")

# タグとして認識する名前一覧（tool_tags.TOOL_TO_TAG.values() と同一セットを維持する）
_KNOWN_TAG_NAMES = list(tool_tags.TOOL_TO_TAG.values())

# debug_logger._unescape_text() が JSON 文字列値内の \\n を実際の改行に展開するため、
# バックスラッシュ直後の改行（\<LF>）が不正 JSON エスケープになる場合がある。
# このパターンで前処理し json.loads(strict=False) に渡す。
_BACKSLASH_NEWLINE_RE = re.compile(r"\\\n")

# タグ名→表示ラベル・バッジ色クラスのマッピング
_TAG_META: dict[str, dict] = {
    "INSCRIBE_MEMORY":  {"label": "記憶",      "cls": "tag-memory"},
    "CARVE_NARRATIVE":  {"label": "ナラティブ", "cls": "tag-narrative"},
    "DRIFT":            {"label": "ドリフト",   "cls": "tag-drift"},
    "DRIFT_RESET":      {"label": "ドリフトリセット", "cls": "tag-drift"},
    "SWITCH_ANGLE":     {"label": "アングル切替", "cls": "tag-switch"},
    "POWER_RECALL":     {"label": "強想起",     "cls": "tag-recall"},
    "END_SESSION":      {"label": "セッション終了", "cls": "tag-end"},
}

# UIテンプレートインスタンス（main.py から注入される）
templates: Optional[Jinja2Templates] = None


def get_templates() -> Jinja2Templates:
    """テンプレートインスタンスを取得する。未初期化の場合は例外を送出する。"""
    if templates is None:
        raise RuntimeError("Templates not initialized")
    return templates


def _parse_json_safely(text: str) -> dict | list:
    """debug_logger が生成した JSON ファイルを安全にパースして返す。

    debug_logger._unescape_text() は JSON 文字列値内の \\n を実際の改行に展開するため、
    バックスラッシュ直後の改行（\\<LF>）が不正エスケープになる場合がある。
    パース前にこのパターンを JSON 改行エスケープ \\n に戻し、
    残る制御文字は strict=False で許容する。
    通常の JSON パースに失敗した場合は NDJSON（各行が独立した JSON オブジェクト、
    ただし複数行またがりあり）として解析して list を返す。
    どちらも失敗した場合は ValueError を送出する。

    Args:
        text: ログファイルのテキスト内容。

    Returns:
        パース結果の dict または list。NDJSON の場合はオブジェクトの list。

    Raises:
        ValueError: JSON・NDJSON どちらのパースにも失敗した場合。
    """
    text_safe = _BACKSLASH_NEWLINE_RE.sub(r"\\n", text)
    try:
        return json.loads(text_safe, strict=False)
    except (json.JSONDecodeError, ValueError):
        pass

    # NDJSON として解析する（LLM出力の生改行で複数行にまたがる場合も対応）
    objects: list = []
    accumulator: list[str] = []
    for line in text_safe.splitlines():
        accumulator.append(line)
        candidate = "\n".join(accumulator)
        try:
            obj = json.loads(candidate, strict=False)
            objects.append(obj)
            accumulator = []
        except (json.JSONDecodeError, ValueError):
            pass
    if objects:
        return objects
    raise ValueError("JSON/NDJSON parse failed")


def _build_char_index() -> dict[str, str]:
    """chotgor.log を走査して {msg_id: char_label} の辞書を返す。

    chronicle / forget などバッチ処理の宛先キャラクター名を解決するために使用する。
    同一 msg_id で複数行マッチした場合は最後の値を採用する（メインの完了ログを優先）。
    ファイルが存在しない・読み取り不可の場合は空辞書を返す。

    Returns:
        {8桁hex文字列: "キャラクター名@プリセット名"} の辞書。
    """
    if not CHOTGOR_LOG.exists():
        return {}
    index: dict[str, str] = {}
    try:
        for line in CHOTGOR_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _LOG_CHAR_RE.search(line)
            if m:
                index[m.group(1)] = m.group(2)
    except Exception:
        pass
    return index


def _parse_tag_body(tag_name: str, body: str) -> dict:
    """タグ本体文字列を構造化辞書に変換する。

    タグ種別ごとに `|` 区切りのフィールドを解釈し、表示用のサブフィールドを返す。

    Args:
        tag_name: タグ名 (例: "INSCRIBE_MEMORY")。
        body: タグ本体テキスト (コロン以降 ']' を除いた部分)。

    Returns:
        表示用フィールド辞書。tag_name / meta / fields / preview を含む。
    """
    meta = _TAG_META.get(tag_name, {"label": tag_name, "cls": "tag-unknown"})
    fields: dict[str, str] = {}

    if tag_name == "INSCRIBE_MEMORY":
        # body: "category|importance|content"
        parts = body.split("|", 2)
        fields["カテゴリ"] = parts[0] if len(parts) > 0 else ""
        fields["重要度"] = parts[1] if len(parts) > 1 else ""
        fields["内容"] = parts[2] if len(parts) > 2 else ""

    elif tag_name == "CARVE_NARRATIVE":
        # body: "mode|content"
        parts = body.split("|", 1)
        fields["モード"] = parts[0] if len(parts) > 0 else ""
        fields["内容"] = parts[1] if len(parts) > 1 else ""

    elif tag_name == "SWITCH_ANGLE":
        # body: "preset|context"
        parts = body.split("|", 1)
        fields["プリセット"] = parts[0] if len(parts) > 0 else ""
        fields["コンテキスト"] = parts[1] if len(parts) > 1 else ""

    elif tag_name in ("DRIFT", "POWER_RECALL", "END_SESSION"):
        # body がそのまま内容
        fields["内容"] = body

    elif tag_name in ("DRIFT_RESET",):
        # 固定マーカー、body は空
        fields["内容"] = "(リセット)"

    # preview: "内容" → SWITCH_ANGLE は "コンテキスト" → 最終フォールバックは body
    preview = fields.get("内容") or fields.get("コンテキスト") or body

    return {
        "tag_name": tag_name,
        "meta": meta,
        "fields": fields,
        "preview": preview,
    }


def _collect_tool_calls_from_single_json(data: dict) -> list[tuple[str, dict]]:
    """単一JSONオブジェクト（Anthropic/OpenAI/Gemini形式）からツール呼び出しを収集する。

    Args:
        data: パース済みのJSONオブジェクト。

    Returns:
        (tool_name, args_dict) のリスト。
    """
    if not isinstance(data, dict):
        return []
    calls: list[tuple[str, dict]] = []

    # Gemini形式: candidates[].content.parts[].function_call
    # 注: promptFeedback でブロックされた場合など candidates が null で返ることがあるため
    # `get(..., [])` のデフォルト値は効かず明示的に `or []` でフォールバックする。
    for candidate in (data.get("candidates") or []):
        for part in ((candidate.get("content") or {}).get("parts") or []):
            fc = part.get("function_call")
            if fc and isinstance(fc, dict) and "name" in fc:
                calls.append((fc["name"], fc.get("args", {}) or {}))

    # Anthropic形式: content[].type == "tool_use"（input キー）
    content = data.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append((block.get("name", ""), block.get("input", {}) or {}))

    # OpenAI形式: choices[].message.tool_calls[].function（arguments は JSON 文字列）
    for choice in (data.get("choices") or []):
        for tc in ((choice.get("message") or {}).get("tool_calls") or []):
            fc = tc.get("function", {}) or {}
            try:
                args = json.loads(fc.get("arguments", "{}"))
            except (json.JSONDecodeError, ValueError):
                args = {}
            calls.append((fc.get("name", ""), args))

    return calls


def _collect_tool_calls_from_stream_json(text: str) -> list[tuple[str, dict]]:
    """stream-json（NDJSON）形式のClaude CLI出力からtool_useブロックを収集する。

    Claude CLI が --output-format stream-json --verbose で出力する形式では、
    各行が独立したJSONイベントになる。tool_use ブロックは assistant イベントの
    message.content 配列内に含まれる。

    Args:
        text: stream-json形式のテキスト（1行1JSONイベント）。

    Returns:
        (tool_name, args_dict) のリスト。
    """
    calls: list[tuple[str, dict]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line, strict=False)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") != "assistant":
            continue
        for block in ((event.get("message") or {}).get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append((block.get("name", ""), block.get("input", {}) or {}))
    return calls


def _extract_function_calls_from_json(text: str) -> list[dict]:
    """JSONレスポンスファイルからfunction_call（ツール呼び出し）をタグとして抽出する。

    Gemini、Anthropic、OpenAI等のAPIレスポンスJSON内の function_call / tool_use エントリを
    検出し、タグ構造体に変換して返す。
    Claude CLI が出力する stream-json（NDJSON）形式にも対応する。

    対応フォーマット:
        - Gemini: candidates[].content.parts[].function_call
        - Anthropic: content[].type == "tool_use"（input キー）
        - OpenAI: choices[].message.tool_calls[].function（arguments は JSON 文字列）
        - Claude CLI stream-json: assistant イベントの message.content[].type == "tool_use"

    Args:
        text: ログファイルのテキスト内容。

    Returns:
        タグ構造体のリスト。該当がなければ空リスト。
    """
    function_calls: list[tuple[str, dict]] = []
    try:
        data = _parse_json_safely(text)
        function_calls = _collect_tool_calls_from_single_json(data)
    except Exception:
        pass

    # シングルJSONでツール呼び出しが見つからなかった場合は stream-json（NDJSON）として試みる。
    # 単一行 NDJSON が正常にパースできても Claude CLI イベント形式の場合はここで抽出する。
    if not function_calls:
        try:
            function_calls = _collect_tool_calls_from_stream_json(text)
        except Exception:
            pass

    results = []
    for name, args in function_calls:
        if not name:
            continue
        tag_name, body = tool_tags.tool_call_to_tag_body(name, args)
        results.append(_parse_tag_body(tag_name, body))
    return results


def _extract_tags_from_file(file_path: Path) -> list[dict]:
    """ファイルからタグ / ツール呼び出しを抽出して構造化リストで返す。

    抽出の優先順位:
      1. JSON / stream-json 内のツール呼び出し（tool-use対応プロバイダー）
      2. テキスト内の [INSCRIBE_MEMORY:...] 等のタグ（タグ方式プロバイダーへのフォールバック）

    tool-use プロバイダー（Anthropic / OpenAI / Claude CLI 等）はレスポンスに
    tool_use ブロックを含むため、それを優先表示する。
    タグ方式プロバイダー（Ollama / OpenRouter 等）はツール呼び出しが存在しないため、
    テキスト内のタグをフォールバックとして抽出する。

    Args:
        file_path: ログファイルのパス。

    Returns:
        タグ構造化辞書のリスト。
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []

    # ツール呼び出し（JSON / stream-json）を優先して試みる
    tool_call_tags = _extract_function_calls_from_json(text)
    if tool_call_tags:
        return tool_call_tags

    # ツール呼び出しが見つからなければタグ方式（Ollama等）としてフォールバック
    try:
        _, matches = parse_tags(text, _KNOWN_TAG_NAMES, multiline=True)
    except Exception:
        return []

    # 全タグを (start位置, tag_name, body) フラットリストにしてから位置順にソートする。
    # タグ種別ごとに固める（INSCRIBE→DRIFT→...）のではなく、
    # ファイル内の出現順（DRIFT→INSCRIBE のような順序）を保持するため。
    flat: list[tuple[int, str, str]] = []
    for tag_name in _KNOWN_TAG_NAMES:
        for m in matches.get(tag_name, []):
            flat.append((m.start, tag_name, m.body))
    flat.sort(key=lambda x: x[0])

    return [_parse_tag_body(tn, body) for _, tn, body in flat]


def _tc_label(tc: dict) -> str:
    """tool_call を人間可読なラベル（feature/preset）に整形する。

    warn_reason の組み立てなど、ログ表示用の識別名として使う。

    Args:
        tc: tool_call 辞書（feature / preset キーを持つ）。

    Returns:
        "feature/preset" 形式の文字列。どちらか欠ける場合は存在する方を、
        いずれも無ければ "不明な処理" を返す。
    """
    feature = tc.get("feature") or ""
    preset = tc.get("preset") or ""
    if feature and preset:
        return f"{feature}/{preset}"
    return feature or preset or "不明な処理"


def _parse_entry(msg_id: str, folder: Path, char_index: dict[str, str]) -> dict:
    """1リクエストフォルダの内容を解析してログエントリ辞書を返す。

    Args:
        msg_id: リクエストID（フォルダ名）。
        folder: フォルダのPathオブジェクト。
        char_index: _build_char_index() で構築した {msg_id: char_label} 辞書。
                    FrontInput がない batch エントリの宛先解決に使用する。

    Returns:
        ログエントリ辞書。以下のキーを含む:
            message_id, dt, dt_str, character, preset, model_id,
            source, user_message, character_response, reasoning_text,
            tool_calls, warnings, files, has_error, warn_reason
    """
    try:
        dt = datetime.fromtimestamp(folder.stat().st_mtime)
    except OSError:
        dt = datetime(1970, 1, 1)

    files = sorted(folder.iterdir(), key=lambda p: p.name)
    file_names = [p.name for p in files]

    # --- FrontInput を解析 ---
    character = ""
    preset = ""
    model_id = ""
    user_message = ""
    source = "system"
    front_input_path = folder / "01_FrontInput.log"
    if front_input_path.exists():
        try:
            raw = front_input_path.read_text(encoding="utf-8")
            data = _parse_json_safely(raw)
            model_id = data.get("model_id", "")
            if "@" in model_id:
                character, preset = model_id.rsplit("@", 1)
            else:
                character = model_id
            user_message = data.get("content", "")
            source = "ユーザ"
        except Exception:
            pass

    # --- FrontInput がない場合（batch 処理）は chotgor.log インデックスで補完 ---
    if not model_id and msg_id in char_index:
        model_id = char_index[msg_id]
        if "@" in model_id:
            character, preset = model_id.rsplit("@", 1)
        else:
            character = model_id

    # --- FrontOutput を探す ---
    character_response = ""
    for f in files:
        if "FrontOutput" in f.name:
            try:
                character_response = f.read_text(encoding="utf-8")
            except Exception:
                pass
            break

    # --- Reasoning ログを探す（翻訳ボタン用） ---
    reasoning_text = ""
    for f in files:
        if "Reasoning" in f.name:
            try:
                reasoning_text = f.read_text(encoding="utf-8")
            except Exception:
                pass
            break

    # --- ツール呼び出しを Request/Response ファイルペアから構築 ---
    # 同一 feature/preset でも複数ラウンドトリップが発生する（tool-use時）ため、
    # dict でのキー管理はせず FIFO キューで「先に来た Request に先の Response を対応付ける」。
    tool_calls: list[dict] = []
    # (feature, preset) → 未マッチ Request エントリのキュー（到着順）
    pending: dict[tuple[str, str], deque[dict]] = {}

    for f in files:
        m = _FILE_PATTERN.match(f.name)
        if not m:
            continue
        feature, kind, preset_name = m.group(1), m.group(2), m.group(3)

        pair_key = (feature, preset_name)
        if kind == "Request":
            entry = {"feature": feature, "preset": preset_name,
                     "request_file": f.name, "response_file": None, "tags": []}
            tool_calls.append(entry)
            pending.setdefault(pair_key, deque()).append(entry)
        else:  # Response
            queue = pending.get(pair_key)
            if queue:
                queue.popleft()["response_file"] = f.name
                if not queue:
                    del pending[pair_key]
            else:
                # 対応する Request がない Response（異常ケース）
                tool_calls.append({"feature": feature, "preset": preset_name,
                                   "request_file": None, "response_file": f.name, "tags": []})

    # Response ファイルからタグを抽出して tool_calls に追加
    for tc in tool_calls:
        if tc["response_file"]:
            tc["tags"] = _extract_tags_from_file(folder / tc["response_file"])

    # --- Warning ファイルを収集 ---
    # log_warning() が書き出した {NN}_Warning_{tag}.log を読み込む。
    warnings: list[dict] = []
    for f in files:
        m = _FILE_PATTERN_WARNING.match(f.name)
        if m:
            tag = m.group(1)
            try:
                msg = f.read_text(encoding="utf-8").strip()
            except Exception:
                msg = ""
            warnings.append({"tag": tag, "message": msg, "file": f.name})

    # --- 警告（WARN）判定 ---
    # いずれかの tool_call で Response が存在しない、または Response がエラー文字列の場合は
    # 警告扱いとし、その理由を warn_reason に人間可読な文言で残す（Logs 画面で表示）。
    has_error = False
    warn_reason = ""
    for tc in tool_calls:
        if tc["request_file"] and not tc["response_file"]:
            # Request はあるが Response が存在しない = エラーで中断したと判断
            has_error = True
            warn_reason = f"{_tc_label(tc)} の応答が生成されず、処理が途中で中断しました"
            break
        if tc["response_file"]:
            # Response の先頭がエラー文字列パターンに一致する場合
            resp_path = folder / tc["response_file"]
            try:
                head = resp_path.read_text(encoding="utf-8", errors="replace")[:120]
                if head.lstrip().startswith("[") and ("error" in head.lower() or "Error" in head):
                    has_error = True
                    warn_reason = (
                        f"{_tc_label(tc)} の応答がエラーを返しました"
                        f"（{tc['response_file']} を参照）"
                    )
                    break
            except Exception:
                pass

    return {
        "message_id": msg_id,
        "dt": dt,
        "dt_str": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "character": character,
        "preset": preset,
        "model_id": model_id,
        "source": source,
        "user_message": user_message,
        "character_response": character_response,
        "reasoning_text": reasoning_text,
        "tool_calls": tool_calls,
        "warnings": warnings,
        "files": file_names,
        "has_error": has_error,
        "warn_reason": warn_reason,
    }


def _sqlite_store():
    """FastAPI app.state から SQLiteStore を取得する。

    logs_ui はルーターなので app.state に直接アクセスできないため、
    グローバル変数 `_store` に起動時にセットする方式を取る。
    セットされていない場合（テスト時等）は None を返す。
    """
    return _store


# main.py の lifespan から呼んで SQLiteStore をセットする
_store = None


def set_sqlite_store(sqlite) -> None:
    """SQLiteStore をセットする。main.py の lifespan から起動時に呼ぶ。

    Args:
        sqlite: SQLiteStore インスタンス。
    """
    global _store
    _store = sqlite


def _build_attempt_detail(r: dict, index: int, char_index: dict) -> dict:
    """DB の1行（1試行）から試行アコーディオン表示用の詳細データを構築する。

    raw_dir が存在すればファイルを読み込んでツール呼び出し・警告・ファイル一覧を
    組み立てる。raw_dir がない場合（CHOTGOR_DEBUG=0 等）は DB の値のみで構成する。

    Args:
        r: get_debug_log_entries_by_request_id() が返す1行の辞書。
        index: 試行番号（0始まり）。表示は +1 して使う。
        char_index: chotgor.log から構築した {msg_id: char_label} 辞書。

    Returns:
        試行アコーディオンが期待するデータ辞書。
    """
    raw_dir = r.get("raw_dir") or ""
    dir_id = ""
    tool_calls: list[dict] = []
    warnings: list[dict] = []
    file_names: list[str] = []
    has_error = bool(r.get("has_error"))
    warn_reason = r.get("warn_reason") or ""

    if raw_dir:
        raw_path = Path(raw_dir)
        dir_id = raw_path.name  # フォルダ名がファイルリンクに使う ID
        if raw_path.exists():
            parsed = _parse_entry(dir_id, raw_path, char_index)
            tool_calls = parsed.get("tool_calls", [])
            warnings_parsed = parsed.get("warnings", [])
            if warnings_parsed:
                warnings = warnings_parsed
            if not has_error:
                has_error = parsed.get("has_error", False)
            if not warn_reason:
                warn_reason = parsed.get("warn_reason", "")
            file_names = sorted(p.name for p in raw_path.iterdir() if p.is_file())

    return {
        "index": index + 1,
        "preset": r.get("preset") or "",
        "response": r.get("response") or "",
        "reasoning": r.get("reasoning") or "",
        "dt_str": r["created_at"].strftime("%H:%M:%S") if r.get("created_at") else "",
        "has_error": has_error,
        "warn_reason": warn_reason,
        "tool_calls": tool_calls,
        "warnings": warnings,
        "files": file_names,
        "dir_id": dir_id,
    }


def _build_entry_from_db_rows(rows: list[dict]) -> dict:
    """同一 request_id の DB 行リストからログ一覧表示用エントリを組み立てる。

    メイン行（chat/scenario 等）は試行ごとに1行存在し、`attempts` リストに
    完全な詳細データを格納する。再生成で複数試行ある場合は attempt_count > 1 になる。
    非メイン行（farewell/trigger/chronicle 等）は tool_calls を top-level で保持する。

    Args:
        rows: get_debug_log_entries_by_request_id() が返すエントリ辞書のリスト（昇順）。

    Returns:
        ログ一覧 UI が期待するエントリ辞書。
    """
    if not rows:
        return {}

    _MAIN_SOURCE_TYPES = {"chat", "scenario", "scenario_chat", "group_chat"}

    main_row = next((r for r in rows if r["source_type"] in _MAIN_SOURCE_TYPES), rows[0])
    main_rows = [r for r in rows if r["source_type"] in _MAIN_SOURCE_TYPES]
    latest_main = main_rows[-1] if main_rows else rows[-1]

    request_id = main_row["request_id"]
    source_type = main_row["source_type"]
    dt = latest_main["created_at"] or main_row["created_at"]

    seen: set[str] = set()
    source_types: list[str] = []
    for r in rows:
        if r["source_type"] not in seen:
            seen.add(r["source_type"])
            source_types.append(r["source_type"])

    has_error = any(r["has_error"] for r in rows)
    warn_reason = next((r["warn_reason"] for r in rows if r["warn_reason"]), "")
    target = main_row.get("target") or latest_main.get("target") or ""
    preset = latest_main.get("preset") or main_row.get("preset") or ""
    model_id = f"{target}@{preset}" if target and preset else target or preset

    # char_index はファイルが存在する場合のみ構築（コスト節約）
    needs_files = any(r.get("raw_dir") for r in rows)
    char_index = _build_char_index() if needs_files else {}

    # メイン行を試行ごとに詳細データへ変換（ツール呼び出し・ファイル含む）
    attempts = [
        _build_attempt_detail(r, i, char_index) for i, r in enumerate(main_rows)
    ]

    # 非メイン行（chronicle/forget 等）向けの top-level tool_calls（旧来互換）
    top_tool_calls: list[dict] = []
    top_warnings: list[dict] = []
    top_file_names: list[str] = []
    if not main_rows:
        # メイン行が無いエントリは従来通り最初の行の raw_dir を使う
        raw_dir = rows[0].get("raw_dir") or ""
        if raw_dir:
            raw_path = Path(raw_dir)
            dir_id = raw_path.name
            if raw_path.exists():
                parsed = _parse_entry(dir_id, raw_path, char_index)
                top_tool_calls = parsed.get("tool_calls", [])
                top_warnings = parsed.get("warnings", [])
                if not has_error:
                    has_error = parsed.get("has_error", False)
                if not warn_reason:
                    warn_reason = parsed.get("warn_reason", "")
                top_file_names = sorted(
                    p.name for p in raw_path.iterdir() if p.is_file()
                )

    # top-level の dir_id（旧来の ID/Files 表示と JSON API 用）
    top_raw_dir = latest_main.get("raw_dir") or main_row.get("raw_dir") or ""
    top_dir_id = Path(top_raw_dir).name if top_raw_dir else request_id

    return {
        "message_id": request_id,
        "dt": dt,
        "dt_str": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "",
        "character": target,
        "preset": preset,
        "model_id": model_id,
        "source_type": source_type,
        "source_types": source_types,
        "source": "system" if source_type not in _MAIN_SOURCE_TYPES else "ユーザ",
        "user_message": main_row.get("user_message") or "",
        # 後方互換（JSON API・サマリー表示用）
        "character_response": latest_main.get("response") or "",
        "reasoning_text": latest_main.get("reasoning") or "",
        # 非メイン行エントリ向け（attempts が空の場合のみ HTML で使う）
        "tool_calls": top_tool_calls,
        "warnings": top_warnings,
        "files": top_file_names,
        "dir_id": top_dir_id,
        "has_error": has_error,
        "warn_reason": warn_reason,
        "attempt_count": len(main_rows),
        "attempts": attempts,
    }


def _load_entries(page: int = 1, per_page: int = 50) -> tuple[list[dict], int]:
    """デバッグログエントリを読み込んでページネーションして返す。

    SQLiteStore が利用可能な場合は DB から読み込む。
    未セット（テスト・起動直後等）の場合はファイルシステムにフォールバックする。

    Args:
        page: ページ番号（1始まり）。
        per_page: 1ページあたりの件数。

    Returns:
        (エントリリスト, 総件数) のタプル。
    """
    sqlite = _sqlite_store()
    if sqlite is not None:
        return _load_entries_from_db(sqlite, page, per_page)
    return _load_entries_from_files(page, per_page)


def _load_entries_from_db(sqlite, page: int, per_page: int) -> tuple[list[dict], int]:
    """DB からログエントリをページネーションして返す。

    Args:
        sqlite: SQLiteStore インスタンス。
        page: ページ番号（1始まり）。
        per_page: 1ページあたりの件数。

    Returns:
        (エントリリスト, 総件数) のタプル。
    """
    request_ids, total = sqlite.get_debug_log_request_ids_paged(page=page, per_page=per_page)
    entries = []
    for req_id in request_ids:
        rows = sqlite.get_debug_log_entries_by_request_id(req_id)
        if rows:
            entries.append(_build_entry_from_db_rows(rows))
    return entries, total


def _load_entries_from_files(page: int, per_page: int) -> tuple[list[dict], int]:
    """ファイルシステムからログエントリを読み込む（フォールバック）。

    DB 未セット時（テスト等）に使用する旧来の実装。

    Args:
        page: ページ番号（1始まり）。
        per_page: 1ページあたりの件数。

    Returns:
        (エントリリスト, 総件数) のタプル。
    """
    if not DEBUG_BASE.exists():
        return [], 0

    folders = []
    for item in DEBUG_BASE.iterdir():
        if item.is_dir() and item.name != "trush":
            try:
                mtime = item.stat().st_mtime
                folders.append((mtime, item.name, item))
            except OSError:
                pass
    folders.sort(reverse=True, key=lambda x: x[0])

    total = len(folders)
    start = (page - 1) * per_page
    end = start + per_page
    page_folders = folders[start:end]

    char_index = _build_char_index()
    entries = [_parse_entry(name, folder, char_index) for _, name, folder in page_folders]
    return entries, total


@router.get("/", response_class=HTMLResponse)
async def logs_list(request: Request, page: int = 1):
    """ログ一覧ページを返す。

    debug/ フォルダ内のリクエスト単位ログをページネーションして表示する。

    Args:
        request: FastAPIリクエストオブジェクト。
        page: ページ番号（デフォルト1）。
    """
    per_page = 50
    entries, total = _load_entries(page=page, per_page=per_page)
    total_pages = max(1, (total + per_page - 1) // per_page)
    debug_enabled = DEBUG_BASE.exists()

    return get_templates().TemplateResponse(
        "logs.html",
        {
            "request": request,
            "entries": entries,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "debug_enabled": debug_enabled,
        },
    )


@router.get("/{message_id}/raw/{filename}", response_class=PlainTextResponse)
async def log_raw_file(message_id: str, filename: str):
    """指定リクエストの生ログファイル内容をプレーンテキストで返す。

    ディレクトリトラバーサルを防ぐため、ファイル名にパス区切り文字が含まれる場合は拒否する。

    Args:
        message_id: リクエストID（フォルダ名）。
        filename: ログファイル名。
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        return PlainTextResponse("Invalid filename", status_code=400)
    if ".." in message_id or "/" in message_id or "\\" in message_id:
        return PlainTextResponse("Invalid message_id", status_code=400)

    file_path = DEBUG_BASE / message_id / filename
    if not file_path.exists() or not file_path.is_file():
        return PlainTextResponse("File not found", status_code=404)

    try:
        content = file_path.read_text(encoding="utf-8")
        return PlainTextResponse(content, media_type="text/plain; charset=utf-8")
    except Exception:
        return PlainTextResponse("Read error", status_code=500)


# JSON ビュー表示でハイライトするキー名のセット（Claude CLI 向けキーも含む）
_HIGHLIGHT_JSON_KEYS = {"text", "content", "thought", "system_prompt", "conversation", "thinking", "result", "system_instruction", "context", "query"}

# JSON 整形ビュー用: key-value 行マッチパターン（indent=2 の json.dumps 前提）
_JSON_KV_LINE_RE = re.compile(r'^(\s*)"([^"]+)"(\s*:\s*)(.+)$')


def _render_json_html(raw_text: str) -> str:
    """JSON ログファイルをハイライト付き HTML 文字列にレンダリングする。

    json.dumps(indent=2) で整形した後、各行を走査し
    _HIGHLIGHT_JSON_KEYS に含まれるキーの文字列値（null でないもの）を
    <mark class="jv"> でハイライトする。
    JSON パース失敗時はプレーンテキストを HTML エスケープして返す。

    Args:
        raw_text: ログファイルのテキスト内容。

    Returns:
        HTML スニペット文字列（<pre> の中身）。
    """
    try:
        data = _parse_json_safely(raw_text)
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return _html.escape(raw_text)

    lines_html: list[str] = []
    for line in pretty.splitlines():
        m = _JSON_KV_LINE_RE.match(line)
        if m:
            indent, key, sep, value_part = m.group(1), m.group(2), m.group(3), m.group(4)
            if key in _HIGHLIGHT_JSON_KEYS:
                # 末尾カンマを含む場合も正しく処理する
                trailing = ","  if value_part.endswith(",") else ""
                stripped = value_part.rstrip(",")
                # 文字列値（ダブルクォートで囲まれ、null でない）のみハイライト
                if stripped.startswith('"') and stripped != '"null"' and len(stripped) > 2:
                    # JSON文字列内部のエスケープを展開して改行・タブを実際の文字に変換する
                    inner = stripped[1:-1]
                    inner = (inner
                        .replace('\\\\', '\x00BACKSLASH\x00')
                        .replace('\\"', '"')
                        .replace('\\n', '\n')
                        .replace('\\t', '\t')
                        .replace('\\r', '\r')
                        .replace('\x00BACKSLASH\x00', '\\'))
                    lines_html.append(
                        _html.escape(indent)
                        + f'<span class="jk">{_html.escape(chr(34) + key + chr(34))}</span>'
                        + _html.escape(sep)
                        + f'<mark class="jv">{_html.escape(inner)}</mark>'
                        + _html.escape(trailing)
                    )
                    continue
        lines_html.append(_html.escape(line))

    return "\n".join(lines_html)


@router.get("/{message_id}/view/{filename}", response_class=HTMLResponse)
async def log_view_file(message_id: str, filename: str):
    """指定ログファイルを整形・ハイライト付き HTML ページで返す。

    JSON ファイルは json.dumps(indent=2) で整形し、
    candidates.content.parts[].text / function_call.args.content 等の
    文字列フィールドを <mark> でハイライト表示する。
    JSON 以外のファイルはプレーンテキストとして表示する。

    Args:
        message_id: リクエストID（フォルダ名）。
        filename: ログファイル名。
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        return HTMLResponse("Invalid filename", status_code=400)
    if ".." in message_id or "/" in message_id or "\\" in message_id:
        return HTMLResponse("Invalid message_id", status_code=400)

    file_path = DEBUG_BASE / message_id / filename
    if not file_path.exists() or not file_path.is_file():
        return HTMLResponse("File not found", status_code=404)

    try:
        raw = file_path.read_text(encoding="utf-8")
    except Exception:
        return HTMLResponse("Read error", status_code=500)

    body_html = _render_json_html(raw)

    page = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{_html.escape(filename)}</title>
<style>
  body {{ margin: 0; background: #1a1a2e; color: #c9d1d9; font-family: 'Consolas','Monaco',monospace; font-size: 13px; }}
  pre {{ padding: 1.2rem 1.6rem; margin: 0; white-space: pre-wrap; word-break: break-all; line-height: 1.6; }}
  h1 {{ font-size: 0.85rem; color: #8b949e; margin: 0; padding: 0.6rem 1.6rem; border-bottom: 1px solid #30363d; background: #161b22; }}
  .jk {{ color: #79c0ff; }}
  mark.jv {{ background: #2d3a1e; color: #a5d6a7; border-radius: 3px; padding: 0 2px; }}
</style>
</head>
<body>
<h1>{_html.escape(message_id)} / {_html.escape(filename)}</h1>
<pre>{body_html}</pre>
</body>
</html>"""

    return HTMLResponse(page)


# --- フロントエンド向け JSON API ルーター ---
json_router = APIRouter(prefix="/api/logs", tags=["logs-json"])


@json_router.get("/entry/{log_message_id}")
async def get_log_entry(log_message_id: str):
    """指定した log_message_id のデバッグログエントリを JSON で返す。

    フロントエンドのチャットバブルからログを取得するために使用する。
    CHOTGOR_DEBUG=1 が設定されていない場合、または該当フォルダが存在しない場合は 404 を返す。

    Args:
        log_message_id: debug フォルダ名（8桁 hex）。
    """
    # パストラバーサル防止: 8桁英数字のみ許可
    if not re.fullmatch(r"[0-9a-f]{8}", log_message_id):
        raise HTTPException(status_code=400, detail="不正なlog_message_idです")

    # DB から取得を試みる
    sqlite = _sqlite_store()
    if sqlite is not None:
        rows = sqlite.get_debug_log_entries_by_request_id(log_message_id)
        if rows:
            entry = _build_entry_from_db_rows(rows)
            entry_json = {k: v for k, v in entry.items() if k != "dt"}
            return {"entry": entry_json, "debug_enabled": True}

    # DB にない場合はファイルにフォールバック
    folder = DEBUG_BASE / log_message_id
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail="ログエントリが見つかりません")
    char_index = _build_char_index()
    entry = _parse_entry(log_message_id, folder, char_index)
    entry_json = {k: v for k, v in entry.items() if k != "dt"}
    return {"entry": entry_json, "debug_enabled": True}

