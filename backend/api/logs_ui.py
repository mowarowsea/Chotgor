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

from fastapi import APIRouter, Request
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

# ファイル名パターン（機能名あり）: "02_chat_Request_ClaudeCode.log"
_FILE_PATTERN = re.compile(r"^\d+_(.+?)_(Request|Response)_(.+)\.log$")
# ファイル名パターン（機能名なし）: "02_Request_claude_cli.log"（旧形式）
_FILE_PATTERN_NOFEATURE = re.compile(r"^\d+_(Request|Response)_(.+)\.log$")
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
    パース失敗時は ValueError を送出する。

    Args:
        text: ログファイルのテキスト内容。

    Returns:
        パース結果の dict または list。

    Raises:
        ValueError: JSON パースに失敗した場合。
    """
    text_safe = _BACKSLASH_NEWLINE_RE.sub(r"\\n", text)
    return json.loads(text_safe, strict=False)


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


def _extract_function_calls_from_json(text: str) -> list[dict]:
    """JSONレスポンスファイルからfunction_call（ツール呼び出し）をタグとして抽出する。

    Gemini、Anthropic、OpenAI等のAPIレスポンスJSON内の function_call / tool_use エントリを
    検出し、タグ構造体に変換して返す。JSON 以外のファイルは無視して空リストを返す。

    対応フォーマット:
        - Gemini: candidates[].content.parts[].function_call
        - Anthropic: content[].type == "tool_use"（input キー）
        - OpenAI: choices[].message.tool_calls[].function（arguments は JSON 文字列）

    Args:
        text: ログファイルのテキスト内容。

    Returns:
        タグ構造体のリスト。該当がなければ空リスト。
    """
    try:
        data = _parse_json_safely(text)
    except (json.JSONDecodeError, ValueError):
        return []

    function_calls: list[tuple[str, dict]] = []

    # Gemini形式: candidates[].content.parts[].function_call
    # 注: promptFeedback でブロックされた場合など candidates が null で返ることがあるため
    # `get(..., [])` のデフォルト値は効かず明示的に `or []` でフォールバックする。
    for candidate in (data.get("candidates") or []):
        for part in ((candidate.get("content") or {}).get("parts") or []):
            fc = part.get("function_call")
            if fc and isinstance(fc, dict) and "name" in fc:
                function_calls.append((fc["name"], fc.get("args", {}) or {}))

    # Anthropic形式: content[].type == "tool_use"（input キー）
    content = data.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                function_calls.append((block.get("name", ""), block.get("input", {}) or {}))

    # OpenAI形式: choices[].message.tool_calls[].function（arguments は JSON 文字列）
    for choice in (data.get("choices") or []):
        for tc in ((choice.get("message") or {}).get("tool_calls") or []):
            fc = tc.get("function", {}) or {}
            try:
                args = json.loads(fc.get("arguments", "{}"))
            except (json.JSONDecodeError, ValueError):
                args = {}
            function_calls.append((fc.get("name", ""), args))

    results = []
    for name, args in function_calls:
        if not name:
            continue
        tag_name, body = tool_tags.tool_call_to_tag_body(name, args)
        results.append(_parse_tag_body(tag_name, body))
    return results


def _extract_tags_from_file(file_path: Path) -> list[dict]:
    """ファイルからタグを抽出して構造化リストで返す。

    `parse_tags` を使い INSCRIBE_MEMORY / CARVE_NARRATIVE 等を検出する。
    ファイル読み込みエラーや解析エラーは無視して空リストを返す。

    Args:
        file_path: ログファイルのパス。

    Returns:
        タグ構造化辞書のリスト。
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
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

    tags = [_parse_tag_body(tn, body) for _, tn, body in flat]
    # JSON形式のレスポンスファイルからfunction_callも抽出する（tool-use対応プロバイダー向け）
    tags.extend(_extract_function_calls_from_json(text))
    return tags


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
            source, user_message, character_response,
            tool_calls, files
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
        # 機能名あり: "02_chat_Request_ClaudeCode.log"
        m = _FILE_PATTERN.match(f.name)
        if m:
            feature, kind, preset_name = m.group(1), m.group(2), m.group(3)
        else:
            # 機能名なし（旧形式）: "02_Request_claude_cli.log"
            m2 = _FILE_PATTERN_NOFEATURE.match(f.name)
            if not m2:
                continue
            kind, preset_name = m2.group(1), m2.group(2)
            feature = ""

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

    # --- エラー判定 ---
    # いずれかの tool_call で Response が存在しない、または Response がエラー文字列の場合は警告扱い
    has_error = False
    for tc in tool_calls:
        if tc["request_file"] and not tc["response_file"]:
            # Request はあるが Response が存在しない = エラーで中断したと判断
            has_error = True
            break
        if tc["response_file"]:
            # Response の先頭がエラー文字列パターンに一致する場合
            resp_path = folder / tc["response_file"]
            try:
                head = resp_path.read_text(encoding="utf-8", errors="replace")[:120]
                if head.lstrip().startswith("[") and ("error" in head.lower() or "Error" in head):
                    has_error = True
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
    }


def _load_entries(page: int = 1, per_page: int = 50) -> tuple[list[dict], int]:
    """debug/ ディレクトリからログエントリを読み込んでページネーションして返す。

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

    # chotgor.log インデックスは1回だけ構築して全エントリで共有する
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


# JSON ビュー表示でハイライトするキー名のセット
_HIGHLIGHT_JSON_KEYS = {"text", "content", "thought"}

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
                    lines_html.append(
                        _html.escape(indent)
                        + f'<span class="jk">{_html.escape(chr(34) + key + chr(34))}</span>'
                        + _html.escape(sep)
                        + f'<mark class="jv">{_html.escape(stripped)}</mark>'
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
    from fastapi import HTTPException
    # パストラバーサル防止: 8桁英数字のみ許可
    if not re.fullmatch(r"[0-9a-f]{8}", log_message_id):
        raise HTTPException(status_code=400, detail="不正なlog_message_idです")
    folder = DEBUG_BASE / log_message_id
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail="ログエントリが見つかりません")
    char_index = _build_char_index()
    entry = _parse_entry(log_message_id, folder, char_index)
    # datetime オブジェクト ("dt" キー) は JSON シリアライズ不可なため除外する
    entry_json = {k: v for k, v in entry.items() if k != "dt"}
    return {"entry": entry_json, "debug_enabled": True}
