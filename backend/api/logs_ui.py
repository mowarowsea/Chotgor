"""ログ閲覧UIのルーター。

debug/ ディレクトリ内のリクエスト単位ログフォルダを読み込み、
Web画面で閲覧できるエンドポイントを提供する。
CHOTGOR_DEBUG=1 の時のみ debug/ フォルダが作成されるため、
フォルダが存在しない場合は空のリストを返す。
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

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

# タグとして認識する名前一覧
_KNOWN_TAG_NAMES = [
    "INSCRIBE_MEMORY",
    "CARVE_NARRATIVE",
    "DRIFT",
    "DRIFT_RESET",
    "SWITCH_ANGLE",
    "POWER_RECALL",
    "END_SESSION",
]

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

    return [_parse_tag_body(tn, body) for _, tn, body in flat]


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
    # debug_logger が _unescape_text() で JSON 文字列内の \n を実際の改行に変換するため、
    # 厳密な JSON としては無効になる。strict=False で制御文字を許容してパースする。
    character = ""
    preset = ""
    model_id = ""
    user_message = ""
    source = "system"
    front_input_path = folder / "01_FrontInput.log"
    if front_input_path.exists():
        try:
            raw = front_input_path.read_text(encoding="utf-8")
            data = json.loads(raw, strict=False)
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

    # --- ツール呼び出しを Request/Response ファイルペアから構築 ---
    # feature → {preset, request_file, response_file}
    tool_map: dict[str, dict] = {}

    for f in files:
        # 機能名あり: "02_chat_Request_ClaudeCode.log"
        m = _FILE_PATTERN.match(f.name)
        if m:
            feature, kind, preset_name = m.group(1), m.group(2), m.group(3)
            key = f"{feature}|{preset_name}"
            if key not in tool_map:
                tool_map[key] = {"feature": feature, "preset": preset_name,
                                 "request_file": None, "response_file": None, "tags": []}
            if kind == "Request":
                tool_map[key]["request_file"] = f.name
            else:
                tool_map[key]["response_file"] = f.name
            continue

        # 機能名なし（旧形式）: "02_Request_claude_cli.log"
        m2 = _FILE_PATTERN_NOFEATURE.match(f.name)
        if m2:
            kind, preset_name = m2.group(1), m2.group(2)
            key = f"|{preset_name}"
            if key not in tool_map:
                tool_map[key] = {"feature": "", "preset": preset_name,
                                 "request_file": None, "response_file": None, "tags": []}
            if kind == "Request":
                tool_map[key]["request_file"] = f.name
            else:
                tool_map[key]["response_file"] = f.name

    # Response ファイルからタグを抽出して tool_map に追加
    for key, tc in tool_map.items():
        if tc["response_file"]:
            tc["tags"] = _extract_tags_from_file(folder / tc["response_file"])

    # ソート: feature の辞書順（tool_map は挿入順なのでキー順に変換）
    tool_calls = list(tool_map.values())

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
        "tool_calls": tool_calls,
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
