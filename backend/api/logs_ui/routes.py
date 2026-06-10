"""ログ閲覧UIのルーター定義。

debug/ ディレクトリ内のリクエスト単位ログフォルダを読み込み、
Web画面（HTML）とフロントエンド向けJSON APIで閲覧できるエンドポイントを提供する。
CHOTGOR_DEBUG=1 の時のみ debug/ フォルダが作成されるため、
フォルダが存在しない場合は空のリストを返す。
"""

import html as _html
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from backend.api.logs_ui import config
from backend.api.logs_ui.entries import (
    _build_char_index,
    _build_entry_from_db_rows,
    _load_entries,
    _parse_entry,
)
from backend.api.logs_ui.json_view import _render_json_html

router = APIRouter(prefix="/ui/logs", tags=["logs-ui"])


@router.get("/", response_class=HTMLResponse)
async def logs_list(request: Request, page: int = 1, type: str = "chat", partial: bool = False):
    """ログ一覧ページを返す。

    debug/ フォルダ内のリクエスト単位ログをページネーションして表示する。
    partial=True のとき、ログリスト部分のHTMLフラグメントのみを返す（AJAXタブ切り替え用）。

    Args:
        request: FastAPIリクエストオブジェクト。
        page: ページ番号（デフォルト1）。
        type: 表示するリクエスト種別（'chat'/'scenario'/'batch'、デフォルト'chat'）。
        partial: True のとき logs_fragment.html のみを返す。
    """
    per_page = 50
    # 不正な値はデフォルトにフォールバック
    valid_types = {"chat", "scenario", "batch"}
    current_type = type if type in valid_types else "chat"

    entries, total = _load_entries(page=page, per_page=per_page, request_type=current_type)
    total_pages = max(1, (total + per_page - 1) // per_page)
    debug_enabled = config.DEBUG_BASE.exists()

    context = {
        "request": request,
        "entries": entries,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "debug_enabled": debug_enabled,
        "current_type": current_type,
    }

    template = "logs_fragment.html" if partial else "logs.html"
    return config.get_templates().TemplateResponse(template, context)


@router.get("/{message_id}/detail", response_class=HTMLResponse)
async def log_entry_detail(request: Request, message_id: str):
    """指定リクエストの試行詳細（tool_calls 含む）を HTML フラグメントで返す。

    アコーディオン開き時のAJAX遅延ロード用。ファイルI/Oを含む完全な詳細を返す。

    Args:
        request: FastAPIリクエストオブジェクト。
        message_id: リクエストID（8桁hex）。
    """
    if not re.fullmatch(r"[0-9a-f]{8}", message_id):
        return HTMLResponse("Invalid message_id", status_code=400)

    sqlite = config.get_sqlite_store()
    if sqlite is None:
        return HTMLResponse("DB未接続", status_code=503)

    rows = sqlite.get_debug_log_entries_by_request_id(message_id)
    if not rows:
        return HTMLResponse("", status_code=404)

    entry = _build_entry_from_db_rows(rows, skip_files=False)
    return config.get_templates().TemplateResponse(
        "logs_entry_detail.html",
        {"request": request, "e": entry},
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

    file_path = config.DEBUG_BASE / message_id / filename
    if not file_path.exists() or not file_path.is_file():
        return PlainTextResponse("File not found", status_code=404)

    try:
        content = file_path.read_text(encoding="utf-8")
        return PlainTextResponse(content, media_type="text/plain; charset=utf-8")
    except Exception:
        return PlainTextResponse("Read error", status_code=500)


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

    file_path = config.DEBUG_BASE / message_id / filename
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

    # DB から取得を試みる。
    # 単一エントリ詳細取得 API なので skip_files=False（一覧高速表示の logs_ui とは違い、
    # tool_calls の tags まで含めた完全な情報をフロントに返す必要がある）。
    sqlite = config.get_sqlite_store()
    if sqlite is not None:
        rows = sqlite.get_debug_log_entries_by_request_id(log_message_id)
        if rows:
            entry = _build_entry_from_db_rows(rows, skip_files=False)
            entry_json = {k: v for k, v in entry.items() if k != "dt"}
            return {"entry": entry_json, "debug_enabled": True}

    # DB にない場合はファイルにフォールバック
    folder = config.DEBUG_BASE / log_message_id
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail="ログエントリが見つかりません")
    char_index = _build_char_index()
    entry = _parse_entry(log_message_id, folder, char_index)
    entry_json = {k: v for k, v in entry.items() if k != "dt"}
    return {"entry": entry_json, "debug_enabled": True}
