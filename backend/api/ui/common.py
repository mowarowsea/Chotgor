"""設定 UI 共通ヘルパー。

テンプレートインスタンスの保持と、各ページルーターが共有する
フォーム/レスポンスユーティリティを提供する。
"""

import base64

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2MB

# UIテンプレートインスタンス（main.py から set_templates() で注入される）
templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    """テンプレートインスタンスをセットする。main.py の起動時に呼ぶ。

    Args:
        t: Jinja2Templates インスタンス。
    """
    global templates
    templates = t


def get_templates() -> Jinja2Templates:
    """テンプレートインスタンスを取得する。未初期化の場合は例外を送出する。"""
    if templates is None:
        raise RuntimeError("Templates not initialized")
    return templates


async def _read_image_data(form, field: str = "image") -> str | None:
    """フォームから画像を読み込みbase64 data URIとして返す。画像がなければNone。

    Args:
        form: await request.form() の結果。
        field: 読み込むファイルフィールド名（既定 "image"。バナーは "banner"）。
    """
    image_file = form.get(field)
    if not image_file or not hasattr(image_file, "read"):
        return None
    content = await image_file.read()
    if not content:
        return None
    if len(content) > MAX_IMAGE_BYTES:
        return None  # サイズ超過は無視
    content_type = image_file.content_type or "image/png"
    b64 = base64.b64encode(content).decode()
    return f"data:{content_type};base64,{b64}"


#: チャットバブルの配色パレット数。フロント側 `cb0〜cb9`（index.css）と対になる。
BUBBLE_PALETTE_SIZE = 10


def _parse_bubble_color(form) -> int | None:
    """フォームの `bubble_color` を 0〜BUBBLE_PALETTE_SIZE-1 の整数か None に正規化する。

    空文字（＝「自動」を選択）・未送信・範囲外・非数値はすべて None（自動配色）にする。
    """
    raw = (form.get("bubble_color") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if 0 <= value < BUBBLE_PALETTE_SIZE else None


def _bubble_color_owners(items) -> dict[int, list[str]]:
    """バブル配色スロット → その色を明示指定している名前一覧、のマップを作る。

    スウォッチUIで「この色は誰かが使用中」を示すために使う（重複選択は禁止しない）。
    自動配色（bubble_color=None）の相手は色が確定しないのでマップに載せない。
    """
    owners: dict[int, list[str]] = {}
    for item in items or []:
        color = getattr(item, "bubble_color", None)
        if color is None:
            continue
        owners.setdefault(int(color), []).append(getattr(item, "name", "") or "")
    return owners


def _is_ajax(request: Request) -> bool:
    """fetch / XHR からのリクエストかどうかを X-Requested-With ヘッダで判定する。"""
    return request.headers.get("x-requested-with", "").lower() in (
        "fetch",
        "xmlhttprequest",
    )


def _save_response(request: Request, redirect_url: str):
    """保存完了レスポンスを返す。

    自動保存（AJAX）の場合は JSON を、通常のフォーム送信の場合は
    従来どおりリダイレクトを返す。これにより 1 つのハンドラが
    「Save ボタン送信」と「フィールド変更ごとの自動保存」の両方に対応する。

    Args:
        request: リクエスト。AJAX 判定に使う。
        redirect_url: 通常送信時のリダイレクト先。

    Returns:
        JSONResponse または RedirectResponse。
    """
    if _is_ajax(request):
        return JSONResponse({"ok": True})
    return RedirectResponse(url=redirect_url, status_code=303)
