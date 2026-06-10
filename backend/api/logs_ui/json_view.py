"""JSONログファイルのハイライト付きHTMLレンダリング。

ログ閲覧UIの「ファイル整形表示」用に、json.dumps(indent=2) で整形した上で
注目キーの文字列値を <mark> でハイライトする。
"""

import html as _html
import json
import re

from backend.api.logs_ui.tag_extract import _parse_json_safely

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
