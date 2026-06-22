"""XML 形式の発話 1 行整形ヘルパ。Chronicle 棚卸し / シナリオ PC モード履歴整形などで共通利用する。

各発話を `<speaker>content</speaker>` で LLM 入力に渡す経路 (Chronicle の 3 セクション
フォーマッタ・pc_runner の履歴整形) では、speaker_name に XML タグ名として不正な文字
（スペース・`「」`・`@`・`<>`）が混入したり、content に `<`/`>`/`&` が含まれることが
普通にある。タグ／本文をそのまま LLM に流すと、セクション構造を誤読されたり、ユーザ
発話による偽装閉じタグ (`</ユーザ>` 等) でプロンプトを乗っ取られたりする。

本モジュールはその受け皿として、speaker_name を「タグ名安全な文字」に正規化し、
content の XML 特殊文字を実体参照にエスケープして 1 行に組み立てる単一の関数を提供する。
タグ名整形・エスケープ規則をここ 1 箇所に集約することで、Chronicle と PC モードで
表記が乖離するのを防ぐ。
"""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)


# XML 特殊文字 → 実体参照。`&` を先に置換しないと `&lt;` が `&amp;lt;` になるため
# 順序維持のために OrderedDict 相当の通常 dict (Python 3.7+ で挿入順保証) を使う。
_XML_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


# XML タグ名として安全とみなす文字集合。
# 英数字・アンダースコア・日本語の主要範囲（ひらがな U+3040-309F / カタカナ U+30A0-30FF /
# CJK 統合漢字 U+4E00-9FFF / CJK 互換漢字 U+F900-FAFF）を許可する。
# `\w` を使うと Unicode カテゴリ全般（句読点系を含む拡張字種）に広がりすぎるので、
# キャラクター運用で実害が出ない範囲に絞って明示する。
_TAG_SAFE_CHAR = re.compile(
    r"[A-Za-z0-9_぀-ゟ゠-ヿ一-鿿豈-﫿]"
)


def _escape_xml_content(content: str) -> str:
    """content の XML 特殊文字 (`&`, `<`, `>`) を実体参照に置換する。

    `&` を最初に処理しないと `&lt;` → `&amp;lt;` に二重エスケープされるため、
    1 文字ずつ走査して置換する方式を採る (str.replace の連鎖で発生する順序事故を防ぐ)。
    """
    return "".join(_XML_ESCAPES.get(ch, ch) for ch in content)


def _sanitize_xml_tag_name(name: str) -> str:
    """speaker_name を XML タグ名として安全な文字列に正規化する。

    許可しない文字（スペース・記号・`「」`・`@`・`<>` など）は `_` に置換する。
    空白だけ・空・全置換で本体が残らない場合は "Unknown" にフォールバック。
    XML 厳密仕様では先頭が数字でも不正だが、LLM 入力としては問題ないので許容する。
    """
    if not name:
        return "Unknown"
    stripped = name.strip()
    if not stripped:
        return "Unknown"
    sanitized = "".join(
        ch if _TAG_SAFE_CHAR.match(ch) else "_" for ch in stripped
    )
    # 置換だけで本体が残らないケース（例: "@@@"）も Unknown 扱い
    if not sanitized.replace("_", ""):
        return "Unknown"
    return sanitized


def format_xml_speech_line(speaker: str, content: str) -> str:
    """1 発話を ``<speaker>content</speaker>`` の XML 行に整形する。

    Chronicle 棚卸し入力 / PC モードの履歴整形で共通利用する。speaker はタグ名安全
    文字に正規化、content は `<`/`>`/`&` を実体参照にエスケープし、LLM のセクション
    構造誤読・偽装閉じタグによるプロンプトインジェクションを防ぐ。content の先頭・
    末尾の空白はトリムする (LLM 入力として無駄な空白を載せない)。

    Args:
        speaker: 話者表示名 (PC 役名・キャラ名・Narrator など)。
        content: 発話本文。

    Returns:
        ``<sanitized_speaker>escaped_content</sanitized_speaker>`` の 1 行。
    """
    safe_speaker = _sanitize_xml_tag_name(speaker)
    safe_content = _escape_xml_content((content or "").strip())
    return f"<{safe_speaker}>{safe_content}</{safe_speaker}>"


__all__ = ["format_xml_speech_line"]
