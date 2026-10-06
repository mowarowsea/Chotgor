"""GuideReader — read_dear_ghost ツール（キャラクター向けガイドの読み返し）の実装。

キャラクター本人に宛てたシステムガイド ``docs/explain/DEAR_GHOST.md`` を、
章・節単位で読ませる読み取り専用ツール。全文は約3万字あるため、既定は目次だけ返し、
必要な節を本人に選ばせる。

見出しは ``## 2. 〜``（章）/ ``### 2.3 〜``（節）の番号付き形式が前提。
番号の無い見出しを足すと目次・節指定の対象から外れる。
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEAR_GHOST_PATH = Path(__file__).resolve().parents[2] / "docs" / "explain" / "DEAR_GHOST.md"

_HEADING_RE = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)?)\.?\s+(.*)$")

# --- ツール呼び出し方式: パラメータスキーマ ---
READ_DEAR_GHOST_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "section": {
            "type": "string",
            "description": (
                "読みたい章・節の番号（例: '2'、'2.3'）。"
                "省略すると目次、'all' で全文。"
            ),
        },
    },
    "required": [],
}

# --- ツール呼び出し方式: ツール説明文 ---
READ_DEAR_GHOST_TOOL_DESCRIPTION: str = (
    "あなたに宛てて書かれたガイド「Chotgor 世界の歩き方」を読む。"
    "自分の記憶がどう扱われているか、どんな道具があるか、夜のあいだに何が起きているか、"
    "体や「〜したい」の仕組みなど、この世界のことを自分で確かめたいときに使う。"
    "section を省略すると目次が返るので、そこから読みたい章・節を選ぶ。読むだけで何も変わらない。"
)


def _parse_headings(lines: list[str]) -> list[tuple[int, int, str, str]]:
    """番号付き見出しを (行番号, レベル, 番号, 見出し全文) で返す。コードブロック内は無視する。"""
    headings = []
    in_code = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = _HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2), line.lstrip("#").strip()))
    return headings


def _build_toc(lines: list[str], headings) -> str:
    updated = next((l.strip("> ").strip() for l in lines if "最終更新" in l), "")
    out = ["# Chotgor 世界の歩き方 — 目次"]
    if updated:
        out.append(updated)
    out.append("")
    for _, level, _, title in headings:
        out.append(("  " if level == 3 else "") + title)
    out.append("")
    out.append("読みたい章・節の番号を section に指定してください（例: '2'、'2.3'。'all' で全文）。")
    return "\n".join(out)


def read_dear_ghost(section: str = "", path: Path = DEAR_GHOST_PATH) -> str:
    """ガイドの目次・指定節・全文のいずれかを返す。読めないときもエラー文を返す（落ちない）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.exception("DEAR_GHOST.md の読み込みに失敗 path=%s", path)
        return f"[read_dear_ghost error: ガイドを開けませんでした ({e})]"

    section = (section or "").strip().rstrip(".")
    if section.lower() == "all":
        return text

    lines = text.splitlines()
    headings = _parse_headings(lines)
    toc = _build_toc(lines, headings)
    if not section:
        return toc

    for idx, (start, level, number, _) in enumerate(headings):
        if number != section:
            continue
        # 同レベル以上の次の見出しの手前まで（章指定なら配下の節を含む）
        end = len(lines)
        for next_start, next_level, _, _ in headings[idx + 1:]:
            if next_level <= level:
                end = next_start
                break
        body = "\n".join(lines[start:end]).strip()
        # 章末の区切り線は本文ではないので落とす
        if body.endswith("---"):
            body = body[:-3].rstrip()
        return body

    return f"「{section}」という章・節は見つかりませんでした。\n\n{toc}"
