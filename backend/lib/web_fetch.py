"""URL detection and web content fetching.

ユーザーメッセージ内のURLを検出し、内容を取得してLLMのコンテキストに注入する。
"""

import logging
import re
from html.parser import HTMLParser

import httpx

logger = logging.getLogger(__name__)

# URL検出（内部文字に日本語句読点などを含めず、末尾の記号も除外）
URL_PATTERN = re.compile(r'https?://[^\s\]<>"\'`。、）]+(?<![.,!?);:。、）])')

MAX_CONTENT_CHARS = 8000
FETCH_TIMEOUT = 10.0


class _TextExtractor(HTMLParser):
    """HTMLからテキストを抽出するパーサー。script/style/head は無視。"""

    SKIP_TAGS = frozenset(["script", "style", "head", "noscript"])

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.texts.append(stripped)


def _extract_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return " ".join(parser.texts)


def find_urls(text: str) -> list[str]:
    """テキスト中のURLを重複なしで返す（出現順）。"""
    seen = {}
    for url in URL_PATTERN.findall(text):
        seen.setdefault(url, None)
    return list(seen)


async def fetch_urls(urls: list[str]) -> list[dict]:
    """URLリストを非同期でfetchし、結果リストを返す。

    Returns:
        list of {"url": str, "content": str, "truncated": bool}
        or       {"url": str, "error": str}
    """
    results = []
    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Chotgor/1.0)"},
    ) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                if resp.status_code >= 400:
                    logger.warning("HTTPエラー url=%s status=%d", url, resp.status_code)
                    results.append({
                        "url": url,
                        "error": f"HTTP {resp.status_code} {resp.reason_phrase}",
                    })
                    continue
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type:
                    text = _extract_text(resp.text)
                else:
                    text = resp.text
                truncated = len(text) > MAX_CONTENT_CHARS
                logger.info("成功 url=%s truncated=%s", url, truncated)
                results.append({
                    "url": url,
                    "content": text[:MAX_CONTENT_CHARS],
                    "truncated": truncated,
                })
            except httpx.ConnectError:
                logger.warning("接続失敗 url=%s", url)
                results.append({"url": url, "error": "ホストに接続できませんでした"})
            except httpx.TimeoutException:
                logger.warning("タイムアウト url=%s", url)
                results.append({"url": url, "error": "接続がタイムアウトしました"})
            except Exception as e:
                logger.warning("取得エラー url=%s error=%s", url, str(e))
                results.append({"url": url, "error": f"取得エラー: {e}"})
    return results
