"""Tests for backend.core.web_fetch — URL detection and fetching."""

import pytest
import pytest_asyncio

from backend.core.web_fetch import find_urls, fetch_urls


class TestFindUrls:
    def test_single_url(self):
        text = "これを見て https://example.com 面白いよ"
        assert find_urls(text) == ["https://example.com"]

    def test_multiple_urls(self):
        text = "https://example.com と https://github.com を参考にした"
        urls = find_urls(text)
        assert "https://example.com" in urls
        assert "https://github.com" in urls

    def test_duplicate_urls_deduplicated(self):
        text = "https://example.com を見て、また https://example.com を見た"
        assert find_urls(text) == ["https://example.com"]

    def test_trailing_punctuation_excluded(self):
        """末尾の句読点はURLに含まない。"""
        text = "https://example.com。これはURLです。"
        urls = find_urls(text)
        assert urls == ["https://example.com"]

    def test_trailing_parenthesis_excluded(self):
        text = "詳細は (https://example.com) を参照"
        urls = find_urls(text)
        assert urls == ["https://example.com"]

    def test_no_url(self):
        assert find_urls("URLなしのテキスト") == []

    def test_http_and_https(self):
        text = "http://insecure.example.com と https://secure.example.com"
        urls = find_urls(text)
        assert len(urls) == 2


class TestFetchUrls:
    @pytest.mark.asyncio
    async def test_fetch_success(self, monkeypatch):
        """正常時: content と truncated フラグが返る。"""
        from unittest.mock import AsyncMock, MagicMock
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason_phrase = "OK"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "ページの内容"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        monkeypatch.setattr(
            "backend.core.web_fetch.httpx.AsyncClient",
            lambda **kwargs: mock_client,
        )

        results = await fetch_urls(["https://example.com"])
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        assert results[0]["content"] == "ページの内容"
        assert results[0]["truncated"] is False

    @pytest.mark.asyncio
    async def test_fetch_http_error(self, monkeypatch):
        """4xx/5xx: error キーが入る。"""
        from unittest.mock import AsyncMock, MagicMock

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        monkeypatch.setattr(
            "backend.core.web_fetch.httpx.AsyncClient",
            lambda **kwargs: mock_client,
        )

        results = await fetch_urls(["https://example.com/notfound"])
        assert "error" in results[0]
        assert "404" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_fetch_connect_error(self, monkeypatch):
        """接続エラー: error キーが日本語メッセージで入る。"""
        from unittest.mock import AsyncMock
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        monkeypatch.setattr(
            "backend.core.web_fetch.httpx.AsyncClient",
            lambda **kwargs: mock_client,
        )

        results = await fetch_urls(["https://unreachable.example.com"])
        assert "error" in results[0]
        assert "接続" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_fetch_empty_url_list(self):
        results = await fetch_urls([])
        assert results == []
