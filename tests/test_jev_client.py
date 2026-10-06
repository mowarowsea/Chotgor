"""JevClient（TypeSafe Jev 判定モデル）のテスト。

実 API は叩かず httpx.MockTransport で応答を差し替える。
"""

import json

import httpx
import pytest

from backend.providers.base import LLMApiError
from backend.providers.jev_client import JevClient, resolve_api_key


class TestJevClient:
    """リクエスト組み立て・認証ヘッダ・エラー変換を検証する。

    - decide は POST /v1/systemone に model / state / questions をそのまま載せ、
      Bearer 認証を付けること
    - list_models は OpenAI 互換の {"data": [{"id": ...}]} から id を取り出すこと
    - 4xx/5xx は本文付きの LLMApiError へ変換されること（422 の原因を見せるため）
    - キー未設定ならネットワークに出る前に LLMApiError になること
    """

    @pytest.mark.asyncio
    async def test_decide_sends_body_and_auth(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.9}}})

        client = JevClient("k", transport=httpx.MockTransport(handler))
        result = await client.decide("text", {"q": {"type": "noul", "instructions": "?"}})

        assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
        assert seen["auth"] == "Bearer k"
        assert seen["body"] == {"model": "jev-latest", "state": "text", "questions": {"q": {"type": "noul", "instructions": "?"}}}
        assert result["answers"]["q"]["noul"] == 0.9

    @pytest.mark.asyncio
    async def test_list_models(self):
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "jev-latest"}, {"id": "jev-preview"}]}))
        assert await JevClient("k", transport=transport).list_models() == ["jev-latest", "jev-preview"]

    @pytest.mark.asyncio
    async def test_http_error_becomes_llm_api_error(self):
        transport = httpx.MockTransport(lambda r: httpx.Response(422, text='{"detail":"questions.q.type"}'))
        with pytest.raises(LLMApiError, match="422.*questions.q.type"):
            await JevClient("k", transport=transport).decide("x", {})

    @pytest.mark.asyncio
    async def test_missing_key(self):
        with pytest.raises(LLMApiError, match="JEV_API_KEY"):
            await JevClient("").list_models()

    def test_resolve_api_key_prefers_settings(self, monkeypatch):
        monkeypatch.setenv("JEV_API_KEY", "env")
        assert resolve_api_key({"jev_api_key": "db"}) == "db"
        assert resolve_api_key({}) == "env"


class TestJevClientDebugLog:
    """decide() が BaseLLMProvider と同じ形式で request / response をログへ残すことを検証する。

    logs 画面で LLM judge と Jev judge を同等に並べるため、Jev もプリセット名
    （既定 "Jev"）で log_provider_request / log_provider_response を呼ぶ。
    失敗時もエラー文字列をレスポンスとして記録し、ログ上で失敗が判別できること。
    """

    @pytest.mark.asyncio
    async def test_logs_request_and_response(self, monkeypatch):
        calls = []
        # インスタンスではなくクラスを差し替える（インスタンス属性に setattr すると、
        # undo 後も束縛済みメソッドがインスタンスに残り、後続テストのクラス patch を隠す）
        from backend.lib.debug_logger import ChotgorLogger
        monkeypatch.setattr(ChotgorLogger, "log_provider_request", lambda self, n, d: calls.append(("req", n, d)))
        monkeypatch.setattr(ChotgorLogger, "log_provider_response", lambda self, n, d: calls.append(("res", n, d)))
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"answers": {}}))
        await JevClient("k", transport=transport).decide("x", {})
        assert [(c[0], c[1]) for c in calls] == [("req", "Jev"), ("res", "Jev")]
        assert calls[0][2]["state"] == "x"

    @pytest.mark.asyncio
    async def test_logs_error_as_response(self, monkeypatch):
        calls = []
        from backend.lib.debug_logger import ChotgorLogger
        monkeypatch.setattr(ChotgorLogger, "log_provider_request", lambda self, n, d: None)
        monkeypatch.setattr(ChotgorLogger, "log_provider_response", lambda self, n, d: calls.append(d))
        transport = httpx.MockTransport(lambda r: httpx.Response(401, text="bad key"))
        with pytest.raises(LLMApiError):
            await JevClient("k", transport=transport).decide("x", {})
        assert "401" in calls[0]
