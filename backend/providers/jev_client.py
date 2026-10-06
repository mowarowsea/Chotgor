"""TypeSafe Jev（判定モデル）クライアント。

Jev は文章を生成しない。テキスト（state）と名前付きの型付き質問（questions）を渡すと、
質問ごとに確率付きの答えを返す:
- noul   : yes/no 確率（0–1）。criteria に {"true": ..., "false": ...} を任意指定
- choice : criteria の選択肢 dict（最大255）から1つ ＋ probabilities / confidence
- score  : criteria の順序付き段階リスト（2–10、低→高）上の位置 ＋ probabilities / confidence

チャット応答を返せないため PROVIDER_REGISTRY（プリセット選択対象）には登録しない。
課金は入力トークンのみ（出力無料）。state は 32k トークンまで。

APIキーは global_settings の `jev_api_key`、無ければ環境変数 `JEV_API_KEY` を使う。

接続確認: `python -m backend.providers.jev_client`
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from backend.providers.base import LLMApiError

BASE_URL = "https://api.typesafe.ai/v1"
DEFAULT_MODEL = "jev-latest"
_ENV_KEY = "JEV_API_KEY"


def resolve_api_key(settings: dict | None = None) -> str:
    return (settings or {}).get("jev_api_key") or os.environ.get(_ENV_KEY, "")


class JevClient:
    """Jev の /v1/systemone を叩く薄い非同期クライアント。"""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # テスト用に httpx.MockTransport を差し込めるようにしている
        self._transport = transport

    @classmethod
    def from_settings(cls, settings: dict | None = None, **kwargs) -> "JevClient":
        return cls(api_key=resolve_api_key(settings), **kwargs)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
            transport=self._transport,
        )

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        if not self.api_key:
            raise LLMApiError(f"[Error: jev_api_key / {_ENV_KEY} が設定されていません]")
        try:
            async with self._client() as client:
                resp = await client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise LLMApiError(f"[Error: Jev への接続に失敗しました: {e}]") from e
        if resp.status_code >= 400:
            # 422 は本文にどのフィールドが不正かが入っているのでそのまま見せる
            raise LLMApiError(f"[Error: Jev API {resp.status_code}: {resp.text[:500]}]")
        return resp.json()

    async def list_models(self) -> list[str]:
        data = await self._request("GET", "/models")
        return [m.get("id", "") for m in data.get("data", [])]

    async def decide(self, state: Any, questions: dict[str, dict], model: str = "") -> dict:
        """state に対して questions を判定し、レスポンス全体（model / answers / usage）を返す。"""
        body = {"model": model or self.model, "state": state, "questions": questions}
        return await self._request("POST", "/systemone", json=body)


async def _connection_check() -> None:
    client = JevClient.from_settings()
    print("models:", await client.list_models())
    result = await client.decide(
        state="もわ: ただいまー、今日めっちゃ疲れた……",
        questions={
            "is_greeting": {"type": "noul", "instructions": "話者は帰宅の挨拶をしていますか？"},
            "mood": {
                "type": "choice",
                "instructions": "話者の気分はどれに近いですか？",
                "criteria": {"happy": "嬉しい・楽しい", "tired": "疲れている", "angry": "怒っている"},
            },
            "energy": {
                "type": "score",
                "instructions": "話者の元気の度合いは？",
                "criteria": ["ぐったり", "やや疲れ", "普通", "元気", "絶好調"],
            },
        },
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_connection_check())
