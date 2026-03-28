"""Ollama プロバイダー — ローカルOllamaサーバーへのHTTPクライアント。

主に司会AI（ディレクターモデル）用途で使用する。
ストリーミングは非対応（司会AIには不要）。
"""

import asyncio
import json
import urllib.error
import urllib.request

from backend.providers.base import BaseLLMProvider

DEFAULT_MODEL = "qwen2.5:latest"


class OllamaProvider(BaseLLMProvider):
    """Ollama経由でローカルLLMを呼び出すプロバイダー。

    Attributes:
        base_url: OllamaサーバーのベースURL（例: http://localhost:11434）。
        model: 使用するモデル名（例: qwen2.5:latest）。
        timeout: リクエストタイムアウト秒数。
    """

    PROVIDER_ID = "ollama"
    DEFAULT_MODEL = DEFAULT_MODEL
    REQUIRES_API_KEY = False

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "", timeout: int = 60):
        """OllamaProviderを初期化する。"""
        self.base_url = base_url.rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout

    @classmethod
    def from_config(cls, model: str, settings: dict, **kwargs) -> "OllamaProvider":
        """設定からOllamaProviderを生成するファクトリメソッド。"""
        base_url = settings.get("ollama_base_url", "http://localhost:11434")
        timeout = int(settings.get("ollama_timeout", 60))
        return cls(base_url=base_url, model=model, timeout=timeout)

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """ローカルOllamaサーバーからモデル一覧を取得して返す。"""
        import httpx
        base_url = settings.get("ollama_base_url", "http://localhost:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
            return sorted(
                [{"id": m["name"], "name": m["name"]} for m in data.get("models", [])],
                key=lambda m: m["id"],
            )
        except Exception:
            return []

    def _call_chat(self, prompt: str) -> str:
        """Ollama /api/chat エンドポイントを同期呼び出しして応答テキストを返す。

        Args:
            prompt: ユーザーターンのプロンプトテキスト。

        Returns:
            モデルの応答テキスト。エラー時はエラーメッセージ文字列。
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self._log_request({"model": self.model, "prompt": prompt[:200]})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result.get("message", {}).get("content", "")
                self._log_response({"content": text[:200]})
                return text
        except urllib.error.URLError as e:
            return f"[Ollama接続エラー: {e.reason}. Ollamaが起動しているか確認してください]"
        except Exception as e:
            return f"[Ollama エラー: {e}]"

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Ollamaから応答テキストを非同期で生成する。

        system_prompt は最初のユーザーメッセージの先頭に挿入する。
        """
        # 全メッセージをフラットなテキストに結合してプロンプトを構築する
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            if role == "user":
                parts.append(f"[ユーザー]: {content}")
            else:
                parts.append(f"[キャラクター]: {content}")
        prompt = "\n\n".join(parts)
        return await asyncio.to_thread(self._call_chat, prompt)

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """generate() の結果を単一チャンクとしてyieldするストリーム互換メソッド。"""
        text = await self.generate(system_prompt, messages)
        yield text

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """generate() の結果を ("text", ...) タプルとしてyieldするstream_typed互換メソッド。"""
        text = await self.generate(system_prompt, messages)
        yield ("text", text)
