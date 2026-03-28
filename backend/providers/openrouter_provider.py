"""OpenRouter provider — OpenAI互換APIを使う薄いサブクラス。

OpenRouterProvider は OpenAIProvider のサブクラスとして、
ベースURL・デフォルトモデル・HTTPヘッダーを上書きする。

OpenRouter は多数のモデルをOpenAI互換APIで提供する。
モデルIDは "プロバイダー/モデル名" 形式（例: openai/gpt-4o, anthropic/claude-opus-4）。
"""

from __future__ import annotations

from backend.providers.openai_provider import OpenAIProvider

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_HEADERS = {
    "X-Title": "Chotgor",
}


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter — OpenAI互換APIで多様なモデルにアクセスするプロバイダー。"""

    PROVIDER_ID = "openrouter"
    DEFAULT_MODEL = "openai/gpt-4o"
    _API_SETTINGS_KEY = "openrouter_api_key"

    # thinking_level / reasoning_effort はモデル依存のため無効化
    _REASONING_MAP: dict = {}

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        super().__init__(
            api_key=api_key,
            model=model or self.DEFAULT_MODEL,
            base_url=_OPENROUTER_BASE_URL,
            thinking_level=thinking_level,
        )

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "OpenRouterProvider":
        return cls(api_key=settings.get("openrouter_api_key", ""), model=model, thinking_level=thinking_level)

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """OpenRouter API からモデル一覧を取得して返す。"""
        import httpx
        api_key = settings.get("openrouter_api_key", "")
        if not api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{_OPENROUTER_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
            models = [
                {"id": m["id"], "name": m.get("name", m["id"])}
                for m in data.get("data", [])
            ]
            return sorted(models, key=lambda m: m["id"])
        except Exception:
            return []

    def _make_openai_client(self):
        """OpenRouterヘッダーを付加したOpenAIクライアントを返す。"""
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url=_OPENROUTER_BASE_URL,
            default_headers=_OPENROUTER_DEFAULT_HEADERS,
        )
