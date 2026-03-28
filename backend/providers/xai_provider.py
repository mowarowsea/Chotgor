"""xAI / Grok provider — OpenAI互換APIを使う薄いサブクラス。

XAIProvider は OpenAIProvider のサブクラスとして、
ベースURLとデフォルトモデルだけを上書きする。
"""

from __future__ import annotations

from backend.providers.openai_provider import OpenAIProvider

# xAI has no "medium" reasoning level; map it to "low"
_XAI_REASONING = {
    "low": "low",
    "medium": "low",
    "high": "high",
}


class XAIProvider(OpenAIProvider):
    """xAI / Grok — OpenAI互換APIを異なるベースURLで呼び出すプロバイダー。"""

    PROVIDER_ID = "xai"
    DEFAULT_MODEL = "grok-2-latest"
    BASE_URL = "https://api.x.ai/v1"
    # _api_guard デコレータがエラーメッセージで参照するキー名
    _API_SETTINGS_KEY = "xai_api_key"

    _REASONING_MAP = _XAI_REASONING

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        super().__init__(
            api_key=api_key,
            model=model or self.DEFAULT_MODEL,
            base_url=self.BASE_URL,
            thinking_level=thinking_level,
        )

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "XAIProvider":
        return cls(api_key=settings.get("xai_api_key", ""), model=model, thinking_level=thinking_level)

    # list_models は OpenAIProvider の実装をそのまま使う。
    # cls._API_SETTINGS_KEY == "xai_api_key"、cls.BASE_URL == "https://api.x.ai/v1" が
    # 自動的に参照されるため、オーバーライド不要。
