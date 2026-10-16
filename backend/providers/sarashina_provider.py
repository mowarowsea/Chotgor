"""Sarashina provider（SB Intuitions / ソフトバンク）— OpenAI互換APIのサブクラス。

Sarashina API（https://api.sarashina.ai/v1）は /v1/chat/completions を OpenAI 互換の
形で提供するが、**仕様書に無いパラメータを 400（Invalid JSON or disallowed fields）で弾く**。
そのため OpenAIProvider の既定送信内容のうち、以下を Sarashina 向けに差し替える。

- 出力上限は `max_completion_tokens`（1〜8000）。旧名 `max_tokens` は 400
- `stream_options` 非対応（400）。usage は送らなくても最終チャンクに入る
- `reasoning_effort` 非対応（推論レベルの概念なし）
- `content` は文字列のみ（配列不可）。画像は受け付けないので、テキストパートだけ連結する
- ツール呼び出し後の assistant メッセージは SDK の model_dump だと refusal / audio 等の
  余計なキーが乗るため、仕様にあるキー（role / content / tool_calls）だけに絞る
- `tool_choice` は文字列のみ。OpenAIProvider は "auto" を明示しているので問題ない
  （省略すると tools 付きで required 扱いになり、ツールを必ず呼ばされる）

/v1/models は 404 のため、モデル一覧は静的に持つ。
レート制限は mini 6RPM / nano 60RPM（2026-10 無償ベータ時点）。ツールループで複数回
呼ぶと mini はすぐ 429 になり得る点に注意。
"""

from __future__ import annotations

from backend.providers.openai_provider import OpenAIProvider

_SARASHINA_BASE_URL = "https://api.sarashina.ai/v1"

# sarashina3-guard は安全性判定専用（文章を生成しない）ため載せない。
_SARASHINA_CHAT_MODELS: list[dict] = [
    {"id": "sarashina3-mini", "label": "sarashina3-mini (標準・6RPM)"},
    {"id": "sarashina3-nano", "label": "sarashina3-nano (軽量・60RPM)"},
]


def _flatten_content(content):
    """マルチパート content をテキストだけ連結した文字列にする（画像パートは捨てる）。"""
    if not isinstance(content, list):
        return content
    texts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "".join(texts)


class SarashinaProvider(OpenAIProvider):
    """Sarashina API — 未知パラメータを拒否する OpenAI 互換サーバ向けのプロバイダー。"""

    PROVIDER_ID = "sarashina"
    DEFAULT_MODEL = "sarashina3-mini"
    BASE_URL = _SARASHINA_BASE_URL
    _API_SETTINGS_KEY = "sarashina_api_key"

    _REASONING_MAP: dict = {}

    # API 側の上限が 8000（超えると 400）。無償ベータでトークン課金もないため上限いっぱいを使う。
    DEFAULT_MAX_TOKENS = 8000
    _MAX_TOKENS_PARAM = "max_completion_tokens"
    _STREAM_INCLUDE_USAGE = False

    # content が文字列しか受けないため画像は不可。テキスト添付は本文へ連結して渡す。
    SUPPORTED_ATTACHMENT_KINDS = {"text"}

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        super().__init__(
            api_key=api_key,
            model=model or self.DEFAULT_MODEL,
            base_url=self.BASE_URL,
            thinking_level=thinking_level,
        )

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "SarashinaProvider":
        return cls(api_key=settings.get("sarashina_api_key", ""), model=model, thinking_level=thinking_level)

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """/v1/models が無いため静的リストを返す（APIキー未設定でも候補を出す）。"""
        return [{"id": m["id"], "name": m["label"]} for m in _SARASHINA_CHAT_MODELS]

    def _prepare_api_messages(self, messages: list[dict]) -> list[dict]:
        """content 配列を文字列へ平坦化する。"""
        prepared = []
        for m in messages:
            if isinstance(m.get("content"), list):
                m = {**m, "content": _flatten_content(m["content"])}
            prepared.append(m)
        return prepared

    def _assistant_tool_message(self, message) -> dict:
        """仕様にあるキーだけで assistant メッセージを組み直す（余計なキーは 400 になる）。"""
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (message.tool_calls or [])
            ],
        }
