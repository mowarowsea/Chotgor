"""SarashinaProvider の送信内容テスト。

Sarashina API は仕様書に無いパラメータ・キーを 400 で弾くため、OpenAIProvider の
既定送信内容から何が差し替わっているかを、SDK クライアントをモックして検証する。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.providers.registry import create_provider, supported_attachment_kinds
from backend.providers.sarashina_provider import SarashinaProvider


def _completion(content="こんにちは", tool_calls=None):
    """chat.completions.create の非ストリーム戻り値を模したオブジェクト。"""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    response = MagicMock()
    response.choices = [SimpleNamespace(message=message)]
    response.usage = None
    response.model_dump.return_value = {}
    return response


@pytest.fixture
def provider():
    p = SarashinaProvider(api_key="sk-test", model="sarashina3-mini", thinking_level="high")
    client = MagicMock()
    with patch.object(SarashinaProvider, "_make_openai_client", return_value=client):
        yield p, client


class TestSarashinaRequestShape:
    """generate / _tool_turn が Sarashina の受け付けるパラメータだけを送ることを検証する。

    - 出力上限は max_completion_tokens=8000（max_tokens は 400 になる）
    - thinking_level を指定しても reasoning_effort は送らない（推論レベル概念なし）
    - content 配列はテキストだけ連結した文字列へ平坦化し、画像パートは捨てる
    """

    async def test_generate_uses_max_completion_tokens(self, provider):
        p, client = provider
        client.chat.completions.create.return_value = _completion()

        text = await p.generate("system", [{"role": "user", "content": "やあ"}])

        assert text == "こんにちは"
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 8000
        assert "max_tokens" not in kwargs
        assert "reasoning_effort" not in kwargs

    async def test_list_content_is_flattened(self, provider):
        p, client = provider
        client.chat.completions.create.return_value = _completion()
        content = [
            {"type": "text", "text": "これ見て"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "\n\n注釈"},
        ]

        await p.generate("system", [{"role": "user", "content": content}])

        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert sent[-1] == {"role": "user", "content": "これ見て\n\n注釈"}

    async def test_tool_turn_sends_auto_tool_choice(self, provider):
        """tool_choice を省略すると Sarashina は required 扱いになるため、auto を明示していること。"""
        p, client = provider
        client.chat.completions.create.return_value = _completion()

        await p._tool_turn("system", [{"role": "user", "content": "やあ"}])

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["max_completion_tokens"] == 8000


class TestSarashinaToolHistory:
    """ツール実行後に積む assistant メッセージが仕様キーだけで組まれることを検証する。

    SDK の model_dump は refusal / audio / function_call などを含み、Sarashina は
    これらを未知キーとして 400 で弾くため、role / content / tool_calls だけに絞る。
    """

    def test_assistant_message_has_only_spec_keys(self):
        p = SarashinaProvider(api_key="sk-test")
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="power_recall", arguments='{"query": "海"}'),
        )
        message = SimpleNamespace(content=None, tool_calls=[tc], refusal=None, audio=None)

        built = p._assistant_tool_message(message)

        assert built == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "power_recall", "arguments": '{"query": "海"}'},
                }
            ],
        }


class TestSarashinaRegistration:
    """レジストリ経由の生成と、添付種別（画像不可）の公開を検証する。"""

    def test_create_provider_reads_api_key(self):
        p = create_provider("sarashina", model="", settings={"sarashina_api_key": "sk-x"})
        assert isinstance(p, SarashinaProvider)
        assert p.api_key == "sk-x"
        assert p.model == "sarashina3-mini"
        assert p.base_url == "https://api.sarashina.ai/v1"

    def test_attachment_kinds_excludes_image(self):
        assert supported_attachment_kinds("sarashina") == ["text"]
