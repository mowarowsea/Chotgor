"""プロバイダー側 usage 記録ヘルパーの単体テスト。

OpenAIProvider / AnthropicProvider / OllamaProvider に追加した
``_record_usage_from_response`` / ``_record_usage_from_result`` が
``backend.lib.usage_recorder.record_usage`` を期待どおりに呼ぶかを検証する。

検証する観点:
    - SDK レスポンスの usage オブジェクト/dict から正しいフィールドが抜き出されること
    - usage が None / 空 dict のときは記録呼び出しが発生しないこと
    - provider / model / preset_name が record_usage へ転送されること
"""

from unittest.mock import MagicMock, patch

from backend.providers.anthropic_provider import AnthropicProvider
from backend.providers.ollama_provider import OllamaProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.sakura_provider import SakuraProvider


class TestOpenAIRecordUsage:
    """OpenAIProvider._record_usage_from_response の挙動を検証する。

    OpenAI SDK の ChatCompletion / ChatCompletionChunk の usage は
    ``prompt_tokens`` / ``completion_tokens`` /
    ``prompt_tokens_details.cached_tokens`` を持つ。これらが
    record_usage へ正しい引数名で渡ることを確認する。
    sakura は OpenAIProvider のサブクラスなので、ここで provider_id が
    "sakura" になることも併せて検証する（過去にここが固定値で
    集計に上がっていなかった経緯がある）。
    """

    def test_records_with_details(self):
        """usage オブジェクトから入力/出力/キャッシュ読込トークンが転送されること。"""
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        provider.preset_name = "default"
        usage = MagicMock()
        usage.prompt_tokens = 1000
        usage.completion_tokens = 200
        details = MagicMock()
        details.cached_tokens = 600
        usage.prompt_tokens_details = details

        with patch("backend.lib.usage_recorder.record_usage") as rec:
            provider._record_usage_from_response(usage)

        rec.assert_called_once_with(
            provider="openai",
            model="gpt-4o",
            preset_name="default",
            input_tokens=1000,
            output_tokens=200,
            cache_read_input_tokens=600,
        )

    def test_no_record_when_usage_none(self):
        """usage=None の呼び出しは record_usage を呼ばないこと。"""
        provider = OpenAIProvider(api_key="sk-test")

        with patch("backend.lib.usage_recorder.record_usage") as rec:
            provider._record_usage_from_response(None)

        rec.assert_not_called()

    def test_sakura_provider_id(self):
        """SakuraProvider 経由の記録は provider="sakura" として記録されること。"""
        provider = SakuraProvider(api_key="sk-test", model="gpt-oss-120b")
        provider.preset_name = "sakura-preset"
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 30
        # prompt_tokens_details は省略 → cached=0 で記録されること
        usage.prompt_tokens_details = None

        with patch("backend.lib.usage_recorder.record_usage") as rec:
            provider._record_usage_from_response(usage)

        rec.assert_called_once_with(
            provider="sakura",
            model="gpt-oss-120b",
            preset_name="sakura-preset",
            input_tokens=50,
            output_tokens=30,
            cache_read_input_tokens=0,
        )


class TestAnthropicRecordUsage:
    """AnthropicProvider._record_usage_from_response の挙動を検証する。

    Anthropic SDK の Usage は input/output に加えて
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` を持つため、
    両者が record_usage へ転送されることを確認する。
    """

    def test_records_cache_fields(self):
        """input/output/cache_read/cache_creation が record_usage に転送されること。"""
        provider = AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-6")
        provider.preset_name = "anthropic-default"
        usage = MagicMock()
        usage.input_tokens = 800
        usage.output_tokens = 200
        usage.cache_read_input_tokens = 400
        usage.cache_creation_input_tokens = 100

        with patch("backend.lib.usage_recorder.record_usage") as rec:
            provider._record_usage_from_response(usage)

        rec.assert_called_once_with(
            provider="anthropic",
            model="claude-sonnet-4-6",
            preset_name="anthropic-default",
            input_tokens=800,
            output_tokens=200,
            cache_read_input_tokens=400,
            cache_creation_input_tokens=100,
        )

    def test_no_record_when_usage_none(self):
        """usage=None の呼び出しは record_usage を呼ばないこと。"""
        provider = AnthropicProvider(api_key="sk-ant-test")

        with patch("backend.lib.usage_recorder.record_usage") as rec:
            provider._record_usage_from_response(None)

        rec.assert_not_called()


class TestOllamaRecordUsage:
    """OllamaProvider._record_usage_from_result の挙動を検証する。

    Ollama /api/generate は ``prompt_eval_count`` / ``eval_count`` を返す。
    これが record_usage の input/output として正しく転送されることを確認する。
    """

    def test_records_eval_counts(self):
        """prompt_eval_count / eval_count が input/output へマップされること。"""
        provider = OllamaProvider(model="qwen2.5:latest")
        provider.preset_name = "ollama-default"
        result = {
            "response": "hello",
            "prompt_eval_count": 120,
            "eval_count": 45,
        }

        with patch("backend.lib.usage_recorder.record_usage") as rec:
            provider._record_usage_from_result(result)

        rec.assert_called_once_with(
            provider="ollama",
            model="qwen2.5:latest",
            preset_name="ollama-default",
            input_tokens=120,
            output_tokens=45,
        )

    def test_no_record_when_result_not_dict(self):
        """dict 以外（None など）の result は record_usage を呼ばないこと。"""
        provider = OllamaProvider()

        with patch("backend.lib.usage_recorder.record_usage") as rec:
            provider._record_usage_from_result(None)  # type: ignore[arg-type]

        rec.assert_not_called()

    def test_missing_counts_treated_as_zero(self):
        """eval count が欠落していても 0 として record_usage を呼ぶこと（recorder 側で握り潰し）。"""
        provider = OllamaProvider(model="qwen2.5:latest")

        with patch("backend.lib.usage_recorder.record_usage") as rec:
            provider._record_usage_from_result({"response": "x"})

        rec.assert_called_once_with(
            provider="ollama",
            model="qwen2.5:latest",
            preset_name="",
            input_tokens=0,
            output_tokens=0,
        )
