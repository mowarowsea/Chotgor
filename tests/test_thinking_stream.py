"""Tests for Issue #25 — ThinkingBlock ストリーミング対応。

各プロバイダーの generate_stream_typed() が
("thinking", text) / ("text", text) タプルを正しくyieldすることを検証する。
外部API・CLIへの実際の通信はモックで置き換える。
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.providers.base import BaseLLMProvider


# ---------------------------------------------------------------------------
# BaseLLMProvider.generate_stream_typed() デフォルト実装
# ---------------------------------------------------------------------------


class _SimpleProvider(BaseLLMProvider):
    """generate_stream() のみをオーバーライドした最小プロバイダー。
    generate_stream_typed() はデフォルト実装を使用する。
    """

    async def generate(self, system_prompt, messages):
        return "hello"

    async def generate_stream(self, system_prompt, messages):
        for chunk in ["hel", "lo"]:
            yield chunk


@pytest.mark.asyncio
async def test_base_generate_stream_typed_wraps_text():
    """デフォルト実装は generate_stream() の各チャンクを ("text", chunk) にラップする。"""
    provider = _SimpleProvider()
    chunks = []
    async for item in provider.generate_stream_typed("sys", []):
        chunks.append(item)

    assert chunks == [("text", "hel"), ("text", "lo")]


@pytest.mark.asyncio
async def test_base_generate_stream_typed_no_thinking_type():
    """デフォルト実装は "thinking" タプルを一切yieldしない。"""
    provider = _SimpleProvider()
    types = set()
    async for t, _ in provider.generate_stream_typed("sys", []):
        types.add(t)

    assert "thinking" not in types


# ---------------------------------------------------------------------------
# AnthropicProvider.generate_stream_typed()
# ---------------------------------------------------------------------------


def _make_delta_event(dtype: str, text: str):
    """Anthropic SDK の ContentBlockDeltaEvent に相当するモックオブジェクトを作成する。"""
    delta = MagicMock()
    delta.type = dtype
    if dtype == "thinking_delta":
        delta.thinking = text
    elif dtype == "text_delta":
        delta.text = text
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = delta
    return event


def _patch_anthropic_module(mock_client):
    """anthropic を sys.modules 経由でモックして mock_client を注入するコンテキストマネージャを返す。

    AnthropicProvider は関数内で `import anthropic` を遅延実行するため、
    モジュール属性ではなく sys.modules に直接注入する必要がある。
    """
    import sys
    mock_module = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    return patch.dict(sys.modules, {"anthropic": mock_module})


@pytest.mark.asyncio
async def test_anthropic_generate_stream_typed_thinking():
    """AnthropicProvider が thinking_delta を ("thinking", ...) としてyieldする。"""
    from backend.core.providers.anthropic_provider import AnthropicProvider

    events = [
        _make_delta_event("thinking_delta", "考え中..."),
        _make_delta_event("text_delta", "応答テキスト"),
    ]

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=iter(events))
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream_ctx

    provider = AnthropicProvider(api_key="dummy", model="claude-sonnet-4-6", thinking_level="medium")

    with _patch_anthropic_module(mock_client):
        chunks = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            chunks.append(item)

    assert ("thinking", "考え中...") in chunks
    assert ("text", "応答テキスト") in chunks


@pytest.mark.asyncio
async def test_anthropic_generate_stream_typed_text_only_when_no_thinking():
    """thinking_level == "default" のとき thinking_delta がなければ ("text", ...) のみyieldされる。"""
    from backend.core.providers.anthropic_provider import AnthropicProvider

    events = [
        _make_delta_event("text_delta", "part1"),
        _make_delta_event("text_delta", "part2"),
    ]

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=iter(events))
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream_ctx

    provider = AnthropicProvider(api_key="dummy", model="claude-sonnet-4-6", thinking_level="default")

    with _patch_anthropic_module(mock_client):
        chunks = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            chunks.append(item)

    types = {t for t, _ in chunks}
    assert types == {"text"}
    assert chunks == [("text", "part1"), ("text", "part2")]


@pytest.mark.asyncio
async def test_anthropic_generate_stream_typed_missing_api_key():
    """APIキー未設定のとき ("text", エラーメッセージ) をyieldして終了する。

    api_key チェックは import の前に行われるため、anthropic モジュールのモックは不要。
    """
    from backend.core.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="", model="claude-sonnet-4-6")

    chunks = []
    # anthropic パッケージが未インストールでもこのパスは通る（api_key 空チェックが先）
    import sys
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"anthropic": mock_module}):
        async for item in provider.generate_stream_typed("sys", []):
            chunks.append(item)

    assert len(chunks) == 1
    t, msg = chunks[0]
    assert t == "text"
    assert "anthropic_api_key" in msg


@pytest.mark.asyncio
async def test_anthropic_generate_stream_typed_sdk_error():
    """SDKがRuntimeErrorを送出した場合、("text", エラーメッセージ) をyieldして終了する。"""
    from backend.core.providers.anthropic_provider import AnthropicProvider

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(side_effect=RuntimeError("connection refused"))
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream_ctx

    provider = AnthropicProvider(api_key="dummy", model="claude-sonnet-4-6")

    with _patch_anthropic_module(mock_client):
        chunks = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            chunks.append(item)

    assert len(chunks) == 1
    t, msg = chunks[0]
    assert t == "text"
    assert "Anthropic API error" in msg


# ---------------------------------------------------------------------------
# ClaudeCliProvider.generate_stream_typed()
# ---------------------------------------------------------------------------


def _cli_line(event_type: str, blocks: list) -> bytes:
    """stream-json 形式の1行JSONバイトを生成するヘルパー。

    Args:
        event_type: "assistant" など。
        blocks: content ブロック辞書のリスト。
    """
    event = {"type": event_type, "message": {"content": blocks}}
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_cli_generate_stream_typed_thinking_and_text():
    """ClaudeCliProvider が stream-json の thinking ブロックを ("thinking", ...) としてyieldする。"""
    from backend.core.providers.claude_cli_provider import ClaudeCliProvider

    lines = [
        _cli_line("assistant", [
            {"type": "thinking", "thinking": "これは思考"},
            {"type": "text", "text": "これは応答"},
        ]),
    ]

    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stderr = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout.readline = MagicMock(side_effect=lines + [b""])
    mock_proc.wait = MagicMock()

    provider = ClaudeCliProvider(model="", character_name="Alice", thinking_level="medium")

    with (
        patch("backend.core.providers.claude_cli_provider.subprocess.Popen", return_value=mock_proc),
        patch("backend.core.providers.claude_cli_provider._clean_env", return_value={}),
    ):
        chunks = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            chunks.append(item)

    assert ("thinking", "これは思考") in chunks
    assert ("text", "これは応答") in chunks


@pytest.mark.asyncio
async def test_cli_generate_stream_typed_text_only():
    """thinking ブロックがない場合は ("text", ...) のみyieldされる。"""
    from backend.core.providers.claude_cli_provider import ClaudeCliProvider

    lines = [
        _cli_line("assistant", [{"type": "text", "text": "テキストのみ"}]),
    ]

    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stderr = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout.readline = MagicMock(side_effect=lines + [b""])
    mock_proc.wait = MagicMock()

    provider = ClaudeCliProvider(model="", character_name="Bob")

    with (
        patch("backend.core.providers.claude_cli_provider.subprocess.Popen", return_value=mock_proc),
        patch("backend.core.providers.claude_cli_provider._clean_env", return_value={}),
    ):
        chunks = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            chunks.append(item)

    types = {t for t, _ in chunks}
    assert types == {"text"}


@pytest.mark.asyncio
async def test_cli_generate_stream_typed_empty_thinking_skipped():
    """thinking フィールドが空文字のブロックはyieldしない。"""
    from backend.core.providers.claude_cli_provider import ClaudeCliProvider

    lines = [
        _cli_line("assistant", [
            {"type": "thinking", "thinking": ""},      # 空 → スキップ
            {"type": "text", "text": "応答あり"},
        ]),
    ]

    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stderr = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout.readline = MagicMock(side_effect=lines + [b""])
    mock_proc.wait = MagicMock()

    provider = ClaudeCliProvider(model="", character_name="Charlie")

    with (
        patch("backend.core.providers.claude_cli_provider.subprocess.Popen", return_value=mock_proc),
        patch("backend.core.providers.claude_cli_provider._clean_env", return_value={}),
    ):
        chunks = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            chunks.append(item)

    types = {t for t, _ in chunks}
    assert "thinking" not in types
    assert ("text", "応答あり") in chunks


@pytest.mark.asyncio
async def test_cli_generate_stream_typed_cli_not_found():
    """CLI が存在しない場合は ("text", エラーメッセージ) をyieldして終了する。"""
    from backend.core.providers.claude_cli_provider import ClaudeCliProvider

    provider = ClaudeCliProvider(model="", character_name="Dana")

    with (
        patch(
            "backend.core.providers.claude_cli_provider.subprocess.Popen",
            side_effect=FileNotFoundError("not found"),
        ),
        patch("backend.core.providers.claude_cli_provider._clean_env", return_value={}),
    ):
        chunks = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            chunks.append(item)

    assert len(chunks) == 1
    t, msg = chunks[0]
    assert t == "text"
    assert "Claude CLI error" in msg


@pytest.mark.asyncio
async def test_cli_generate_stream_typed_nonzero_exit():
    """CLI が非ゼロ終了コードで終了した場合、エラーチャンクをyieldする。"""
    from backend.core.providers.claude_cli_provider import ClaudeCliProvider

    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stderr = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout.readline = MagicMock(side_effect=[b""])
    mock_proc.stderr.read = MagicMock(return_value=b"some error")
    mock_proc.wait = MagicMock()

    provider = ClaudeCliProvider(model="", character_name="Eve")

    with (
        patch("backend.core.providers.claude_cli_provider.subprocess.Popen", return_value=mock_proc),
        patch("backend.core.providers.claude_cli_provider._clean_env", return_value={}),
    ):
        chunks = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            chunks.append(item)

    assert len(chunks) == 1
    t, msg = chunks[0]
    assert t == "text"
    assert "Claude CLI error" in msg


# ---------------------------------------------------------------------------
# GoogleProvider.generate_stream_typed()
# ---------------------------------------------------------------------------


def _make_google_chunk(parts_spec: list[tuple[bool, str]]):
    """Google Gemini の GenerateContentResponse チャンクに相当するモックオブジェクトを作成する。

    Args:
        parts_spec: (is_thought, text) のタプルリスト。
            is_thought=True のとき part.thought == True になる。
    """
    parts = []
    for is_thought, text in parts_spec:
        part = MagicMock()
        part.text = text
        part.thought = is_thought
        parts.append(part)

    content = MagicMock()
    content.parts = parts

    candidate = MagicMock()
    candidate.content = content

    chunk = MagicMock()
    chunk.candidates = [candidate]
    # chunk.text は非思考パートのみ結合したもの（SDK の挙動を模倣）
    chunk.text = "".join(t for is_t, t in parts_spec if not is_t)
    return chunk


from contextlib import contextmanager


@contextmanager
def _patch_google_module(mock_client):
    """google.genai を sys.modules 経由でモックして mock_client を注入するコンテキストマネージャ。

    GoogleProvider は関数内で `from google import genai` を遅延実行するため
    sys.modules に直接注入する必要がある。

    また log_provider_request は _json_default で MagicMock の __dict__ を
    再帰的に展開しようとするため、合わせて no-op にパッチする。
    """
    import sys

    mock_types = MagicMock()
    mock_types.ThinkingConfig = MagicMock(return_value=MagicMock())
    mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
    mock_types.SafetySetting = MagicMock(return_value=MagicMock())
    mock_types.Content = MagicMock(return_value=MagicMock())
    mock_types.Part = MagicMock(return_value=MagicMock())

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_genai.types = mock_types

    mock_google = MagicMock()
    mock_google.genai = mock_genai

    with (
        patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }),
        # MagicMock の __dict__ を再帰展開して無限ループするのを防ぐ
        # log_provider_request は base.py の名前空間で呼ばれるためそちらをパッチ
        patch("backend.core.debug_logger.ChotgorLogger.log_provider_request"),
    ):
        yield


@pytest.mark.asyncio
async def test_google_generate_stream_typed_thinking_and_text():
    """GoogleProvider が thought=True のパートを ("thinking", ...) としてyieldする。"""
    from backend.core.providers.google_provider import GoogleProvider

    chunks_from_api = [
        _make_google_chunk([(True, "思考中...")]),
        _make_google_chunk([(False, "応答テキスト")]),
    ]

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter(chunks_from_api)

    provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash-thinking-exp", thinking_level="medium")

    with _patch_google_module(mock_client):
        result = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            result.append(item)

    assert ("thinking", "思考中...") in result
    assert ("text", "応答テキスト") in result


@pytest.mark.asyncio
async def test_google_generate_stream_typed_text_only_when_no_thinking():
    """thinking_level == "default" のとき thought パートがなければ ("text", ...) のみyieldされる。"""
    from backend.core.providers.google_provider import GoogleProvider

    chunks_from_api = [
        _make_google_chunk([(False, "part1")]),
        _make_google_chunk([(False, "part2")]),
    ]

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter(chunks_from_api)

    provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash", thinking_level="default")

    with _patch_google_module(mock_client):
        result = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            result.append(item)

    types = {t for t, _ in result}
    assert types == {"text"}
    assert result == [("text", "part1"), ("text", "part2")]


@pytest.mark.asyncio
async def test_google_generate_stream_typed_mixed_parts_in_one_chunk():
    """1チャンクに thought パートと text パートが混在しても正しく分離してyieldされる。"""
    from backend.core.providers.google_provider import GoogleProvider

    chunks_from_api = [
        _make_google_chunk([
            (True, "思考A"),
            (False, "テキストA"),
        ]),
    ]

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter(chunks_from_api)

    provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash-thinking-exp", thinking_level="high")

    with _patch_google_module(mock_client):
        result = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            result.append(item)

    assert ("thinking", "思考A") in result
    assert ("text", "テキストA") in result
    # 順序: thinking が text より先にyieldされること
    assert result.index(("thinking", "思考A")) < result.index(("text", "テキストA"))


@pytest.mark.asyncio
async def test_google_generate_stream_typed_empty_text_part_skipped():
    """text が空のパートはyieldしない。"""
    from backend.core.providers.google_provider import GoogleProvider

    chunks_from_api = [
        _make_google_chunk([(True, ""), (False, "応答")]),  # 空thought → スキップ
    ]

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter(chunks_from_api)

    provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash-thinking-exp", thinking_level="medium")

    with _patch_google_module(mock_client):
        result = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            result.append(item)

    types = {t for t, _ in result}
    assert "thinking" not in types
    assert ("text", "応答") in result


@pytest.mark.asyncio
async def test_google_generate_stream_typed_parts_fallback_to_chunk_text():
    """candidates へのアクセスが失敗したとき chunk.text にフォールバックして ("text", ...) をyieldする。"""
    from backend.core.providers.google_provider import GoogleProvider

    # candidates アクセスで AttributeError を発生させるチャンク
    bad_chunk = MagicMock()
    bad_chunk.candidates = []  # 空リスト → IndexError
    bad_chunk.text = "フォールバックテキスト"

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter([bad_chunk])

    provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash", thinking_level="default")

    with _patch_google_module(mock_client):
        result = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            result.append(item)

    assert ("text", "フォールバックテキスト") in result


@pytest.mark.asyncio
async def test_google_generate_stream_typed_include_thoughts_set_when_thinking():
    """thinking_level != "default" のとき ThinkingConfig が include_thoughts=True で呼ばれること。"""
    import sys
    from backend.core.providers.google_provider import GoogleProvider

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter([])

    captured_thinking_config_kwargs: dict = {}

    mock_types = MagicMock()

    def capture_thinking_config(**kwargs):
        captured_thinking_config_kwargs.update(kwargs)
        return MagicMock()

    mock_types.ThinkingConfig = capture_thinking_config
    mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
    mock_types.SafetySetting = MagicMock(return_value=MagicMock())
    mock_types.Content = MagicMock(return_value=MagicMock())
    mock_types.Part = MagicMock(return_value=MagicMock())

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_genai.types = mock_types

    mock_google = MagicMock()
    mock_google.genai = mock_genai

    provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash-thinking-exp", thinking_level="high")

    with patch.dict(sys.modules, {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }):
        async for _ in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            pass

    assert captured_thinking_config_kwargs.get("include_thoughts") is True
    assert captured_thinking_config_kwargs.get("thinking_budget") == 16000  # "high"


@pytest.mark.asyncio
async def test_google_generate_stream_typed_missing_api_key():
    """APIキー未設定のとき ("text", エラーメッセージ) をyieldして終了する。"""
    import sys
    from backend.core.providers.google_provider import GoogleProvider

    provider = GoogleProvider(api_key="", model="gemini-2.0-flash")

    mock_genai = MagicMock()
    with patch.dict(sys.modules, {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai}):
        result = []
        async for item in provider.generate_stream_typed("sys", []):
            result.append(item)

    assert len(result) == 1
    t, msg = result[0]
    assert t == "text"
    assert "google_api_key" in msg


@pytest.mark.asyncio
async def test_google_generate_stream_typed_sdk_error():
    """SDK が RuntimeError を送出した場合、("text", エラーメッセージ) をyieldして終了する。"""
    from backend.core.providers.google_provider import GoogleProvider

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.side_effect = RuntimeError("quota exceeded")

    provider = GoogleProvider(api_key="dummy", model="gemini-2.0-flash-thinking-exp", thinking_level="medium")

    with _patch_google_module(mock_client):
        result = []
        async for item in provider.generate_stream_typed("sys", [{"role": "user", "content": "hi"}]):
            result.append(item)

    assert len(result) == 1
    t, msg = result[0]
    assert t == "text"
    assert "Google API error" in msg
