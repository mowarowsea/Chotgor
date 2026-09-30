"""Claude CLI プロバイダの逐次表示（stream_event 差分）のテスト。

背景: ``--include-partial-messages`` で CLI はトークン差分（stream_event）を出していたが、
provider は完成版の assistant イベントしか読まず、さらに 1on1 はツール方式
（generate_with_tools＝CLI 終了まで待つ）を通っていたため、応答が一括表示になっていた。
本テストは以下を検証する:
- _StreamChunkParser: 差分の切り出しと、完成版 assistant イベントとの二重出力防止
- generate_with_tools_stream: 差分を逐次 yield し、MCP 用の env（batch_context）を渡す
- ChatFlow.execute_stream: STREAMS_TOOL_TURNS のプロバイダで本文を逐次流し、
  予想タグは表示から除去しつつ従来どおり抽出する
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.providers.claude_cli_provider import ClaudeCliProvider, _StreamChunkParser
from backend.services.chat.models import ChatRequest, Message
from backend.services.chat.service import ChatService


def _message_start(mid: str) -> dict:
    return {"type": "stream_event", "event": {"type": "message_start", "message": {"id": mid}}}


def _delta(dtype: str, key: str, text: str) -> dict:
    return {
        "type": "stream_event",
        "event": {"type": "content_block_delta", "index": 0, "delta": {"type": dtype, key: text}},
    }


def _text_delta(text: str) -> dict:
    return _delta("text_delta", "text", text)


def _thinking_delta(text: str) -> dict:
    return _delta("thinking_delta", "thinking", text)


def _assistant(mid: str, blocks: list) -> dict:
    return {"type": "assistant", "message": {"id": mid, "content": blocks}}


class TestStreamChunkParser:
    """_StreamChunkParser が stream-json のイベント列から表示用チャンクを正しく切り出すことを検証する。

    CLI は同じ本文を「stream_event の差分」と「完成版の assistant イベント」の二重で出す。
    差分を流したメッセージの完成版を捨てないと本文が二重に表示されるため、その判定
    （message.id 単位）が肝。逆に差分が一切来ないメッセージ（partial 非対応の CLI）は
    完成版から拾わないと本文が消えるので、そのフォールバックも確認する。
    """

    def _feed_all(self, events: list[dict]) -> list[tuple[str, str]]:
        parser = _StreamChunkParser()
        out: list[tuple[str, str]] = []
        for ev in events:
            out.extend(parser.feed(ev))
        return out

    def test_deltas_are_emitted_and_assistant_duplicate_is_dropped(self):
        """差分はその都度 yield され、同じ message.id の完成版 assistant は捨てられること。"""
        chunks = self._feed_all([
            _message_start("m1"),
            _thinking_delta("考え"),
            _text_delta("こん"),
            _text_delta("にちは"),
            _assistant("m1", [
                {"type": "thinking", "thinking": "考え"},
                {"type": "text", "text": "こんにちは"},
            ]),
        ])
        assert chunks == [("thinking", "考え"), ("text", "こん"), ("text", "にちは")]

    def test_assistant_without_deltas_falls_back(self):
        """差分が来なかったメッセージは完成版 assistant から本文を拾うこと（partial 非対応 CLI）。"""
        chunks = self._feed_all([
            _assistant("m1", [{"type": "text", "text": "一括"}]),
        ])
        assert chunks == [("text", "一括")]

    def test_tool_use_between_messages(self):
        """ツールを挟む応答（前置き→tool_use→続き）で、各メッセージの差分が順に流れること。

        tool_use のみの完成版や input_json_delta は表示対象外。
        """
        chunks = self._feed_all([
            _message_start("m1"),
            _text_delta("調べるね。"),
            _delta("input_json_delta", "partial_json", '{"query":'),
            _assistant("m1", [{"type": "text", "text": "調べるね。"}]),
            _assistant("m1", [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]),
            {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
            _message_start("m2"),
            _text_delta("わかった！"),
            _assistant("m2", [{"type": "text", "text": "わかった！"}]),
        ])
        assert chunks == [("text", "調べるね。"), ("text", "わかった！")]

    def test_empty_and_unknown_events_ignored(self):
        """空の差分・signature_delta・system/result イベントは何も出さないこと。"""
        chunks = self._feed_all([
            {"type": "system", "subtype": "init"},
            _message_start("m1"),
            _text_delta(""),
            _delta("signature_delta", "signature", "sig"),
            {"type": "result", "result": "x"},
        ])
        assert chunks == []


def _mock_proc(events: list[dict]) -> MagicMock:
    lines = [(json.dumps(e, ensure_ascii=False) + "\n").encode("utf-8") for e in events]
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout.readline = MagicMock(side_effect=lines + [b""])
    return proc


class TestGenerateWithToolsStream:
    """ClaudeCliProvider.generate_with_tools_stream の振る舞いを検証する。

    1on1 の逐次表示に使う経路。generate_with_tools と同じく MCP サーバーへ
    batch_context / origin を env で渡す必要がある（渡らないと force_insert 等が効かない）。
    """

    @pytest.mark.asyncio
    async def test_yields_deltas_and_passes_batch_context_env(self):
        """差分チャンクを順に yield し、Popen の env に CHOTGOR_BATCH_CONTEXT が載ること。"""
        proc = _mock_proc([
            _message_start("m1"),
            _text_delta("やあ"),
            _text_delta("、もわ"),
            _assistant("m1", [{"type": "text", "text": "やあ、もわ"}]),
        ])
        tool_executor = MagicMock()
        tool_executor.batch_context = {"force_insert_memory": True}
        tool_executor.default_origin = "real"
        provider = ClaudeCliProvider(model="", character_name="はる", character_id="c1")

        with (
            patch("backend.providers.claude_cli_provider.subprocess.Popen", return_value=proc) as popen,
            patch("backend.providers.claude_cli_provider._clean_env", return_value={}),
        ):
            chunks = [c async for c in provider.generate_with_tools_stream(
                "sys", [{"role": "user", "content": "hi"}], tool_executor,
            )]

        assert chunks == [("text", "やあ"), ("text", "、もわ")]
        env = popen.call_args.kwargs["env"]
        assert json.loads(env["CHOTGOR_BATCH_CONTEXT"]) == {"force_insert_memory": True}
        assert env["CHOTGOR_CHARACTER_ID"] == "c1"


async def _stream_provider(chunks):
    for c in chunks:
        yield c


def _request() -> ChatRequest:
    return ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="claude_cli",
        model="",
        messages=[Message(role="user", content="hello")],
    )


async def _run_flow(fake_provider):
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])
    with (
        patch("backend.services.chat_flow.preparation.create_provider", return_value=fake_provider),
        patch("backend.services.chat_flow.preparation.build_system_prompt", return_value="sys"),
        patch("backend.services.chat_flow.preparation.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        return [ev async for ev in service.execute_stream(_request())]


class TestFlowToolStreaming:
    """ChatFlow.execute_stream が STREAMS_TOOL_TURNS=True のプロバイダで逐次表示することを検証する。

    従来のツール方式は generate_with_tools の戻り値（本文全体）を1回で yield していた。
    逐次化後も、予想タグ（ANTICIPATE_RESPONSE）は画面に出さず anticipation として抽出され、
    プロバイダ由来エラーは provider_error として分離されること（既存契約の維持）を確認する。
    """

    @pytest.mark.asyncio
    async def test_text_is_streamed_in_chunks_and_tags_are_hidden(self):
        """本文は複数チャンクで流れ、予想タグは表示から除去されて anticipation として出ること。
        generate_with_tools（一括経路）は呼ばれないこと。"""
        fake_provider = AsyncMock()
        fake_provider.SUPPORTS_TOOLS = True
        fake_provider.STREAMS_TOOL_TURNS = True
        fake_provider.generate_with_tools = AsyncMock(side_effect=AssertionError("呼ばれてはいけない"))
        fake_provider.generate_with_tools_stream = MagicMock(return_value=_stream_provider([
            ("thinking", "ふむ"),
            ("text", "こんに"),
            ("text", "ちは[ANTICIPATE_RESPONSE:喜ぶ"),
            ("text", "はず]"),
        ]))

        events = await _run_flow(fake_provider)

        texts = [c for t, c in events if t == "text"]
        assert len(texts) >= 2
        assert "".join(texts).strip() == "こんにちは"
        assert ("thinking", "ふむ") in events
        assert ("anticipation", "喜ぶはず") in events

    @pytest.mark.asyncio
    async def test_error_chunk_becomes_provider_error_after_partial_text(self):
        """途中でエラーチャンクが来たら、それまでの本文は流しつつ provider_error で終わること。"""
        fake_provider = AsyncMock()
        fake_provider.SUPPORTS_TOOLS = True
        fake_provider.STREAMS_TOOL_TURNS = True
        fake_provider.generate_with_tools_stream = MagicMock(return_value=_stream_provider([
            ("text", "途中まで"),
            ("error", "[Claude CLI error: boom]"),
        ]))

        events = await _run_flow(fake_provider)

        assert ("text", "途中まで") in events
        assert events[-1] == ("provider_error", "[Claude CLI error: boom]")
