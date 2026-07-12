"""switch_angle — ChatService 再ディスパッチとシステムプロンプトブロックのテスト。

テスト対象:
  - ChatService.execute() の switch 再ディスパッチ
  - ChatService.execute_stream() の switch 再ディスパッチ
  - build_system_prompt() の switch_angle ブロック生成
"""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.services.chat.models import ChatRequest, Message
from backend.services.chat.service import ChatService
from backend.character_actions.executor import ToolCall
from backend.services.chat.request_builder import _build_switch_angle_block, build_system_prompt

from tests._switch_angle_helpers import _SAMPLE_PRESETS, _make_executor, _make_request

# ---------------------------------------------------------------------------
# ChatService.execute() の switch_angle 再ディスパッチテスト
# ---------------------------------------------------------------------------

class TestChatServiceExecuteWithSwitch:
    """ChatService.execute() で switch_angle が発動したときの再ディスパッチを検証する。"""

    @pytest.mark.asyncio
    async def test_execute_switch_via_tag_redispatches(self, monkeypatch):
        """タグ方式: [SWITCH_ANGLE:...] が応答に含まれる場合、新プロバイダーで再ディスパッチされる。

        第1プロバイダーの応答テキスト（タグ除去済み）が第2プロバイダーへの messages に
        assistant ターンとして追加されることを検証する。
        """
        first_provider = AsyncMock()
        first_provider.SUPPORTS_TOOLS = False
        first_provider.generate = AsyncMock(return_value="本文[SWITCH_ANGLE:fastModel|軽く]")

        second_captured_messages = []

        async def second_generate(system_prompt, messages):
            """第2プロバイダーに渡された messages を記録する。"""
            second_captured_messages.extend(messages)
            return "軽い応答"

        second_provider = AsyncMock()
        second_provider.SUPPORTS_TOOLS = False
        second_provider.generate = second_generate

        def fake_create_provider(provider_id, model, settings, **kwargs):
            if provider_id == "google":
                return second_provider
            return first_provider

        monkeypatch.setattr("backend.services.chat_flow.preparation.create_provider", fake_create_provider)
        monkeypatch.setattr("backend.services.chat_flow.preparation.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.services.chat_flow.preparation.find_urls", lambda t: [])
        service = ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        result = await service.execute(request)

        # 結果は新プロバイダー（google）の応答
        assert result == "軽い応答"
        # 最初のプロバイダー（anthropic）は1回呼ばれる
        first_provider.generate.assert_called_once()
        # 第2プロバイダーに渡された messages に第1プロバイダーの応答が assistant ターンとして含まれる
        assistant_msgs = [m for m in second_captured_messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert any("本文" in (m.get("content") or "") for m in assistant_msgs)

    @pytest.mark.asyncio
    async def test_execute_switch_via_tool_use_redispatches(self, monkeypatch):
        """tool-use 方式: switch_angle ツールが呼ばれた場合、新プロバイダーで再ディスパッチされる。"""
        switch_tc = ToolCall(
            id="sw", name="switch_angle",
            input={"preset_name": "fastModel", "self_instruction": "軽く"}
        )

        call_log = []

        class FirstProvider:
            SUPPORTS_TOOLS = True

            async def generate_with_tools(self, system_prompt, messages, tool_executor):
                call_log.append("first")
                tool_executor.execute("switch_angle", {"preset_name": "fastModel", "self_instruction": "軽く"})
                return "", ""

        second_provider = AsyncMock()
        second_provider.SUPPORTS_TOOLS = False
        second_provider.generate = AsyncMock(return_value="軽い応答")

        def fake_create_provider(provider_id, model, settings, **kwargs):
            if provider_id == "google":
                return second_provider
            return FirstProvider()

        monkeypatch.setattr("backend.services.chat_flow.preparation.create_provider", fake_create_provider)
        monkeypatch.setattr("backend.services.chat_flow.preparation.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.services.chat_flow.preparation.find_urls", lambda t: [])
        service = ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        result = await service.execute(request)

        assert result == "軽い応答"
        assert "first" in call_log

    @pytest.mark.asyncio
    async def test_execute_no_switch_when_preset_unknown(self, monkeypatch):
        """存在しないプリセット名が指定された場合は再ディスパッチせずタグ除去済みテキストを返す。"""
        provider = AsyncMock()
        provider.SUPPORTS_TOOLS = False
        provider.generate = AsyncMock(return_value="本文[SWITCH_ANGLE:unknown|何か]")

        monkeypatch.setattr("backend.services.chat_flow.preparation.create_provider", lambda *a, **kw: provider)
        monkeypatch.setattr("backend.services.chat_flow.preparation.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.services.chat_flow.preparation.find_urls", lambda t: [])
        service = ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        result = await service.execute(request)

        # 未知プリセットなので再ディスパッチなし。タグ除去済みテキストが返る
        assert "[SWITCH_ANGLE:" not in result
        assert "本文" in result


# ---------------------------------------------------------------------------
# ChatService.execute_stream() の switch_angle 再ディスパッチテスト
# ---------------------------------------------------------------------------

class TestChatServiceExecuteStreamWithSwitch:
    """ChatService.execute_stream() で switch_angle が発動したときの挙動を検証する。"""

    @pytest.mark.asyncio
    async def test_execute_stream_switch_yields_angle_switched_event(self, monkeypatch):
        """switch_angle 発動後、("angle_switched", model_id) イベントが yield される。"""
        async def first_stream(system_prompt, messages):
            yield ("text", "本文[SWITCH_ANGLE:fastModel|軽く]")

        async def second_stream(system_prompt, messages):
            yield ("text", "軽い応答")

        first_provider = MagicMock()
        first_provider.SUPPORTS_TOOLS = False

        async def typed_stream_first(sp, msgs):
            yield ("text", "本文[SWITCH_ANGLE:fastModel|軽く]")

        first_provider.generate_stream_typed = typed_stream_first

        second_provider = MagicMock()
        second_provider.SUPPORTS_TOOLS = False

        async def typed_stream_second(sp, msgs):
            yield ("text", "軽い応答")

        second_provider.generate_stream_typed = typed_stream_second

        def fake_create_provider(provider_id, model, settings, **kwargs):
            if provider_id == "google":
                return second_provider
            return first_provider

        monkeypatch.setattr("backend.services.chat_flow.preparation.create_provider", fake_create_provider)
        monkeypatch.setattr("backend.services.chat_flow.preparation.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.services.chat_flow.preparation.find_urls", lambda t: [])
        service = ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())
        request = _make_request(
            character_name="Alice",
            available_presets=_SAMPLE_PRESETS,
        )

        events = []
        async for event in service.execute_stream(request):
            events.append(event)

        event_types = [e[0] for e in events]
        assert "text" in event_types
        assert "angle_switched" in event_types

        angle_switched_event = next(e for e in events if e[0] == "angle_switched")
        payload = angle_switched_event[1]
        assert "Alice" in payload["model_id"]
        assert "fastModel" in payload["preset_name"]

    @pytest.mark.asyncio
    async def test_execute_stream_switch_text_from_second_provider(self, monkeypatch):
        """switch 後のテキストは新プロバイダーからのものである。

        switch_angle 発動前の第1プロバイダーのテキストはそのまま UI に残り（clear しない）、
        第2プロバイダーのテキストが連続してストリームされる。
        第1プロバイダーの応答は assistant ターンとして第2プロバイダーへの messages に追加される。
        """
        second_captured_messages = []

        first_provider = MagicMock()
        first_provider.SUPPORTS_TOOLS = False

        async def typed_stream_first(sp, msgs):
            yield ("text", "最初の応答[SWITCH_ANGLE:fastModel|軽く]")

        first_provider.generate_stream_typed = typed_stream_first

        second_provider = MagicMock()
        second_provider.SUPPORTS_TOOLS = False

        async def typed_stream_second(sp, msgs):
            second_captured_messages.extend(msgs)
            yield ("text", "軽い応答テキスト")

        second_provider.generate_stream_typed = typed_stream_second

        def fake_create_provider(provider_id, model, settings, **kwargs):
            if provider_id == "google":
                return second_provider
            return first_provider

        monkeypatch.setattr("backend.services.chat_flow.preparation.create_provider", fake_create_provider)
        monkeypatch.setattr("backend.services.chat_flow.preparation.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.services.chat_flow.preparation.find_urls", lambda t: [])
        service = ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())
        request = _make_request(available_presets=_SAMPLE_PRESETS)

        events = []
        async for event in service.execute_stream(request):
            events.append(event)

        event_types = [e[0] for e in events]
        all_texts = [e[1] for e in events if e[0] == "text"]

        # clear は yield されない（第1プロバイダーのテキストを消さない）
        assert "clear" not in event_types

        # 第1プロバイダーのテキストが含まれる（[SWITCH_ANGLE:...] タグは除去済み）
        assert any("最初の応答" in t for t in all_texts)

        # 第2プロバイダーのテキストも連続してストリームされる
        assert any("軽い応答テキスト" in t for t in all_texts)

        # 第2プロバイダーに渡された messages に第1プロバイダーの応答が assistant ターンとして含まれる
        assistant_msgs = [m for m in second_captured_messages if isinstance(m, dict) and m.get("role") == "assistant"]
        assert any("最初の応答" in (m.get("content") or "") for m in assistant_msgs)

    @pytest.mark.asyncio
    async def test_execute_stream_tools_no_switch_when_presets_empty(self, monkeypatch):
        """SUPPORTS_TOOLS=True で available_presets が空のとき、switch_angle ツール呼び出しを無視する。

        Bug Fix: SUPPORTS_TOOLS プロバイダーは available_presets が空でも switch_angle ツールを
        LLM に渡すため、LLM が誤呼び出しした場合でも switch / clear を発生させてはならない。
        第1プロバイダーの応答テキストがそのまま返り、angle_switched も clear も yield されない。
        """
        provider = MagicMock()
        provider.SUPPORTS_TOOLS = True

        async def mock_generate_with_tools(sys, msgs, tool_executor):
            # LLM が switch_angle を誤呼び出しする状況をシミュレートする
            tool_executor.execute("switch_angle", {"preset_name": "fastModel", "self_instruction": "軽く"})
            return "プロバイダーからの応答", ""

        provider.generate_with_tools = mock_generate_with_tools

        monkeypatch.setattr("backend.services.chat_flow.preparation.create_provider", lambda *a, **kw: provider)
        monkeypatch.setattr("backend.services.chat_flow.preparation.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.services.chat_flow.preparation.find_urls", lambda t: [])

        service = ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())
        # available_presets が空 = switch 無効
        request = _make_request(available_presets=[])

        events = []
        async for event in service.execute_stream(request):
            events.append(event)

        event_types = [e[0] for e in events]
        # switch は発生しない
        assert "angle_switched" not in event_types
        assert "clear" not in event_types
        # 第1プロバイダーのテキストがそのまま返る
        text_events = [e[1] for e in events if e[0] == "text"]
        assert any("プロバイダーからの応答" in t for t in text_events)

    @pytest.mark.asyncio
    async def test_execute_stream_no_switch_when_presets_empty(self, monkeypatch):
        """available_presets が空のとき switch_angle タグはスキャンされない。"""
        provider = MagicMock()
        provider.SUPPORTS_TOOLS = False

        async def typed_stream(sp, msgs):
            yield ("text", "本文[SWITCH_ANGLE:fastModel|軽く]")

        provider.generate_stream_typed = typed_stream

        monkeypatch.setattr("backend.services.chat_flow.preparation.create_provider", lambda *a, **kw: provider)
        monkeypatch.setattr("backend.services.chat_flow.preparation.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.services.chat_flow.preparation.find_urls", lambda t: [])
        service = ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())
        # available_presets が空 = switch 無効
        request = _make_request(available_presets=[])

        events = []
        async for event in service.execute_stream(request):
            events.append(event)

        event_types = [e[0] for e in events]
        # angle_switched は yield されない
        assert "angle_switched" not in event_types
        # clear も yield されない
        assert "clear" not in event_types
        # StreamingTagStripper が [SWITCH_ANGLE:] を除去するためテキストにタグは含まれない
        text_events = [e[1] for e in events if e[0] == "text"]
        assert not any("[SWITCH_ANGLE:" in t for t in text_events)


# ---------------------------------------------------------------------------
# build_system_prompt の switch_angle ブロック生成テスト
# ---------------------------------------------------------------------------

class TestSwitchAngleSystemPromptBlock:
    """build_system_prompt() と _build_switch_angle_block() の出力を検証する。"""

    def test_block_not_generated_when_no_presets(self):
        """available_presets が空のとき switch_angle ブロックは含まれない。"""
        prompt = build_system_prompt(
            "You are Alice.",
            available_presets=[],
        )
        assert "switch_angle" not in prompt
        assert "プリセット切り替え" not in prompt

    def test_block_generated_when_presets_exist(self):
        """available_presets が非空のとき switch_angle ブロックが含まれる。"""
        prompt = build_system_prompt(
            "You are Alice.",
            available_presets=_SAMPLE_PRESETS,
            current_preset_name="slowModel",
        )
        assert "プリセット切り替え" in prompt

    def test_available_presets_listed(self):
        """利用可能なプリセット名がブロックに列挙される。"""
        prompt = build_system_prompt(
            "You are Alice.",
            available_presets=_SAMPLE_PRESETS,
        )
        assert "fastModel" in prompt
        assert "deepModel" in prompt

    def test_when_to_switch_shown_when_set(self):
        """when_to_switch が設定されているプリセットはその説明が表示される。"""
        prompt = build_system_prompt(
            "You are Alice.",
            available_presets=_SAMPLE_PRESETS,
        )
        assert "軽い雑談のとき" in prompt

    def test_tool_use_instructions_when_use_tools_true(self):
        """use_tools=True のとき tool-use 形式の説明が含まれる。"""
        prompt = build_system_prompt(
            "You are Alice.",
            available_presets=_SAMPLE_PRESETS,
            use_tools=True,
        )
        assert "switch_angle" in prompt
        assert "[SWITCH_ANGLE:" not in prompt

    def test_tag_instructions_when_use_tools_false(self):
        """use_tools=False のときタグ形式の説明が含まれる。"""
        prompt = build_system_prompt(
            "You are Alice.",
            available_presets=_SAMPLE_PRESETS,
            use_tools=False,
        )
        assert "[SWITCH_ANGLE:" in prompt

    def test_build_switch_angle_block_direct_tool_use(self):
        """_build_switch_angle_block の tool-use 形式出力を直接検証する。"""
        block = _build_switch_angle_block(_SAMPLE_PRESETS, use_tools=True)
        assert "fastModel" in block
        assert "軽い雑談のとき" in block
        assert "switch_angle" in block
        assert "[SWITCH_ANGLE:" not in block

    def test_build_switch_angle_block_direct_tag_mode(self):
        """_build_switch_angle_block のタグ方式出力を直接検証する。"""
        block = _build_switch_angle_block(_SAMPLE_PRESETS, use_tools=False)
        assert "[SWITCH_ANGLE:" in block
        assert "gemini2FlashLite" in block or "fastModel" in block

    def test_preset_without_when_to_switch_has_no_empty_label(self):
        """when_to_switch が空のプリセットはコロン付きの説明が表示されない。"""
        block = _build_switch_angle_block(_SAMPLE_PRESETS, use_tools=False)
        # deepModel は when_to_switch が空 → "deepModel: " という形式にならない
        lines = block.split("\n")
        deep_lines = [l for l in lines if "deepModel" in l]
        assert len(deep_lines) > 0
        for line in deep_lines:
            assert not line.strip().endswith(":")
