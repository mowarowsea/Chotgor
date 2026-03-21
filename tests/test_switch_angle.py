"""switch_angle 機能の統合テスト。

switch_angle ツール / タグ方式 / ChatService 再ディスパッチロジック /
システムプロンプトブロック生成の全レイヤーを検証する。

テスト対象:
  - ToolExecutor._switch_angle() と switch_request 属性
  - ANTHROPIC_TOOLS / OPENAI_TOOLS への switch_angle 自動組み込み
  - BaseLLMProvider.generate_with_tools() の switch 検知ループ中断
  - ChatService._extract_switch_info()
  - ChatService._build_switched_request()
  - ChatService.execute() の switch 再ディスパッチ
  - ChatService.execute_stream() の switch 再ディスパッチ
  - build_system_prompt() の switch_angle ブロック生成
"""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.chat.models import ChatRequest, Message
from backend.core.chat.service import ChatService
from backend.core.system_prompt import _build_switch_angle_block, build_system_prompt
from backend.core.tools import ANTHROPIC_TOOLS, OPENAI_TOOLS, ToolCall, ToolExecutor, ToolTurnResult


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_executor(memory_manager=None, drift_manager=None, session_id="sess-1"):
    """テスト用 ToolExecutor を生成するヘルパー。"""
    return ToolExecutor(
        character_id="char-1",
        session_id=session_id,
        memory_manager=memory_manager or MagicMock(),
        drift_manager=drift_manager or MagicMock(),
    )


def _make_request(**kwargs) -> ChatRequest:
    """テスト用 ChatRequest を生成するヘルパー。"""
    defaults = dict(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="こんにちは")],
        session_id="sess-1",
    )
    defaults.update(kwargs)
    return ChatRequest(**defaults)


_SAMPLE_PRESETS = [
    {
        "preset_id": "preset-b",
        "preset_name": "fastModel",
        "provider": "google",
        "model_id": "gemini-2.0-flash-lite",
        "additional_instructions": "簡潔に",
        "thinking_level": "default",
        "when_to_switch": "軽い雑談のとき",
    },
    {
        "preset_id": "preset-c",
        "preset_name": "deepModel",
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "additional_instructions": "",
        "thinking_level": "high",
        "when_to_switch": "",
    },
]


# ---------------------------------------------------------------------------
# ToolExecutor.switch_angle テスト
# ---------------------------------------------------------------------------

class TestToolExecutorSwitchAngle:
    """switch_angle ツールの ToolExecutor への組み込みと挙動を検証する。"""

    def test_switch_angle_in_anthropic_tools(self):
        """ANTHROPIC_TOOLS に switch_angle エントリが存在する。"""
        names = {t["name"] for t in ANTHROPIC_TOOLS}
        assert "switch_angle" in names

    def test_switch_angle_in_openai_tools(self):
        """OPENAI_TOOLS に switch_angle エントリが自動生成されている。"""
        names = {t["function"]["name"] for t in OPENAI_TOOLS}
        assert "switch_angle" in names

    def test_switch_angle_schema_required_fields(self):
        """switch_angle ツールが preset_name と self_instruction を必須引数として持つ。"""
        tool = next(t for t in ANTHROPIC_TOOLS if t["name"] == "switch_angle")
        required = tool["input_schema"]["required"]
        assert "preset_name" in required
        assert "self_instruction" in required

    def test_switch_request_initially_none(self):
        """初期状態では switch_request は None である。"""
        executor = _make_executor()
        assert executor.switch_request is None

    def test_switch_angle_sets_switch_request(self):
        """switch_angle ツールを呼び出すと switch_request にリクエストが格納される。"""
        executor = _make_executor()
        result = executor.execute(
            "switch_angle",
            {"preset_name": "fastModel", "self_instruction": "軽くさっぱりと"},
        )
        assert executor.switch_request == ("fastModel", "軽くさっぱりと")
        assert "fastModel" in result

    def test_switch_angle_does_not_touch_drift_manager(self):
        """switch_angle はdrift_managerを呼び出さない（drift操作は再ディスパッチ後に行う）。"""
        dm = MagicMock()
        executor = _make_executor(drift_manager=dm)
        executor.execute("switch_angle", {"preset_name": "p", "self_instruction": "i"})
        dm.add_drift.assert_not_called()
        dm.reset_drifts.assert_not_called()

    def test_switch_angle_does_not_touch_memory_manager(self):
        """switch_angle はmemory_managerを呼び出さない。"""
        mm = MagicMock()
        executor = _make_executor(memory_manager=mm)
        executor.execute("switch_angle", {"preset_name": "p", "self_instruction": "i"})
        mm.write_memory.assert_not_called()


# ---------------------------------------------------------------------------
# generate_with_tools ループの switch_angle 中断テスト
# ---------------------------------------------------------------------------

class TestGenerateWithToolsSwitchBreak:
    """generate_with_tools() が switch_angle 検知後にループを中断するかを検証する。"""

    def _make_provider(self, turn_results):
        """指定した ToolTurnResult を順に返すモックプロバイダーを生成するヘルパー。"""
        from backend.core.providers.base import BaseLLMProvider

        class MockProvider(BaseLLMProvider):
            SUPPORTS_TOOLS = True

            def __init__(self):
                self._turns = iter(turn_results)
                self.extend_call_count = 0

            async def _tool_turn(self, system_prompt, messages):
                return next(self._turns)

            def _extend_messages_with_results(self, messages, turn_result, results):
                self.extend_call_count += 1
                return messages + [{"role": "tool_result", "content": "ok"}]

        return MockProvider()

    def test_switch_angle_breaks_loop_immediately(self):
        """switch_angle 呼び出し後に _extend_messages_with_results が呼ばれない。"""
        switch_tc = ToolCall(
            id="tc-sw", name="switch_angle",
            input={"preset_name": "fastModel", "self_instruction": "軽く"}
        )
        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[switch_tc]),
            # このターンは呼ばれてはいけない
            ToolTurnResult(text="不正なターン", tool_calls=[]),
        ])
        executor = _make_executor()
        asyncio.run(
            provider.generate_with_tools("sys", [], executor)
        )
        # switch 後は _extend_messages_with_results が呼ばれない
        assert provider.extend_call_count == 0

    def test_switch_angle_sets_executor_switch_request(self):
        """generate_with_tools() 後に tool_executor.switch_request が設定される。"""
        switch_tc = ToolCall(
            id="tc-sw", name="switch_angle",
            input={"preset_name": "deepModel", "self_instruction": "深く考える"}
        )
        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[switch_tc]),
        ])
        executor = _make_executor()
        asyncio.run(
            provider.generate_with_tools("sys", [], executor)
        )
        assert executor.switch_request == ("deepModel", "深く考える")

    def test_non_switch_tools_still_extend_messages(self):
        """switch_angle 以外のツールでは通常通りメッセージが拡張される。"""
        carve_tc = ToolCall(
            id="tc-1", name="carve_memory",
            input={"content": "test", "category": "user", "impact": 1.0}
        )
        provider = self._make_provider([
            ToolTurnResult(text="", tool_calls=[carve_tc]),
            ToolTurnResult(text="完了", tool_calls=[]),
        ])
        executor = _make_executor()
        asyncio.run(
            provider.generate_with_tools("sys", [], executor)
        )
        assert provider.extend_call_count == 1
        assert executor.switch_request is None


# ---------------------------------------------------------------------------
# ChatService._extract_switch_info テスト
# ---------------------------------------------------------------------------

class TestExtractSwitchInfo:
    """ChatService._extract_switch_info() の挙動を検証する。"""

    def _make_service(self):
        """テスト用 ChatService を生成するヘルパー。"""
        return ChatService(memory_manager=MagicMock(), drift_manager=MagicMock())

    def test_tool_executor_with_switch_request(self):
        """SUPPORTS_TOOLS=True 方式: tool_executor.switch_request が設定されていれば返す。"""
        service = self._make_service()
        executor = _make_executor()
        executor.switch_request = ("fastModel", "軽く")
        clean_text, switch_info = service._extract_switch_info(executor, "some text", True)
        assert switch_info == ("fastModel", "軽く")
        assert clean_text == "some text"

    def test_tool_executor_no_switch_request_returns_none(self):
        """SUPPORTS_TOOLS=True 方式: switch_request が None なら None を返す。"""
        service = self._make_service()
        executor = _make_executor()
        clean_text, switch_info = service._extract_switch_info(executor, "some text", True)
        assert switch_info is None

    def test_tag_mode_extracts_tag(self):
        """SUPPORTS_TOOLS=False 方式: テキストから [SWITCH_ANGLE:...] を抽出する。"""
        service = self._make_service()
        text = "こんにちは[SWITCH_ANGLE:fastModel|軽くさっぱりと]"
        clean_text, switch_info = service._extract_switch_info(None, text, True)
        assert switch_info == ("fastModel", "軽くさっぱりと")
        assert "[SWITCH_ANGLE:" not in clean_text

    def test_tag_mode_no_tag_returns_none(self):
        """SUPPORTS_TOOLS=False 方式: タグなしなら None を返す。"""
        service = self._make_service()
        text = "タグなしのテキスト"
        clean_text, switch_info = service._extract_switch_info(None, text, True)
        assert switch_info is None
        assert clean_text == text

    def test_tag_mode_skipped_when_no_presets(self):
        """has_angle_presets=False のときはタグスキャンをスキップして None を返す。"""
        service = self._make_service()
        text = "テキスト[SWITCH_ANGLE:fastModel|軽く]"
        clean_text, switch_info = service._extract_switch_info(None, text, False)
        # スキャンをスキップするのでタグは除去されず switch_info も None
        assert switch_info is None
        assert "[SWITCH_ANGLE:" in clean_text


# ---------------------------------------------------------------------------
# ChatService._build_switched_request テスト
# ---------------------------------------------------------------------------

class TestBuildSwitchedRequest:
    """ChatService._build_switched_request() の挙動を検証する。"""

    def _make_service(self, drift_manager=None):
        """テスト用 ChatService を生成するヘルパー。"""
        dm = drift_manager or MagicMock()
        return ChatService(memory_manager=MagicMock(), drift_manager=dm)

    def test_returns_none_when_preset_not_found(self):
        """存在しないプリセット名のとき None を返す。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        result = service._build_switched_request(request, "nonexistent", "instruction")
        assert result is None

    def test_switches_provider_and_model(self):
        """切り替え後のリクエストが正しい provider / model を持つ。"""
        service = self._make_service()
        request = _make_request(
            provider="anthropic",
            model="claude-sonnet-4-6",
            available_presets=_SAMPLE_PRESETS,
        )
        switched = service._build_switched_request(request, "fastModel", "軽く")
        assert switched is not None
        assert switched.provider == "google"
        assert switched.model == "gemini-2.0-flash-lite"

    def test_additional_instructions_updated(self):
        """切り替え後のリクエストが新プリセットの additional_instructions を持つ。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(request, "fastModel", "軽く")
        assert switched is not None
        assert switched.provider_additional_instructions == "簡潔に"

    def test_thinking_level_updated(self):
        """切り替え後のリクエストが新プリセットの thinking_level を持つ。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(request, "deepModel", "深く考える")
        assert switched is not None
        assert switched.thinking_level == "high"

    def test_active_drifts_set_to_self_instruction(self):
        """切り替え後のリクエストの active_drifts に self_instruction が設定される。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(request, "fastModel", "軽くさっぱり")
        assert switched is not None
        assert switched.active_drifts == ["軽くさっぱり"]

    def test_active_drifts_empty_when_no_self_instruction(self):
        """self_instruction が空文字の場合、active_drifts は空リストになる。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(request, "fastModel", "")
        assert switched is not None
        assert switched.active_drifts == []

    def test_available_presets_cleared_for_no_infinite_loop(self):
        """再ディスパッチでの無限ループ防止のため available_presets が空になる。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(request, "fastModel", "軽く")
        assert switched is not None
        assert switched.available_presets == []

    def test_current_preset_name_updated(self):
        """current_preset_name が切り替え先プリセット名に更新される。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS, current_preset_name="slowModel")
        switched = service._build_switched_request(request, "fastModel", "軽く")
        assert switched is not None
        assert switched.current_preset_name == "fastModel"

    def test_drift_manager_reset_and_add_called(self):
        """切り替え時に drift_manager.reset_drifts と add_drift が呼ばれる。"""
        dm = MagicMock()
        service = self._make_service(drift_manager=dm)
        request = _make_request(available_presets=_SAMPLE_PRESETS, session_id="sess-abc")
        service._build_switched_request(request, "fastModel", "軽く")
        dm.reset_drifts.assert_called_once_with("sess-abc", "char-1")
        dm.add_drift.assert_called_once_with("sess-abc", "char-1", "軽く")

    def test_drift_manager_add_skipped_when_no_self_instruction(self):
        """self_instruction が空の場合は add_drift を呼ばない（reset のみ）。"""
        dm = MagicMock()
        service = self._make_service(drift_manager=dm)
        request = _make_request(available_presets=_SAMPLE_PRESETS, session_id="sess-abc")
        service._build_switched_request(request, "fastModel", "")
        dm.reset_drifts.assert_called_once()
        dm.add_drift.assert_not_called()

    def test_original_messages_preserved(self):
        """切り替え後も元のメッセージリストが保持される。"""
        service = self._make_service()
        msgs = [Message(role="user", content="テスト")]
        request = _make_request(messages=msgs, available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(request, "fastModel", "軽く")
        assert switched is not None
        assert switched.messages == msgs


# ---------------------------------------------------------------------------
# ChatService.execute() の switch_angle 再ディスパッチテスト
# ---------------------------------------------------------------------------

class TestChatServiceExecuteWithSwitch:
    """ChatService.execute() で switch_angle が発動したときの再ディスパッチを検証する。"""

    @pytest.mark.asyncio
    async def test_execute_switch_via_tag_redispatches(self, monkeypatch):
        """タグ方式: [SWITCH_ANGLE:...] が応答に含まれる場合、新プロバイダーで再ディスパッチされる。"""
        first_provider = AsyncMock()
        first_provider.SUPPORTS_TOOLS = False
        first_provider.generate = AsyncMock(return_value="本文[SWITCH_ANGLE:fastModel|軽く]")

        second_provider = AsyncMock()
        second_provider.SUPPORTS_TOOLS = False
        second_provider.generate = AsyncMock(return_value="軽い応答")

        def fake_create_provider(provider_id, model, settings, **kwargs):
            if provider_id == "google":
                return second_provider
            return first_provider

        monkeypatch.setattr("backend.core.chat.service.create_provider", fake_create_provider)
        monkeypatch.setattr("backend.core.chat.service.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.core.chat.service.find_urls", lambda t: [])
        mock_inscriber = MagicMock()
        mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
        mock_carver = MagicMock()
        mock_carver.carve_narrative_from_text.side_effect = lambda text: text
        monkeypatch.setattr("backend.core.chat.service.Inscriber", lambda *_: mock_inscriber)
        monkeypatch.setattr("backend.core.chat.service.Carver", lambda *_: mock_carver)

        service = ChatService(memory_manager=MagicMock(), drift_manager=MagicMock())
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        result = await service.execute(request)

        # 結果は新プロバイダー（google）の応答
        assert result == "軽い応答"
        # 最初のプロバイダー（anthropic）は1回呼ばれる
        first_provider.generate.assert_called_once()
        # 新プロバイダー（google）も1回呼ばれる
        second_provider.generate.assert_called_once()

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
                return ""

        second_provider = AsyncMock()
        second_provider.SUPPORTS_TOOLS = False
        second_provider.generate = AsyncMock(return_value="軽い応答")

        def fake_create_provider(provider_id, model, settings, **kwargs):
            if provider_id == "google":
                return second_provider
            return FirstProvider()

        monkeypatch.setattr("backend.core.chat.service.create_provider", fake_create_provider)
        monkeypatch.setattr("backend.core.chat.service.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.core.chat.service.find_urls", lambda t: [])
        mock_inscriber = MagicMock()
        mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
        mock_carver = MagicMock()
        mock_carver.carve_narrative_from_text.side_effect = lambda text: text
        monkeypatch.setattr("backend.core.chat.service.Inscriber", lambda *_: mock_inscriber)
        monkeypatch.setattr("backend.core.chat.service.Carver", lambda *_: mock_carver)

        service = ChatService(memory_manager=MagicMock(), drift_manager=MagicMock())
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

        monkeypatch.setattr("backend.core.chat.service.create_provider", lambda *a, **kw: provider)
        monkeypatch.setattr("backend.core.chat.service.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.core.chat.service.find_urls", lambda t: [])
        mock_inscriber = MagicMock()
        mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
        mock_carver = MagicMock()
        mock_carver.carve_narrative_from_text.side_effect = lambda text: text
        monkeypatch.setattr("backend.core.chat.service.Inscriber", lambda *_: mock_inscriber)
        monkeypatch.setattr("backend.core.chat.service.Carver", lambda *_: mock_carver)

        service = ChatService(memory_manager=MagicMock(), drift_manager=MagicMock())
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

        monkeypatch.setattr("backend.core.chat.service.create_provider", fake_create_provider)
        monkeypatch.setattr("backend.core.chat.service.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.core.chat.service.find_urls", lambda t: [])
        mock_inscriber = MagicMock()
        mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
        mock_carver = MagicMock()
        mock_carver.carve_narrative_from_text.side_effect = lambda text: text
        monkeypatch.setattr("backend.core.chat.service.Inscriber", lambda *_: mock_inscriber)
        monkeypatch.setattr("backend.core.chat.service.Carver", lambda *_: mock_carver)

        service = ChatService(memory_manager=MagicMock(), drift_manager=MagicMock())
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
        assert "Alice" in angle_switched_event[1]
        assert "fastModel" in angle_switched_event[1]

    @pytest.mark.asyncio
    async def test_execute_stream_switch_text_from_second_provider(self, monkeypatch):
        """switch 後のテキストは新プロバイダーからのものである。"""
        first_provider = MagicMock()
        first_provider.SUPPORTS_TOOLS = False

        async def typed_stream_first(sp, msgs):
            yield ("text", "最初の応答[SWITCH_ANGLE:fastModel|軽く]")

        first_provider.generate_stream_typed = typed_stream_first

        second_provider = MagicMock()
        second_provider.SUPPORTS_TOOLS = False

        async def typed_stream_second(sp, msgs):
            yield ("text", "軽い応答テキスト")

        second_provider.generate_stream_typed = typed_stream_second

        def fake_create_provider(provider_id, model, settings, **kwargs):
            if provider_id == "google":
                return second_provider
            return first_provider

        monkeypatch.setattr("backend.core.chat.service.create_provider", fake_create_provider)
        monkeypatch.setattr("backend.core.chat.service.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.core.chat.service.find_urls", lambda t: [])
        mock_inscriber = MagicMock()
        mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
        mock_carver = MagicMock()
        mock_carver.carve_narrative_from_text.side_effect = lambda text: text
        monkeypatch.setattr("backend.core.chat.service.Inscriber", lambda *_: mock_inscriber)
        monkeypatch.setattr("backend.core.chat.service.Carver", lambda *_: mock_carver)

        service = ChatService(memory_manager=MagicMock(), drift_manager=MagicMock())
        request = _make_request(available_presets=_SAMPLE_PRESETS)

        text_events = []
        async for chunk_type, content in service.execute_stream(request):
            if chunk_type == "text":
                text_events.append(content)

        # 最初の応答("最初の応答...")はユーザーに渡らず、新プロバイダーの応答のみ
        assert any("軽い応答テキスト" in t for t in text_events)
        assert not any("最初の応答" in t for t in text_events)

    @pytest.mark.asyncio
    async def test_execute_stream_no_switch_when_presets_empty(self, monkeypatch):
        """available_presets が空のとき switch_angle タグはスキャンされない。"""
        provider = MagicMock()
        provider.SUPPORTS_TOOLS = False

        async def typed_stream(sp, msgs):
            yield ("text", "本文[SWITCH_ANGLE:fastModel|軽く]")

        provider.generate_stream_typed = typed_stream

        monkeypatch.setattr("backend.core.chat.service.create_provider", lambda *a, **kw: provider)
        monkeypatch.setattr("backend.core.chat.service.build_system_prompt", lambda **kw: "sys")
        monkeypatch.setattr("backend.core.chat.service.find_urls", lambda t: [])
        mock_inscriber = MagicMock()
        mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
        mock_carver = MagicMock()
        mock_carver.carve_narrative_from_text.side_effect = lambda text: text
        monkeypatch.setattr("backend.core.chat.service.Inscriber", lambda *_: mock_inscriber)
        monkeypatch.setattr("backend.core.chat.service.Carver", lambda *_: mock_carver)

        service = ChatService(memory_manager=MagicMock(), drift_manager=MagicMock())
        # available_presets が空 = switch 無効
        request = _make_request(available_presets=[])

        events = []
        async for event in service.execute_stream(request):
            events.append(event)

        event_types = [e[0] for e in events]
        # angle_switched は yield されない
        assert "angle_switched" not in event_types
        # タグはスキャンされないのでテキストにそのまま含まれる
        text_events = [e[1] for e in events if e[0] == "text"]
        assert any("[SWITCH_ANGLE:" in t for t in text_events)


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

    def test_current_preset_name_shown(self):
        """現在のプリセット名がブロックに表示される。"""
        prompt = build_system_prompt(
            "You are Alice.",
            available_presets=_SAMPLE_PRESETS,
            current_preset_name="slowModel",
        )
        assert "slowModel" in prompt

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
        block = _build_switch_angle_block(_SAMPLE_PRESETS, "currentPreset", use_tools=True)
        assert "currentPreset" in block
        assert "fastModel" in block
        assert "軽い雑談のとき" in block
        assert "switch_angle" in block
        assert "[SWITCH_ANGLE:" not in block

    def test_build_switch_angle_block_direct_tag_mode(self):
        """_build_switch_angle_block のタグ方式出力を直接検証する。"""
        block = _build_switch_angle_block(_SAMPLE_PRESETS, "currentPreset", use_tools=False)
        assert "currentPreset" in block
        assert "[SWITCH_ANGLE:" in block
        assert "gemini2FlashLite" in block or "fastModel" in block

    def test_preset_without_when_to_switch_has_no_empty_label(self):
        """when_to_switch が空のプリセットはコロン付きの説明が表示されない。"""
        block = _build_switch_angle_block(_SAMPLE_PRESETS, "", use_tools=False)
        # deepModel は when_to_switch が空 → "deepModel: " という形式にならない
        lines = block.split("\n")
        deep_lines = [l for l in lines if "deepModel" in l]
        assert len(deep_lines) > 0
        for line in deep_lines:
            assert not line.strip().endswith(":")
