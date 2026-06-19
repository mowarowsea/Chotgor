"""switch_angle — ToolExecutor・ツール定義・switch 情報抽出・リクエスト再構築のテスト。

テスト対象:
  - ToolExecutor._switch_angle() と switch_request 属性
  - ANTHROPIC_TOOLS / OPENAI_TOOLS への switch_angle 自動組み込み
  - BaseLLMProvider.generate_with_tools() の switch 検知ループ中断
  - ChatService._extract_switch_info()
  - ChatService._build_switched_request()
"""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.services.chat.models import ChatRequest, Message
from backend.services.chat.service import ChatService
from backend.character_actions.executor import ANTHROPIC_TOOLS, OPENAI_TOOLS, ToolCall, ToolExecutor, ToolTurnResult

from tests._switch_angle_helpers import _SAMPLE_PRESETS, _make_executor, _make_request

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

    def test_switch_angle_does_not_touch_memory_manager(self):
        """switch_angle はmemory_managerを呼び出さない。"""
        mm = MagicMock()
        executor = _make_executor(memory_manager=mm)
        executor.execute("switch_angle", {"preset_name": "p", "self_instruction": "i"})
        mm.write_inscribed_memory.assert_not_called()


# ---------------------------------------------------------------------------
# generate_with_tools ループの switch_angle 中断テスト
# ---------------------------------------------------------------------------

class TestGenerateWithToolsSwitchBreak:
    """generate_with_tools() が switch_angle 検知後にループを中断するかを検証する。"""

    def _make_provider(self, turn_results):
        """指定した ToolTurnResult を順に返すモックプロバイダーを生成するヘルパー。"""
        from backend.providers.base import BaseLLMProvider

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
        return ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())

    def test_tool_executor_with_switch_request(self):
        """SUPPORTS_TOOLS=True 方式: tool_executor.switch_request が設定されていれば返す。"""
        service = self._make_service()
        executor = _make_executor()
        # switch_angle を呼び出して switch_request を設定する
        executor.execute("switch_angle", {"preset_name": "fastModel", "self_instruction": "軽く"})
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
        """SUPPORTS_TOOLS=False 方式: apply_switch_angle_tags でタグ抽出 → executor.switch_request セット。"""
        service = self._make_service()
        executor = _make_executor()
        text = "こんにちは[SWITCH_ANGLE:fastModel|軽くさっぱりと]"
        clean_text = executor.apply_switch_angle_tags(text)
        _, switch_info = service._extract_switch_info(executor, clean_text, True)
        assert switch_info == ("fastModel", "軽くさっぱりと")
        assert "[SWITCH_ANGLE:" not in clean_text

    def test_tag_mode_no_tag_returns_none(self):
        """SUPPORTS_TOOLS=False 方式: タグなしなら switch_info は None。"""
        service = self._make_service()
        executor = _make_executor()
        text = "タグなしのテキスト"
        clean_text = executor.apply_switch_angle_tags(text)
        _, switch_info = service._extract_switch_info(executor, clean_text, True)
        assert switch_info is None
        assert clean_text == text

    def test_tag_mode_skipped_when_no_presets(self):
        """has_angle_presets=False のときは switch_request をスキップして None を返す。

        テキスト中のタグ自体は apply_switch_angle_tags で除去できるが、本テストでは
        apply を呼ばない（呼び出し側が has_angle_presets=False を判断するパターン）
        ことで _extract_switch_info の挙動だけ検証する。
        """
        service = self._make_service()
        executor = _make_executor()
        text = "テキスト[SWITCH_ANGLE:fastModel|軽く]"
        # has_angle_presets=False なので即座に None を返し、タグは触らない
        clean_text, switch_info = service._extract_switch_info(executor, text, False)
        assert switch_info is None
        assert "[SWITCH_ANGLE:" in clean_text

    def test_tool_executor_returns_none_when_no_presets(self):
        """SUPPORTS_TOOLS=True 方式: has_angle_presets=False のとき switch_request があっても None を返す。

        Bug Fix: SUPPORTS_TOOLS プロバイダーは available_presets が空でも switch_angle ツールを
        LLM に渡し続けるため、LLM が誤呼び出しした場合に switch を黙って無視するガードが必要。
        """
        service = self._make_service()
        executor = _make_executor()
        executor.execute("switch_angle", {"preset_name": "fastModel", "self_instruction": "軽く"})
        # switch_request は設定されているが has_angle_presets=False → None を返すべき
        clean_text, switch_info = service._extract_switch_info(executor, "some text", False)
        assert switch_info is None
        assert clean_text == "some text"


# ---------------------------------------------------------------------------
# ChatService._build_switched_request テスト
# ---------------------------------------------------------------------------

class TestBuildSwitchedRequest:
    """ChatService._build_switched_request() の挙動を検証する。"""

    def _make_service(self):
        """テスト用 ChatService を生成するヘルパー。"""
        return ChatService(memory_manager=MagicMock(), working_memory_manager=MagicMock())

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
        # self_instruction が空なら畳み込みは起きず、プリセットの追記がそのまま入る
        switched = service._build_switched_request(request, "fastModel", "")
        assert switched is not None
        assert switched.provider_additional_instructions == "簡潔に"

    def test_thinking_level_updated(self):
        """切り替え後のリクエストが新プリセットの thinking_level を持つ。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(request, "deepModel", "深く考える")
        assert switched is not None
        assert switched.thinking_level == "high"

    def test_self_instruction_folded_into_additional_instructions(self):
        """self_instruction はプロバイダー固有追記（Block 5）の末尾に畳み込まれる。

        ワーキングメモリ移行で SELF_DRIFT / active_drifts は廃止され、
        切り替え後モデルへの自己指針は additional_instructions に統合される。
        """
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        # fastModel の additional_instructions は "簡潔に"
        switched = service._build_switched_request(request, "fastModel", "軽くさっぱり")
        assert switched is not None
        assert switched.provider_additional_instructions == "簡潔に\n\n軽くさっぱり"

    def test_self_instruction_alone_when_preset_has_no_instructions(self):
        """プリセットに additional_instructions が無い場合、self_instruction だけが入る。"""
        service = self._make_service()
        request = _make_request(available_presets=_SAMPLE_PRESETS)
        # deepModel の additional_instructions は空文字
        switched = service._build_switched_request(request, "deepModel", "深く考える")
        assert switched is not None
        assert switched.provider_additional_instructions == "深く考える"

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

    def test_original_messages_preserved_when_no_first_response(self):
        """first_response_text が空のとき元のメッセージリストがそのまま引き継がれる。"""
        service = self._make_service()
        msgs = [Message(role="user", content="テスト")]
        request = _make_request(messages=msgs, available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(request, "fastModel", "軽く", first_response_text="")
        assert switched is not None
        assert switched.messages == msgs

    def test_first_response_text_appended_to_messages(self):
        """first_response_text が指定された場合、assistant ターンとして messages の末尾に追加される。

        第1プロバイダーの応答を第2プロバイダーに会話文脈として引き継ぐための仕組み。
        """
        service = self._make_service()
        msgs = [Message(role="user", content="テスト")]
        request = _make_request(messages=msgs, available_presets=_SAMPLE_PRESETS)
        switched = service._build_switched_request(
            request, "fastModel", "軽く", first_response_text="第1プロバイダーの返答"
        )
        assert switched is not None
        assert len(switched.messages) == 2
        assert switched.messages[-1].role == "assistant"
        assert switched.messages[-1].content == "第1プロバイダーの返答"


