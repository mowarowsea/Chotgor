"""SelfReflector — 会話整形と run()（disabled / always モード）のユニットテスト。

テスト方針:
    - LLMプロバイダーは AsyncMock で差し替える
    - _run_reflection() はプロバイダーの SUPPORTS_TOOLS に応じて分岐する:
        - tool-use対応（claude_cli / anthropic 等）→ ask_character_with_tools() を呼ぶ
        - tool-use非対応（ollama / openrouter 等）→ ask_character() + タグパース
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.character_actions.reflector import _format_conversation

from tests._self_reflection_helpers import (  # noqa: F401
    _make_provider,
    char_id,
    google_trigger_preset_id,
    memory_manager,
    reflection_preset_id,
    reflector,
    tools_reflection_preset_id,
    trigger_preset_id,
    working_memory_manager,
)

# ─── _format_conversation ─────────────────────────────────────────────────────


class TestFormatConversation:
    """_format_conversation() のテキスト整形を検証する。"""

    def test_user_message_is_labeled_correctly(self):
        """userロールが「[ユーザー]:」で出力されること。"""
        messages = [{"role": "user", "content": "こんにちは"}]
        result = _format_conversation(messages)
        assert "[ユーザー]: こんにちは" in result

    def test_assistant_message_is_labeled_correctly(self):
        """assistantロールが「[キャラクター]:」で出力されること。"""
        messages = [{"role": "assistant", "content": "やあ"}]
        result = _format_conversation(messages)
        assert "[キャラクター]: やあ" in result

    def test_multiple_turns_are_joined_with_newlines(self):
        """複数ターンが改行区切りで結合されること。"""
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = _format_conversation(messages)
        lines = result.split("\n")
        assert len(lines) == 3

    def test_list_content_extracts_text_parts(self):
        """contentがリスト形式（マルチモーダル）の場合、textパートのみ抽出されること。"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "テキスト部分"},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                ],
            }
        ]
        result = _format_conversation(messages)
        assert "テキスト部分" in result
        assert "image_url" not in result

    def test_empty_messages_returns_empty_string(self):
        """空リストを渡すと空文字列が返ること。"""
        assert _format_conversation([]) == ""

    def test_system_role_is_excluded(self):
        """systemロールのメッセージは出力に含まれないこと。"""
        messages = [{"role": "system", "content": "システムプロンプト"}]
        result = _format_conversation(messages)
        assert result == ""


# ─── SelfReflector.run() — disabled モード ───────────────────────────────────


class TestSelfReflectorDisabled:
    """self_reflection_mode が disabled のとき、LLMコールが一切発生しないことを検証する。"""

    def test_disabled_mode_skips_ask_character(self, reflector):
        """disabled モードでは ask_character() が呼ばれないこと。"""
        with patch("backend.character_actions.reflector.SelfReflector._run_reflection") as mock_run:
            asyncio.run(
                reflector.run(
                    request_mode="disabled",
                    trigger_preset_id="",
                    n_turns=5,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id="char-1",
                    session_id="sess-1",
                    current_preset_id="",
                )
            )
        mock_run.assert_not_called()

    def test_disabled_mode_returns_none(self, reflector):
        """disabled モードは None を返すこと（副作用なし）。"""
        result = asyncio.run(
            reflector.run(
                request_mode="disabled",
                trigger_preset_id="",
                n_turns=5,
                settings={},
                messages=[],
                character_id="char-1",
                session_id="sess-1",
                current_preset_id="",
            )
        )
        assert result is None


# ─── SelfReflector.run() — always モード ─────────────────────────────────────


class TestSelfReflectorAlways:
    """self_reflection_mode が always のとき、ask_character() が必ず呼ばれることを検証する。"""

    def test_always_mode_calls_ask_character(self, reflector, sqlite_store, char_id, reflection_preset_id):
        """always モードでは ask_character() が必ず呼ばれること。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character",
            new=AsyncMock(return_value=""),
        ) as mock_ask:
            asyncio.run(
                reflector.run(
                    request_mode="always",
                    trigger_preset_id="",
                    n_turns=5,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    current_preset_id=reflection_preset_id,
                )
            )
        mock_ask.assert_called_once()

    def test_always_mode_does_not_require_trigger_preset(self, reflector, sqlite_store, char_id, reflection_preset_id):
        """always モードでは trigger_preset_id が空でも正常に動作すること（例外なし）。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch("backend.character_actions.reflector.ask_character", new=AsyncMock(return_value="")):
            asyncio.run(
                reflector.run(
                    request_mode="always",
                    trigger_preset_id="",  # 空でもOK
                    n_turns=5,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    current_preset_id=reflection_preset_id,
                )
            )

    def test_always_mode_applies_carve_narrative_tag(self, reflector, sqlite_store, char_id, reflection_preset_id):
        """always モードで返された CARVE_NARRATIVE タグが inner_narrative に反映されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character",
            new=AsyncMock(return_value="[CARVE_NARRATIVE:append|自己参照による気づき]"),
        ):
            asyncio.run(
                reflector.run(
                    request_mode="always",
                    trigger_preset_id="",
                    n_turns=5,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    current_preset_id=reflection_preset_id,
                )
            )
        char = sqlite_store.get_character(char_id)
        assert "自己参照による気づき" in char.inner_narrative

    def test_always_mode_empty_response_does_nothing(self, reflector, sqlite_store, char_id, reflection_preset_id):
        """always モードで ask_character() が空文字を返した場合、DB変更がないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character",
            new=AsyncMock(return_value=""),
        ):
            asyncio.run(
                reflector.run(
                    request_mode="always",
                    trigger_preset_id="",
                    n_turns=5,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    current_preset_id=reflection_preset_id,
                )
            )
        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == ""

    def test_always_mode_n_turns_limits_window(self, reflector, sqlite_store, char_id, reflection_preset_id):
        """always モードで n_turns=2 のとき、最新2ターンのみが会話ウィンドウとして渡されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        captured_messages = []

        async def capture_ask(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return ""

        messages = [
            {"role": "user", "content": "古い発言1"},
            {"role": "assistant", "content": "古い応答1"},
            {"role": "user", "content": "新しい発言"},
            {"role": "assistant", "content": "新しい応答"},
        ]
        with patch("backend.character_actions.reflector.ask_character", side_effect=capture_ask):
            asyncio.run(
                reflector.run(
                    request_mode="always",
                    trigger_preset_id="",
                    n_turns=2,
                    settings={},
                    messages=messages,
                    character_id=char_id,
                    session_id="sess-1",
                    current_preset_id=reflection_preset_id,
                )
            )
        # ask_character に渡された messages の content（会話テキストに変換済み）を確認する
        assert len(captured_messages) == 1
        content = captured_messages[0]["content"]
        assert "古い発言1" not in content
        assert "新しい発言" in content


