"""SelfReflector — 契機判断（_detect_trigger / local_trigger）とエラー処理のユニットテスト。

テスト方針:
    - _detect_trigger() は create_provider() を直接呼ぶため、これを patch して検証する
    - エラー系（接続失敗・空応答・preset未発見・キャラ未発見）が
      サイレントにスキップされることを確認する
    - 契機判断のプロンプト構造: キャラクター設定がユーザーメッセージに入ること・
      システムプロンプトが中立の判断依頼であることを確認する
"""

import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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

# ─── SelfReflector._detect_trigger() — 契機判断 ──────────────────────────────


class TestDetectTrigger:
    """SelfReflector._detect_trigger() の YES/NO 解析とプロンプト構造を検証する。"""

    def _run_detect(self, reflector, response_text, trigger_preset_id, character_id, settings=None):
        """_detect_trigger() を同期的に実行するヘルパー。

        Args:
            reflector: テスト対象の SelfReflector インスタンス。
            response_text: モック契機判断プロバイダーが返すテキスト。
            trigger_preset_id: 使用するプリセットID。
            character_id: 対象キャラクターID。
            settings: グローバル設定 dict（省略時は空 dict）。

        Returns:
            _detect_trigger() の戻り値（True / False）。
        """
        with patch("backend.character_actions.reflector.create_provider") as mock_create:
            provider = _make_provider(response_text)
            mock_create.return_value = provider
            return asyncio.run(
                reflector._detect_trigger(
                    trigger_preset_id=trigger_preset_id,
                    settings=settings or {},
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=character_id,
                )
            )

    def test_yes_response_returns_true(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """契機判断プロバイダーが「YES」を返したとき True になること。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = self._run_detect(reflector, "YES", trigger_preset_id, char_id)
        assert result is True

    def test_yes_with_trailing_text_returns_true(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """「YES\n理由テキスト」のようにYESで始まる応答も True になること。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = self._run_detect(reflector, "YES\n感情の変化が見られます", trigger_preset_id, char_id)
        assert result is True

    def test_no_response_returns_false(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """契機判断プロバイダーが「NO」を返したとき False になること。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = self._run_detect(reflector, "NO", trigger_preset_id, char_id)
        assert result is False

    def test_empty_response_returns_false(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """契機判断プロバイダーが空文字を返したとき False になること。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = self._run_detect(reflector, "", trigger_preset_id, char_id)
        assert result is False

    def test_preset_not_found_returns_false(self, reflector, sqlite_store, char_id):
        """存在しないプリセットIDを渡したとき False を返すこと（例外を送出しない）。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = asyncio.run(
            reflector._detect_trigger(
                trigger_preset_id="nonexistent-preset-id",
                settings={},
                conversation_window=[{"role": "user", "content": "テスト"}],
                character_id=char_id,
            )
        )
        assert result is False

    def test_character_not_found_returns_false(self, reflector, sqlite_store, trigger_preset_id):
        """存在しないキャラクターIDを渡したとき False を返すこと（例外を送出しない）。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = asyncio.run(
            reflector._detect_trigger(
                trigger_preset_id=trigger_preset_id,
                settings={},
                conversation_window=[{"role": "user", "content": "テスト"}],
                character_id="nonexistent-char",
            )
        )
        assert result is False

    def test_trigger_connection_error_returns_false(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """契機判断プロバイダーへの接続が失敗したとき False を返すこと（例外を送出しない）。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch("backend.character_actions.reflector.create_provider") as mock_create:
            provider = MagicMock()
            provider.generate = AsyncMock(side_effect=ConnectionError("provider unreachable"))
            mock_create.return_value = provider
            result = asyncio.run(
                reflector._detect_trigger(
                    trigger_preset_id=trigger_preset_id,
                    settings={},
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                )
            )
        assert result is False

    def test_trigger_uses_neutral_system_prompt(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """契機判断のシステムプロンプトがキャラクター設定を含まない中立の文言であること。

        キャラクター設定はユーザーメッセージ側に移動したため、
        システムプロンプトはキャラクターの人物設定を持たないことを検証する。
        """
        reflector.memory_manager.sqlite = sqlite_store
        captured = {}

        def capture_generate(*args, **kwargs):
            captured["system_prompt"] = args[0] if args else kwargs.get("system_prompt", "")
            return AsyncMock(return_value="NO")()

        with patch("backend.character_actions.reflector.create_provider") as mock_create:
            provider = MagicMock()
            provider.generate = capture_generate
            mock_create.return_value = provider
            asyncio.run(
                reflector._detect_trigger(
                    trigger_preset_id=trigger_preset_id,
                    settings={},
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                )
            )
        # システムプロンプトはキャラクターの設定テキストを含まないこと
        assert "テスト用設定" not in captured.get("system_prompt", "")

    def test_trigger_user_message_contains_character_context(
        self, reflector, sqlite_store, char_id, trigger_preset_id
    ):
        """契機判断のユーザーメッセージにキャラクター設定が含まれること。

        キャラクター設定がシステムプロンプトではなくユーザーメッセージに
        入力として渡されていることを検証する。
        """
        reflector.memory_manager.sqlite = sqlite_store
        captured = {}

        def capture_generate(system_prompt, messages, **kwargs):
            captured["messages"] = messages
            f = AsyncMock(return_value="NO")
            return f()

        with patch("backend.character_actions.reflector.create_provider") as mock_create:
            provider = MagicMock()
            provider.generate = capture_generate
            mock_create.return_value = provider
            asyncio.run(
                reflector._detect_trigger(
                    trigger_preset_id=trigger_preset_id,
                    settings={},
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                )
            )
        user_content = captured.get("messages", [{}])[0].get("content", "")
        # キャラクターの設定テキストがユーザーメッセージに含まれること
        assert "テスト用設定" in user_content

    def test_non_ollama_preset_uses_correct_provider(
        self, reflector, sqlite_store, char_id, google_trigger_preset_id
    ):
        """GoogleプリセットのIDを渡したとき、create_provider が "google" で呼ばれること。

        Ollamaにハードコードされていたバグのリグレッションテストとして機能する。
        """
        reflector.memory_manager.sqlite = sqlite_store
        with patch("backend.character_actions.reflector.create_provider") as mock_create:
            mock_create.return_value = _make_provider("NO")
            asyncio.run(
                reflector._detect_trigger(
                    trigger_preset_id=google_trigger_preset_id,
                    settings={},
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                )
            )
        called_provider_id = mock_create.call_args[0][0]
        assert called_provider_id == "google"

    def test_ollama_preset_uses_ollama_provider(
        self, reflector, sqlite_store, char_id, trigger_preset_id
    ):
        """OllamaプリセットのIDを渡したとき、create_provider が "ollama" で呼ばれること。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch("backend.character_actions.reflector.create_provider") as mock_create:
            mock_create.return_value = _make_provider("NO")
            asyncio.run(
                reflector._detect_trigger(
                    trigger_preset_id=trigger_preset_id,
                    settings={},
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                )
            )
        called_provider_id = mock_create.call_args[0][0]
        assert called_provider_id == "ollama"


# ─── SelfReflector.run() — local_trigger モード ──────────────────────────────


class TestSelfReflectorLocalTrigger:
    """self_reflection_mode が local_trigger のとき、契機判断の結果で分岐することを検証する。"""

    def test_local_trigger_yes_calls_run_reflection(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """local_trigger + 契機判断=YES のとき、_run_reflection() が呼ばれること。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch.object(reflector, "_detect_trigger", new=AsyncMock(return_value=True)):
            with patch.object(reflector, "_run_reflection", new=AsyncMock()) as mock_reflect:
                asyncio.run(
                    reflector.run(
                        request_mode="local_trigger",
                        trigger_preset_id=trigger_preset_id,
                        n_turns=5,
                        settings={},
                        messages=[{"role": "user", "content": "テスト"}],
                        character_id=char_id,
                        session_id="sess-1",
                        current_preset_id="preset-x",
                    )
                )
        mock_reflect.assert_called_once()

    def test_local_trigger_no_skips_run_reflection(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """local_trigger + 契機判断=NO のとき、_run_reflection() が呼ばれないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch.object(reflector, "_detect_trigger", new=AsyncMock(return_value=False)):
            with patch.object(reflector, "_run_reflection", new=AsyncMock()) as mock_reflect:
                asyncio.run(
                    reflector.run(
                        request_mode="local_trigger",
                        trigger_preset_id=trigger_preset_id,
                        n_turns=5,
                        settings={},
                        messages=[{"role": "user", "content": "テスト"}],
                        character_id=char_id,
                        session_id="sess-1",
                        current_preset_id="preset-x",
                    )
                )
        mock_reflect.assert_not_called()

    def test_local_trigger_without_preset_id_skips_silently(self, reflector, sqlite_store, char_id):
        """local_trigger で trigger_preset_id が未設定のとき、黙ってスキップすること（例外なし）。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch.object(reflector, "_run_reflection", new=AsyncMock()) as mock_reflect:
            asyncio.run(
                reflector.run(
                    request_mode="local_trigger",
                    trigger_preset_id="",  # 未設定
                    n_turns=5,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    current_preset_id="",
                )
            )
        mock_reflect.assert_not_called()

    def test_local_trigger_yes_applies_tags_to_db(
        self, reflector, sqlite_store, char_id, trigger_preset_id, reflection_preset_id
    ):
        """local_trigger + 契機判断=YES で返されたタグがDBに反映されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch.object(reflector, "_detect_trigger", new=AsyncMock(return_value=True)):
            with patch(
                "backend.character_actions.reflector.ask_character",
                new=AsyncMock(return_value="[CARVE_NARRATIVE:append|契機判断後の気づき]"),
            ):
                asyncio.run(
                    reflector.run(
                        request_mode="local_trigger",
                        trigger_preset_id=trigger_preset_id,
                        n_turns=5,
                        settings={},
                        messages=[{"role": "user", "content": "テスト"}],
                        character_id=char_id,
                        session_id="sess-1",
                        current_preset_id=reflection_preset_id,
                    )
                )
        char = sqlite_store.get_character(char_id)
        assert "契機判断後の気づき" in char.inner_narrative


# ─── SelfReflector._run_reflection() — エラーハンドリング ────────────────────


class TestRunReflectionErrorHandling:
    """_run_reflection() が ask_character() 失敗時にサイレントにスキップすることを検証する。"""

    def test_ask_character_none_does_not_crash(self, reflector, sqlite_store, char_id, reflection_preset_id):
        """ask_character() が None を返しても _run_reflection() が例外を伝播しないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character",
            new=AsyncMock(return_value=None),
        ):
            asyncio.run(
                reflector._run_reflection(
                    preset_id=reflection_preset_id,
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    settings={},
                )
            )
        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == ""  # DB変更なし

    def test_whitespace_only_response_does_nothing(self, reflector, sqlite_store, char_id, reflection_preset_id):
        """ask_character() が空白のみを返した場合、DB変更がないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character",
            new=AsyncMock(return_value="   \n  "),
        ):
            asyncio.run(
                reflector._run_reflection(
                    preset_id=reflection_preset_id,
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    settings={},
                )
            )
        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == ""


# ─── SelfReflector._run_reflection() — MCPツール方式 ────────────────────────


class TestRunReflectionToolsPath:
    """SUPPORTS_TOOLS=True のプロバイダー使用時に ask_character_with_tools() が呼ばれることを検証する。

    MCP ツール方式では ask_character() は呼ばれず、
    carve_narrative / drift はツールコール経由でキャラクター自身が実行する。
    """

    def test_tools_provider_calls_ask_character_with_tools(
        self, reflector, sqlite_store, char_id, tools_reflection_preset_id
    ):
        """SUPPORTS_TOOLS=True プリセット使用時に ask_character_with_tools() が呼ばれること。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character_with_tools",
            new=AsyncMock(return_value=True),
        ) as mock_with_tools, patch(
            "backend.character_actions.reflector.ask_character",
            new=AsyncMock(return_value=""),
        ) as mock_without_tools:
            asyncio.run(
                reflector._run_reflection(
                    preset_id=tools_reflection_preset_id,
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    settings={},
                )
            )
        mock_with_tools.assert_called_once()
        mock_without_tools.assert_not_called()

    def test_tools_path_passes_session_id(
        self, reflector, sqlite_store, char_id, tools_reflection_preset_id
    ):
        """MCPツール方式では session_id が ask_character_with_tools() に渡されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character_with_tools",
            new=AsyncMock(return_value=True),
        ) as mock_with_tools:
            asyncio.run(
                reflector._run_reflection(
                    preset_id=tools_reflection_preset_id,
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="my-session-id",
                    settings={},
                )
            )
        call_kwargs = mock_with_tools.call_args.kwargs
        assert call_kwargs.get("session_id") == "my-session-id"

    def test_tools_path_feature_label_is_reflection(
        self, reflector, sqlite_store, char_id, tools_reflection_preset_id
    ):
        """MCPツール方式では feature_label="reflection" が渡されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character_with_tools",
            new=AsyncMock(return_value=True),
        ) as mock_with_tools:
            asyncio.run(
                reflector._run_reflection(
                    preset_id=tools_reflection_preset_id,
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    settings={},
                )
            )
        call_kwargs = mock_with_tools.call_args.kwargs
        assert call_kwargs.get("feature_label") == "reflection"

    def test_non_tools_provider_does_not_call_ask_character_with_tools(
        self, reflector, sqlite_store, char_id, reflection_preset_id
    ):
        """SUPPORTS_TOOLS=False プリセット使用時は ask_character_with_tools() が呼ばれないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch(
            "backend.character_actions.reflector.ask_character_with_tools",
            new=AsyncMock(return_value=True),
        ) as mock_with_tools, patch(
            "backend.character_actions.reflector.ask_character",
            new=AsyncMock(return_value=""),
        ):
            asyncio.run(
                reflector._run_reflection(
                    preset_id=reflection_preset_id,
                    conversation_window=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    settings={},
                )
            )
        mock_with_tools.assert_not_called()


# ─── SQLite スキーマ — 自己参照カラムのデフォルト値 ──────────────────────────


class TestSelfReflectionSchema:
    """自己参照フィールドのデフォルト値とDBへの読み書きを検証する。"""

    def test_new_character_has_disabled_mode_by_default(self, sqlite_store):
        """新規キャラクターの self_reflection_mode デフォルト値が 'disabled' であること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="デフォルトキャラ")
        char = sqlite_store.get_character(cid)
        assert char.self_reflection_mode == "disabled"

    def test_new_character_has_null_preset_id_by_default(self, sqlite_store):
        """新規キャラクターの self_reflection_preset_id デフォルト値が None であること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="デフォルトキャラ2")
        char = sqlite_store.get_character(cid)
        assert char.self_reflection_preset_id is None

    def test_new_character_has_n_turns_5_by_default(self, sqlite_store):
        """新規キャラクターの self_reflection_n_turns デフォルト値が 5 であること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="デフォルトキャラ3")
        char = sqlite_store.get_character(cid)
        assert char.self_reflection_n_turns == 5

    def test_create_character_with_local_trigger_mode(self, sqlite_store):
        """local_trigger モードで作成したキャラクターが正しくDBに保存されること。"""
        cid = str(uuid.uuid4())
        sqlite_store.create_character(
            character_id=cid,
            name="LocalTriggerキャラ",
            self_reflection_mode="local_trigger",
            self_reflection_preset_id="preset-abc",
            self_reflection_n_turns=10,
        )
        char = sqlite_store.get_character(cid)
        assert char.self_reflection_mode == "local_trigger"
        assert char.self_reflection_preset_id == "preset-abc"
        assert char.self_reflection_n_turns == 10

    def test_update_character_changes_reflection_mode(self, sqlite_store, char_id):
        """update_character() で self_reflection_mode を変更できること。"""
        sqlite_store.update_character(char_id, self_reflection_mode="always")
        char = sqlite_store.get_character(char_id)
        assert char.self_reflection_mode == "always"

    def test_update_character_changes_n_turns(self, sqlite_store, char_id):
        """update_character() で self_reflection_n_turns を変更できること。"""
        sqlite_store.update_character(char_id, self_reflection_n_turns=8)
        char = sqlite_store.get_character(char_id)
        assert char.self_reflection_n_turns == 8
