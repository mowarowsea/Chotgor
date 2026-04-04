"""backend.character_actions.reflector モジュールのユニットテスト。

自己参照ループ（SelfReflector）の動作を検証する。

対象クラス・関数:
    SelfReflector.run()              — エントリポイント。モード・設定に応じて処理を振り分ける
    SelfReflector._detect_trigger()  — 中立プロバイダーによる契機判断（YES/NO応答の解析を含む）
    SelfReflector._run_reflection()  — ask_character() 経由の自己参照コール＋タグ処理
    _format_conversation()           — メッセージリストを整形する内部ユーティリティ

テスト方針:
    - LLMプロバイダーは AsyncMock で差し替える
    - _run_reflection() は内部で ask_character() を呼ぶため、これを patch して検証する
    - _detect_trigger() は create_provider() を直接呼ぶため、これを patch して検証する
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
    - エラー系（接続失敗・空応答・preset未発見・キャラ未発見）がサイレントにスキップされることを確認する
    - 契機判断のプロンプト構造: キャラクター設定がユーザーメッセージに入ること・
      システムプロンプトが中立の判断依頼であることを確認する
"""

import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.character_actions.reflector import SelfReflector, _format_conversation


# ─── フィクスチャ ──────────────────────────────────────────────────────────────
# sqlite_store は conftest.py で定義済み。


@pytest.fixture
def char_id(sqlite_store):
    """テスト用キャラクターをSQLiteに作成し、そのIDを返すフィクスチャ。"""
    cid = "reflector-test-char"
    sqlite_store.create_character(
        character_id=cid,
        name="自己参照テストキャラ",
        system_prompt_block1="テスト用設定",
    )
    return cid


@pytest.fixture
def trigger_preset_id(sqlite_store):
    """テスト用モデルプリセット（Ollama）をSQLiteに作成し、そのIDを返すフィクスチャ。"""
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Ollama",
        provider="ollama",
        model_id="qwen2.5:3b",
    )
    return pid


@pytest.fixture
def reflection_preset_id(sqlite_store):
    """テスト用モデルプリセット（Anthropic）をSQLiteに作成し、そのIDを返すフィクスチャ。

    _run_reflection() で ask_character() に渡す preset_id として使用する。
    """
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Anthropic",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
    )
    return pid


@pytest.fixture
def google_trigger_preset_id(sqlite_store):
    """テスト用モデルプリセット（Google）をSQLiteに作成し、そのIDを返すフィクスチャ。

    Ollamaに限らず任意プロバイダーを契機判断に使えることの検証用。
    """
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Google",
        provider="google",
        model_id="gemini-2.0-flash",
    )
    return pid


@pytest.fixture
def memory_manager(sqlite_store):
    """MagicMock の MemoryManager（sqlite は実DBに差し替え）を返すフィクスチャ。"""
    mm = MagicMock()
    mm.sqlite = sqlite_store
    return mm


@pytest.fixture
def drift_manager():
    """MagicMock の DriftManager を返すフィクスチャ。"""
    return MagicMock()


@pytest.fixture
def reflector(memory_manager, drift_manager):
    """テスト用 SelfReflector インスタンスを返すフィクスチャ。"""
    return SelfReflector(memory_manager=memory_manager, drift_manager=drift_manager)


def _make_provider(response_text: str = "") -> MagicMock:
    """指定テキストを返す非同期 generate() を持つモックプロバイダーを生成する。

    Args:
        response_text: generate() が返すテキスト。

    Returns:
        AsyncMock の generate() を持つ MagicMock プロバイダー。
    """
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=response_text)
    return provider


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

    def test_always_mode_applies_drift_tag(
        self, reflector, sqlite_store, char_id, reflection_preset_id, drift_manager
    ):
        """always モードで返された DRIFT タグが drift_manager に渡されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        reflector.drift_manager = drift_manager
        with patch(
            "backend.character_actions.reflector.ask_character",
            new=AsyncMock(return_value="[DRIFT:少しざわざわしている]"),
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
        drift_manager.add_drift.assert_called_once()
        assert "少しざわざわしている" in str(drift_manager.add_drift.call_args)

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
