"""backend.character_actions.reflector モジュールのユニットテスト。

自己参照ループ（SelfReflector）の動作を検証する。
対象クラス・関数:
    SelfReflector.run()              — エントリポイント。モード・設定に応じて処理を振り分ける
    SelfReflector._detect_trigger()  — Ollama契機判断（YES/NO応答の解析を含む）
    SelfReflector._run_reflection()  — パブリックLLMへの自己参照コール＋タグ処理
    _format_conversation()           — メッセージリストを整形する内部ユーティリティ

テスト方針:
    - LLMプロバイダーは asyncio coroutine を返す AsyncMock で差し替える
    - SQLite操作は実際の一時DBを使い、Carver/Drifterの副作用もあわせて検証する
    - エラー系（Ollama接続失敗・空応答・preset未発見）がサイレントにskipされることを確認する
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.character_actions.reflector import SelfReflector, _format_conversation


# ─── フィクスチャ ──────────────────────────────────────────────────────────────
# sqlite_store は conftest.py で定義済み。


@pytest.fixture
def char_id(sqlite_store):
    """テスト用キャラクターを SQLite に作成し、その ID を返すフィクスチャ。"""
    cid = "reflector-test-char"
    sqlite_store.create_character(
        character_id=cid,
        name="自己参照テストキャラ",
        system_prompt_block1="テスト用設定",
    )
    return cid


@pytest.fixture
def trigger_preset_id(sqlite_store):
    """テスト用Ollamaモデルプリセットを作成し、そのIDを返すフィクスチャ。"""
    import uuid
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Ollama",
        provider="ollama",
        model_id="qwen2.5:3b",
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


def _make_provider(response_text: str = ""):
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

    def test_disabled_mode_does_not_call_provider(self, reflector):
        """disabled モードではパブリックLLMプロバイダーが呼ばれないこと。"""
        provider = _make_provider()
        asyncio.run(
            reflector.run(
                request_mode="disabled",
                trigger_preset_id="",
                n_turns=5,
                public_provider=provider,
                settings={},
                messages=[{"role": "user", "content": "テスト"}],
                character_id="char-1",
                session_id="sess-1",
                current_preset_id="",
            )
        )
        provider.generate.assert_not_called()

    def test_disabled_mode_returns_none(self, reflector):
        """disabled モードは None を返すこと（副作用なし）。"""
        provider = _make_provider()
        result = asyncio.run(
            reflector.run(
                request_mode="disabled",
                trigger_preset_id="",
                n_turns=5,
                public_provider=provider,
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
    """self_reflection_mode が always のとき、Ollama不要でパブリックLLMが呼ばれることを検証する。"""

    def test_always_mode_calls_public_provider(self, reflector):
        """always モードではパブリックLLMが必ず呼ばれること。"""
        provider = _make_provider(response_text="")
        asyncio.run(
            reflector.run(
                request_mode="always",
                trigger_preset_id="",
                n_turns=5,
                public_provider=provider,
                settings={},
                messages=[{"role": "user", "content": "テスト"}],
                character_id="char-1",
                session_id="sess-1",
                current_preset_id="",
            )
        )
        provider.generate.assert_called_once()

    def test_always_mode_does_not_require_trigger_preset(self, reflector):
        """always モードでは trigger_preset_id が空でも正常に動作すること。"""
        provider = _make_provider(response_text="")
        # 例外が送出されないこと
        asyncio.run(
            reflector.run(
                request_mode="always",
                trigger_preset_id="",  # 空でもOK
                n_turns=5,
                public_provider=provider,
                settings={},
                messages=[{"role": "user", "content": "テスト"}],
                character_id="char-1",
                session_id="sess-1",
                current_preset_id="",
            )
        )

    def test_always_mode_applies_carve_narrative_tag(self, reflector, sqlite_store, char_id):
        """always モードで返されたCARVE_NARRATIVEタグが inner_narrative に反映されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        provider = _make_provider("[CARVE_NARRATIVE:append|自己参照による気づき]")

        asyncio.run(
            reflector.run(
                request_mode="always",
                trigger_preset_id="",
                n_turns=5,
                public_provider=provider,
                settings={},
                messages=[{"role": "user", "content": "テスト"}],
                character_id=char_id,
                session_id="sess-1",
                current_preset_id="",
            )
        )

        char = sqlite_store.get_character(char_id)
        assert "自己参照による気づき" in char.inner_narrative

    def test_always_mode_applies_drift_tag(self, reflector, sqlite_store, char_id, drift_manager):
        """always モードで返されたDRIFTタグが drift_manager に渡されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        reflector.drift_manager = drift_manager
        provider = _make_provider("[DRIFT:少しざわざわしている]")

        asyncio.run(
            reflector.run(
                request_mode="always",
                trigger_preset_id="",
                n_turns=5,
                public_provider=provider,
                settings={},
                messages=[{"role": "user", "content": "テスト"}],
                character_id=char_id,
                session_id="sess-1",
                current_preset_id="",
            )
        )

        drift_manager.add_drift.assert_called_once()
        call_kwargs = drift_manager.add_drift.call_args
        assert "少しざわざわしている" in str(call_kwargs)

    def test_always_mode_empty_response_does_nothing(self, reflector, sqlite_store, char_id):
        """always モードでパブリックLLMが空文字を返した場合、DB変更がないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        provider = _make_provider("")  # 内省なし

        asyncio.run(
            reflector.run(
                request_mode="always",
                trigger_preset_id="",
                n_turns=5,
                public_provider=provider,
                settings={},
                messages=[{"role": "user", "content": "テスト"}],
                character_id=char_id,
                session_id="sess-1",
                current_preset_id="",
            )
        )

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == ""

    def test_always_mode_n_turns_limits_window(self, reflector):
        """always モードで n_turns=2 のとき、最新2ターンのみがLLMに渡されること。"""
        provider = _make_provider("")
        messages = [
            {"role": "user", "content": "古い発言1"},
            {"role": "assistant", "content": "古い応答1"},
            {"role": "user", "content": "新しい発言"},
            {"role": "assistant", "content": "新しい応答"},
        ]

        asyncio.run(
            reflector.run(
                request_mode="always",
                trigger_preset_id="",
                n_turns=2,  # 最新2ターンのみ
                public_provider=provider,
                settings={},
                messages=messages,
                character_id="char-1",
                session_id="sess-1",
                current_preset_id="",
            )
        )

        # generate() に渡されたメッセージ (第2引数) を確認する
        call_args = provider.generate.call_args
        passed_messages = call_args[0][1]  # generate(system_prompt, messages)
        user_content = passed_messages[0]["content"]
        assert "古い発言1" not in user_content
        assert "新しい発言" in user_content


# ─── SelfReflector._detect_trigger() — 契機判断 ──────────────────────────────


class TestDetectTrigger:
    """SelfReflector._detect_trigger() のYES/NO解析を検証する。"""

    def _run_detect(self, reflector, response_text, trigger_preset_id, settings=None):
        """_detect_trigger() を同期的に実行するヘルパー。"""
        with patch("backend.character_actions.reflector.create_provider") as mock_create:
            provider = _make_provider(response_text)
            mock_create.return_value = provider
            return asyncio.run(
                reflector._detect_trigger(
                    trigger_preset_id=trigger_preset_id,
                    settings=settings or {},
                    conversation_window=[{"role": "user", "content": "テスト"}],
                )
            )

    def test_yes_response_returns_true(self, reflector, sqlite_store, trigger_preset_id):
        """Ollama が「YES」を返したとき True になること。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = self._run_detect(reflector, "YES", trigger_preset_id)
        assert result is True

    def test_yes_with_trailing_text_returns_true(self, reflector, sqlite_store, trigger_preset_id):
        """「YES\n理由テキスト」のようにYESで始まる応答も True になること。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = self._run_detect(reflector, "YES\n感情の変化が見られます", trigger_preset_id)
        assert result is True

    def test_no_response_returns_false(self, reflector, sqlite_store, trigger_preset_id):
        """Ollama が「NO」を返したとき False になること。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = self._run_detect(reflector, "NO", trigger_preset_id)
        assert result is False

    def test_empty_response_returns_false(self, reflector, sqlite_store, trigger_preset_id):
        """Ollama が空文字を返したとき False になること（NOT YES）。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = self._run_detect(reflector, "", trigger_preset_id)
        assert result is False

    def test_preset_not_found_returns_false(self, reflector, sqlite_store):
        """存在しないプリセットIDを渡したとき False を返すこと（例外を送出しない）。"""
        reflector.memory_manager.sqlite = sqlite_store
        result = asyncio.run(
            reflector._detect_trigger(
                trigger_preset_id="nonexistent-preset-id",
                settings={},
                conversation_window=[{"role": "user", "content": "テスト"}],
            )
        )
        assert result is False

    def test_ollama_connection_error_returns_false(self, reflector, sqlite_store, trigger_preset_id):
        """Ollamaへの接続が失敗したとき False を返すこと（例外を送出しない）。"""
        reflector.memory_manager.sqlite = sqlite_store
        with patch("backend.character_actions.reflector.create_provider") as mock_create:
            provider = MagicMock()
            provider.generate = AsyncMock(side_effect=ConnectionError("Ollama unreachable"))
            mock_create.return_value = provider
            result = asyncio.run(
                reflector._detect_trigger(
                    trigger_preset_id=trigger_preset_id,
                    settings={},
                    conversation_window=[{"role": "user", "content": "テスト"}],
                )
            )
        assert result is False


# ─── SelfReflector.run() — local_trigger モード ──────────────────────────────


class TestSelfReflectorLocalTrigger:
    """self_reflection_mode が local_trigger のとき、Ollama契機判断の結果で分岐することを検証する。"""

    def test_local_trigger_yes_calls_public_provider(self, reflector, sqlite_store, trigger_preset_id):
        """local_trigger + Ollama=YES のとき、パブリックLLMが呼ばれること。"""
        reflector.memory_manager.sqlite = sqlite_store
        public_provider = _make_provider("")

        with patch.object(reflector, "_detect_trigger", new=AsyncMock(return_value=True)):
            asyncio.run(
                reflector.run(
                    request_mode="local_trigger",
                    trigger_preset_id=trigger_preset_id,
                    n_turns=5,
                    public_provider=public_provider,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id="char-1",
                    session_id="sess-1",
                    current_preset_id="",
                )
            )

        public_provider.generate.assert_called_once()

    def test_local_trigger_no_skips_public_provider(self, reflector, sqlite_store, trigger_preset_id):
        """local_trigger + Ollama=NO のとき、パブリックLLMが呼ばれないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        public_provider = _make_provider("")

        with patch.object(reflector, "_detect_trigger", new=AsyncMock(return_value=False)):
            asyncio.run(
                reflector.run(
                    request_mode="local_trigger",
                    trigger_preset_id=trigger_preset_id,
                    n_turns=5,
                    public_provider=public_provider,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id="char-1",
                    session_id="sess-1",
                    current_preset_id="",
                )
            )

        public_provider.generate.assert_not_called()

    def test_local_trigger_without_preset_id_skips_silently(self, reflector, sqlite_store):
        """local_trigger で trigger_preset_id が未設定のとき、黙ってスキップすること（例外なし）。"""
        reflector.memory_manager.sqlite = sqlite_store
        public_provider = _make_provider("")

        asyncio.run(
            reflector.run(
                request_mode="local_trigger",
                trigger_preset_id="",  # 未設定
                n_turns=5,
                public_provider=public_provider,
                settings={},
                messages=[{"role": "user", "content": "テスト"}],
                character_id="char-1",
                session_id="sess-1",
                current_preset_id="",
            )
        )

        public_provider.generate.assert_not_called()

    def test_local_trigger_yes_applies_tags_to_db(self, reflector, sqlite_store, char_id, trigger_preset_id):
        """local_trigger + Ollama=YES で返されたタグがDBに反映されること。"""
        reflector.memory_manager.sqlite = sqlite_store
        public_provider = _make_provider("[CARVE_NARRATIVE:append|契機判断後の気づき]")

        with patch.object(reflector, "_detect_trigger", new=AsyncMock(return_value=True)):
            asyncio.run(
                reflector.run(
                    request_mode="local_trigger",
                    trigger_preset_id=trigger_preset_id,
                    n_turns=5,
                    public_provider=public_provider,
                    settings={},
                    messages=[{"role": "user", "content": "テスト"}],
                    character_id=char_id,
                    session_id="sess-1",
                    current_preset_id="",
                )
            )

        char = sqlite_store.get_character(char_id)
        assert "契機判断後の気づき" in char.inner_narrative


# ─── SelfReflector._run_reflection() — エラーハンドリング ────────────────────


class TestRunReflectionErrorHandling:
    """_run_reflection() がLLM失敗時にサイレントにスキップすることを検証する。"""

    def test_llm_failure_does_not_raise(self, reflector, sqlite_store, char_id):
        """パブリックLLMコールが例外を送出しても _run_reflection() が例外を伝播しないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=RuntimeError("LLM error"))

        # 例外が送出されないこと
        asyncio.run(
            reflector._run_reflection(
                public_provider=provider,
                conversation_window=[{"role": "user", "content": "テスト"}],
                character_id=char_id,
                session_id="sess-1",
                current_preset_id="",
            )
        )

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == ""  # DB変更なし

    def test_whitespace_only_response_does_nothing(self, reflector, sqlite_store, char_id):
        """パブリックLLMが空白のみを返した場合、DB変更がないこと。"""
        reflector.memory_manager.sqlite = sqlite_store
        provider = _make_provider("   \n  ")

        asyncio.run(
            reflector._run_reflection(
                public_provider=provider,
                conversation_window=[{"role": "user", "content": "テスト"}],
                character_id=char_id,
                session_id="sess-1",
                current_preset_id="",
            )
        )

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == ""


# ─── SQLite スキーマ — 自己参照カラムのデフォルト値 ──────────────────────────


class TestSelfReflectionSchema:
    """自己参照フィールドのデフォルト値とDBへの読み書きを検証する。"""

    def test_new_character_has_disabled_mode_by_default(self, sqlite_store):
        """新規キャラクターの self_reflection_mode デフォルト値が 'disabled' であること。"""
        import uuid
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="デフォルトキャラ")
        char = sqlite_store.get_character(cid)
        assert char.self_reflection_mode == "disabled"

    def test_new_character_has_null_preset_id_by_default(self, sqlite_store):
        """新規キャラクターの self_reflection_preset_id デフォルト値が None であること。"""
        import uuid
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="デフォルトキャラ2")
        char = sqlite_store.get_character(cid)
        assert char.self_reflection_preset_id is None

    def test_new_character_has_n_turns_5_by_default(self, sqlite_store):
        """新規キャラクターの self_reflection_n_turns デフォルト値が 5 であること。"""
        import uuid
        cid = str(uuid.uuid4())
        sqlite_store.create_character(character_id=cid, name="デフォルトキャラ3")
        char = sqlite_store.get_character(cid)
        assert char.self_reflection_n_turns == 5

    def test_create_character_with_local_trigger_mode(self, sqlite_store):
        """local_trigger モードで作成したキャラクターが正しくDBに保存されること。"""
        import uuid
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
