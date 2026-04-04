"""backend.services.character_query モジュールのユニットテスト。

ask_character() の動作を検証する。

対象関数:
    ask_character() — プリセット解決・システムプロンプト構築・LLMコールを一本化する
                      共通エントリポイント

テスト方針:
    - SQLite は conftest.py の sqlite_store フィクスチャで実際の一時DBを使用する
    - LLMプロバイダーは AsyncMock で差し替える
    - 記憶想起は MemoryManager.recall_with_identity() を MagicMock で差し替える
    - キャラクター未発見・プリセット未発見・プロバイダー生成失敗・LLMコール失敗の
      いずれも例外を送出せず None を返すことを確認する
    - recall_query の有無で記憶想起の実行・スキップが切り替わることを確認する
    - recall_query あり時にシステムプロンプトへ記憶が注入されることを確認する
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.character_query import ask_character


# ─── フィクスチャ ──────────────────────────────────────────────────────────────
# sqlite_store は conftest.py で定義済み。


@pytest.fixture
def char_id(sqlite_store):
    """テスト用キャラクターをSQLiteに作成し、そのIDを返すフィクスチャ。

    system_prompt_block1 / inner_narrative / self_history / relationship_state を
    設定して、システムプロンプト構築の検証に使用する。
    """
    cid = str(uuid.uuid4())
    sqlite_store.create_character(
        character_id=cid,
        name="クエリテストキャラ",
        system_prompt_block1="テスト用キャラクター設定",
    )
    sqlite_store.update_character(
        cid,
        inner_narrative="inner_narrativeの内容",
        self_history="self_historyの内容",
        relationship_state="relationship_stateの内容",
    )
    return cid


@pytest.fixture
def preset_id(sqlite_store):
    """テスト用モデルプリセットをSQLiteに作成し、そのIDを返すフィクスチャ。"""
    pid = str(uuid.uuid4())
    sqlite_store.create_model_preset(
        preset_id=pid,
        name="Test-Preset",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
    )
    return pid


def _mock_provider(response_text: str = "テスト応答") -> MagicMock:
    """指定テキストを返す AsyncMock generate() を持つモックプロバイダーを生成する。

    Args:
        response_text: generate() が返す応答テキスト。

    Returns:
        generate が AsyncMock の MagicMock プロバイダー。
    """
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=response_text)
    return provider


# ─── エラー系 — None を返す ───────────────────────────────────────────────────


class TestAskCharacterErrorCases:
    """ask_character() がエラー時に例外を送出せず None を返すことを検証する。"""

    @pytest.mark.asyncio
    async def test_character_not_found_returns_none(self, sqlite_store, preset_id):
        """存在しないキャラクターIDを渡すと None を返すこと。"""
        result = await ask_character(
            character_id="nonexistent-char",
            preset_id=preset_id,
            messages=[{"role": "user", "content": "テスト"}],
            sqlite=sqlite_store,
            settings={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_preset_not_found_returns_none(self, sqlite_store, char_id):
        """存在しないプリセットIDを渡すと None を返すこと。"""
        result = await ask_character(
            character_id=char_id,
            preset_id="nonexistent-preset",
            messages=[{"role": "user", "content": "テスト"}],
            sqlite=sqlite_store,
            settings={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_provider_creation_failure_returns_none(self, sqlite_store, char_id, preset_id):
        """create_provider() が例外を送出したとき None を返すこと（例外が伝播しないこと）。"""
        with patch(
            "backend.services.character_query.create_provider",
            side_effect=RuntimeError("provider init failed"),
        ):
            result = await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_call_failure_returns_none(self, sqlite_store, char_id, preset_id):
        """provider.generate() が例外を送出したとき None を返すこと（例外が伝播しないこと）。"""
        failing_provider = MagicMock()
        failing_provider.generate = AsyncMock(side_effect=ConnectionError("LLM unreachable"))
        with patch(
            "backend.services.character_query.create_provider",
            return_value=failing_provider,
        ):
            result = await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
            )
        assert result is None


# ─── 正常系 — 応答テキストを返す ─────────────────────────────────────────────


class TestAskCharacterSuccess:
    """ask_character() が正常にLLM応答を返すことを検証する。"""

    @pytest.mark.asyncio
    async def test_returns_llm_response_text(self, sqlite_store, char_id, preset_id):
        """LLMが返したテキストをそのまま返すこと。"""
        provider = _mock_provider("キャラクターからの応答です")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            result = await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "こんにちは"}],
                sqlite=sqlite_store,
                settings={},
            )
        assert result == "キャラクターからの応答です"

    @pytest.mark.asyncio
    async def test_messages_are_passed_to_provider(self, sqlite_store, char_id, preset_id):
        """渡したmessagesがprovider.generate()の第2引数に正しく渡されること。"""
        provider = _mock_provider("")
        input_messages = [{"role": "user", "content": "テストメッセージ"}]
        with patch("backend.services.character_query.create_provider", return_value=provider):
            await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=input_messages,
                sqlite=sqlite_store,
                settings={},
            )
        _, call_messages = provider.generate.call_args[0]
        assert call_messages == input_messages

    @pytest.mark.asyncio
    async def test_system_prompt_contains_character_block1(self, sqlite_store, char_id, preset_id):
        """生成されたシステムプロンプトにキャラクター設定（block1）が含まれること。"""
        provider = _mock_provider("")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
            )
        system_prompt, _ = provider.generate.call_args[0]
        assert "テスト用キャラクター設定" in system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_inner_narrative(self, sqlite_store, char_id, preset_id):
        """生成されたシステムプロンプトに inner_narrative が含まれること。"""
        provider = _mock_provider("")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
            )
        system_prompt, _ = provider.generate.call_args[0]
        assert "inner_narrativeの内容" in system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_self_history(self, sqlite_store, char_id, preset_id):
        """生成されたシステムプロンプトに self_history が含まれること。"""
        provider = _mock_provider("")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
            )
        system_prompt, _ = provider.generate.call_args[0]
        assert "self_historyの内容" in system_prompt


# ─── recall_query — 記憶想起のオン・オフ ─────────────────────────────────────


class TestAskCharacterRecall:
    """recall_query の有無で記憶想起の実行・スキップが切り替わることを検証する。"""

    @pytest.mark.asyncio
    async def test_no_recall_query_skips_memory_recall(self, sqlite_store, char_id, preset_id):
        """recall_query=None のとき、記憶想起が実行されないこと。"""
        memory_manager = MagicMock()
        memory_manager.recall_with_identity = MagicMock(return_value=([], []))
        provider = _mock_provider("")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
                memory_manager=memory_manager,
                recall_query=None,
            )
        memory_manager.recall_with_identity.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_query_triggers_memory_recall(self, sqlite_store, char_id, preset_id):
        """recall_query を渡すと recall_with_identity() が呼ばれること。"""
        memory_manager = MagicMock()
        memory_manager.recall_with_identity = MagicMock(return_value=([], []))
        provider = _mock_provider("")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
                memory_manager=memory_manager,
                recall_query="ユーザーの最後の発言",
            )
        memory_manager.recall_with_identity.assert_called_once_with(char_id, "ユーザーの最後の発言")

    @pytest.mark.asyncio
    async def test_recalled_memories_injected_into_system_prompt(self, sqlite_store, char_id, preset_id):
        """recall_query あり時に想起記憶がシステムプロンプトに注入されること。"""
        memory_manager = MagicMock()
        # recall_with_identity が identity記憶とその他記憶を返すようにする
        identity_mems = [{"id": "m1", "content": "アイデンティティ記憶", "metadata": {"category": "identity"}}]
        other_mems = [{"id": "m2", "content": "その他の記憶", "metadata": {"category": "event"}}]
        memory_manager.recall_with_identity = MagicMock(return_value=(identity_mems, other_mems))
        provider = _mock_provider("")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
                memory_manager=memory_manager,
                recall_query="テストクエリ",
            )
        system_prompt, _ = provider.generate.call_args[0]
        assert "アイデンティティ記憶" in system_prompt
        assert "その他の記憶" in system_prompt

    @pytest.mark.asyncio
    async def test_recall_failure_does_not_abort(self, sqlite_store, char_id, preset_id):
        """recall_with_identity() が例外を送出しても、記憶なしで続行してLLM応答を返すこと。"""
        memory_manager = MagicMock()
        memory_manager.recall_with_identity = MagicMock(side_effect=RuntimeError("chroma down"))
        provider = _mock_provider("記憶なしで応答")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            result = await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
                memory_manager=memory_manager,
                recall_query="クエリ",
            )
        # LLMコールは続行されて応答が返ること
        assert result == "記憶なしで応答"

    @pytest.mark.asyncio
    async def test_no_recall_query_but_no_memory_manager_still_works(
        self, sqlite_store, char_id, preset_id
    ):
        """recall_query=None かつ memory_manager=None でも正常に動作すること。"""
        provider = _mock_provider("応答")
        with patch("backend.services.character_query.create_provider", return_value=provider):
            result = await ask_character(
                character_id=char_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": "テスト"}],
                sqlite=sqlite_store,
                settings={},
                memory_manager=None,
                recall_query=None,
            )
        assert result == "応答"
