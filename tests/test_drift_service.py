"""ChatService の _apply_drifts メソッドおよび SELF_DRIFT 統合テスト。

test_chat_service.py と同じスタイルで、以下を検証する:
  - _apply_drifts の単体テスト（マーカー除去・DB反映・スキップ条件・例外処理）
  - ChatService.execute での drift 統合テスト（LLM応答からのマーカー除去と drift_manager 呼び出し）
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.chat.models import ChatRequest, Message
from backend.services.chat.service import ChatService


# --- パス・スルーモックヘルパー ---


def _passthrough_inscriber():
    """inscribe_memory_from_text をパス・スルーする Inscriber モックを生成する。"""
    m = MagicMock()
    m.inscribe_memory_from_text.side_effect = lambda text, *_: text
    return m


def _passthrough_carver():
    """carve_narrative_from_text をパス・スルーする Carver モックを生成する。"""
    m = MagicMock()
    m.carve_narrative_from_text.side_effect = lambda text: text
    return m


# --- _apply_drifts の単体テスト ---

class TestApplyDrifts:
    """ChatService._apply_drifts の振る舞いを検証するテストスイート。

    ChatService を直接インスタンス化し、_apply_drifts メソッドを呼び出すことで
    LLMプロバイダーへの依存なしにテストする。
    """

    def _make_service(self, drift_manager=None) -> ChatService:
        """テスト用の ChatService インスタンスを作成するヘルパー。

        memory_manager は MagicMock で代替する。
        """
        memory_manager = MagicMock()
        return ChatService(memory_manager=memory_manager, drift_manager=drift_manager)

    def _make_request(self, session_id: str = "session-1", character_id: str = "char-1") -> ChatRequest:
        """テスト用の ChatRequest を作成するヘルパー。"""
        return ChatRequest(
            character_id=character_id,
            character_name="Alice",
            provider="anthropic",
            model="",
            messages=[Message(role="user", content="hello")],
            session_id=session_id,
        )

    def test_drift_marker_is_removed_from_response(self):
        """[DRIFT:...] マーカーが応答テキストから除去されること。"""
        drift_manager = MagicMock()
        service = self._make_service(drift_manager=drift_manager)
        request = self._make_request()

        text = "了解です。[DRIFT:もっとクールに話す]今後そうします。"
        clean = service._apply_drifts(text, request)

        assert "[DRIFT:" not in clean
        assert "了解です。" in clean
        assert "今後そうします。" in clean

    def test_drift_manager_add_drift_is_called(self):
        """[DRIFT:...] マーカーがある場合に drift_manager.add_drift が呼ばれること。"""
        drift_manager = MagicMock()
        service = self._make_service(drift_manager=drift_manager)
        request = self._make_request()

        service._apply_drifts("[DRIFT:指針A]テキスト", request)

        drift_manager.add_drift.assert_called_once_with("session-1", "char-1", "指針A")

    def test_drift_reset_calls_reset_drifts(self):
        """[DRIFT_RESET] マーカーがある場合に drift_manager.reset_drifts が呼ばれること。"""
        drift_manager = MagicMock()
        service = self._make_service(drift_manager=drift_manager)
        request = self._make_request()

        service._apply_drifts("[DRIFT_RESET]リセットします。", request)

        drift_manager.reset_drifts.assert_called_once_with("session-1", "char-1")

    def test_skip_db_write_when_session_id_is_empty(self):
        """session_id が空の場合はDB書き込みをスキップするが、マーカーは除去されること。

        session_id がなくても [DRIFT:...] がユーザーに見えてはいけないため、
        マーカーのストリップ自体は常に実行される。
        """
        drift_manager = MagicMock()
        service = self._make_service(drift_manager=drift_manager)
        request = self._make_request(session_id="")

        result = service._apply_drifts("テキスト[DRIFT:指針A]", request)

        # マーカーが除去されること
        assert "[DRIFT:" not in result
        assert "テキスト" in result
        # DB書き込みは呼ばれないこと
        drift_manager.add_drift.assert_not_called()
        drift_manager.reset_drifts.assert_not_called()

    def test_skip_db_write_when_drift_manager_is_none(self):
        """drift_manager が None の場合もマーカーは除去されること。

        drift_manager がなくてもマーカーをユーザーに見せてはいけないため、
        ストリップ処理は実行される。
        """
        service = self._make_service(drift_manager=None)
        request = self._make_request()

        result = service._apply_drifts("テキスト[DRIFT:指針A]", request)

        # マーカーが除去されること
        assert "[DRIFT:" not in result
        assert "テキスト" in result

    def test_drift_and_reset_coexist_reset_before_add(self):
        """DRIFT と DRIFT_RESET が共存する場合、reset 後に add が呼ばれること。

        reset_drifts が add_drift より先に呼ばれる（コード上の実装順序）ことを
        call_args_list で検証する。
        """
        drift_manager = MagicMock()
        service = self._make_service(drift_manager=drift_manager)
        request = self._make_request()

        service._apply_drifts("[DRIFT_RESET][DRIFT:新しい指針]テキスト", request)

        # 両方が呼ばれること
        drift_manager.reset_drifts.assert_called_once()
        drift_manager.add_drift.assert_called_once()

        # reset が add より先に呼ばれていること（call_args_list の順序で検証）
        all_calls = drift_manager.method_calls
        reset_idx = next(i for i, c in enumerate(all_calls) if c[0] == "reset_drifts")
        add_idx = next(i for i, c in enumerate(all_calls) if c[0] == "add_drift")
        assert reset_idx < add_idx

    def test_add_drift_exception_is_swallowed(self):
        """drift_manager.add_drift が例外を出してもクラッシュせず clean_text が返ること。"""
        drift_manager = MagicMock()
        drift_manager.add_drift.side_effect = RuntimeError("DB error!")
        service = self._make_service(drift_manager=drift_manager)
        request = self._make_request()

        # 例外が外に漏れないこと
        result = service._apply_drifts("[DRIFT:指針A]テキスト", request)

        # クリーンなテキストが返ること
        assert "[DRIFT:" not in result
        assert "テキスト" in result


# --- ChatService.execute での drift 統合テスト ---

@pytest.mark.asyncio
async def test_execute_removes_drift_marker_and_calls_drift_manager():
    """LLM応答に [DRIFT:...] が含まれる場合、マーカーが除去されて drift_manager が呼ばれること。"""
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    drift_manager = MagicMock()

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="こんにちは")],
        session_id="session-abc",
    )

    # LLM応答に [DRIFT:...] マーカーを含める
    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(return_value="了解です。[DRIFT:もっとクールに話す]")

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_passthrough_inscriber()),
        patch("backend.services.chat.service.Carver", return_value=_passthrough_carver()),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        result = await service.execute(request)

    # 返却テキストに [DRIFT:...] が含まれないこと
    assert "[DRIFT:" not in result
    assert "了解です。" in result

    # drift_manager.add_drift が呼ばれること
    drift_manager.add_drift.assert_called_once_with("session-abc", "char-1", "もっとクールに話す")


@pytest.mark.asyncio
async def test_execute_drift_marker_stripped_when_no_session_id():
    """session_id が空の場合、[DRIFT:...] マーカーは除去されるがDB書き込みはスキップされること。

    session_id なしでも [DRIFT:...] マーカーをユーザーに返してはいけないため、
    マーカー除去は実行される。ただし drift_manager.add_drift は呼ばれない。
    """
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    drift_manager = MagicMock()

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hi")],
        session_id="",  # session_id なし
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(return_value="応答テキスト[DRIFT:指針A]")

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_passthrough_inscriber()),
        patch("backend.services.chat.service.Carver", return_value=_passthrough_carver()),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        result = await service.execute(request)

    # マーカーが除去されていること
    assert "[DRIFT:" not in result
    assert "応答テキスト" in result
    # DB書き込みは呼ばれないこと
    drift_manager.add_drift.assert_not_called()


# --- execute での active_drifts 自動ロードテスト ---

@pytest.mark.asyncio
async def test_execute_loads_active_drifts_from_db():
    """session_id があり active_drifts が未指定の場合、DBから自動ロードして build_system_prompt に渡すこと。"""
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    drift_manager = MagicMock()
    drift_manager.list_active_drifts.return_value = ["もっとクールに話す", "短く答える"]

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="こんにちは")],
        session_id="session-abc",
        # active_drifts は未指定（デフォルト空リスト）
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(return_value="了解です。")

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys") as mock_build,
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_passthrough_inscriber()),
        patch("backend.services.chat.service.Carver", return_value=_passthrough_carver()),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        await service.execute(request)

    # drift_manager.list_active_drifts が呼ばれること
    drift_manager.list_active_drifts.assert_called_once_with("session-abc", "char-1")

    # build_system_prompt に active_drifts が渡されること
    _, kwargs = mock_build.call_args
    assert kwargs["active_drifts"] == ["もっとクールに話す", "短く答える"]


@pytest.mark.asyncio
async def test_execute_does_not_overwrite_active_drifts_if_provided():
    """リクエストに active_drifts が既に含まれている場合、DBロードをスキップすること。

    Chotgorフロントは将来的には active_drifts を渡さず session_id だけ渡すが、
    万が一両方渡された場合は既存値を優先する（DBロードは不要）。
    """
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    drift_manager = MagicMock()

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="こんにちは")],
        session_id="session-abc",
        active_drifts=["既存の指針"],  # 既に指定済み
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(return_value="了解です。")

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys") as mock_build,
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_passthrough_inscriber()),
        patch("backend.services.chat.service.Carver", return_value=_passthrough_carver()),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        await service.execute(request)

    # DBロードは呼ばれないこと
    drift_manager.list_active_drifts.assert_not_called()

    # 既存の active_drifts がそのまま渡されること
    _, kwargs = mock_build.call_args
    assert kwargs["active_drifts"] == ["既存の指針"]


@pytest.mark.asyncio
async def test_execute_stream_loads_active_drifts_from_db():
    """execute_stream でも session_id があれば DBから active_drifts を自動ロードすること。"""
    memory_manager = MagicMock()
    memory_manager.recall_memory.return_value = []

    drift_manager = MagicMock()
    drift_manager.list_active_drifts.return_value = ["統計的に答える"]

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="こんにちは")],
        session_id="session-xyz",
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False

    async def fake_stream(*_):
        yield ("text", "了解です。")

    fake_provider.generate_stream_typed = fake_stream

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys") as mock_build,
        patch("backend.services.chat.service.find_urls", return_value=[]),
        patch("backend.services.chat.service.Inscriber", return_value=_passthrough_inscriber()),
        patch("backend.services.chat.service.Carver", return_value=_passthrough_carver()),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        chunks = [c async for c in service.execute_stream(request)]

    # drift_manager.list_active_drifts が呼ばれること
    drift_manager.list_active_drifts.assert_called_once_with("session-xyz", "char-1")

    # build_system_prompt に active_drifts が渡されること
    _, kwargs = mock_build.call_args
    assert kwargs["active_drifts"] == ["統計的に答える"]

    # テキストが返ること
    text_chunks = [c for t, c in chunks if t == "text"]
    assert text_chunks == ["了解です。"]
