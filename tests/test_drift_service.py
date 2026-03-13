"""ChatService の _apply_drifts メソッドおよび SELF_DRIFT 統合テスト。

test_chat_service.py と同じスタイルで、以下を検証する:
  - _apply_drifts の単体テスト（マーカー除去・DB反映・スキップ条件・例外処理）
  - ChatService.execute での drift 統合テスト（LLM応答からのマーカー除去と drift_manager 呼び出し）
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.chat.models import ChatRequest, Message
from backend.core.chat.service import ChatService


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

    def test_skip_when_session_id_is_empty(self):
        """session_id が空の場合は drift 処理をスキップし、テキストをそのまま返すこと。"""
        drift_manager = MagicMock()
        service = self._make_service(drift_manager=drift_manager)
        request = self._make_request(session_id="")

        text = "テキスト[DRIFT:指針A]"
        result = service._apply_drifts(text, request)

        # テキストがそのまま返ること（マーカー除去なし）
        assert result == text
        # drift_manager は呼ばれないこと
        drift_manager.add_drift.assert_not_called()
        drift_manager.reset_drifts.assert_not_called()

    def test_skip_when_drift_manager_is_none(self):
        """drift_manager が None の場合は drift 処理をスキップし、テキストをそのまま返すこと。"""
        service = self._make_service(drift_manager=None)
        request = self._make_request()

        text = "テキスト[DRIFT:指針A]"
        result = service._apply_drifts(text, request)

        # テキストがそのまま返ること（マーカー除去なし）
        assert result == text

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
    fake_provider.generate = AsyncMock(return_value="了解です。[DRIFT:もっとクールに話す]")

    with (
        patch("backend.core.chat.service.create_provider", return_value=fake_provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=lambda text, *_: text),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        result = await service.execute(request)

    # 返却テキストに [DRIFT:...] が含まれないこと
    assert "[DRIFT:" not in result
    assert "了解です。" in result

    # drift_manager.add_drift が呼ばれること
    drift_manager.add_drift.assert_called_once_with("session-abc", "char-1", "もっとクールに話す")


@pytest.mark.asyncio
async def test_execute_drift_skipped_when_no_session_id():
    """session_id が空の場合に drift 処理がスキップされること。

    session_id なしのリクエストでは [DRIFT:...] マーカーがあっても
    drift_manager.add_drift が呼ばれないことを検証する。
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
    fake_provider.generate = AsyncMock(return_value="応答テキスト[DRIFT:指針A]")

    with (
        patch("backend.core.chat.service.create_provider", return_value=fake_provider),
        patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.core.chat.service.find_urls", return_value=[]),
        patch("backend.core.chat.service.carve", side_effect=lambda text, *_: text),
    ):
        service = ChatService(memory_manager=memory_manager, drift_manager=drift_manager)
        result = await service.execute(request)

    # drift_manager.add_drift が呼ばれないこと
    drift_manager.add_drift.assert_not_called()
    # テキストはそのまま返ること（_apply_drifts がスキップされるのでマーカーが残る）
    assert "[DRIFT:指針A]" in result
