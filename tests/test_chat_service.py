"""backend.services.chat.service と backend.services.chat.models のテスト。

ChatService のフロー全体を検証する:
- SUPPORTS_TOOLS=False パス: タグ抽出後 ToolExecutor.apply_*_tags() で execute() 経由
- SUPPORTS_TOOLS=True パス: generate_with_tools / ToolExecutor 経由
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.character_actions.executor import ToolCall, ToolTurnResult
from backend.services.chat.models import ChatRequest, Message
from backend.services.chat.service import ChatService, extract_text_content


# --- extract_text_content (moved from test_llm_service_multimodal) ---

def test_extract_text_content_string():
    """文字列をそのまま返すこと。"""
    assert extract_text_content("hello") == "hello"

def test_extract_text_content_none():
    """None は空文字列を返すこと。"""
    assert extract_text_content(None) == ""

def test_extract_text_content_list():
    """リスト形式の content からテキスト部分だけ抽出すること。"""
    content = [
        {"type": "text", "text": "Hello "},
        {"type": "image_url", "image_url": {"url": "..."}},
        "world",
    ]
    assert extract_text_content(content) == "Hello world"

def test_extract_text_content_empty_list():
    """空リストは空文字列を返すこと。"""
    assert extract_text_content([]) == ""


# --- ChatRequest のデフォルト値確認 ---

def test_chat_request_defaults():
    """ChatRequest のデフォルト値が正しいこと。inner_narrative フィールドが存在すること。"""
    req = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="hi")],
    )
    assert req.character_system_prompt == ""
    assert req.inner_narrative == ""
    assert req.provider_additional_instructions == ""
    assert req.settings == {}


# --- ChatService.execute (SUPPORTS_TOOLS=False パス) ---

@pytest.mark.asyncio
async def test_chat_service_execute_returns_text():
    """SUPPORTS_TOOLS=False のプロバイダーは generate() を呼び、クリーンテキストを返すこと。

    タグ方式の実行口は ToolExecutor.apply_*_tags() に統一されたが、応答にマーカーが
    含まれなければ何も呼ばれない（memory_manager の write が触られない）ためそのまま動く。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(return_value="Hi there!")

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "Hi there!"


@pytest.mark.asyncio
async def test_chat_service_execute_provider_error_returns_error_string():
    """SUPPORTS_TOOLS=False でプロバイダーが例外を送出した場合、エラー文字列を返すこと。"""
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="boom")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(side_effect=RuntimeError("oops"))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result.startswith("[Error: RuntimeError: oops")


# --- ChatService.execute (SUPPORTS_TOOLS=True パス) ---

@pytest.mark.asyncio
async def test_chat_service_execute_with_tools_returns_text():
    """SUPPORTS_TOOLS=True のプロバイダーは generate_with_tools を呼び、その戻り値を返すこと。

    タグ抽出後の apply_*_tags は呼ばれない経路（tool-use ループ内で直接 execute() される）。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value=("Hi via tools!", ""))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "Hi via tools!"
    fake_provider.generate_with_tools.assert_awaited_once()
    fake_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_chat_service_execute_with_tools_error_returns_error_string():
    """SUPPORTS_TOOLS=True でプロバイダーが例外を送出した場合、エラー文字列を返すこと。"""
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="boom")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(side_effect=RuntimeError("tools oops"))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result.startswith("[Error: RuntimeError: tools oops")


# --- ChatService.execute — ToolExecutor 統合テスト ---

def _make_tool_provider(turn_results: list):
    """指定した ToolTurnResult リストを順に返すフェイクプロバイダーを生成する。

    SUPPORTS_TOOLS=True のプロバイダーとして振る舞い、
    generate_with_tools のツールループ（BaseLLMProvider 実装）をそのまま使う。

    Args:
        turn_results: _tool_turn() が順に返す ToolTurnResult のリスト。

    Returns:
        BaseLLMProvider サブクラスのインスタンス。
    """
    from backend.providers.base import BaseLLMProvider

    class FakeToolProvider(BaseLLMProvider):
        """テスト用ツール対応プロバイダー。"""

        SUPPORTS_TOOLS = True

        def __init__(self):
            """フェイクプロバイダーを初期化する。"""
            self._turns = iter(turn_results)

        async def _tool_turn(self, system_prompt, messages):
            """事前に設定した ToolTurnResult を順に返す。"""
            return next(self._turns)

        def _extend_messages_with_results(self, messages, turn_result, results):
            """ツール結果をダミーメッセージとして追加する。"""
            return messages + [{"role": "tool_result_dummy", "content": str(results)}]

    return FakeToolProvider()


@pytest.mark.asyncio
async def test_execute_with_tools_calls_memory_manager_via_inscribe_memory():
    """SUPPORTS_TOOLS=True のプロバイダーが inscribe_memory ツールを呼び出したとき、
    memory_manager.write_inscribed_memory が実際に呼ばれること。

    サービス → generate_with_tools → ToolExecutor → inscribe_memory → memory_manager という
    一連の呼び出しチェーンをサービスレベルで統合検証する。
    """

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="覚えてね")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-1", name="inscribe_memory", input={"content": "ユーザは猫が好き", "category": "user", "impact": 1.0})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="覚えたよ", tool_calls=[]),
    ])

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "覚えたよ"
    memory_manager.write_inscribed_memory.assert_called_once()
    call_kwargs = memory_manager.write_inscribed_memory.call_args
    assert call_kwargs.kwargs.get("content") == "ユーザは猫が好き"


@pytest.mark.asyncio
async def test_execute_with_tools_calls_carve_narrative_via_tool_executor():
    """SUPPORTS_TOOLS=True のプロバイダーが carve_narrative ツールを呼び出したとき、
    sqlite_store.update_character が実際に呼ばれること。

    サービス → generate_with_tools → ToolExecutor → carve_narrative → Carver → sqlite という
    一連の呼び出しチェーンをサービスレベルで統合検証する。
    """

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])
    mock_char = MagicMock()
    mock_char.inner_narrative = ""
    memory_manager.sqlite.get_character.return_value = mock_char

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="自分を書き直したい")],
        session_id="session-abc",
    )

    tc = ToolCall(id="tc-2", name="carve_narrative", input={"mode": "append", "content": "知的好奇心を大切にする"})
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="指針を彫り込んだ", tool_calls=[]),
    ])

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "指針を彫り込んだ"
    memory_manager.sqlite.update_character.assert_called_once()


@pytest.mark.asyncio
async def test_execute_with_tools_creates_thread_via_post_working_memory_thread():
    """SUPPORTS_TOOLS=True のプロバイダーが post_working_memory_thread ツール（新規作成）を呼び出したとき、
    working_memory_manager.create_thread が実際に呼ばれること。

    サービス → generate_with_tools → ToolExecutor → Threader → WorkingMemoryManager という
    一連の呼び出しチェーンをサービスレベルで統合検証する。
    """

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])
    working_memory_manager = MagicMock()
    working_memory_manager.create_thread.return_value = {"id": "thread-1"}

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="この件、気になってる")],
        session_id="session-abc",
    )

    tc = ToolCall(
        id="tc-3",
        name="post_working_memory_thread",
        input={"type": "topic", "summary": "気になっている件", "importance": 0.7},
    )
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="スレッドに残した", tool_calls=[]),
    ])

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(
            memory_manager=memory_manager, working_memory_manager=working_memory_manager
        )
        result = await service.execute(request)

    assert result == "スレッドに残した"
    working_memory_manager.create_thread.assert_called_once()
    kwargs = working_memory_manager.create_thread.call_args.kwargs
    assert kwargs["character_id"] == "char-1"
    assert kwargs["type"] == "topic"
    assert kwargs["summary"] == "気になっている件"


@pytest.mark.asyncio
async def test_execute_with_tools_adds_post_via_post_working_memory_thread():
    """SUPPORTS_TOOLS=True のプロバイダーが post_working_memory_thread ツール（既存更新）を呼び出したとき、
    working_memory_manager.add_post が実際に呼ばれること。
    """

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])
    working_memory_manager = MagicMock()

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="さっきの件、進展あった")],
        session_id="session-abc",
    )

    tc = ToolCall(
        id="tc-4",
        name="post_working_memory_thread",
        input={"thread_id": "thread-9", "content": "進展があった"},
    )
    provider = _make_tool_provider([
        ToolTurnResult(text="", tool_calls=[tc]),
        ToolTurnResult(text="追記した", tool_calls=[]),
    ])

    with (
        patch("backend.services.chat.service.create_provider", return_value=provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(
            memory_manager=memory_manager, working_memory_manager=working_memory_manager
        )
        result = await service.execute(request)

    assert result == "追記した"
    working_memory_manager.add_post.assert_called_once_with("thread-9", "進展があった")


@pytest.mark.asyncio
async def test_execute_without_tools_does_not_call_generate_with_tools():
    """SUPPORTS_TOOLS=False のプロバイダーは generate() のみを呼び、
    generate_with_tools は呼ばれないこと。

    タグ抽出後の実行は ToolExecutor.apply_*_tags() を経由するが、ここではマーカーがないので
    実際の execute() 呼び出しは発生しない（マーカーなしテキストでパススルー）。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="claude_cli",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = False
    fake_provider.generate = AsyncMock(return_value="Hi!")
    fake_provider.generate_with_tools = AsyncMock(side_effect=AssertionError("呼ばれてはいけない"))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        result = await service.execute(request)

    assert result == "Hi!"
    fake_provider.generate.assert_called_once()
    fake_provider.generate_with_tools.assert_not_called()


# --- ChatService.execute_stream — 想起失敗の UI 通知（recall_error） ---

async def _collect_stream_events(service, request):
    """execute_stream が yield する (type, content) タプルを全件リストに集める補助関数。

    非同期ジェネレーターをテスト内で扱いやすくするため、全イベントを同期的なリストへ展開する。
    """
    events = []
    async for ev in service.execute_stream(request):
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_execute_stream_yields_recall_error_on_embedding_failure():
    """記憶想起時に embedding が失敗（EmbeddingError）した場合、
    execute_stream が embedding 専用のエラーメッセージを ("recall_error", ...) として yield すること。

    背景: infinity 等の embedding サーバが落ちると _embed_query が EmbeddingError を送出する。
    従来はこれが service 層で握り潰され、UI には何も出ず「想起ゼロ」と区別できなかった。
    本テストは、その失敗が UI 表示用メッセージとしてストリームに乗ることを保証する
    （フロントは想起欄のスケッチ行としてこのメッセージを表示する）。
    """
    from backend.repositories.lance.store import EmbeddingError

    memory_manager = MagicMock()
    # 想起呼び出しが embedding 失敗で例外送出するケースを模す
    memory_manager.recall_with_identity.side_effect = EmbeddingError("connection refused")

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value=("Hi there!", ""))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        events = await _collect_stream_events(service, request)

    # embedding 専用メッセージが recall_error として流れること
    assert ("recall_error", "記憶の想起に失敗しました（embedding接続エラー）") in events
    # 想起は空なので inscribed_memories は流れないこと
    assert not any(t == "inscribed_memories" for t, _ in events)
    # 応答テキスト自体は通常通り流れること
    assert ("text", "Hi there!") in events


@pytest.mark.asyncio
async def test_execute_stream_yields_generic_recall_error_on_other_failure():
    """embedding 以外の理由で想起が失敗した場合、execute_stream が汎用エラーメッセージを
    ("recall_error", ...) として yield すること。

    EmbeddingError 以外（例: LanceDB 破損や予期せぬ例外）は原因を embedding と断定できないため、
    embedding 専用メッセージではなく汎用の「記憶の想起に失敗しました」を表示する。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.side_effect = RuntimeError("boom")

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value=("Hi there!", ""))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        events = await _collect_stream_events(service, request)

    assert ("recall_error", "記憶の想起に失敗しました") in events


@pytest.mark.asyncio
async def test_execute_stream_no_recall_error_on_success():
    """想起が成功した通常時は recall_error イベントが一切流れないこと（回帰防止）。"""
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value=("Hi there!", ""))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        events = await _collect_stream_events(service, request)

    assert not any(t == "recall_error" for t, _ in events)


@pytest.mark.asyncio
async def test_execute_stream_yields_anticipation_event():
    """応答に [ANTICIPATE_RESPONSE:...] が含まれるとき、本文からタグが除去され、
    予想が ("anticipation", ...) イベントとして yield されること（tool-use 方式）。

    予想タグは全プロバイダー一律のため tool-use 方式の応答テキストにも本文末尾に
    出現する。execute_stream はそれを抽出し、本文（text）からは除去したうえで
    anticipation を別イベントとして流すことを検証する。
    """
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(
        return_value=("こんにちは！[ANTICIPATE_RESPONSE:次は質問が来ると思う]", "")
    )

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys"),
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        events = await _collect_stream_events(service, request)

    # 本文からタグが除去されて流れること
    assert ("text", "こんにちは！") in events
    # 予想が anticipation イベントとして流れること
    assert ("anticipation", "次は質問が来ると思う") in events


# --- ChatService — 記憶縮退のキャラクター向け告知（memory_degraded フラグの伝搬） ---

@pytest.mark.asyncio
async def test_prepare_context_passes_memory_degraded_on_recall_embedding_failure():
    """長期記憶の想起が EmbeddingError で失敗した場合、build_system_prompt に
    memory_degraded=True が渡ること。

    記憶縮退の通知先はユーザ（recall_error → UIスケッチ欄）だけでなく、
    キャラクター本人（システムプロンプト内の運用告知ブロック）でもある、という
    Chotgor の思想（記憶へのアクセス断絶は本人の身に起きている事象）を回帰防止する。
    """
    from backend.repositories.lance.store import EmbeddingError

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.side_effect = EmbeddingError("connection refused")

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value=("Hi there!", ""))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys") as mock_build,
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        await _collect_stream_events(service, request)

    assert mock_build.call_args.kwargs["memory_degraded"] is True


@pytest.mark.asyncio
async def test_prepare_context_passes_memory_degraded_on_wm_embedding_failure():
    """ワーキングメモリの heat 想起が EmbeddingError で失敗した場合も memory_degraded=True が
    渡り、ユーザ向けには WM 用の recall_error イベントが流れること。

    従来は WM 取得失敗が warning ログのみで、ユーザにもキャラクターにも完全に無通知だった
    （Block 8 が静かに消えるだけ）。本テストは「WM 失敗も縮退として両系統に通知される」
    ことを保証する。長期記憶の想起自体は成功しているケースなので、recall 系の
    エラーメッセージではなく WM 専用メッセージが選ばれることも確認する。
    """
    from backend.repositories.lance.store import EmbeddingError

    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    working_memory_manager = MagicMock()
    working_memory_manager.list_all_threads.return_value = []
    working_memory_manager.get_fixed_threads.return_value = []
    working_memory_manager.recall_threads.side_effect = EmbeddingError("connection refused")

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value=("Hi there!", ""))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys") as mock_build,
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(
            memory_manager=memory_manager,
            working_memory_manager=working_memory_manager,
        )
        events = await _collect_stream_events(service, request)

    assert mock_build.call_args.kwargs["memory_degraded"] is True
    assert ("recall_error", "ワーキングメモリの取得に失敗しました（embedding接続エラー）") in events


@pytest.mark.asyncio
async def test_prepare_context_memory_degraded_false_on_success():
    """記憶・WM の取得がすべて成功した通常時は memory_degraded=False が渡ること（回帰防止）。"""
    memory_manager = MagicMock()
    memory_manager.recall_with_identity.return_value = ([], [])

    request = ChatRequest(
        character_id="char-1",
        character_name="Alice",
        provider="anthropic",
        model="",
        messages=[Message(role="user", content="hello")],
    )

    fake_provider = AsyncMock()
    fake_provider.SUPPORTS_TOOLS = True
    fake_provider.generate_with_tools = AsyncMock(return_value=("Hi there!", ""))

    with (
        patch("backend.services.chat.service.create_provider", return_value=fake_provider),
        patch("backend.services.chat.service.build_system_prompt", return_value="sys") as mock_build,
        patch("backend.services.chat.service.find_urls", return_value=[]),
    ):
        service = ChatService(memory_manager=memory_manager)
        await _collect_stream_events(service, request)

    assert mock_build.call_args.kwargs["memory_degraded"] is False
