"""chronicled_at フラグ — 未棚卸しメッセージの抽出・マーキング・更新挙動のテスト。

検証する観点:
    - get_unchronicled_messages_for_character の抽出条件
    - mark_messages_as_chronicled のタイムスタンプ付与・冪等性
    - run_chronicle 成功/失敗時のフラグ更新有無
    - run_pending_chronicles のフラグベース処理
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import run_chronicle, run_pending_chronicles

from tests._ghost_model_helpers import (  # noqa: F401
    _NO_UPDATE_RESPONSE,
    _setup_char_with_messages,
    working_memory_manager,
)

# ---------------------------------------------------------------------------
# get_unchronicled_messages_for_character のテスト
# ---------------------------------------------------------------------------

def test_get_unchronicled_returns_new_messages(sqlite_store):
    """新規メッセージ（chronicled_at IS NULL）がすべて返されることを確認する。"""
    _, _, message_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=3)

    messages = sqlite_store.get_unchronicled_messages_for_character("Alice")

    assert {m.id for m in messages} == set(message_ids)


def test_get_unchronicled_excludes_already_chronicled(sqlite_store):
    """chronicled_at が設定済みのメッセージは除外されることを確認する。"""
    _, _, message_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=3)
    sqlite_store.mark_messages_as_chronicled([message_ids[0]])

    messages = sqlite_store.get_unchronicled_messages_for_character("Alice")

    ids = {m.id for m in messages}
    assert message_ids[0] not in ids
    assert message_ids[1] in ids
    assert message_ids[2] in ids


def test_get_unchronicled_excludes_system_messages(sqlite_store):
    """is_system_message=True のメッセージは chronicled_at が NULL でも除外されることを確認する。"""
    _, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)
    sys_mid = str(uuid.uuid4())
    sqlite_store.create_chat_message(
        message_id=sys_mid,
        session_id=session_id,
        role="character",
        content="退席しました",
        is_system_message=True,
    )

    messages = sqlite_store.get_unchronicled_messages_for_character("Alice")

    assert all(m.id != sys_mid for m in messages)


def test_get_unchronicled_returns_empty_when_all_chronicled(sqlite_store):
    """全メッセージが chronicled_at 設定済みの場合は空リストを返すことを確認する。"""
    _, _, message_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    sqlite_store.mark_messages_as_chronicled(message_ids)

    messages = sqlite_store.get_unchronicled_messages_for_character("Alice")

    assert messages == []


def test_get_unchronicled_filters_by_character(sqlite_store):
    """別キャラクターのメッセージは返されないことを確認する。"""
    _, _, alice_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    _setup_char_with_messages(sqlite_store, "Bob", n_messages=2)

    alice_messages = sqlite_store.get_unchronicled_messages_for_character("Alice")

    assert {m.id for m in alice_messages} == set(alice_ids)


# ---------------------------------------------------------------------------
# mark_messages_as_chronicled のテスト
# ---------------------------------------------------------------------------

def test_mark_messages_as_chronicled_sets_timestamp(sqlite_store):
    """mark 後に chronicled_at が非 NULL のタイムスタンプになることを確認する。"""
    _, session_id, message_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    sqlite_store.mark_messages_as_chronicled(message_ids)

    msgs = sqlite_store.list_chat_messages(session_id)
    for msg in msgs:
        assert msg.chronicled_at is not None


def test_mark_messages_as_chronicled_empty_list_is_noop(sqlite_store):
    """空リストを渡してもエラーが発生しないことを確認する。"""
    sqlite_store.mark_messages_as_chronicled([])  # 例外が出なければ OK


def test_mark_messages_as_chronicled_is_idempotent(sqlite_store):
    """同じメッセージを2回 mark しても問題ないことを確認する。"""
    _, session_id, message_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)
    sqlite_store.mark_messages_as_chronicled(message_ids)
    sqlite_store.mark_messages_as_chronicled(message_ids)  # 2回目も例外が出なければ OK

    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is not None for m in msgs)


# ---------------------------------------------------------------------------
# run_chronicle: chronicled_at フラグの更新挙動テスト
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chronicle_marks_messages_on_success(sqlite_store, working_memory_manager):
    """chronicle 成功後、処理対象メッセージの chronicled_at が設定されることを確認する。"""
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id, sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "success"
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is not None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_does_not_mark_messages_on_llm_error(sqlite_store, working_memory_manager):
    """LLM 呼び出し失敗時は chronicled_at が NULL のままであることを確認する。"""
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=Exception("network error"))

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id, sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "error"
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_does_not_mark_messages_on_json_parse_failure(sqlite_store, working_memory_manager):
    """JSON パース失敗時は chronicled_at が NULL のままであることを確認する。"""
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value="これはJSONではありません")

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id, sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "error"
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_marks_even_when_no_updates(sqlite_store, working_memory_manager):
    """LLM が全配列とも空で返した場合でも chronicled_at はセットされることを確認する。"""
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id, sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "success"
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is not None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_only_processes_unchronicled_messages(sqlite_store, working_memory_manager):
    """既に chronicled_at が設定済みのメッセージは LLM への入力に含まれないことを確認する。"""
    char_id, _, message_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=3)
    # message_ids[0] を処理済みにする
    sqlite_store.mark_messages_as_chronicled([message_ids[0]])

    captured_prompt: list[str] = []

    async def fake_generate(sys_prompt, messages):
        captured_prompt.append(messages[0]["content"])
        return _NO_UPDATE_RESPONSE

    mock_provider = AsyncMock()
    mock_provider.generate = fake_generate

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        await run_chronicle(
            character_id=char_id, sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert len(captured_prompt) == 1
    assert "メッセージ 0" not in captured_prompt[0]
    assert "メッセージ 1" in captured_prompt[0]
    assert "メッセージ 2" in captured_prompt[0]


@pytest.mark.asyncio
async def test_run_chronicle_all_chronicled_still_calls_llm(sqlite_store, working_memory_manager):
    """未処理メッセージがなくても LLM は呼ばれ（空会話として反芻）、success を返すことを確認する。"""
    char_id, _, message_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    sqlite_store.mark_messages_as_chronicled(message_ids)

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id, sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "success"
    mock_provider.generate.assert_called_once()


# ---------------------------------------------------------------------------
# run_chronicle: LLM 出力の数値堅牢性・失敗計上テスト（リファクタ回帰防止）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chronicle_non_numeric_importance_does_not_crash(
    sqlite_store, working_memory_manager,
):
    """new_threads の importance が非数値（"high" 等）でも棚卸し全体が中断しないことを確認する。

    リファクタで float() 変換が try の外に出てしまい、LLM が "importance": "high" のような
    非数値を返すと ValueError が run_chronicle を貫通 → メッセージが未chronicle のまま
    翌晩も同じ会話で再クラッシュする恒久スタックを起こしていた。_safe_float による堅牢化を検証する。
    """
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)
    # importance が非数値の new_thread を返す chronicle 応答
    bad_importance_response = (
        '{"thread_updates": [], "new_threads": ['
        '{"type": "topic", "summary": "気になる話題", "atmosphere_tag": "もやもや",'
        ' "importance": "high", "post": "考え中", "origin": "real"}'
        '], "merges": [], "inscribe": [], "farewell_config": null}'
    )
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=bad_importance_response)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id, sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    # クラッシュせず success。非数値 importance は既定値(0.5)へ丸めてスレッドは作成される。
    assert result["status"] == "success"
    assert result["counts"].get("created") == 1
    # メッセージは chronicled 済みになる（恒久スタックしない）
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is not None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_update_to_missing_thread_not_counted(
    sqlite_store, working_memory_manager,
):
    """存在しない thread_id への更新は「成功」として計上されないことを確認する。

    ToolExecutor.execute() はツール失敗を例外ではなくエラー文字列で返すため、
    例外捕捉だけでは失敗を検知できない。result_looks_like_error チェックにより
    counts['updated'] が水増しされないことを検証する（リファクタ回帰防止）。
    """
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)
    missing_thread_response = (
        '{"thread_updates": ['
        '{"id": "nonexistent-thread-id", "summary": "存在しないスレッドへの更新",'
        ' "atmosphere_tag": null, "importance": null, "new_post": null, "is_open": null}'
        '], "new_threads": [], "merges": [], "inscribe": [], "farewell_config": null}'
    )
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=missing_thread_response)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id, sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "success"
    # 失敗した更新は計上されない
    assert result["counts"].get("updated", 0) == 0


# ---------------------------------------------------------------------------
# run_pending_chronicles: スケジューラー経由のフラグベース処理テスト
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pending_chronicles_marks_unchronicled_messages(sqlite_store, working_memory_manager):
    """run_pending_chronicles 実行後、全未処理メッセージの chronicled_at が設定されることを確認する。"""
    _, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        await run_pending_chronicles(
            sqlite=sqlite_store, working_memory_manager=working_memory_manager,
        )

    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is not None for m in msgs)


@pytest.mark.asyncio
async def test_run_pending_chronicles_skips_chars_without_ghost_model(sqlite_store):
    """ghost_model 未設定のキャラクターはスキップされ、エラーにならないことを確認する。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "NoGhost")
    session_id = str(uuid.uuid4())
    sqlite_store.create_chat_session(session_id=session_id, model_id="NoGhost@default")
    sqlite_store.create_chat_message(
        message_id=str(uuid.uuid4()), session_id=session_id, role="user", content="hello"
    )

    # ghost_model なしのキャラクターのみの場合、例外なく完了すればよい
    await run_pending_chronicles(sqlite=sqlite_store)  # 例外が出なければ OK

    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is None for m in msgs)


