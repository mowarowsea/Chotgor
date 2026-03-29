"""ghost_model の DB 永続化・chronicle / forget ルーティングのテスト。

digest が削除されたため、夜間バッチのテストは chronicle に差し替え済み。
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import run_chronicle, run_pending_chronicles, _parse_chronicle_response
from backend.batch.forget_job import run_forget_process, _call_llm_for_forget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_manager(sqlite_store):
    """テスト用 MemoryManager（ChromaDB はモック）。"""
    from unittest.mock import MagicMock
    from backend.services.memory.manager import MemoryManager
    chroma = MagicMock()
    chroma.add_memory = MagicMock()
    chroma.delete_memory = MagicMock()
    chroma.search = MagicMock(return_value=[])
    return MemoryManager(sqlite_store, chroma)


# ---------------------------------------------------------------------------
# ghost_model DB 永続化テスト
# ---------------------------------------------------------------------------

def test_ghost_model_saved_and_retrieved(sqlite_store):
    """ghost_model を指定してキャラクターを作成し、正しく取得できることを確認する。"""
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    char = sqlite_store.get_character(char_id)
    assert char.ghost_model == preset_id


def test_ghost_model_default_is_none(sqlite_store):
    """ghost_model 未指定で作成したキャラクターのデフォルトが None であることを確認する。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")

    char = sqlite_store.get_character(char_id)
    assert char.ghost_model is None


def test_ghost_model_updated(sqlite_store):
    """ghost_model を後から update_character で設定できることを確認する。"""
    char_id = str(uuid.uuid4())
    preset_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")
    sqlite_store.update_character(char_id, ghost_model=preset_id)

    char = sqlite_store.get_character(char_id)
    assert char.ghost_model == preset_id


def test_ghost_model_cleared(sqlite_store):
    """ghost_model を None に更新できることを確認する。"""
    char_id = str(uuid.uuid4())
    preset_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)
    sqlite_store.update_character(char_id, ghost_model=None)

    char = sqlite_store.get_character(char_id)
    assert char.ghost_model is None


# ---------------------------------------------------------------------------
# _parse_chronicle_response テスト
# ---------------------------------------------------------------------------

def test_parse_chronicle_response_plain_json():
    """コードブロックなしの JSON をパースできることを確認する。"""
    raw = '{"self_history": {"update": true, "text": "hello"}, "relationship_state": {"update": false, "text": null}}'
    result = _parse_chronicle_response(raw)
    assert result["self_history"]["update"] is True
    assert result["self_history"]["text"] == "hello"
    assert result["relationship_state"]["update"] is False


def test_parse_chronicle_response_with_code_block():
    """```json ブロックで囲まれた JSON もパースできることを確認する。"""
    raw = '説明文\n```json\n{"self_history": {"update": false, "text": null}, "relationship_state": {"update": true, "text": "changed"}}\n```'
    result = _parse_chronicle_response(raw)
    assert result["relationship_state"]["update"] is True
    assert result["relationship_state"]["text"] == "changed"


def test_parse_chronicle_response_invalid_returns_empty():
    """不正な JSON の場合は空辞書を返すことを確認する。"""
    result = _parse_chronicle_response("これはJSONではありません")
    assert result == {}


# ---------------------------------------------------------------------------
# run_chronicle: ghost_model 未設定・プリセット不在のエラーハンドリング
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chronicle_skipped_when_no_ghost_model(sqlite_store):
    """ghost_model が未設定のキャラクターは skipped を返すことを確認する。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")

    result = await run_chronicle(
        character_id=char_id,
        target_date="2026-01-01",
        sqlite=sqlite_store,
    )

    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_run_chronicle_error_when_preset_not_found(sqlite_store):
    """存在しないプリセット ID を ghost_model に指定した場合は error を返すことを確認する。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar", ghost_model="nonexistent-id")

    result = await run_chronicle(
        character_id=char_id,
        target_date="2026-01-01",
        sqlite=sqlite_store,
    )

    assert result["status"] == "error"
    assert "nonexistent-id" in result["error"]


@pytest.mark.asyncio
async def test_run_chronicle_error_when_character_not_found(sqlite_store):
    """存在しないキャラクター ID を指定した場合は error を返すことを確認する。"""
    result = await run_chronicle(
        character_id="does-not-exist",
        target_date="2026-01-01",
        sqlite=sqlite_store,
    )

    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# run_chronicle: LLM が更新不要と回答した場合
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chronicle_no_update_when_llm_returns_no_update(sqlite_store):
    """LLM が両フィールドとも update: false を返した場合、updated_fields が空であることを確認する。"""
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    no_update_response = '{"self_history": {"update": false, "text": null}, "relationship_state": {"update": false, "text": null}}'
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=no_update_response)

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id,
            target_date="2026-01-01",
            sqlite=sqlite_store,
        )

    assert result["status"] == "success"
    assert result["updated_fields"] == []

    # DB が変わっていないことを確認する
    char = sqlite_store.get_character(char_id)
    assert char.self_history == ""
    assert char.relationship_state == ""


# ---------------------------------------------------------------------------
# run_chronicle: LLM が更新を指示した場合
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chronicle_updates_fields_when_llm_requests_update(sqlite_store):
    """LLM が self_history を更新するよう指示した場合、DB に反映されることを確認する。"""
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    update_response = '{"self_history": {"update": true, "text": "新しい歴史"}, "relationship_state": {"update": false, "text": null}}'
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=update_response)

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id,
            target_date="2026-01-01",
            sqlite=sqlite_store,
        )

    assert result["status"] == "success"
    assert "self_history" in result["updated_fields"]
    assert "relationship_state" not in result["updated_fields"]

    char = sqlite_store.get_character(char_id)
    assert char.self_history == "新しい歴史"
    assert char.relationship_state == ""


@pytest.mark.asyncio
async def test_run_chronicle_uses_ghost_model_preset(sqlite_store):
    """run_chronicle が正しいプロバイダー・モデルで create_provider を呼ぶことを確認する。"""
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "anthropic", "claude-3-5-haiku-latest")
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    no_update_response = '{"self_history": {"update": false, "text": null}, "relationship_state": {"update": false, "text": null}}'
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=no_update_response)

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider) as mock_cp:
        await run_chronicle(
            character_id=char_id,
            target_date="2026-01-01",
            sqlite=sqlite_store,
        )

    mock_cp.assert_called_once_with(
        "anthropic", "claude-3-5-haiku-latest",
        sqlite_store.get_all_settings(),
        thinking_level="default",
    )


# ---------------------------------------------------------------------------
# _call_llm_for_forget: エラーケース
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_llm_for_forget_no_ghost_model(sqlite_store):
    """ghost_model が None の場合に RuntimeError を送出することを確認する。"""
    with pytest.raises(RuntimeError, match="ghost_model"):
        await _call_llm_for_forget("sys", "mem_text", None, sqlite_store, {})


@pytest.mark.asyncio
async def test_call_llm_for_forget_unknown_preset(sqlite_store):
    """存在しないプリセット ID の場合に RuntimeError を送出することを確認する。"""
    with pytest.raises(RuntimeError, match="プリセット"):
        await _call_llm_for_forget("sys", "mem_text", "nonexistent-id", sqlite_store, {})


@pytest.mark.asyncio
async def test_call_llm_for_forget_uses_ghost_model_preset(sqlite_store):
    """_call_llm_for_forget が正しいプロバイダー・モデルで呼ばれることを確認する。"""
    preset_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "anthropic", "claude-3-5-haiku-latest")

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value="[KEEP: NONE]")

    settings = sqlite_store.get_all_settings()
    with patch("backend.batch.forget_job.create_provider", return_value=mock_provider) as mock_cp:
        result = await _call_llm_for_forget("sys_prompt", "candidates text", preset_id, sqlite_store, settings)

    assert result == "[KEEP: NONE]"
    mock_cp.assert_called_once_with(
        "anthropic", "claude-3-5-haiku-latest",
        settings,
        thinking_level="default",
    )


# ---------------------------------------------------------------------------
# run_forget_process: ghost_model 未設定のエラーハンドリング
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_forget_process_error_when_no_ghost_model(sqlite_store, memory_manager):
    """ghost_model が None の場合に error ステータスを返すことを確認する。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "TestChar")

    from datetime import datetime, timedelta
    sqlite_store.create_memory(
        memory_id=str(uuid.uuid4()),
        character_id=char_id,
        content="faint memory",
        contextual_importance=0.01,
        user_importance=0.01,
        semantic_importance=0.01,
        identity_importance=0.01,
    )
    with sqlite_store.get_session() as session:
        from backend.repositories.sqlite.store import Memory
        m = session.query(Memory).filter_by(character_id=char_id).first()
        m.created_at = datetime.now() - timedelta(days=180)
        m.last_accessed_at = datetime.now() - timedelta(days=180)
        session.commit()

    result = await run_forget_process(
        character_id=char_id,
        character_name="TestChar",
        character_system_prompt="You are TestChar.",
        memory_manager=memory_manager,
        sqlite=sqlite_store,
        settings={},
        threshold=10.0,
        ghost_model=None,
    )

    assert result["status"] == "error"
    assert "ghost_model" in result["error"]


# ---------------------------------------------------------------------------
# chronicled_at フラグ — ユーティリティ
# ---------------------------------------------------------------------------

def _setup_char_with_messages(sqlite_store, char_name: str = "Alice", n_messages: int = 2):
    """テスト用キャラクター・セッション・メッセージを一括作成するヘルパー。

    Returns:
        (char_id, session_id, message_ids, preset_id) のタプル。
    """
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(char_id, char_name, ghost_model=preset_id)

    session_id = str(uuid.uuid4())
    sqlite_store.create_chat_session(session_id=session_id, model_id=f"{char_name}@default")

    message_ids = []
    for i in range(n_messages):
        mid = str(uuid.uuid4())
        role = "user" if i % 2 == 0 else "character"
        sqlite_store.create_chat_message(
            message_id=mid,
            session_id=session_id,
            role=role,
            content=f"メッセージ {i}",
            character_name=char_name if role == "character" else None,
        )
        message_ids.append(mid)

    return char_id, session_id, message_ids, preset_id


_NO_UPDATE_RESPONSE = (
    '{"self_history": {"update": false, "text": null},'
    ' "relationship_state": {"update": false, "text": null}}'
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
async def test_run_chronicle_marks_messages_on_success(sqlite_store):
    """chronicle 成功後、処理対象メッセージの chronicled_at が設定されることを確認する。"""
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        result = await run_chronicle(character_id=char_id, sqlite=sqlite_store)

    assert result["status"] == "success"
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is not None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_does_not_mark_messages_on_llm_error(sqlite_store):
    """LLM 呼び出し失敗時は chronicled_at が NULL のままであることを確認する。"""
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=Exception("network error"))

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        result = await run_chronicle(character_id=char_id, sqlite=sqlite_store)

    assert result["status"] == "error"
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_does_not_mark_messages_on_json_parse_failure(sqlite_store):
    """JSON パース失敗時は chronicled_at が NULL のままであることを確認する。"""
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value="これはJSONではありません")

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        result = await run_chronicle(character_id=char_id, sqlite=sqlite_store)

    assert result["status"] == "error"
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_marks_even_when_no_field_updates(sqlite_store):
    """LLM が update: false を返した場合でも chronicled_at はセットされることを確認する。"""
    char_id, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        result = await run_chronicle(character_id=char_id, sqlite=sqlite_store)

    assert result["status"] == "success"
    assert result["updated_fields"] == []
    msgs = sqlite_store.list_chat_messages(session_id)
    assert all(m.chronicled_at is not None for m in msgs)


@pytest.mark.asyncio
async def test_run_chronicle_only_processes_unchronicled_messages(sqlite_store):
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

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        await run_chronicle(character_id=char_id, sqlite=sqlite_store)

    assert len(captured_prompt) == 1
    assert "メッセージ 0" not in captured_prompt[0]
    assert "メッセージ 1" in captured_prompt[0]
    assert "メッセージ 2" in captured_prompt[0]


@pytest.mark.asyncio
async def test_run_chronicle_all_chronicled_still_calls_llm(sqlite_store):
    """未処理メッセージがなくても LLM は呼ばれ（空会話として反芻）、success を返すことを確認する。"""
    char_id, _, message_ids, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    sqlite_store.mark_messages_as_chronicled(message_ids)

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        result = await run_chronicle(character_id=char_id, sqlite=sqlite_store)

    assert result["status"] == "success"
    mock_provider.generate.assert_called_once()


# ---------------------------------------------------------------------------
# run_pending_chronicles: スケジューラー経由のフラグベース処理テスト
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pending_chronicles_marks_unchronicled_messages(sqlite_store):
    """run_pending_chronicles 実行後、全未処理メッセージの chronicled_at が設定されることを確認する。"""
    _, session_id, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=2)
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.batch.chronicle_job.create_provider", return_value=mock_provider):
        await run_pending_chronicles(sqlite=sqlite_store)

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
