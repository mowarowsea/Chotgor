"""ghost_model の DB 永続化・chronicle / forget ルーティングのテスト。

digest が削除されたため、夜間バッチのテストは chronicle に差し替え済み。
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import (
    run_chronicle,
    run_pending_chronicles,
    _parse_chronicle_response,
    _format_memories,
)
from backend.batch.forget_job import run_forget_process


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_manager(sqlite_store):
    """テスト用 InscribedMemoryManager（LanceDB はモック）。"""
    from unittest.mock import MagicMock
    from backend.services.memory.manager import InscribedMemoryManager
    vector_store = MagicMock()
    vector_store.add_inscribed_memory = MagicMock()
    vector_store.delete_inscribed_memory = MagicMock()
    vector_store.search = MagicMock(return_value=[])
    return InscribedMemoryManager(sqlite_store, vector_store)


@pytest.fixture
def working_memory_manager(sqlite_store):
    """テスト用 WorkingMemoryManager。

    スレッド CRUD は実 SQLite（インメモリ一時DB）で動かし、embedding 層の
    LanceStore のみ MagicMock に置き換える。chronicle が棚卸し結果を
    実際にスレッドへ反映する挙動をそのまま検証できる。
    """
    from unittest.mock import MagicMock
    from backend.services.memory.working_memory_manager import WorkingMemoryManager
    return WorkingMemoryManager(sqlite_store, MagicMock())


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
    raw = '{"new_threads": [{"type": "topic", "summary": "hello"}], "carve": null}'
    result = _parse_chronicle_response(raw)
    assert result["new_threads"][0]["type"] == "topic"
    assert result["new_threads"][0]["summary"] == "hello"
    assert result["carve"] is None


def test_parse_chronicle_response_with_code_block():
    """```json ブロックで囲まれた JSON もパースできることを確認する。"""
    raw = '説明文\n```json\n{"thread_updates": [], "carve": {"mode": "append", "text": "changed"}}\n```'
    result = _parse_chronicle_response(raw)
    assert result["thread_updates"] == []
    assert result["carve"]["text"] == "changed"


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
# run_chronicle: LLM が棚卸し不要と回答した場合
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chronicle_no_update_when_llm_returns_empty(sqlite_store, working_memory_manager):
    """LLM が全配列とも空で返した場合、counts が全て 0 でスレッドも作られないことを確認する。"""
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id,
            target_date="2026-01-01",
            sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "success"
    assert all(v == 0 for v in result["counts"].values())

    # ワーキングメモリにスレッドが作られていないことを確認する
    assert working_memory_manager.list_threads_by_type(char_id) == []


# ---------------------------------------------------------------------------
# run_chronicle: LLM が棚卸し・蒸留を指示した場合
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chronicle_applies_new_thread_and_ignores_carve(
    sqlite_store, working_memory_manager
):
    """Chronicle が new_threads は反映し、carve は無視することを確認する。

    三段階の蒸留パイプライン（WorkingMemory → InscribedMemory → InnerNarrative）上、
    Chronicle は第1段（WM 整理）と第2段（長期記憶への昇格）のみを担い、第3段の
    inner_narrative への昇華（carve）は Forget バッチへ移譲された。本テストでは、LLM 応答に
    レガシーな carve フィールドが紛れ込んでも以下が保たれることを保証する:
      - new_threads（WM 整理）は通常どおり反映される
      - carve は完全に無視され、inner_narrative は変更されない
      - 反映件数 counts に "carved" キーは現れない
    """
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    # 敢えて carve フィールドを含む応答を返させ、それが無視されることを確認する
    update_response = (
        '{"thread_updates": [], "new_threads": ['
        '{"type": "topic", "summary": "気になっている問い", "atmosphere": "モヤモヤ",'
        ' "importance": 0.6, "post": "今日の会話で浮かんだ", "relation_target": null}],'
        ' "merges": [], "inscribe": [],'
        ' "carve": {"mode": "append", "text": "Chronicleでは無視されるべき自己像テキスト"},'
        ' "farewell_config": null}'
    )
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=update_response)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id,
            target_date="2026-01-01",
            sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "success"
    assert result["counts"]["created"] == 1
    # carve は実行されないため、件数辞書に carved キーは存在しない
    assert "carved" not in result["counts"]

    # 新規スレッドが実際に作られていること（WM 整理は通常どおり機能する）
    threads = working_memory_manager.list_threads_by_type(char_id)
    assert len(threads) == 1
    assert threads[0]["summary"] == "気になっている問い"
    assert threads[0]["type"] == "topic"

    # inner_narrative は変更されていないこと（carve テキストが書き込まれていない）
    char = sqlite_store.get_character(char_id)
    assert "Chronicleでは無視されるべき自己像テキスト" not in (char.inner_narrative or "")


@pytest.mark.asyncio
async def test_run_chronicle_prompt_includes_closed_threads(sqlite_store, working_memory_manager):
    """Close 済みスレッドが棚卸しプロンプト（ユーザメッセージ）に参照用として含まれることを確認する。"""
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    # Close 済みスレッドを1本用意する
    thread = working_memory_manager.create_thread(
        character_id=char_id, type="task", summary="決着済みの課題XYZ", importance=0.5,
    )
    working_memory_manager.set_open(thread["id"], False)

    captured_prompt: list[str] = []

    async def fake_generate(sys_prompt, messages):
        captured_prompt.append(messages[0]["content"])
        return _NO_UPDATE_RESPONSE

    mock_provider = AsyncMock()
    mock_provider.generate = fake_generate

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        await run_chronicle(
            character_id=char_id, target_date="2026-01-01", sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert len(captured_prompt) == 1
    assert "決着済みの課題XYZ" in captured_prompt[0]
    assert "最近 Close したスレッド" in captured_prompt[0]


@pytest.mark.asyncio
async def test_run_chronicle_system_prompt_includes_working_memory_threads(
    sqlite_store, working_memory_manager
):
    """chronicle のシステムプロンプトが 1on1 基準に統一されていることを確認する。

    emotion 固定注入（Block 7）に加え、Close 済み task スレッドも含む全スレッド一覧
    （Block 6）がシステムプロンプトに入る。「私は過去こういうことがあった」という
    自己認識を 1on1 チャットと同じ形で持たせる。
    """
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    working_memory_manager.create_thread(
        character_id=char_id, type="emotion", summary="落ち着いた高揚感ABC", importance=0.5,
    )
    closed = working_memory_manager.create_thread(
        character_id=char_id, type="task", summary="決着済みの課題GHI", importance=0.5,
    )
    working_memory_manager.set_open(closed["id"], False)

    captured_system: list[str] = []

    async def fake_generate(sys_prompt, messages):
        captured_system.append(sys_prompt)
        return _NO_UPDATE_RESPONSE

    mock_provider = AsyncMock()
    mock_provider.generate = fake_generate

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        await run_chronicle(
            character_id=char_id, target_date="2026-01-01", sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    assert len(captured_system) == 1
    # Block 7: emotion 固定注入
    assert "落ち着いた高揚感ABC" in captured_system[0]
    # Block 6: Close 済み task も含む全スレッド一覧
    assert "決着済みの課題GHI" in captured_system[0]


@pytest.mark.asyncio
async def test_run_chronicle_uses_ghost_model_preset(sqlite_store, working_memory_manager):
    """run_chronicle が正しいプロバイダー・モデルで create_provider を呼ぶことを確認する。"""
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "anthropic", "claude-3-5-haiku-latest")
    sqlite_store.create_character(char_id, "TestChar", ghost_model=preset_id)

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

    with patch("backend.services.character_query.create_provider", return_value=mock_provider) as mock_cp:
        await run_chronicle(
            character_id=char_id,
            target_date="2026-01-01",
            sqlite=sqlite_store,
            working_memory_manager=working_memory_manager,
        )

    mock_cp.assert_called_once_with(
        "anthropic", "claude-3-5-haiku-latest",
        sqlite_store.get_all_settings(),
        thinking_level="default",
        preset_name="TestPreset",
        timeout_seconds=300,
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
    sqlite_store.create_inscribed_memory(
        memory_id=str(uuid.uuid4()),
        character_id=char_id,
        content="faint memory",
        contextual_importance=0.01,
        user_importance=0.01,
        semantic_importance=0.01,
        identity_importance=0.01,
    )
    with sqlite_store.get_session() as session:
        from backend.repositories.sqlite.store import InscribedMemory
        m = session.query(InscribedMemory).filter_by(character_id=char_id).first()
        m.created_at = datetime.now() - timedelta(days=180)
        m.last_accessed_at = datetime.now() - timedelta(days=180)
        session.commit()

    result = await run_forget_process(
        character_id=char_id,
        character_name="TestChar",
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


# 棚卸し・蒸留とも「変更なし」を表す chronicle 応答。
_NO_UPDATE_RESPONSE = (
    '{"thread_updates": [], "new_threads": [], "merges": [],'
    ' "inscribe": [], "carve": null, "farewell_config": null}'
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


# ---------------------------------------------------------------------------
# _format_memories テスト
# ---------------------------------------------------------------------------

def test_format_memories_empty_list_returns_placeholder():
    """空リストを渡した場合、「記憶なし」プレースホルダーが返されることを確認する。"""
    result = _format_memories([])
    assert result == "（記憶なし）"


def test_format_memories_formats_category_and_score():
    """記憶オブジェクトのカテゴリ・スコア・コンテンツが正しく整形されることを確認する。

    _decayed_score 属性が float 形式（小数点2桁）で、カテゴリと内容とともに
    "[カテゴリ|スコア] 内容" の形式で出力されることを確認する。
    """
    from unittest.mock import MagicMock
    m = MagicMock()
    m.memory_category = "identity"
    m.content = "私はキャラクターAです"
    m._decayed_score = 0.75
    result = _format_memories([m])
    assert "[identity|0.75]" in result
    assert "私はキャラクターAです" in result


def test_format_memories_multiple_entries_one_per_line():
    """複数の記憶が1件1行で出力されることを確認する。

    記憶の件数と出力行数が一致することを確認する。
    """
    from unittest.mock import MagicMock
    memories = []
    for i in range(3):
        m = MagicMock()
        m.memory_category = "contextual"
        m.content = f"記憶 {i}"
        m._decayed_score = 0.5
        memories.append(m)
    result = _format_memories(memories)
    assert result.count("\n") == 2  # 3件 → 改行2本


# ---------------------------------------------------------------------------
# InscribedMemoryManager.get_top_memorable テスト
# ---------------------------------------------------------------------------

def test_get_top_memorable_empty_returns_empty_list(sqlite_store, memory_manager):
    """記憶が1件もないキャラクターに対して空リストを返すことを確認する。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "MemChar")
    result = memory_manager.get_top_memorable(char_id, limit=10)
    assert result == []


def test_get_top_memorable_returns_descending_order(sqlite_store, memory_manager):
    """importance が高い記憶が先頭に来るよう降順で返されることを確認する。

    全importance=0.0（低スコア）と全importance=1.0（高スコア）の2件を登録し、
    高スコア記憶が先頭に来ることを確認する。
    """
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "MemChar")

    low_id = str(uuid.uuid4())
    sqlite_store.create_inscribed_memory(
        memory_id=low_id, character_id=char_id, content="低スコア記憶",
        contextual_importance=0.0, semantic_importance=0.0,
        identity_importance=0.0, user_importance=0.0,
    )
    high_id = str(uuid.uuid4())
    sqlite_store.create_inscribed_memory(
        memory_id=high_id, character_id=char_id, content="高スコア記憶",
        contextual_importance=1.0, semantic_importance=1.0,
        identity_importance=1.0, user_importance=1.0,
    )

    results = memory_manager.get_top_memorable(char_id, limit=10)

    assert len(results) == 2
    assert results[0].id == high_id
    assert results[1].id == low_id


def test_get_top_memorable_respects_limit(sqlite_store, memory_manager):
    """limit を超える件数の記憶がある場合、limit 件のみ返されることを確認する。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "MemChar")

    for i in range(5):
        sqlite_store.create_inscribed_memory(
            memory_id=str(uuid.uuid4()), character_id=char_id,
            content=f"記憶 {i}",
            contextual_importance=0.5, semantic_importance=0.5,
            identity_importance=0.5, user_importance=0.5,
        )

    results = memory_manager.get_top_memorable(char_id, limit=3)
    assert len(results) == 3


def test_get_top_memorable_excludes_deleted(sqlite_store, memory_manager):
    """ソフトデリート済みの記憶は結果に含まれないことを確認する。

    削除済み記憶の importance を意図的に高く設定し、
    削除されていなければ必ず先頭に来るはずの状況で、
    結果に含まれないことを検証する。
    """
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "MemChar")

    live_id = str(uuid.uuid4())
    sqlite_store.create_inscribed_memory(
        memory_id=live_id, character_id=char_id, content="生きている記憶",
        contextual_importance=0.5, semantic_importance=0.5,
        identity_importance=0.5, user_importance=0.5,
    )
    deleted_id = str(uuid.uuid4())
    sqlite_store.create_inscribed_memory(
        memory_id=deleted_id, character_id=char_id, content="削除済み記憶",
        contextual_importance=1.0, semantic_importance=1.0,
        identity_importance=1.0, user_importance=1.0,
    )
    sqlite_store.soft_delete_inscribed_memory(deleted_id)

    results = memory_manager.get_top_memorable(char_id, limit=10)

    ids = [m.id for m in results]
    assert live_id in ids
    assert deleted_id not in ids


def test_get_top_memorable_attaches_decayed_score(sqlite_store, memory_manager):
    """返された記憶オブジェクトに _decayed_score 属性が付与されていることを確認する。

    _format_memories がこの属性を参照するため、必ず付与されている必要がある。
    """
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(char_id, "MemChar")
    sqlite_store.create_inscribed_memory(
        memory_id=str(uuid.uuid4()), character_id=char_id, content="スコア付き記憶",
        contextual_importance=0.8, semantic_importance=0.5,
        identity_importance=0.3, user_importance=0.4,
    )

    results = memory_manager.get_top_memorable(char_id, limit=10)

    assert len(results) == 1
    assert hasattr(results[0], '_decayed_score')
    assert isinstance(results[0]._decayed_score, float)
    assert results[0]._decayed_score >= 0.0


# ---------------------------------------------------------------------------
# run_chronicle: memory_manager とプロンプト連携テスト
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_chronicle_includes_memories_in_prompt_when_memory_manager_given(
    sqlite_store, memory_manager, working_memory_manager
):
    """memory_manager を渡した場合、記憶内容がプロンプトに含まれることを確認する。

    記憶を1件登録したうえで run_chronicle を呼び、LLM に渡されるプロンプトに
    その記憶コンテンツが含まれていることを検証する。
    """
    char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)
    sqlite_store.create_inscribed_memory(
        memory_id=str(uuid.uuid4()), character_id=char_id,
        content="印象的な記憶コンテンツXYZ",
        contextual_importance=0.9, semantic_importance=0.5,
        identity_importance=0.5, user_importance=0.5,
    )

    captured_prompt: list[str] = []

    async def fake_generate(sys_prompt, messages):
        captured_prompt.append(messages[0]["content"])
        return _NO_UPDATE_RESPONSE

    mock_provider = AsyncMock()
    mock_provider.generate = fake_generate

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_chronicle(
            character_id=char_id,
            sqlite=sqlite_store,
            memory_manager=memory_manager,
            working_memory_manager=working_memory_manager,
        )

    assert result["status"] == "success"
    assert len(captured_prompt) == 1
    assert "印象的な記憶コンテンツXYZ" in captured_prompt[0]


@pytest.mark.asyncio
async def test_run_chronicle_uses_placeholder_when_no_memory_manager(sqlite_store, working_memory_manager):
    """memory_manager=None の場合、「記憶データなし」プレースホルダーがプロンプトに入ることを確認する。"""
    char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)

    captured_prompt: list[str] = []

    async def fake_generate(sys_prompt, messages):
        captured_prompt.append(messages[0]["content"])
        return _NO_UPDATE_RESPONSE

    mock_provider = AsyncMock()
    mock_provider.generate = fake_generate

    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        await run_chronicle(
            character_id=char_id,
            sqlite=sqlite_store,
            memory_manager=None,
            working_memory_manager=working_memory_manager,
        )

    assert len(captured_prompt) == 1
    assert "記憶データなし" in captured_prompt[0]


# ---------------------------------------------------------------------------
# run_pending_chronicles: memory_manager 伝達テスト
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pending_chronicles_passes_memory_manager_to_run_chronicle(
    sqlite_store, memory_manager
):
    """run_pending_chronicles に渡した memory_manager が各 run_chronicle 呼び出しに伝達されることを確認する。

    run_chronicle をモックし、呼び出し引数に memory_manager が含まれていることを検証する。
    """
    _setup_char_with_messages(sqlite_store, "Alice", n_messages=1)

    with patch(
        "backend.batch.chronicle_job.run_chronicle", new_callable=AsyncMock
    ) as mock_rc:
        mock_rc.return_value = {"status": "success", "counts": {}}
        await run_pending_chronicles(sqlite=sqlite_store, memory_manager=memory_manager)

    assert mock_rc.called
    call_kwargs = mock_rc.call_args.kwargs
    assert call_kwargs.get("memory_manager") is memory_manager
