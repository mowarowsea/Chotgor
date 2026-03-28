"""ghost_model の DB 永続化・chronicle / forget ルーティングのテスト。

digest が削除されたため、夜間バッチのテストは chronicle に差し替え済み。
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import run_chronicle, _parse_chronicle_response
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
