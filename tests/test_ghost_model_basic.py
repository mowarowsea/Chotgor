"""ghost_model の DB 永続化・chronicle / forget ルーティングのテスト。

検証する観点:
    - ghost_model カラムの保存・取得・更新・クリア
    - _parse_chronicle_response の JSON / コードブロック / 不正入力
    - run_chronicle のエラーハンドリング（ghost_model 未設定・プリセット不在）
    - run_chronicle の棚卸し分岐（不要回答 / 蒸留指示）
    - run_forget_process のエラーハンドリング
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import (
    run_chronicle,
    _parse_chronicle_response,
)
from backend.batch.forget_job import run_forget_process

from tests._ghost_model_helpers import (  # noqa: F401
    _NO_UPDATE_RESPONSE,
    memory_manager,
    working_memory_manager,
)

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


