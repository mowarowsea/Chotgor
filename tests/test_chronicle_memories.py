"""chronicle と保存記憶の連携 — _format_memories / get_top_memorable / プロンプト注入のテスト。

検証する観点:
    - _format_memories の整形（空リスト・カテゴリ/スコア・複数件）
    - InscribedMemoryManager.get_top_memorable の順序・件数制限・削除除外・減衰スコア
    - run_chronicle が memory_manager から記憶をプロンプトへ注入すること
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import (
    run_chronicle,
    run_pending_chronicles,
    _format_memories,
)

from tests._ghost_model_helpers import (  # noqa: F401
    _NO_UPDATE_RESPONSE,
    _setup_char_with_messages,
    memory_manager,
    working_memory_manager,
)

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
