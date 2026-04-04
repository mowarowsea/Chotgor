"""backend.batch.forget_job モジュールのユニットテスト。

忘却バッチ処理（run_forget_process）の動作を検証する。

対象関数:
    run_forget_process()        — キャラクターの忘れかけた記憶を特定し、キャラクター自身に
                                  手放すか判断させてソフトデリートする
    calculate_decayed_score()   — MemoryManager の減衰スコア計算ロジック
    get_forgotten_candidates()  — 候補記憶の抽出ロジック

テスト方針:
    - LLMコールは backend.batch.forget_job.ask_character を AsyncMock でパッチする
    - キャラクターとプリセットは SQLiteStore の実DBに作成して run_forget_process() に渡す
    - [DELETE: ID] 形式のパース・ソフトデリート・last_accessed_at 更新を検証する
    - LLM が何も消さないと判断した場合、削除が行われないことを確認する
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from backend.repositories.sqlite.store import Memory
from backend.services.memory.manager import MemoryManager
from backend.batch.forget_job import run_forget_process


@pytest.fixture
def memory_manager(sqlite_store):
    """ChromaDB を MagicMock に差し替えた MemoryManager を返すフィクスチャ。"""
    chroma = MagicMock()
    return MemoryManager(sqlite_store, chroma)


def test_calculate_decayed_score(memory_manager):
    """減衰スコアが経過日数に応じて正しく計算されること。

    14日経過後の各軸スコア（importance=1.0）:
        contextual: exp(-ln2/4 * 14) * 1.0 ≈ 0.088
        user:       exp(-ln2/10 * 14) * 0.8 ≈ 0.303
        semantic:   exp(-ln2/20 * 14) * 0.6 ≈ 0.370
        identity:   exp(-ln2/90 * 14) * 0.3 ≈ 0.269
        合計 ≈ 1.030
    """
    manager = memory_manager

    now = datetime.now()
    old_time = now - timedelta(days=14)

    m = Memory(
        id="test-mem",
        character_id="char1",
        content="test",
        contextual_importance=1.0,
        user_importance=1.0,
        semantic_importance=1.0,
        identity_importance=1.0,
        created_at=old_time,
        last_accessed_at=old_time,
    )

    score = manager.calculate_decayed_score(m, now)
    assert score > 1.0
    assert score < 1.1

    # 直後の想起はほぼ最大スコア（weights合計: 1.0+0.8+0.6+0.3 = 2.7）
    m2 = Memory(
        id="test-mem2",
        character_id="char1",
        content="test",
        contextual_importance=1.0,
        user_importance=1.0,
        semantic_importance=1.0,
        identity_importance=1.0,
        created_at=now,
        last_accessed_at=now,
    )
    score2 = manager.calculate_decayed_score(m2, now)
    assert score2 > 2.6


def test_get_forgotten_candidates(memory_manager):
    """閾値以下の decayed_score を持つ古い記憶のみが候補として抽出されること。"""
    manager = memory_manager
    char_id = "test-char"
    manager.sqlite.create_character(char_id, "Test")

    # 記憶1: 古くて重要度低
    manager.sqlite.create_memory(
        memory_id="mem1",
        character_id=char_id,
        content="old stuff",
        contextual_importance=0.1,
        user_importance=0.1,
        semantic_importance=0.1,
        identity_importance=0.1,
    )

    # 記憶2: 新しくて重要度高
    manager.sqlite.create_memory(
        memory_id="mem2",
        character_id=char_id,
        content="new stuff",
        contextual_importance=1.0,
        user_importance=1.0,
        semantic_importance=1.0,
        identity_importance=1.0,
    )

    # 記憶1を60日前に更新
    now = datetime.now()
    old_time = now - timedelta(days=60)
    with manager.sqlite.get_session() as session:
        m1 = session.query(Memory).filter_by(id="mem1").first()
        m1.created_at = old_time
        m1.last_accessed_at = old_time
        session.commit()

    candidates = manager.get_forgotten_candidates(char_id, threshold=1.0)
    ids = [c.id for c in candidates]

    assert "mem1" in ids
    assert "mem2" not in ids


# ─── run_forget_process テスト用ヘルパー ─────────────────────────────────────


def _make_manager_with_candidates(sqlite_store, char_id, memory_ids):
    """テスト用キャラクター・記憶・MemoryManager を作成して返すヘルパー。

    get_forgotten_candidates() が memory_ids の記憶を返すよう、
    作成日時・最終アクセス日を60日前に設定する。

    Args:
        sqlite_store: テスト用 SQLiteStore。
        char_id: 作成するキャラクターID。
        memory_ids: 作成する記憶IDのリスト。

    Returns:
        作成した MemoryManager インスタンス。
    """
    chroma = MagicMock()
    manager = MemoryManager(sqlite_store, chroma)
    sqlite_store.create_character(char_id, "TestChar")

    now = datetime.now()
    old_time = now - timedelta(days=60)

    for mid in memory_ids:
        sqlite_store.create_memory(
            memory_id=mid,
            character_id=char_id,
            content=f"content of {mid}",
            contextual_importance=0.1,
            user_importance=0.1,
            semantic_importance=0.1,
            identity_importance=0.1,
        )
        with sqlite_store.get_session() as session:
            m = session.query(Memory).filter_by(id=mid).first()
            m.created_at = old_time
            m.last_accessed_at = old_time
            session.commit()

    return manager


def _make_preset(sqlite_store, preset_id="test-ghost-preset"):
    """テスト用 ghost_model プリセットをSQLiteに作成するヘルパー。

    Args:
        sqlite_store: テスト用 SQLiteStore。
        preset_id: 作成するプリセットID。

    Returns:
        作成したプリセットID文字列。
    """
    sqlite_store.create_model_preset(
        preset_id=preset_id,
        name="Test-Ghost",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
    )
    return preset_id


# ─── run_forget_process — LLM 応答パターン ───────────────────────────────────


@pytest.mark.asyncio
async def test_forget_default_keeps_all_when_no_delete(sqlite_store):
    """LLMが [DELETE: ...] を返さない場合、全候補が削除されないこと。"""
    char_id = "char-keep-all"
    preset_id = _make_preset(sqlite_store)
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2", "m3"])

    with patch("backend.batch.forget_job.ask_character", new=AsyncMock(return_value="何も手放しません。")):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            memory_manager=manager,
            sqlite=sqlite_store,
            settings={},
            threshold=1.0,
            ghost_model=preset_id,
        )

    assert result["status"] == "success"
    assert result["deleted_count"] == 0
    assert result["kept_count"] == 3

    active = sqlite_store.get_all_active_memories(char_id)
    ids = [m.id for m in active]
    assert "m1" in ids
    assert "m2" in ids
    assert "m3" in ids


@pytest.mark.asyncio
async def test_forget_deletes_only_explicitly_listed(sqlite_store):
    """[DELETE: ID] で明示されたIDのみが削除され、他は残ること。"""
    char_id = "char-partial-delete"
    preset_id = _make_preset(sqlite_store)
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2", "m3"])

    with patch("backend.batch.forget_job.ask_character", new=AsyncMock(return_value="[DELETE: m2]")):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            memory_manager=manager,
            sqlite=sqlite_store,
            settings={},
            threshold=1.0,
            ghost_model=preset_id,
        )

    assert result["status"] == "success"
    assert result["deleted_count"] == 1
    assert result["kept_count"] == 2

    active = sqlite_store.get_all_active_memories(char_id)
    ids = [m.id for m in active]
    assert "m1" in ids
    assert "m2" not in ids
    assert "m3" in ids


@pytest.mark.asyncio
async def test_forget_delete_none_keeps_all(sqlite_store):
    """[DELETE: NONE] を返した場合、全候補が削除されないこと。"""
    char_id = "char-delete-none"
    preset_id = _make_preset(sqlite_store)
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2"])

    with patch("backend.batch.forget_job.ask_character", new=AsyncMock(return_value="[DELETE: NONE]")):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            memory_manager=manager,
            sqlite=sqlite_store,
            settings={},
            threshold=1.0,
            ghost_model=preset_id,
        )

    assert result["deleted_count"] == 0
    assert result["kept_count"] == 2


@pytest.mark.asyncio
async def test_forget_parse_failure_keeps_all(sqlite_store):
    """LLMが解釈不能な応答を返した場合（[DELETE:] なし）、何も削除されないこと。"""
    char_id = "char-garbled"
    preset_id = _make_preset(sqlite_store)
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2"])

    garbled = "うーん、どれも大切かな... KEEP m1 DELETE maybe m2??"
    with patch("backend.batch.forget_job.ask_character", new=AsyncMock(return_value=garbled)):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            memory_manager=manager,
            sqlite=sqlite_store,
            settings={},
            threshold=1.0,
            ghost_model=preset_id,
        )

    assert result["deleted_count"] == 0
    assert result["kept_count"] == 2


@pytest.mark.asyncio
async def test_forget_ask_character_none_returns_error(sqlite_store):
    """ask_character() が None を返した場合（LLM接続失敗等）、error ステータスを返すこと。"""
    char_id = "char-none-response"
    preset_id = _make_preset(sqlite_store)
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1"])

    with patch("backend.batch.forget_job.ask_character", new=AsyncMock(return_value=None)):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            memory_manager=manager,
            sqlite=sqlite_store,
            settings={},
            threshold=1.0,
            ghost_model=preset_id,
        )

    assert result["status"] == "error"


# ─── run_forget_process — last_accessed_at 更新 ──────────────────────────────


@pytest.mark.asyncio
async def test_forget_kept_memories_have_updated_last_accessed_at(sqlite_store):
    """「残す」と判断された記憶の last_accessed_at が更新されること。

    Issue #56: キャラクターが能動的に「残す」と判断した記憶のみ decay タイマーをリセットする。
    これにより、その記憶は次の忘却サイクルで再び候補になりにくくなる。
    """
    char_id = "char-keep-updated"
    preset_id = _make_preset(sqlite_store)
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2"])

    old_last_accessed = sqlite_store.get_memory("m1").last_accessed_at

    with patch("backend.batch.forget_job.ask_character", new=AsyncMock(return_value="[DELETE: NONE]")):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            memory_manager=manager,
            sqlite=sqlite_store,
            settings={},
            threshold=1.0,
            ghost_model=preset_id,
        )

    assert result["kept_count"] == 2
    m1_after = sqlite_store.get_memory("m1")
    m2_after = sqlite_store.get_memory("m2")
    # 「残す」判断後は decay タイマーがリセットされている
    assert m1_after.last_accessed_at > old_last_accessed
    assert m2_after.last_accessed_at > old_last_accessed


@pytest.mark.asyncio
async def test_forget_deleted_memories_do_not_update_last_accessed_at(sqlite_store):
    """削除された記憶の last_accessed_at は更新されないこと。

    削除対象記憶は recall() を呼ばれないため、last_accessed_at が古いままであることを検証する。
    """
    char_id = "char-delete-check"
    preset_id = _make_preset(sqlite_store)
    manager = _make_manager_with_candidates(sqlite_store, char_id, ["m1", "m2"])

    old_last_accessed = sqlite_store.get_memory("m1").last_accessed_at

    with patch("backend.batch.forget_job.ask_character", new=AsyncMock(return_value="[DELETE: m1]")):
        result = await run_forget_process(
            character_id=char_id,
            character_name="TestChar",
            memory_manager=manager,
            sqlite=sqlite_store,
            settings={},
            threshold=1.0,
            ghost_model=preset_id,
        )

    assert result["deleted_count"] == 1
    m1 = sqlite_store.get_memory("m1")
    # 削除されているが last_accessed_at は更新されていないこと
    assert m1.deleted_at is not None
    assert m1.last_accessed_at == old_last_accessed
