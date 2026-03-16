"""MemoryManagerのユニットテスト。

ChromaStoreはモック化し、SQLiteStoreは実インメモリDBを使用する。
write_memoryのupsertロジック（Issue #50）を中心にテストする。
"""
from unittest.mock import MagicMock, patch
import pytest
from backend.core.memory.manager import MemoryManager

@pytest.fixture
def mock_chroma():
    """ChromaStoreのモック。デフォルトでは類似記憶なし（新規作成ケース）。"""
    store = MagicMock()
    # recall_memoryはhybridスコア計算用（1件返す）
    store.recall_memory.return_value = [
        {"id": "mem-1", "content": "test content", "distance": 0.1, "metadata": {"category": "general"}}
    ]
    # find_similar_in_categoryはデフォルトで「類似なし」
    store.find_similar_in_category.return_value = None
    return store

@pytest.fixture
def manager(sqlite_store, mock_chroma):
    """MemoryManagerのフィクスチャ。"""
    import backend.core.memory.manager
    return MemoryManager(sqlite=sqlite_store, chroma=mock_chroma)


def test_write_memory_new(manager, sqlite_store, mock_chroma):
    """類似記憶がない場合、新規レコードを作成することを確認する。

    ChromaDBが類似なし（None）を返したとき、SQLiteに新規レコードが1件作成され、
    chroma.add_memoryが1回呼ばれること。
    """
    char_id = "char-001"
    mock_chroma.find_similar_in_category.return_value = None

    manager.write_memory(char_id, "Hello world", category="user")

    mems = sqlite_store.list_memories(char_id)
    assert len(mems) == 1
    assert mems[0].content == "Hello world"
    assert mems[0].memory_category == "user"
    mock_chroma.add_memory.assert_called_once()


def test_write_memory_update_when_similar_found(manager, sqlite_store, mock_chroma):
    """類似記憶がある場合、既存レコードを上書き更新することを確認する。

    ChromaDBが既存のmemory_idを返したとき、SQLiteに新規レコードは作成されず、
    既存レコードのcontentと各importanceが更新され、
    access_countとcreated_atは変化しないこと。

    コサイン距離の目安:
      ~0.05 : ほぼ同じ文（「コーヒーが好き」vs「コーヒーが大好き」）
      ~0.15 : 同系統の内容（「コーヒーが好き」vs「カフェラテが好き」）← 更新対象
      ~0.2  : ボーダーライン（「昨日映画を見た」vs「先週映画を観た」）← 更新しない
      ~0.6+ : 別トピック（「コーヒーが好き」vs「犬を飼っている」）← 新規作成
    """
    char_id = "char-001"

    # 既存記憶をSQLiteに直接作成しておく
    sqlite_store.create_memory(
        "existing-mem",
        char_id,
        "コーヒーが好き",
        memory_category="user",
        contextual_importance=0.3,
        semantic_importance=0.3,
        identity_importance=0.3,
        user_importance=0.3,
    )
    existing = sqlite_store.get_memory("existing-mem")
    original_created_at = existing.created_at
    original_access_count = existing.access_count

    # ChromaDBが「コーヒーが好き」→「カフェラテが好き」で類似ありと判定した想定
    mock_chroma.find_similar_in_category.return_value = "existing-mem"

    returned_id = manager.write_memory(
        char_id,
        "カフェラテが好き",
        category="user",
        contextual_importance=0.8,
        semantic_importance=0.7,
        identity_importance=0.6,
        user_importance=0.9,
    )

    # 返却IDは既存IDであること
    assert returned_id == "existing-mem"

    # SQLiteに新規レコードは増えていないこと
    mems = sqlite_store.list_memories(char_id)
    assert len(mems) == 1

    # contentと各importanceが更新されていること
    updated = sqlite_store.get_memory("existing-mem")
    assert updated.content == "カフェラテが好き"
    assert updated.contextual_importance == 0.8
    assert updated.semantic_importance == 0.7
    assert updated.identity_importance == 0.6
    assert updated.user_importance == 0.9

    # access_countとcreated_atは引き継がれていること
    assert updated.access_count == original_access_count
    assert updated.created_at == original_created_at

    # last_accessed_atが設定されていること
    assert updated.last_accessed_at is not None

    # ChromaDBのupsertが既存IDで呼ばれていること
    mock_chroma.add_memory.assert_called_once()
    call_kwargs = mock_chroma.add_memory.call_args
    assert call_kwargs.kwargs["memory_id"] == "existing-mem"


def test_recall_memory_hybrid(manager, sqlite_store, mock_chroma):
    """recall_memoryがhybridスコアと減衰スコアを返すことを確認する。"""
    char_id = "char-001"
    sqlite_store.create_memory("mem-1", char_id, "real content")

    results = manager.recall_memory(char_id, "test")
    assert len(results) == 1
    assert "hybrid_score" in results[0]
    assert "decayed_score" in results[0]


def test_recall_with_identity_calls_recall_memory_twice(manager, mock_chroma):
    """recall_with_identity が identity / 非identity それぞれで chroma.recall_memory を呼ぶことを確認する。

    chroma.recall_memory はカテゴリ別にフィルタされた2回の呼び出しが行われ、
    返り値が (identity_list, other_list) のタプルであることを検証する。
    """
    char_id = "char-001"
    identity_mem = {"id": "id-mem", "content": "私はプログラマーだ", "distance": 0.02, "metadata": {"category": "identity"}}
    other_mem = {"id": "other-mem", "content": "昨日カフェに行った", "distance": 0.1, "metadata": {"category": "contextual"}}

    # 1回目（identity）と2回目（others）で異なる結果を返す
    mock_chroma.recall_memory.side_effect = [[identity_mem], [other_mem]]

    identity_results, other_results = manager.recall_with_identity(char_id, "テスト", identity_top_k=5, other_top_k=5)

    # chroma.recall_memory が2回呼ばれていること
    assert mock_chroma.recall_memory.call_count == 2

    # 1回目は identity フィルタ付き
    first_call = mock_chroma.recall_memory.call_args_list[0]
    assert first_call.kwargs.get("where") == {"category": "identity"}

    # 2回目は identity 除外フィルタ付き
    second_call = mock_chroma.recall_memory.call_args_list[1]
    assert second_call.kwargs.get("where") == {"category": {"$ne": "identity"}}


def test_recall_with_identity_returns_tuple_of_two_lists(manager, sqlite_store, mock_chroma):
    """recall_with_identity の戻り値が (list, list) のタプルであることを確認する。

    identity が空、others に1件ある場合でも正しくアンパックできることを検証する。
    """
    char_id = "char-001"
    sqlite_store.create_memory("m1", char_id, "test content")
    other_mem = {"id": "m1", "content": "test content", "distance": 0.1, "metadata": {"category": "contextual"}}
    mock_chroma.recall_memory.side_effect = [[], [other_mem]]

    identity_results, other_results = manager.recall_with_identity(char_id, "query")

    assert isinstance(identity_results, list)
    assert isinstance(other_results, list)
    assert identity_results == []
    assert len(other_results) == 1


def test_delete_memory(manager, sqlite_store, mock_chroma):
    """delete_memoryがSQLiteをソフト削除し、ChromaDBを物理削除することを確認する。"""
    char_id = "char-001"
    sqlite_store.create_memory("mem-1", char_id, "to be deleted")

    ok = manager.delete_memory("mem-1", char_id)
    assert ok is True
    assert sqlite_store.get_memory("mem-1").deleted_at is not None
    mock_chroma.delete_memory.assert_called_once_with("mem-1", char_id)
