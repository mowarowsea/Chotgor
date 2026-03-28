"""MemoryManagerのユニットテスト。

ChromaStoreはモック化し、SQLiteStoreは実インメモリDBを使用する。
write_memoryのupsertロジック（Issue #50）を中心にテストする。
"""
from unittest.mock import MagicMock, patch
import pytest
from backend.services.memory.manager import MemoryManager

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
    import backend.services.memory.manager
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


def test_write_memory_delete_insert_when_similar_found(manager, sqlite_store, mock_chroma):
    """類似記憶がある場合、旧レコードをsoft-deleteして新規レコードを作成することを確認する。

    delete-insert方式（Issue #38対応）の動作を検証する。

    ChromaDBが既存のmemory_idを返したとき:
    - 旧レコード（"existing-mem"）がSQLiteでsoft-deleteされること
    - ChromaDBの旧レコードが物理削除されること
    - 新しいUUIDでSQLiteに新規レコードが作成されること
    - 新しいIDでChromaDBにembeddingが追加されること
    - アクティブな記憶は1件のみであること（旧レコードは削除済み）

    コサイン距離の目安:
      ~0.05 : ほぼ同じ文（「コーヒーが好き」vs「コーヒーが大好き」）
      ~0.15 : 同系統の内容（「コーヒーが好き」vs「カフェラテが好き」）← delete-insert対象
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

    # 返却IDは新しいUUIDであること（旧IDではない）
    assert returned_id != "existing-mem"

    # アクティブな記憶は新規レコード1件のみであること
    active_mems = sqlite_store.list_memories(char_id)
    assert len(active_mems) == 1
    assert active_mems[0].id == returned_id
    assert active_mems[0].content == "カフェラテが好き"
    assert active_mems[0].contextual_importance == 0.8
    assert active_mems[0].semantic_importance == 0.7
    assert active_mems[0].identity_importance == 0.6
    assert active_mems[0].user_importance == 0.9

    # 旧レコードはsoft-deleteされていること
    old_mem = sqlite_store.get_memory("existing-mem")
    assert old_mem.deleted_at is not None

    # ChromaDBで旧レコードが削除され、新IDで追加されていること
    mock_chroma.delete_memory.assert_called_once_with("existing-mem", char_id)
    mock_chroma.add_memory.assert_called_once()
    call_kwargs = mock_chroma.add_memory.call_args
    assert call_kwargs.kwargs["memory_id"] == returned_id


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


# --- remember / recall メソッドの単体テスト（Issue #56） ---

def test_remember_increments_count_without_updating_date(sqlite_store):
    """remember() は access_count のみ更新し、last_accessed_at は変更しないことを確認する。

    システムによる自動想起（remember）では decay タイマーをリセットしてはならない。
    last_accessed_at が None のまま維持され、access_count だけが増えることを検証する。
    """
    sqlite_store.create_memory("mem-r", "char-x", "content")
    mem_before = sqlite_store.get_memory("mem-r")
    assert (mem_before.access_count or 0) == 0
    assert mem_before.last_accessed_at is None

    sqlite_store.remember("mem-r")

    mem_after = sqlite_store.get_memory("mem-r")
    assert mem_after.access_count == 1
    assert mem_after.last_accessed_at is None  # decay タイマー保持


def test_remember_is_cumulative(sqlite_store):
    """remember() を複数回呼んだとき、access_count が正しく累積されることを確認する。

    3回呼び出すと access_count が 3 になり、last_accessed_at は依然として None のままであること。
    """
    sqlite_store.create_memory("mem-r", "char-x", "content")

    sqlite_store.remember("mem-r")
    sqlite_store.remember("mem-r")
    sqlite_store.remember("mem-r")

    mem = sqlite_store.get_memory("mem-r")
    assert mem.access_count == 3
    assert mem.last_accessed_at is None


def test_recall_updates_both_count_and_date(sqlite_store):
    """recall() は access_count と last_accessed_at 両方を更新することを確認する。

    キャラクターが能動的に「残す」と判断した際（忘却バッチ保持）は
    decay タイマーをリセットする必要がある。両フィールドが更新されることを検証する。
    """
    sqlite_store.create_memory("mem-c", "char-x", "content")
    mem_before = sqlite_store.get_memory("mem-c")
    assert (mem_before.access_count or 0) == 0
    assert mem_before.last_accessed_at is None

    sqlite_store.recall("mem-c")

    mem_after = sqlite_store.get_memory("mem-c")
    assert mem_after.access_count == 1
    assert mem_after.last_accessed_at is not None  # decay タイマーリセット済み


# --- recall_memory() の統合テスト（Issue #56） ---

def test_recall_memory_does_not_update_last_accessed_at(manager, sqlite_store, mock_chroma):
    """recall_memory() 後、想起された記憶の last_accessed_at が更新されないことを確認する。

    Issue #56: システムによる自動想起では decay タイマーをリセットしてはならない。
    last_accessed_at が変わらないことで、decay が継続して忘却候補に残り続けることを保証する。
    """
    char_id = "char-001"
    sqlite_store.create_memory("mem-1", char_id, "real content")
    mem_before = sqlite_store.get_memory("mem-1")
    last_accessed_before = mem_before.last_accessed_at  # None のはず

    manager.recall_memory(char_id, "test")

    mem_after = sqlite_store.get_memory("mem-1")
    assert mem_after.last_accessed_at == last_accessed_before


def test_recall_memory_increments_access_count(manager, sqlite_store, mock_chroma):
    """recall_memory() 後、想起された記憶の access_count がインクリメントされることを確認する。

    Issue #56: last_accessed_at は更新しないが、想起回数（access_count）は記録する。
    何回思い出されたかの統計は保持したまま、decay タイマーのみ保護する。
    """
    char_id = "char-001"
    sqlite_store.create_memory("mem-1", char_id, "real content")
    mem_before = sqlite_store.get_memory("mem-1")
    count_before = mem_before.access_count or 0

    manager.recall_memory(char_id, "test")

    mem_after = sqlite_store.get_memory("mem-1")
    assert mem_after.access_count == count_before + 1
