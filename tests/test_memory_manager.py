"""InscribedMemoryManager のユニットテスト。

# テスト方針

LanceStore はモック化し、SQLiteStore は実インメモリ DB を使用する。
write_inscribed_memory の upsert ロジック（in-place 更新）と、
recall_memory の soft-delete 除外、recall_with_identity の where フィルタを中心に検証する。
LanceStore は書き込みが原子的で、失敗時は例外として呼び出し側に伝播する。
"""
from unittest.mock import MagicMock

import pytest

from backend.services.memory.manager import InscribedMemoryManager


@pytest.fixture
def mock_vector_store():
    """LanceStore のモック。デフォルトでは類似記憶なし（新規作成ケース）。"""
    store = MagicMock()
    # recall_memory は hybrid スコア計算用（1件返す）
    store.recall_inscribed_memory.return_value = [
        {"id": "mem-1", "content": "test content", "distance": 0.1, "metadata": {"category": "general"}}
    ]
    # find_similar_in_category はデフォルトで「類似なし」
    store.find_similar_in_category.return_value = None
    return store


@pytest.fixture
def manager(sqlite_store, mock_vector_store):
    """InscribedMemoryManager のフィクスチャ。"""
    return InscribedMemoryManager(sqlite=sqlite_store, vector_store=mock_vector_store)


# ---------------------------------------------------------------------------
# write_inscribed_memory — upsert ロジック（欠陥 C 対策）
# ---------------------------------------------------------------------------


def test_write_inscribed_memory_new(manager, sqlite_store, mock_vector_store):
    """類似記憶がない場合、新規レコードを作成することを確認する。

    LanceStore が類似なし（None）を返したとき、SQLite に新規レコードが 1 件作成され、
    vector_store.add_inscribed_memory が 1 回呼ばれること。
    """
    char_id = "char-001"
    mock_vector_store.find_similar_in_category.return_value = None

    manager.write_inscribed_memory(char_id, "Hello world", category="user")

    mems = sqlite_store.list_inscribed_memories(char_id)
    assert len(mems) == 1
    assert mems[0].content == "Hello world"
    assert mems[0].memory_category == "user"
    mock_vector_store.add_inscribed_memory.assert_called_once()


def test_write_inscribed_memory_inplace_update_when_similar_found(manager, sqlite_store, mock_vector_store):
    """類似記憶がある場合、既存 ID で in-place 更新されることを確認する（欠陥 C 対策）。

    既存 ID をそのまま再利用し、SQLite 側は in-place 更新、ベクトル DB 側は
    同一 ID での upsert（add_memory が内部で merge_insert を呼ぶ）で delete を排除する。

    ベクトル DB が既存の memory_id を返したとき:
    - 返却 ID は既存 ID（"existing-mem"）と同一であること
    - SQLite 側は soft-delete されず、in-place で content/category/importances が上書きされていること
    - access_count / created_at は維持されること
    - vector_store.delete_inscribed_memory は **呼ばれないこと**
    - vector_store.add_inscribed_memory が同一 ID で呼ばれること（内部 upsert で置き換わる）
    """
    char_id = "char-001"

    # 既存記憶を SQLite に直接作成しておく
    sqlite_store.create_inscribed_memory(
        "existing-mem",
        char_id,
        "コーヒーが好き",
        memory_category="user",
        contextual_importance=0.3,
        semantic_importance=0.3,
        identity_importance=0.3,
        user_importance=0.3,
    )
    # access_count を 5 に積み上げて、in-place 更新で維持されることを後で確認する
    for _ in range(5):
        sqlite_store.remember_inscribed_memory("existing-mem")
    pre_mem = sqlite_store.get_inscribed_memory("existing-mem")
    pre_created_at = pre_mem.created_at
    assert pre_mem.access_count == 5

    # ベクトル DB が「コーヒーが好き」→「カフェラテが好き」で類似ありと判定した想定
    mock_vector_store.find_similar_in_category.return_value = "existing-mem"

    returned_id = manager.write_inscribed_memory(
        char_id,
        "カフェラテが好き",
        category="user",
        contextual_importance=0.8,
        semantic_importance=0.7,
        identity_importance=0.6,
        user_importance=0.9,
    )

    # 返却 ID は既存 ID と同一であること（新規 UUID ではない）
    assert returned_id == "existing-mem"

    # アクティブな記憶は in-place 更新された 1 件のみ
    active_mems = sqlite_store.list_inscribed_memories(char_id)
    assert len(active_mems) == 1
    assert active_mems[0].id == "existing-mem"
    assert active_mems[0].content == "カフェラテが好き"
    assert active_mems[0].contextual_importance == 0.8
    assert active_mems[0].semantic_importance == 0.7
    assert active_mems[0].identity_importance == 0.6
    assert active_mems[0].user_importance == 0.9

    # in-place 更新で deleted_at は付与されないこと
    assert active_mems[0].deleted_at is None
    # access_count / created_at は維持されていること
    assert active_mems[0].access_count == 5
    assert active_mems[0].created_at == pre_created_at

    # vector_store.delete_inscribed_memory は呼ばれないこと（欠陥 C の主目的）
    mock_vector_store.delete_inscribed_memory.assert_not_called()
    # add_memory は同一 ID で呼ばれていること（内部 upsert により置き換わる）
    mock_vector_store.add_inscribed_memory.assert_called_once()
    call_kwargs = mock_vector_store.add_inscribed_memory.call_args
    assert call_kwargs.kwargs["memory_id"] == "existing-mem"
    assert call_kwargs.kwargs["content"] == "カフェラテが好き"


def test_write_inscribed_memory_falls_back_to_new_when_existing_is_soft_deleted(
    manager, sqlite_store, mock_vector_store
):
    """find_similar が既存 ID を返したが SQLite 側が soft-delete 済みのとき、新規 UUID で作り直すこと。

    ベクトル DB と SQLite の整合が崩れている場合のフォールバック動作の確認。
    update_memory_for_overwrite は False を返し、write_inscribed_memory は新規 UUID で
    create_memory に進む。delete は呼ばれない。
    """
    char_id = "char-001"
    sqlite_store.create_inscribed_memory(
        "ghost-existing",
        char_id,
        "古い記憶",
        memory_category="user",
    )
    sqlite_store.soft_delete_inscribed_memory("ghost-existing")
    mock_vector_store.find_similar_in_category.return_value = "ghost-existing"

    returned_id = manager.write_inscribed_memory(
        char_id,
        "新しい記憶",
        category="user",
    )

    # 新規 UUID が返却されていること
    assert returned_id != "ghost-existing"
    # アクティブな記憶は新規 1 件
    active_mems = sqlite_store.list_inscribed_memories(char_id)
    assert len(active_mems) == 1
    assert active_mems[0].id == returned_id
    # add_memory は新規 ID で呼ばれ、delete_memory は呼ばれていないこと
    mock_vector_store.add_inscribed_memory.assert_called_once()
    assert mock_vector_store.add_inscribed_memory.call_args.kwargs["memory_id"] == returned_id
    mock_vector_store.delete_inscribed_memory.assert_not_called()


def test_write_inscribed_memory_force_insert_skips_similarity_and_creates_new(
    manager, sqlite_store, mock_vector_store
):
    """force_insert=True を渡すと、類似既存があっても上書きせず新規 UUID を発行することを確認する。

    forget 蒸留バッチでは、キャラが蒸留として登録した記憶が「削除対象として提示した候補記憶」に
    上書きされてしまうと、続く全件削除フェーズで蒸留物まで一緒に消えてしまう。
    このバグを回避するために write_inscribed_memory には force_insert オプションがあり、
    True を渡したときは類似検索（find_similar_in_category）すら呼ばずに必ず新規作成パスへ
    抜けて新規 UUID で insert されることを検証する。

    検証項目:
    - find_similar_in_category が **一度も呼ばれない** こと（早期スキップ）
    - 既存記憶（existing-mem）は完全に無関係に保たれること（content/importances が一切上書きされない）
    - 戻り値が新規 UUID（既存 ID とは別）であること
    - SQLite に新規記憶レコードが追加されていること
    - vector_store.add_inscribed_memory が新規 UUID で 1 回呼ばれること
    - vector_store.delete_inscribed_memory は呼ばれないこと
    """
    char_id = "char-001"

    # 類似マッチがあっても上書きされてはならない、という挙動を担保するため、
    # 既存記憶を SQLite に置き、ベクトル DB は「類似あり」と返すよう設定する。
    sqlite_store.create_inscribed_memory(
        "existing-mem",
        char_id,
        "コーヒーが好き",
        memory_category="user",
        contextual_importance=0.3,
        semantic_importance=0.3,
        identity_importance=0.3,
        user_importance=0.3,
    )
    mock_vector_store.find_similar_in_category.return_value = "existing-mem"

    returned_id = manager.write_inscribed_memory(
        char_id,
        "カフェラテが好き",
        category="user",
        contextual_importance=0.8,
        semantic_importance=0.7,
        identity_importance=0.6,
        user_importance=0.9,
        force_insert=True,
    )

    # 類似検索パス自体がスキップされていること（force_insert の本質）
    mock_vector_store.find_similar_in_category.assert_not_called()

    # 既存記憶は無傷で残ること（上書きされていない）
    pre = sqlite_store.get_inscribed_memory("existing-mem")
    assert pre is not None
    assert pre.content == "コーヒーが好き"
    assert pre.contextual_importance == 0.3

    # 戻り値は既存 ID とは別の新規 UUID であること
    assert returned_id != "existing-mem"

    # アクティブな記憶が 2 件（既存 + 新規）になり、新規側に蒸留結果が入っていること
    active_mems = {m.id: m for m in sqlite_store.list_inscribed_memories(char_id)}
    assert "existing-mem" in active_mems
    assert returned_id in active_mems
    new_mem = active_mems[returned_id]
    assert new_mem.content == "カフェラテが好き"
    assert new_mem.contextual_importance == 0.8

    # ベクトル DB 側は新規 ID で add_memory が呼ばれ、delete_memory は呼ばれないこと
    mock_vector_store.add_inscribed_memory.assert_called_once()
    assert mock_vector_store.add_inscribed_memory.call_args.kwargs["memory_id"] == returned_id
    mock_vector_store.delete_inscribed_memory.assert_not_called()


def test_write_inscribed_memory_force_insert_false_keeps_default_upsert(
    manager, sqlite_store, mock_vector_store
):
    """force_insert=False（デフォルト）を明示的に渡しても、従来の類似検索＋in-place 更新が動くこと。

    後方互換の最低ラインを担保するための回帰テスト。
    既存呼び出し（chat 系・通常の inscribe_memory ツール）は force_insert を渡さない、
    あるいは False を渡すが、いずれの場合も類似検索が呼ばれ、類似マッチ時は既存 ID が再利用される。
    """
    char_id = "char-001"
    sqlite_store.create_inscribed_memory(
        "existing-mem",
        char_id,
        "コーヒーが好き",
        memory_category="user",
    )
    mock_vector_store.find_similar_in_category.return_value = "existing-mem"

    returned_id = manager.write_inscribed_memory(
        char_id,
        "カフェラテが好き",
        category="user",
        force_insert=False,
    )

    mock_vector_store.find_similar_in_category.assert_called_once()
    assert returned_id == "existing-mem"


# ---------------------------------------------------------------------------
# recall_memory — hybrid スコア / soft-delete 除外
# ---------------------------------------------------------------------------


def test_recall_inscribed_memory_hybrid(manager, sqlite_store, mock_vector_store):
    """recall_memory が hybrid スコアと減衰スコアを返すことを確認する。"""
    char_id = "char-001"
    sqlite_store.create_inscribed_memory("mem-1", char_id, "real content")

    results = manager.recall_inscribed_memory(char_id, "test")
    assert len(results) == 1
    assert "hybrid_score" in results[0]
    assert "decayed_score" in results[0]


# ---------------------------------------------------------------------------
# recall_with_identity — identity / 非 identity の分離
# ---------------------------------------------------------------------------


def test_recall_with_identity_calls_recall_memory_twice(manager, mock_vector_store):
    """recall_with_identity が identity / 非 identity それぞれで vector_store.recall_inscribed_memory を呼ぶことを確認する。

    vector_store.recall_inscribed_memory はカテゴリ別にフィルタされた 2 回の呼び出しが行われ、
    返り値が ``(identity_list, other_list)`` のタプルであることを検証する。
    """
    char_id = "char-001"
    identity_mem = {"id": "id-mem", "content": "私はプログラマーだ", "distance": 0.02, "metadata": {"category": "identity"}}
    other_mem = {"id": "other-mem", "content": "昨日カフェに行った", "distance": 0.1, "metadata": {"category": "contextual"}}

    # 1 回目（identity）と 2 回目（others）で異なる結果を返す
    mock_vector_store.recall_inscribed_memory.side_effect = [[identity_mem], [other_mem]]

    identity_results, other_results = manager.recall_with_identity(char_id, "テスト", identity_top_k=5, other_top_k=5)

    # vector_store.recall_inscribed_memory が 2 回呼ばれていること
    assert mock_vector_store.recall_inscribed_memory.call_count == 2

    # 1 回目は identity フィルタ付き
    first_call = mock_vector_store.recall_inscribed_memory.call_args_list[0]
    assert first_call.kwargs.get("where") == {"category": "identity"}

    # 2 回目は identity 除外フィルタ付き
    second_call = mock_vector_store.recall_inscribed_memory.call_args_list[1]
    assert second_call.kwargs.get("where") == {"category": {"$ne": "identity"}}


def test_recall_with_identity_returns_tuple_of_two_lists(manager, sqlite_store, mock_vector_store):
    """recall_with_identity の戻り値が ``(list, list)`` のタプルであることを確認する。

    identity が空、others に 1 件ある場合でも正しくアンパックできることを検証する。
    """
    char_id = "char-001"
    sqlite_store.create_inscribed_memory("m1", char_id, "test content")
    other_mem = {"id": "m1", "content": "test content", "distance": 0.1, "metadata": {"category": "contextual"}}
    mock_vector_store.recall_inscribed_memory.side_effect = [[], [other_mem]]

    identity_results, other_results = manager.recall_with_identity(char_id, "query")

    assert isinstance(identity_results, list)
    assert isinstance(other_results, list)
    assert identity_results == []
    assert len(other_results) == 1


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------


def test_delete_memory(manager, sqlite_store, mock_vector_store):
    """delete_memory が SQLite をソフト削除し、ベクトル DB を物理削除することを確認する。"""
    char_id = "char-001"
    sqlite_store.create_inscribed_memory("mem-1", char_id, "to be deleted")

    ok = manager.delete_inscribed_memory("mem-1", char_id)
    assert ok is True
    assert sqlite_store.get_inscribed_memory("mem-1").deleted_at is not None
    mock_vector_store.delete_inscribed_memory.assert_called_once_with(memory_id="mem-1", character_id=char_id)


# ---------------------------------------------------------------------------
# remember / recall（SQLite 側のメソッド）
# ---------------------------------------------------------------------------


def test_remember_increments_count_without_updating_date(sqlite_store):
    """remember() は access_count のみ更新し、last_accessed_at は変更しないことを確認する。

    システムによる自動想起（remember）では decay タイマーをリセットしてはならない。
    last_accessed_at が None のまま維持され、access_count だけが増えることを検証する。
    """
    sqlite_store.create_inscribed_memory("mem-r", "char-x", "content")
    mem_before = sqlite_store.get_inscribed_memory("mem-r")
    assert (mem_before.access_count or 0) == 0
    assert mem_before.last_accessed_at is None

    sqlite_store.remember_inscribed_memory("mem-r")

    mem_after = sqlite_store.get_inscribed_memory("mem-r")
    assert mem_after.access_count == 1
    assert mem_after.last_accessed_at is None  # decay タイマー保持


def test_remember_is_cumulative(sqlite_store):
    """remember() を複数回呼んだとき、access_count が正しく累積されることを確認する。

    3 回呼び出すと access_count が 3 になり、last_accessed_at は依然として None のままであること。
    """
    sqlite_store.create_inscribed_memory("mem-r", "char-x", "content")

    sqlite_store.remember_inscribed_memory("mem-r")
    sqlite_store.remember_inscribed_memory("mem-r")
    sqlite_store.remember_inscribed_memory("mem-r")

    mem = sqlite_store.get_inscribed_memory("mem-r")
    assert mem.access_count == 3
    assert mem.last_accessed_at is None


def test_recall_updates_both_count_and_date(sqlite_store):
    """recall() は access_count と last_accessed_at 両方を更新することを確認する。

    キャラクターが能動的に「残す」と判断した際（忘却バッチ保持）は
    decay タイマーをリセットする必要がある。両フィールドが更新されることを検証する。
    """
    sqlite_store.create_inscribed_memory("mem-c", "char-x", "content")
    mem_before = sqlite_store.get_inscribed_memory("mem-c")
    assert (mem_before.access_count or 0) == 0
    assert mem_before.last_accessed_at is None

    sqlite_store.recall_inscribed_memory("mem-c")

    mem_after = sqlite_store.get_inscribed_memory("mem-c")
    assert mem_after.access_count == 1
    assert mem_after.last_accessed_at is not None  # decay タイマーリセット済み


# ---------------------------------------------------------------------------
# recall_memory の Issue #56（last_accessed_at 非更新）
# ---------------------------------------------------------------------------


def test_recall_inscribed_memory_does_not_update_last_accessed_at(manager, sqlite_store, mock_vector_store):
    """recall_memory() 後、想起された記憶の last_accessed_at が更新されないことを確認する。

    Issue #56: システムによる自動想起では decay タイマーをリセットしてはならない。
    last_accessed_at が変わらないことで、decay が継続して忘却候補に残り続けることを保証する。
    """
    char_id = "char-001"
    sqlite_store.create_inscribed_memory("mem-1", char_id, "real content")
    mem_before = sqlite_store.get_inscribed_memory("mem-1")
    last_accessed_before = mem_before.last_accessed_at  # None のはず

    manager.recall_inscribed_memory(char_id, "test")

    mem_after = sqlite_store.get_inscribed_memory("mem-1")
    assert mem_after.last_accessed_at == last_accessed_before


def test_recall_inscribed_memory_increments_access_count(manager, sqlite_store, mock_vector_store):
    """recall_memory() 後、想起された記憶の access_count がインクリメントされることを確認する。

    Issue #56: last_accessed_at は更新しないが、想起回数（access_count）は記録する。
    何回思い出されたかの統計は保持したまま、decay タイマーのみ保護する。
    """
    char_id = "char-001"
    sqlite_store.create_inscribed_memory("mem-1", char_id, "real content")
    mem_before = sqlite_store.get_inscribed_memory("mem-1")
    count_before = mem_before.access_count or 0

    manager.recall_inscribed_memory(char_id, "test")

    mem_after = sqlite_store.get_inscribed_memory("mem-1")
    assert mem_after.access_count == count_before + 1


# ---------------------------------------------------------------------------
# soft-delete 済み記憶の除外
# ---------------------------------------------------------------------------


def test_recall_inscribed_memory_skips_soft_deleted(manager, sqlite_store, mock_vector_store):
    """recall_memory() は soft-delete 済みの記憶を想起結果から除外することを確認する。

    ベクトル DB が soft-delete 済みの memory_id を返した場合、
    deleted_at が設定されているため reranking でスキップされ、
    最終的な想起結果に含まれないことを検証する。
    「忘れた記憶」がシステムプロンプトに混入しないことを保証する。
    """
    char_id = "char-001"
    sqlite_store.create_inscribed_memory("mem-deleted", char_id, "忘れたはずの記憶")
    sqlite_store.soft_delete_inscribed_memory("mem-deleted")

    mock_vector_store.recall_inscribed_memory.return_value = [
        {"id": "mem-deleted", "content": "忘れたはずの記憶", "distance": 0.05, "metadata": {"category": "general"}},
    ]

    results = manager.recall_inscribed_memory(char_id, "テスト")

    assert results == []


def test_recall_inscribed_memory_skips_record_not_in_sqlite(manager, sqlite_store, mock_vector_store):
    """recall_memory() はベクトル DB にあるが SQLite に存在しない記憶をスキップすることを確認する。

    ベクトル DB と SQLite の乖離（例: SQLite 側のみ削除された場合）が発生したとき、
    session.get() が None を返す記憶は reranking でスキップされ、
    最終結果に含まれないことを検証する。
    """
    char_id = "char-001"
    # SQLite には登録しない（ベクトル DB にだけある想定）
    mock_vector_store.recall_inscribed_memory.return_value = [
        {"id": "ghost-id", "content": "SQLiteにない記憶", "distance": 0.05, "metadata": {"category": "general"}},
    ]

    results = manager.recall_inscribed_memory(char_id, "テスト")

    assert results == []


def test_recall_with_identity_excludes_soft_deleted_from_other(manager, sqlite_store, mock_vector_store):
    """recall_with_identity() の other 枠で soft-delete 済み記憶がスキップされることを確認する。

    本事象の再現テスト: ベクトル DB の non-identity 検索結果が全て soft-delete 済みだった場合、
    スキップされて other=0 になることを検証する（identity 枠には影響しない）。
    """
    char_id = "char-001"
    sqlite_store.create_inscribed_memory("mem-identity", char_id, "私はエンジニアだ", memory_category="identity")
    sqlite_store.create_inscribed_memory("mem-ctx-deleted", char_id, "soft-delete済みの記憶", memory_category="contextual")
    sqlite_store.soft_delete_inscribed_memory("mem-ctx-deleted")

    identity_mem = {"id": "mem-identity", "content": "私はエンジニアだ", "distance": 0.02, "metadata": {"category": "identity"}}
    soft_deleted_mem = {"id": "mem-ctx-deleted", "content": "soft-delete済みの記憶", "distance": 0.05, "metadata": {"category": "contextual"}}
    mock_vector_store.recall_inscribed_memory.side_effect = [[identity_mem], [soft_deleted_mem]]

    identity_results, other_results = manager.recall_with_identity(char_id, "テスト")

    # identity は正常に返る
    assert len(identity_results) == 1
    assert identity_results[0]["id"] == "mem-identity"
    # soft-delete 済みは除外され other は空
    assert other_results == []


def test_recall_with_identity_both_sections_appear_when_valid_memories_exist(manager, sqlite_store, mock_vector_store):
    """recall_with_identity() が identity と other 両方に有効な記憶を返すことを確認する。

    soft-delete 済み記憶が混在していても、有効な記憶だけが各セクションに入ることを検証する。
    「はる」ケースの正常系: identity 枠と other 枠が両方揃うことで
    システムプロンプトに ## Identity / ## Other Memories の両セクションが出ることを保証する。
    """
    char_id = "char-001"
    sqlite_store.create_inscribed_memory("mem-identity", char_id, "私はエンジニアだ", memory_category="identity")
    sqlite_store.create_inscribed_memory("mem-other-valid", char_id, "昨日コードを書いた", memory_category="contextual")
    sqlite_store.create_inscribed_memory("mem-other-deleted", char_id, "soft-delete済み", memory_category="contextual")
    sqlite_store.soft_delete_inscribed_memory("mem-other-deleted")

    identity_mem = {"id": "mem-identity", "content": "私はエンジニアだ", "distance": 0.02, "metadata": {"category": "identity"}}
    valid_other = {"id": "mem-other-valid", "content": "昨日コードを書いた", "distance": 0.1, "metadata": {"category": "contextual"}}
    deleted_other = {"id": "mem-other-deleted", "content": "soft-delete済み", "distance": 0.05, "metadata": {"category": "contextual"}}
    # ベクトル DB は soft-delete 済みを先に返す（スコアが高い想定）が、スキップされて valid が残る
    mock_vector_store.recall_inscribed_memory.side_effect = [[identity_mem], [deleted_other, valid_other]]

    identity_results, other_results = manager.recall_with_identity(char_id, "テスト")

    assert len(identity_results) == 1
    assert identity_results[0]["id"] == "mem-identity"
    assert len(other_results) == 1
    assert other_results[0]["id"] == "mem-other-valid"


# ---------------------------------------------------------------------------
# delete_character_with_inscribed_memories — SQLite 先 → ベクトル DB 後の順序
# ---------------------------------------------------------------------------


def test_delete_character_sqlite_first_then_vector_store(manager, mock_vector_store):
    """delete_character_with_inscribed_memories が SQLite 削除後にベクトル DB 削除を行うことを確認する。

    SQLite = source of truth のため、SQLite の削除を先に確定させる。
    SQLite 削除が False（存在しない）のときはベクトル DB 削除が呼ばれないことも検証する。
    """
    call_order: list[str] = []

    original_sqlite = manager.sqlite
    mock_sqlite = MagicMock(wraps=original_sqlite)
    mock_sqlite.delete_character = MagicMock(
        side_effect=lambda cid: (call_order.append("sqlite"), False)[1]
    )
    manager.sqlite = mock_sqlite
    mock_vector_store.delete_all_inscribed_memories = MagicMock(
        side_effect=lambda cid: call_order.append("vector_store")
    )

    result = manager.delete_character_with_inscribed_memories("char-nonexistent")

    assert result is False
    assert "sqlite" in call_order
    assert "vector_store" not in call_order  # SQLite 削除が失敗したのでベクトル DB は呼ばれない

    # 後始末
    manager.sqlite = original_sqlite
