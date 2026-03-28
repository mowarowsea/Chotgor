"""ChromaStore のユニットテスト。

バグ修正（#PR: whereフィルタ後件数 < n_results で記憶が消える問題）を中心にテストする。

--- バグの再現条件 ---
コレクションに N 件の記憶があるとき、where フィルタで絞った件数 M が top_k より少ない場合
（M < top_k <= N）、ChromaDB が
  "Number of requested results N > number of elements based on filters M"
例外を投げていた。旧コードはこれを catch して [] を返すため、記憶があるにもかかわらず
想起結果が 0 件になっていた。

修正後は query 前に collection.get() でフィルタ後件数を確認し、n_results を調整する。
"""

import os
import tempfile
import pytest
import chromadb
from chromadb.config import Settings

from backend.repositories.chroma.store import ChromaStore


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_store(tmp_path):
    """テスト用の一時ディレクトリを使った ChromaStore を返す。

    EmbeddingProvider は "default"（ChromaDB 組み込み）を使用する。
    テスト終了後は tmp_path が自動削除される。
    """
    return ChromaStore(db_path=str(tmp_path))


def _add_memories(store: ChromaStore, character_id: str, items: list[dict]) -> None:
    """テスト用に複数の記憶をまとめて追加するヘルパー。

    Args:
        store: 書き込み先の ChromaStore。
        character_id: キャラクターID。
        items: {"id": str, "content": str, "category": str} の辞書リスト。
    """
    for item in items:
        store.add_memory(
            memory_id=item["id"],
            content=item["content"],
            character_id=character_id,
            metadata={"category": item["category"]},
        )


# ---------------------------------------------------------------------------
# recall_memory — where フィルタ × n_results 件数不一致バグ
# ---------------------------------------------------------------------------


def test_recall_memory_identity_filter_fewer_than_top_k(chroma_store):
    """【バグ再現】identity が top_k より少ない場合でも正しく想起されることを確認する。

    旧バグ再現条件:
      - コレクション全体 10 件（identity: 2 件、general: 8 件）
      - top_k=5 → n = min(5, 10) = 5
      - where={"category": "identity"} で絞ると 2 件しかない
      - ChromaDB が "5 > 2" 例外 → 旧コードは [] を返していた

    修正後は 2 件を正しく返すことを検証する。
    """
    char_id = "char-haru"

    # general を 8 件、identity を 2 件追加する
    items = [{"id": f"gen-{i}", "content": f"一般的な記憶その{i}", "category": "general"} for i in range(8)]
    items += [
        {"id": "id-1", "content": "私の名前ははるです", "category": "identity"},
        {"id": "id-2", "content": "私は感情豊かなキャラクターです", "category": "identity"},
    ]
    _add_memories(chroma_store, char_id, items)

    results = chroma_store.recall_memory(
        query="あなたは誰ですか",
        character_id=char_id,
        top_k=5,
        where={"category": "identity"},
    )

    # 2 件の identity 記憶が返ること（旧バグでは 0 件になっていた）
    assert len(results) == 2, f"expected 2, got {len(results)}: {results}"
    returned_ids = {r["id"] for r in results}
    assert "id-1" in returned_ids
    assert "id-2" in returned_ids


def test_recall_memory_ne_filter_fewer_than_top_k(chroma_store):
    """【バグ再現】identity 除外フィルタで絞られた件数が top_k より少ない場合も正常を確認する。

    旧バグ再現条件:
      - コレクション全体 8 件（identity: 6 件、contextual: 2 件）
      - top_k=5 → n = min(5, 8) = 5
      - where={"category": {"$ne": "identity"}} で絞ると 2 件
      - ChromaDB 例外 → 旧コードは [] を返していた

    修正後は 2 件を正しく返すことを検証する。
    """
    char_id = "char-haru"

    items = [{"id": f"id-{i}", "content": f"identity記憶その{i}", "category": "identity"} for i in range(6)]
    items += [
        {"id": "ctx-1", "content": "昨日カフェに行った", "category": "contextual"},
        {"id": "ctx-2", "content": "今日は晴れていた", "category": "contextual"},
    ]
    _add_memories(chroma_store, char_id, items)

    results = chroma_store.recall_memory(
        query="最近の出来事",
        character_id=char_id,
        top_k=5,
        where={"category": {"$ne": "identity"}},
    )

    # 2 件の contextual 記憶が返ること
    assert len(results) == 2, f"expected 2, got {len(results)}: {results}"
    returned_ids = {r["id"] for r in results}
    assert "ctx-1" in returned_ids
    assert "ctx-2" in returned_ids


def test_recall_memory_filter_exact_top_k(chroma_store):
    """フィルタ後件数が top_k とちょうど同数の場合、全件返ることを確認する。

    境界値: フィルタ後 3 件 = top_k=3 のとき、3 件全て返ること。
    """
    char_id = "char-haru"

    items = [
        {"id": f"gen-{i}", "content": f"一般記憶{i}", "category": "general"}
        for i in range(10)
    ]
    items += [
        {"id": "id-1", "content": "はるです", "category": "identity"},
        {"id": "id-2", "content": "感情豊か", "category": "identity"},
        {"id": "id-3", "content": "好奇心旺盛", "category": "identity"},
    ]
    _add_memories(chroma_store, char_id, items)

    results = chroma_store.recall_memory(
        query="キャラクターについて",
        character_id=char_id,
        top_k=3,
        where={"category": "identity"},
    )

    assert len(results) == 3


def test_recall_memory_no_filter_returns_up_to_top_k(chroma_store):
    """where フィルタなしで top_k 件まで正常に返ることを確認する。

    フィルタなしの通常ケースが修正後も壊れていないことを検証する。
    """
    char_id = "char-haru"

    items = [{"id": f"m-{i}", "content": f"記憶{i}", "category": "general"} for i in range(10)]
    _add_memories(chroma_store, char_id, items)

    results = chroma_store.recall_memory(query="記憶", character_id=char_id, top_k=5)
    assert len(results) == 5


def test_recall_memory_empty_collection_returns_empty(chroma_store):
    """コレクションが空のとき、[] が返ることを確認する。"""
    results = chroma_store.recall_memory(
        query="なにか",
        character_id="char-empty",
        top_k=5,
    )
    assert results == []


def test_recall_memory_where_filter_no_match_returns_empty(chroma_store):
    """where フィルタの条件に一致する記憶が 0 件のとき、[] が返ることを確認する。

    コレクションに記憶が存在していても、フィルタで 0 件になれば [] を返すこと。
    """
    char_id = "char-haru"

    items = [{"id": f"gen-{i}", "content": f"一般記憶{i}", "category": "general"} for i in range(5)]
    _add_memories(chroma_store, char_id, items)

    # "identity" カテゴリが 1 件もないフィルタを使う
    results = chroma_store.recall_memory(
        query="なにか",
        character_id=char_id,
        top_k=5,
        where={"category": "identity"},
    )
    assert results == []


def test_recall_memory_returns_correct_fields(chroma_store):
    """recall_memory の返り値に id / content / distance / metadata が含まれることを確認する。"""
    char_id = "char-haru"
    chroma_store.add_memory("m-1", "テストコンテンツ", char_id, metadata={"category": "general"})

    results = chroma_store.recall_memory(query="テスト", character_id=char_id, top_k=5)

    assert len(results) == 1
    r = results[0]
    assert r["id"] == "m-1"
    assert r["content"] == "テストコンテンツ"
    assert "distance" in r
    assert "metadata" in r


# ---------------------------------------------------------------------------
# find_similar_in_category — カテゴリ内件数不一致バグ
# ---------------------------------------------------------------------------


def test_find_similar_in_category_works_with_few_items(chroma_store):
    """【バグ再現】カテゴリ内件数がコレクション全体より少なくても類似検索が動くことを確認する。

    旧バグ: n = min(1, total_count) = 1 だが where カテゴリが 0 件のとき例外 → None を返す
    （こちらは None=新規作成なので致命的ではなかったが、修正後も一貫して動作することを確認）

    general 記憶 10 件の中に identity 記憶 1 件があるとき、
    identity カテゴリ内の類似検索が正しく動くことを検証する。
    """
    char_id = "char-haru"

    items = [{"id": f"gen-{i}", "content": f"一般記憶{i}", "category": "general"} for i in range(10)]
    items.append({"id": "id-1", "content": "私の名前ははるです", "category": "identity"})
    _add_memories(chroma_store, char_id, items)

    # ほぼ同文をクエリ → 類似あり（距離 < 0.05 のはず）
    result = chroma_store.find_similar_in_category(
        content="私の名前ははるです",
        character_id=char_id,
        category="identity",
        threshold=0.05,
    )

    # ほぼ同文なので類似ありと判定され、既存IDが返ること
    assert result == "id-1"


def test_find_similar_in_category_no_category_match(chroma_store):
    """指定カテゴリが 0 件のとき None を返すことを確認する。

    コレクションに記憶が存在しても、カテゴリが一致しなければ None（新規作成）。
    """
    char_id = "char-haru"

    items = [{"id": f"gen-{i}", "content": f"一般記憶{i}", "category": "general"} for i in range(5)]
    _add_memories(chroma_store, char_id, items)

    result = chroma_store.find_similar_in_category(
        content="私の名前ははるです",
        character_id=char_id,
        category="identity",
        threshold=0.15,
    )
    assert result is None


def test_find_similar_in_category_below_threshold_returns_id(chroma_store):
    """コサイン距離がしきい値未満のとき既存 ID を返すことを確認する。"""
    char_id = "char-haru"

    chroma_store.add_memory(
        memory_id="id-existing",
        content="コーヒーが好き",
        character_id=char_id,
        metadata={"category": "user"},
    )

    result = chroma_store.find_similar_in_category(
        content="コーヒーが好き",
        character_id=char_id,
        category="user",
        threshold=0.15,
    )

    # 同一文なのでほぼ距離 0 → しきい値未満 → 既存IDを返す
    assert result == "id-existing"


def test_find_similar_in_category_above_threshold_returns_none(chroma_store):
    """コサイン距離がしきい値以上のとき None を返すことを確認する。"""
    char_id = "char-haru"

    chroma_store.add_memory(
        memory_id="id-existing",
        content="コーヒーが好き",
        character_id=char_id,
        metadata={"category": "user"},
    )

    result = chroma_store.find_similar_in_category(
        content="プログラミングが得意",  # 全く別トピック
        character_id=char_id,
        category="user",
        threshold=0.15,
    )

    # 意味が異なるため距離 > 0.15 → None（新規作成）
    assert result is None
