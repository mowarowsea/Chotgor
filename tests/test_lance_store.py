"""LanceStore のユニットテスト。

# テスト方針

ChromaStore からの置き換えとして、以下を網羅的に検証する。

- パブリック API の入出力互換性（ChromaStore との戻り値形状一致）
- where フィルタの $eq / $ne / $in 等の SQL 変換が正しいこと
- merge_insert（upsert）が同一ID再投入で更新動作になること
- 過去 ChromaStore で発生した「whereフィルタ後件数 < top_k で結果が消える」バグが
  LanceStore では発生しないこと（プリフィルタ + LIMIT で安全に動く）
- delete / delete_all_memories の物理削除挙動
- キャラクター定義（estranged 検出）の正しい動作

# 埋め込みの扱い

外部 embedding サーバ（infinity / Gemini）に依存しないよう、テストでは
``FakeEmbeddingFunction`` を ``embedding_fn=`` 経由で注入する。
固定次元のハッシュベース疑似ベクトルを返し、距離関係を制御可能にする。
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import pytest

from backend.repositories.lance.store import (
    LanceStore,
    _where_dict_to_sql,
    _quote_id,
)


# ---------------------------------------------------------------------------
# ヘルパ・フィクスチャ
# ---------------------------------------------------------------------------


_DIM = 16


class FakeEmbeddingFunction:
    """テスト用の決定的な擬似 embedding。

    入力テキストを SHA256 でハッシュし、16 次元の正規化ベクトルにする。
    同じテキストは常に同じベクトルを返し、テキストが似ているほど類似度が高くなることは
    期待しない。**距離関係を厳密に制御するテストでは _put_vector で直接ベクトルを刷り込む。**
    """

    def __call__(self, texts: Iterable[str]) -> list[list[float]]:
        """ChromaDB EmbeddingFunction 互換: テキストリスト → ベクトルリスト。"""
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # 16 バイト × 2 を 16 個の float (-1, 1) に変換
            vec = [(b - 128) / 128.0 for b in h[:_DIM]]
            # 正規化（cosine 距離安定化のため）
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def store(tmp_path):
    """テスト用の一時 LanceStore を返す。Fake embedding fn を注入する。"""
    return LanceStore(db_path=str(tmp_path), embedding_fn=FakeEmbeddingFunction())


def _add_memories(s: LanceStore, char_id: str, items: list[dict]) -> None:
    """記憶をまとめて追加するヘルパ。

    Args:
        s: 書き込み先の LanceStore。
        char_id: キャラクター ID。
        items: ``{"id", "content", "category"}`` を含む辞書リスト。
    """
    for it in items:
        s.add_memory(
            memory_id=it["id"],
            content=it["content"],
            character_id=char_id,
            metadata={"category": it["category"]},
        )


# ---------------------------------------------------------------------------
# _where_dict_to_sql のユニットテスト
# ---------------------------------------------------------------------------


class TestWhereDictToSql:
    """ChromaDB 形式 where dict → LanceDB SQL 変換ロジックのテスト。

    ``MemoryManager.recall_with_identity`` が ``{"category": {"$ne": "identity"}}`` 形式を
    使うため、$ne の正しい変換が特に重要。
    """

    def test_empty_returns_empty(self):
        """空 dict は空文字列を返す（LanceDB の where 句省略相当）。"""
        assert _where_dict_to_sql({}) == ""
        assert _where_dict_to_sql(None) == ""

    def test_simple_equality(self):
        """単純な ``{key: value}`` は ``key = 'value'`` になる。"""
        assert _where_dict_to_sql({"category": "identity"}) == "category = 'identity'"

    def test_string_quote_escaping(self):
        """シングルクォート入りの値は ``''`` で安全にエスケープする（SQL injection 対策）。"""
        sql = _where_dict_to_sql({"name": "O'Reilly"})
        assert sql == "name = 'O''Reilly'"

    def test_ne_operator(self):
        """``{key: {$ne: v}}`` → ``key != 'v'`` 変換（recall_with_identity が依存する）。"""
        assert _where_dict_to_sql({"category": {"$ne": "identity"}}) == "category != 'identity'"

    def test_comparison_operators(self):
        """$gt / $gte / $lt / $lte の各演算子が SQL 比較演算子に変換される。"""
        assert _where_dict_to_sql({"score": {"$gt": 0.5}}) == "score > 0.5"
        assert _where_dict_to_sql({"score": {"$gte": 0.5}}) == "score >= 0.5"
        assert _where_dict_to_sql({"score": {"$lt": 0.5}}) == "score < 0.5"
        assert _where_dict_to_sql({"score": {"$lte": 0.5}}) == "score <= 0.5"

    def test_in_operator(self):
        """$in は IN (...) リテラルリストに変換される。"""
        sql = _where_dict_to_sql({"category": {"$in": ["a", "b", "c"]}})
        assert sql == "category IN ('a', 'b', 'c')"

    def test_in_empty_list(self):
        """空 $in は常偽（``1 = 0``）に変換される。"""
        assert _where_dict_to_sql({"x": {"$in": []}}) == "1 = 0"

    def test_nin_operator(self):
        """$nin は NOT IN リテラルリストに変換される。"""
        sql = _where_dict_to_sql({"category": {"$nin": ["a", "b"]}})
        assert sql == "category NOT IN ('a', 'b')"

    def test_multiple_keys_and_join(self):
        """複数キーは AND で結合される。"""
        sql = _where_dict_to_sql({"a": "1", "b": "2"})
        assert sql == "a = '1' AND b = '2'"

    def test_unknown_operator_raises(self):
        """未対応演算子は ValueError。"""
        with pytest.raises(ValueError):
            _where_dict_to_sql({"x": {"$unknown": 1}})


def test_quote_id_escapes_single_quote():
    """``_quote_id`` がシングルクォートを ``''`` でエスケープして安全にクオートする。"""
    assert _quote_id("abc") == "'abc'"
    assert _quote_id("ab'c") == "'ab''c'"


# ---------------------------------------------------------------------------
# add_memory / recall_memory の基本動作
# ---------------------------------------------------------------------------


class TestAddRecallMemory:
    """add_memory と recall_memory の往復が成立することを検証する。"""

    def test_add_then_recall_returns_added_memory(self, store):
        """add_memory した記憶が recall_memory で取得できる。"""
        store.add_memory(
            memory_id="m1",
            content="コーヒーが好き",
            character_id="char_a",
            metadata={"category": "user"},
        )
        results = store.recall_memory(query="コーヒー", character_id="char_a", top_k=5)
        assert len(results) == 1
        r = results[0]
        assert r["id"] == "m1"
        assert r["content"] == "コーヒーが好き"
        # ChromaStore 互換の戻り値形状
        assert "distance" in r
        assert r["metadata"]["category"] == "user"
        assert r["metadata"]["character_id"] == "char_a"

    def test_recall_filters_by_character_id(self, store):
        """recall_memory は character_id で必ずフィルタする（他キャラの記憶は混入しない）。"""
        store.add_memory(memory_id="ma", content="char A の記憶", character_id="char_a", metadata={"category": "x"})
        store.add_memory(memory_id="mb", content="char B の記憶", character_id="char_b", metadata={"category": "x"})
        results = store.recall_memory(query="記憶", character_id="char_a", top_k=10)
        ids = [r["id"] for r in results]
        assert "ma" in ids
        assert "mb" not in ids

    def test_recall_on_empty_store_returns_empty(self, store):
        """テーブル未作成の状態で recall を呼んでも例外を出さず空リストを返す。"""
        assert store.recall_memory(query="anything", character_id="char_x", top_k=5) == []

    def test_add_same_id_updates_inplace(self, store):
        """同一 ID で add_memory を再呼び出しすると既存レコードが in-place 更新される。

        旧 ChromaStore の欠陥 C（旧ID delete + 新ID add で HNSW ゴーストID共存）を
        構造的に避けるため、merge_insert による upsert になっていることを保証する。
        """
        store.add_memory(memory_id="m1", content="初期内容", character_id="c", metadata={"category": "user"})
        store.add_memory(memory_id="m1", content="更新内容", character_id="c", metadata={"category": "identity"})

        results = store.recall_memory(query="内容", character_id="c", top_k=5)
        # ID 重複ではなく1件にマージされていること
        assert len([r for r in results if r["id"] == "m1"]) == 1
        target = next(r for r in results if r["id"] == "m1")
        assert target["content"] == "更新内容"
        assert target["metadata"]["category"] == "identity"


# ---------------------------------------------------------------------------
# where フィルタ（$ne を含む）
# ---------------------------------------------------------------------------


class TestRecallMemoryWithWhereFilter:
    """recall_memory の where 引数が正しく動くことを検証する。

    特に旧 ChromaStore で発生していた「フィルタ後件数 < top_k で結果が消える」バグが
    LanceStore では発生しないことを保証する（LanceDB の prefilter + LIMIT で安全）。
    """

    def test_identity_only_filter(self, store):
        """``{"category": "identity"}`` で identity 記憶のみ取得できる。"""
        char_id = "c"
        items = [{"id": f"g-{i}", "content": f"general記憶{i}", "category": "general"} for i in range(8)]
        items += [
            {"id": "id-1", "content": "私の名前ははるです", "category": "identity"},
            {"id": "id-2", "content": "私は感情豊かなキャラクター", "category": "identity"},
        ]
        _add_memories(store, char_id, items)

        results = store.recall_memory(query="名前", character_id=char_id, top_k=5, where={"category": "identity"})
        assert len(results) == 2
        for r in results:
            assert r["metadata"]["category"] == "identity"

    def test_filtered_count_less_than_top_k_returns_filtered_count(self, store):
        """旧バグ再現: フィルタ後件数 (2) < top_k (5) でも結果が消えないことを保証する。

        ChromaStore では `Number of requested results > number of elements based on filters`
        例外が発生し、空リストが返る不具合があった。LanceDB は LIMIT で勝手に絞るため
        この問題は構造的に発生しない。
        """
        char_id = "c"
        items = [{"id": f"g-{i}", "content": f"general{i}", "category": "general"} for i in range(8)]
        items += [
            {"id": "id-1", "content": "id記憶1", "category": "identity"},
            {"id": "id-2", "content": "id記憶2", "category": "identity"},
        ]
        _add_memories(store, char_id, items)

        results = store.recall_memory(
            query="記憶",
            character_id=char_id,
            top_k=5,
            where={"category": "identity"},
        )
        # フィルタ後の件数 = 2 件が返る（top_k=5 でも、2 件しか該当しないので 2 件）
        assert len(results) == 2

    def test_ne_operator_filter(self, store):
        """``{"category": {"$ne": "identity"}}`` で identity 以外を取得できる。

        recall_with_identity が「identity 以外の記憶を取得する」ためにこの形式を使う。
        """
        char_id = "c"
        items = [
            {"id": "g-1", "content": "general記憶1", "category": "general"},
            {"id": "g-2", "content": "general記憶2", "category": "general"},
            {"id": "id-1", "content": "identity記憶", "category": "identity"},
        ]
        _add_memories(store, char_id, items)

        results = store.recall_memory(
            query="記憶",
            character_id=char_id,
            top_k=10,
            where={"category": {"$ne": "identity"}},
        )
        cats = [r["metadata"]["category"] for r in results]
        assert "identity" not in cats
        assert all(c == "general" for c in cats)


# ---------------------------------------------------------------------------
# find_similar_in_category / 重複排除
# ---------------------------------------------------------------------------


class TestFindSimilarInCategory:
    """find_similar_in_category の閾値判定とカテゴリ絞り込みを検証する。"""

    def test_returns_none_on_empty(self, store):
        """テーブル未作成の状態では None を返す（例外を出さない）。"""
        assert store.find_similar_in_category(
            content="x", character_id="c", category="user", threshold=0.15
        ) is None

    def test_returns_existing_id_for_identical_content(self, store):
        """同一テキストを与えると既存 ID を返す（距離 0 < threshold）。"""
        store.add_memory(memory_id="m1", content="identical", character_id="c", metadata={"category": "user"})
        result = store.find_similar_in_category(
            content="identical", character_id="c", category="user", threshold=0.15
        )
        assert result == "m1"

    def test_skips_other_category(self, store):
        """指定カテゴリ以外は検索対象外（identity を探しているのに user は対象外）。"""
        store.add_memory(memory_id="m1", content="x", character_id="c", metadata={"category": "user"})
        result = store.find_similar_in_category(
            content="x", character_id="c", category="identity", threshold=0.15
        )
        assert result is None


# ---------------------------------------------------------------------------
# delete / delete_all_memories
# ---------------------------------------------------------------------------


class TestDelete:
    """物理削除が正しく動くことを検証する。"""

    def test_delete_memory_removes_record(self, store):
        """delete_memory で指定 ID が消える（recall に出てこない）。"""
        store.add_memory(memory_id="m1", content="残る記憶", character_id="c", metadata={"category": "x"})
        store.add_memory(memory_id="m2", content="消す記憶", character_id="c", metadata={"category": "x"})
        store.delete_memory(memory_id="m2", character_id="c")
        results = store.recall_memory(query="記憶", character_id="c", top_k=5)
        ids = [r["id"] for r in results]
        assert "m1" in ids
        assert "m2" not in ids

    def test_delete_all_memories_removes_only_target_character(self, store):
        """delete_all_memories は指定キャラのみ削除し、他キャラは残す（単一テーブル方式の検証）。"""
        store.add_memory(memory_id="ma", content="A", character_id="char_a", metadata={"category": "x"})
        store.add_memory(memory_id="mb", content="B", character_id="char_b", metadata={"category": "x"})
        store.delete_all_memories("char_a")

        a_results = store.recall_memory(query="A", character_id="char_a", top_k=5)
        b_results = store.recall_memory(query="B", character_id="char_b", top_k=5)
        assert a_results == []
        assert len(b_results) == 1 and b_results[0]["id"] == "mb"

    def test_delete_on_missing_table_is_noop(self, store):
        """テーブル未作成での delete 呼び出しは例外を出さず黙って戻る。"""
        store.delete_memory(memory_id="x", character_id="c")
        store.delete_all_memories("c")


# ---------------------------------------------------------------------------
# チャット履歴
# ---------------------------------------------------------------------------


class TestChatTurns:
    """add_chat_turn / recall_chat_turns の動作検証。"""

    def test_add_then_recall(self, store):
        """add_chat_turn したターンが recall_chat_turns で取れる。metadata も保持される。"""
        store.add_chat_turn(
            message_id="msg1",
            content="ユーザ: こんにちは",
            character_id="c",
            metadata={
                "session_id": "sess1",
                "role": "user",
                "speaker_name": "ユーザ",
                "created_at": "2026-05-04T10:00:00",
            },
        )
        results = store.recall_chat_turns(query="挨拶", character_id="c", top_k=5)
        assert len(results) == 1
        r = results[0]
        assert r["id"] == "msg1"
        assert r["content"] == "ユーザ: こんにちは"
        assert r["metadata"]["session_id"] == "sess1"
        assert r["metadata"]["role"] == "user"

    def test_recall_filters_by_character(self, store):
        """recall_chat_turns も character_id でフィルタされる。"""
        store.add_chat_turn(
            message_id="m1", content="A発言", character_id="char_a",
            metadata={"session_id": "s1", "role": "user", "speaker_name": "u", "created_at": ""},
        )
        store.add_chat_turn(
            message_id="m2", content="B発言", character_id="char_b",
            metadata={"session_id": "s2", "role": "user", "speaker_name": "u", "created_at": ""},
        )
        results = store.recall_chat_turns(query="発言", character_id="char_a", top_k=5)
        ids = [r["id"] for r in results]
        assert "m1" in ids
        assert "m2" not in ids


# ---------------------------------------------------------------------------
# キャラクター定義（estranged 検出）
# ---------------------------------------------------------------------------


class TestDefinitions:
    """upsert_character_definition / find_similar_definition / mark_definition_estranged のテスト。"""

    def test_upsert_returns_character_id(self, store):
        """upsert_character_definition は引数の character_id を返す。"""
        result = store.upsert_character_definition("c1", "私はキャラクターA", status="active")
        assert result == "c1"

    def test_find_similar_only_includes_estranged(self, store):
        """active キャラは検索対象外、estranged のみ閾値以下なら返る。"""
        store.upsert_character_definition("c1", "私はキャラクターA", status="active")
        store.upsert_character_definition("c2", "私はキャラクターB", status="estranged")
        # 完全一致クエリ
        result = store.find_similar_definition("私はキャラクターA", threshold=0.5)
        # c1 は active なので候補から外れる
        ids = [r["character_id"] for r in result]
        assert "c1" not in ids

    def test_find_similar_excludes_self(self, store):
        """exclude_character_id 指定で自分自身を検索結果から除外する。"""
        store.upsert_character_definition("c1", "同じテキスト", status="estranged")
        store.upsert_character_definition("c2", "同じテキスト", status="estranged")
        result = store.find_similar_definition(
            "同じテキスト", exclude_character_id="c1", threshold=0.5
        )
        ids = [r["character_id"] for r in result]
        assert "c1" not in ids

    def test_mark_definition_estranged_changes_status(self, store):
        """mark_definition_estranged 後は find_similar_definition の対象になる。"""
        store.upsert_character_definition("c1", "テキスト", status="active")
        # active 状態では類似検索の対象外
        before = store.find_similar_definition("テキスト", threshold=0.5)
        assert all(r["character_id"] != "c1" for r in before)
        # estranged に変更後は対象になる
        store.mark_definition_estranged("c1")
        after = store.find_similar_definition("テキスト", threshold=0.5)
        ids = [r["character_id"] for r in after]
        assert "c1" in ids


# ---------------------------------------------------------------------------
# 永続化
# ---------------------------------------------------------------------------


def test_persistence_across_reopen(tmp_path):
    """LanceStore を再 open してもデータが永続化されている。

    `lancedb.connect()` を別インスタンスで再度開いても、テーブルとレコードが残ることを確認する。
    backend 再起動後も記憶が消えないことの最低限の保証。
    """
    s1 = LanceStore(db_path=str(tmp_path), embedding_fn=FakeEmbeddingFunction())
    s1.add_memory(memory_id="m1", content="残るべき記憶", character_id="c", metadata={"category": "user"})
    del s1

    s2 = LanceStore(db_path=str(tmp_path), embedding_fn=FakeEmbeddingFunction())
    results = s2.recall_memory(query="記憶", character_id="c", top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "m1"
    assert results[0]["content"] == "残るべき記憶"
