"""``ChromaStore.find_similar_in_category`` の HNSW 破損自動修復テスト。

--- 背景・目的（詳細） ---
``find_similar_in_category`` は ``MemoryManager.write_memory`` から呼ばれる
重要な分岐関数で、戻り値が ``None`` か既存IDかで「新規 add」か「上書き update」
かが決まる。

この関数は以前、内部の ``collection.query`` で発生した例外を
``except Exception: return None`` で完全に黙殺していた。一方、姉妹関数の
``recall_memory`` は HNSW 破損エラー（"Error finding id ..."）を検出すると
``rebuild_memory_collection`` を呼び自動修復する非対称な実装になっていた。

この非対称が破損連鎖を増幅させていた：
  1. HNSW にゴーストID が混入する
  2. 次回の write_memory で ``find_similar_in_category`` が呼ばれる
  3. query が "Error finding id" を投げるが、関数は黙殺して ``None`` を返す
  4. write_memory は「類似なし」と判断し、新規 UUID で add する
  5. ChromaDB に「ゴーストID + 新規ID」が共存する状態に
  6. 後の recall や別の find_similar で再び破損エラーが発生

この事象は実運用でキャラクターのコレクション全消失（count=0）にまで発展
した。本テストは ``find_similar_in_category`` が ``recall_memory`` と同じ
復旧パターンを取ることを CI レベルで保証する。

--- テスト戦略 ---
ChromaDB 実体（tmp_path）に記憶を 1 件登録した上で、最初の 1 回だけ
``collection.query`` が "Error finding id" を模した例外を投げるよう
モンキーパッチする。修正前なら関数は ``None`` を返して黙って失敗するが、
修正後なら ``rebuild_memory_collection`` を呼びリトライで成功する。

``rebuild_memory_collection`` 呼び出しの観測を最重要 assert とし、
副次的に「最終結果が None / 例外で外に出ない」ことも確認する。
"""

from __future__ import annotations

import pytest

from backend.repositories.chroma.store import ChromaStore


@pytest.fixture
def chroma_store(tmp_path):
    """テスト用の一時 ChromaStore を返す（embedding は default）。

    test_chroma_store.py と同じ fixture 流儀。tmp_path は pytest が自動で
    後始末する。
    """
    return ChromaStore(db_path=str(tmp_path))


class TestFindSimilarHnswRecovery:
    """``find_similar_in_category`` が HNSW 破損から自動回復することの検証。

    本クラスのテストは「破損を黙殺せず rebuild_memory_collection を呼ぶ」
    という構造的修正を保証する。recall_memory と同じ非対称解消の砦。
    関数の中の query 呼び出しを差し替えることで、本物の HNSW 破損なしに
    破損相当のシナリオを再現する点が肝。
    """

    def test_rebuild_called_on_error_finding_id(self, chroma_store, monkeypatch):
        """query が "Error finding id" を投げた際に rebuild_memory_collection が呼ばれること。

        --- 検証ステップ ---
        1. 記憶を 1 件登録する。
        2. ``_get_collection`` が返す Collection の ``query`` を、初回呼び出しのみ
           "Error finding id ..." を投げる関数に差し替える（2 回目以降は本物）。
        3. ``rebuild_memory_collection`` を spy で wrap し、呼び出し回数を観測する。
        4. ``find_similar_in_category`` を呼ぶ。
        5. spy の呼び出し回数が 1 以上であること（=黙殺ではなく自動修復が走った）を assert。

        本テストが赤くなった場合、欠陥 B（find_similar の例外黙殺）が再発したか、
        rebuild の呼び出し経路が変更されたことを意味する。
        """
        character_id = "char-test-recovery"
        chroma_store.add_memory(
            memory_id="m1",
            content="コーヒーが好き",
            character_id=character_id,
            metadata={"category": "identity"},
        )

        # rebuild_memory_collection を spy で包む
        rebuild_calls = {"count": 0}
        real_rebuild = chroma_store.rebuild_memory_collection

        def spy_rebuild(cid: str) -> int:
            """rebuild_memory_collection の呼び出しを記録する spy。"""
            rebuild_calls["count"] += 1
            return real_rebuild(cid)

        monkeypatch.setattr(chroma_store, "rebuild_memory_collection", spy_rebuild)

        # _get_collection が返す collection の query を最初の 1 回だけ破損例外に差し替える。
        # 2 回目以降（rebuild 後のリトライ）は本物の query を呼ばせる。
        real_get_collection = chroma_store._get_collection
        query_call_count = {"n": 0}

        def patched_get_collection(cid: str):
            """破損動作する collection を返すラッパ。"""
            collection = real_get_collection(cid)
            real_query = collection.query

            def maybe_failing_query(*args, **kwargs):
                """初回呼び出しのみ "Error finding id" を投げる。"""
                query_call_count["n"] += 1
                if query_call_count["n"] == 1:
                    raise RuntimeError(
                        "Error finding id deadbeef-1234 in segment"
                    )
                return real_query(*args, **kwargs)

            collection.query = maybe_failing_query
            return collection

        monkeypatch.setattr(chroma_store, "_get_collection", patched_get_collection)

        # 例外が外に漏れないこと（修正前は内部黙殺で漏れない／修正後は rebuild 経由で復旧）
        result = chroma_store.find_similar_in_category(
            content="コーヒーが大好き",
            character_id=character_id,
            category="identity",
            threshold=0.05,
        )

        # 最重要 assert: rebuild が呼ばれた = 破損を検出して自動修復した
        assert rebuild_calls["count"] >= 1, (
            "rebuild_memory_collection が呼ばれていません。"
            "find_similar_in_category が HNSW 破損エラーを黙殺している可能性があります。"
        )
        # query は最低 2 回呼ばれている（初回失敗 + リトライ）
        assert query_call_count["n"] >= 2, (
            f"query 呼び出し回数が想定外: {query_call_count['n']} (期待: >= 2)"
        )
        # 戻り値は実際の類似判定結果（distance < 0.05 でなければ None）。
        # 値そのものではなく「例外が外に漏れず None または既存IDが返る」型を確認する。
        assert result is None or isinstance(result, str), (
            f"戻り値は None または str であること。実際: {type(result).__name__}"
        )

    def test_no_rebuild_for_non_corruption_error(self, chroma_store, monkeypatch):
        """HNSW 破損以外のエラーでは rebuild を呼ばないこと。

        ``recall_memory`` の挙動と対称にするため、``find_similar_in_category`` も
        "Error finding id" 以外の例外（一時的なネットワークエラー、内部バグ等）に
        対しては rebuild を呼ばずに ``None`` を返すべき。安易に rebuild を呼ぶと
        全データ消失リスクがあるため、誤発火しないことを保証する。
        """
        character_id = "char-test-noerr"
        chroma_store.add_memory(
            memory_id="m1",
            content="ねこが好き",
            character_id=character_id,
            metadata={"category": "identity"},
        )

        rebuild_calls = {"count": 0}
        real_rebuild = chroma_store.rebuild_memory_collection

        def spy_rebuild(cid: str) -> int:
            """rebuild の呼び出しを記録するだけの spy。"""
            rebuild_calls["count"] += 1
            return real_rebuild(cid)

        monkeypatch.setattr(chroma_store, "rebuild_memory_collection", spy_rebuild)

        real_get_collection = chroma_store._get_collection

        def patched_get_collection(cid: str):
            """破損ではない別系統エラーを投げる collection ラッパ。"""
            collection = real_get_collection(cid)

            def always_failing_query(*args, **kwargs):
                """破損とは無関係な例外を投げる。"""
                raise RuntimeError("temporary network glitch")

            collection.query = always_failing_query
            return collection

        monkeypatch.setattr(chroma_store, "_get_collection", patched_get_collection)

        result = chroma_store.find_similar_in_category(
            content="ねこが大好き",
            character_id=character_id,
            category="identity",
            threshold=0.05,
        )

        assert rebuild_calls["count"] == 0, (
            "破損以外のエラーで rebuild が呼ばれています。誤発火は全データ消失リスク。"
        )
        assert result is None, "破損以外のエラー時は None を返すこと"
