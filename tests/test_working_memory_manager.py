"""WorkingMemoryManager の一覧取得・heat 想起に関するテスト。

システムプロンプトへ載る WM ブロックの「分量」を決める2つの制御を検証する
（current-spec/memory_recall_algorithm.md §4.3）:

  - list_all_threads(closed_limit): Close 済みスレッドは決着済みだが件数が
    増え続けるため、一覧へ載せるのは直近ぶんだけに絞る。省いた本数は
    告知行に使うので、絞り込みと同時に「いくつ省いたか」を返す契約とする。
  - recall_threads(min_heat): heat 上位から機械的に TopK を取ると、関連の薄い
    スレッドまで前景へ上がる。下限を設けて 0 件のターンを許容する。

加えて、Chronicle 棚卸しの「気づき誘導」で使う逆向きの検索も検証する
（docs/planned/wm_repeat_awareness_plan.md）:

  - find_similar_closed_threads(min_relevance): Open スレッド → 意味的に近い
    Close 済みスレッド。「もう決着済みの話題を Open のまま抱えていないか」の
    材料を作るだけで、close はしない。

SQLite / LanceStore は本テストの関心ではないためモックで置き換え、
スレッド ORM は属性アクセスだけを満たす SimpleNamespace で代用する。
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.services.memory.working_memory_manager import (
    DEFAULT_CLOSED_INDEX_LIMIT,
    DEFAULT_SIMILAR_CLOSED_MIN_RELEVANCE,
    DEFAULT_WM_RECALL_MIN_HEAT,
    WorkingMemoryManager,
)


def _thread(id: str, *, type: str = "topic", is_open: bool = True,
            importance: float = 0.5, touched: datetime | None = None):
    """スレッド ORM の代役（_thread_to_dict が触る属性だけを持つ）。"""
    now = datetime.now()
    return SimpleNamespace(
        id=id,
        character_id="char-1",
        type=type,
        summary=f"{id} の要約",
        atmosphere_tag="",
        importance=importance,
        is_open=is_open,
        relation_target=None,
        origin="real",
        created_at=now,
        updated_at=now,
        last_touched_at=touched or now,
    )


class TestListAllThreadsClosedLimit:
    """一覧注入用の取得（Open 全件 + Close 直近ぶん）の検証。

    Open スレッドは「いま抱えていること」なので必ず全件残す。Close 済みだけを
    上限で切り、切った本数を第2要素として返す。呼び出し側（request_builder）は
    この本数から告知行「ほかに N 本あります」を組み立てる。
    """

    def _manager(self, threads):
        sqlite = MagicMock()
        sqlite.list_working_memory_threads.return_value = threads
        return WorkingMemoryManager(sqlite=sqlite, vector_store=MagicMock())

    def test_open_threads_are_never_dropped(self):
        """Close 上限をどれだけ小さくしても Open は全件返ること。"""
        threads = [_thread(f"open-{i}") for i in range(4)]
        threads += [_thread(f"closed-{i}", is_open=False) for i in range(3)]
        wm = self._manager(threads)

        result, omitted = wm.list_all_threads("char-1", closed_limit=1)

        open_ids = [t["id"] for t in result if t["is_open"]]
        assert open_ids == ["open-0", "open-1", "open-2", "open-3"]
        assert omitted == 2

    def test_closed_threads_are_kept_in_given_order(self):
        """Close 済みは渡された順（updated_at 降順）の先頭から残ること。"""
        threads = [_thread(f"closed-{i}", is_open=False) for i in range(5)]
        wm = self._manager(threads)

        result, omitted = wm.list_all_threads("char-1", closed_limit=2)

        assert [t["id"] for t in result] == ["closed-0", "closed-1"]
        assert omitted == 3

    def test_none_limit_returns_everything(self):
        """closed_limit=None は従来どおりの全件返し（省略数は 0）。"""
        threads = [_thread("open-0"), _thread("closed-0", is_open=False)]
        wm = self._manager(threads)

        result, omitted = wm.list_all_threads("char-1", closed_limit=None)

        assert len(result) == 2
        assert omitted == 0

    def test_default_limit_is_applied_without_argument(self):
        """引数を省いても既定の上限が効くこと（呼び出し側の書き忘れで肥大させない）。"""
        threads = [
            _thread(f"closed-{i}", is_open=False)
            for i in range(DEFAULT_CLOSED_INDEX_LIMIT + 5)
        ]
        wm = self._manager(threads)

        result, omitted = wm.list_all_threads("char-1")

        assert len(result) == DEFAULT_CLOSED_INDEX_LIMIT
        assert omitted == 5


class TestRecallThreadsMinHeat:
    """heat 想起の下限フィルタの検証。

    heat = importance × 時間減衰 × クエリ類似度。ベクトル検索が返した候補でも、
    重要度が低い・古い・関連が薄いものは前景へ上げない。「思い出すものが無い」
    ターンがあってよい、という設計を固定する。
    """

    def _manager(self, threads_by_id, hits):
        sqlite = MagicMock()
        sqlite.get_working_memory_thread.side_effect = lambda tid: threads_by_id.get(tid)
        vector = MagicMock()
        vector.recall_working_memory_threads.return_value = hits
        sqlite.get_latest_working_memory_post.return_value = None
        return WorkingMemoryManager(sqlite=sqlite, vector_store=vector)

    def test_low_heat_thread_is_dropped(self):
        """下限未満のスレッドは TopK に空きがあっても返らないこと。"""
        hot = _thread("hot", importance=0.9)
        cold = _thread("cold", importance=0.01, touched=datetime.now() - timedelta(days=30))
        wm = self._manager(
            {"hot": hot, "cold": cold},
            [{"id": "hot", "distance": 0.2}, {"id": "cold", "distance": 1.6}],
        )

        result = wm.recall_threads("char-1", "クエリ", min_heat=DEFAULT_WM_RECALL_MIN_HEAT)

        assert [t["id"] for t in result] == ["hot"]

    def test_all_dropped_returns_empty(self):
        """全候補が下限未満なら空リスト（0 件のターンを許容する）。"""
        cold = _thread("cold", importance=0.02)
        wm = self._manager({"cold": cold}, [{"id": "cold", "distance": 1.9}])

        assert wm.recall_threads("char-1", "クエリ") == []

    def test_default_top_k_is_three(self):
        """既定の返却件数は 3 件（heat 降順で上位のみ前景へ上げる）。"""
        threads = {f"t{i}": _thread(f"t{i}", importance=0.9 - i * 0.05) for i in range(6)}
        hits = [{"id": f"t{i}", "distance": 0.2} for i in range(6)]
        wm = self._manager(threads, hits)

        result = wm.recall_threads("char-1", "クエリ")

        assert len(result) == 3
        assert [t["id"] for t in result] == ["t0", "t1", "t2"]


def _distance_for(relevance: float) -> float:
    """狙った relevance を返す cosine 距離を逆算する（distance_to_similarity の逆関数）。

    relevance = 1 - distance/2 なので distance = 2 × (1 - relevance)。
    閾値の境界をテストで直に書けるようにするためのヘルパー。
    """
    return 2.0 * (1.0 - relevance)


class TestFindSimilarClosedThreads:
    """Open スレッド → 類似 Close 済みスレッドの検索（重複疑いの検出）の検証。

    Chronicle 棚卸しで「もう Close 済みの話題を Open のまま抱え続けていないか」を
    本人に気づかせるための材料作り。判断は本人に委ねる設計なので、ここでの関心は
    「拾うべきものを拾い、拾ってはいけないものを混ぜないこと」に限られる:

      - min_relevance の境界（無関係ペアを材料に混ぜない）
      - Close 済み以外（Open のまま index が古いスレッド）を混ぜない
      - 検索が対象キャラのスコープ・Close 済み条件で発行されている
      - 材料が無いとき（クエリ素材が空）に embedding 検索を撃たない
    """

    def _manager(self, threads_by_id, hits):
        sqlite = MagicMock()
        sqlite.get_working_memory_thread.side_effect = lambda tid: threads_by_id.get(tid)
        sqlite.get_latest_working_memory_post.return_value = None
        vector = MagicMock()
        vector.recall_working_memory_threads.return_value = hits
        return WorkingMemoryManager(sqlite=sqlite, vector_store=vector)

    def test_relevance_above_threshold_is_returned(self):
        """閾値をわずかに上回るペアは材料として返ること。"""
        closed = _thread("closed-1", is_open=False)
        wm = self._manager(
            {"closed-1": closed},
            [{"id": "closed-1", "distance": _distance_for(DEFAULT_SIMILAR_CLOSED_MIN_RELEVANCE + 0.01)}],
        )

        result = wm.find_similar_closed_threads("char-1", {"summary": "日食なつこの実験"})

        assert [t["id"] for t in result] == ["closed-1"]
        assert result[0]["relevance"] > DEFAULT_SIMILAR_CLOSED_MIN_RELEVANCE

    def test_relevance_below_threshold_is_dropped(self):
        """閾値をわずかに下回るペアは捨てること（無関係ペアを材料に混ぜない）。"""
        closed = _thread("closed-1", is_open=False)
        wm = self._manager(
            {"closed-1": closed},
            [{"id": "closed-1", "distance": _distance_for(DEFAULT_SIMILAR_CLOSED_MIN_RELEVANCE - 0.01)}],
        )

        assert wm.find_similar_closed_threads("char-1", {"summary": "日食なつこの実験"}) == []

    def test_still_open_thread_is_excluded(self):
        """index が古く Open スレッドが返っても、SQLite 側の is_open で弾くこと。

        LanceStore の index 更新はバックグラウンド（fire-and-forget）なので、
        close 直後などに is_open の食い違いが起こりうる。source of truth は SQLite。
        """
        stale_open = _thread("stale", is_open=True)
        closed = _thread("closed-1", is_open=False)
        wm = self._manager(
            {"stale": stale_open, "closed-1": closed},
            [
                {"id": "stale", "distance": _distance_for(0.99)},
                {"id": "closed-1", "distance": _distance_for(0.90)},
            ],
        )

        result = wm.find_similar_closed_threads("char-1", {"summary": "何かの話題"}, top_k=5)

        assert [t["id"] for t in result] == ["closed-1"]

    def test_search_is_scoped_to_character_and_closed_threads(self):
        """検索が「このキャラの」「Close 済み task/topic」に限定して発行されること。

        他キャラのスレッドを材料に混ぜないためのスコープは LanceStore 側の
        where 句で効く。ここではその条件が正しく渡ることを固定する。
        """
        wm = self._manager({}, [])

        wm.find_similar_closed_threads("char-1", {"summary": "話題", "latest_post": "続き"})

        args, kwargs = wm.vector_store.recall_working_memory_threads.call_args
        assert args[0] == "話題\n続き"      # summary + 最新ポストがクエリ素材
        assert args[1] == "char-1"
        assert kwargs["where"] == {"type": {"$in": ["task", "topic"]}, "is_open": 0}

    def test_empty_query_text_skips_search(self):
        """クエリ素材が空なら embedding 検索を撃たずに空を返すこと。"""
        wm = self._manager({}, [])

        assert wm.find_similar_closed_threads("char-1", {"summary": "  ", "latest_post": ""}) == []
        wm.vector_store.recall_working_memory_threads.assert_not_called()

    def test_results_are_sorted_and_capped_by_top_k(self):
        """relevance 降順で top_k 件に切ること（既定は 1 件）。"""
        threads = {f"c{i}": _thread(f"c{i}", is_open=False) for i in range(3)}
        hits = [
            {"id": "c0", "distance": _distance_for(0.85)},
            {"id": "c1", "distance": _distance_for(0.95)},
            {"id": "c2", "distance": _distance_for(0.90)},
        ]
        wm = self._manager(threads, hits)

        assert [t["id"] for t in wm.find_similar_closed_threads(
            "char-1", {"summary": "話題"}, top_k=3,
        )] == ["c1", "c2", "c0"]
        assert [t["id"] for t in wm.find_similar_closed_threads(
            "char-1", {"summary": "話題"},
        )] == ["c1"]
