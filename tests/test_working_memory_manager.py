"""WorkingMemoryManager の一覧取得・heat 想起に関するテスト。

システムプロンプトへ載る WM ブロックの「分量」を決める2つの制御を検証する
（current-spec/memory_recall_algorithm.md §4.3）:

  - list_all_threads(closed_limit): Close 済みスレッドは決着済みだが件数が
    増え続けるため、一覧へ載せるのは直近ぶんだけに絞る。省いた本数は
    告知行に使うので、絞り込みと同時に「いくつ省いたか」を返す契約とする。
  - recall_threads(min_heat): heat 上位から機械的に TopK を取ると、関連の薄い
    スレッドまで前景へ上がる。下限を設けて 0 件のターンを許容する。

SQLite / LanceStore は本テストの関心ではないためモックで置き換え、
スレッド ORM は属性アクセスだけを満たす SimpleNamespace で代用する。
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.services.memory.working_memory_manager import (
    DEFAULT_CLOSED_INDEX_LIMIT,
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
