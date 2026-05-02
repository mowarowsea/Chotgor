"""``ChromaStore`` の書き込み並行性テスト。

--- 背景・目的（詳細） ---
ChromaDB の HNSW バイナリ（link_lists.bin / data.bin など）は multi-writer-unsafe で、
同一プロセス内の複数スレッド（chat indexer の asyncio.to_thread / MemoryManager の
リトライワーカー / メインスレッドの同期 inscribe など）が同時に upsert / delete を
走らせるとファイル整合性が崩れて Error finding id を引き起こす（欠陥 D）。

修正後、``ChromaStore`` は内部に ``threading.RLock`` を持ち、書き込み系メソッド
（add_memory / delete_memory / add_chat_turn / rebuild_memory_collection /
upsert_character_definition / mark_definition_estranged / delete_all_memories）の
入口でロックを取って書き込みを直列化する。さらに ``_safe_get_or_create_collection``
内の破損検出 → delete_collection 経路も保護する（read 系から発火しても並行 write を
排他するため）。

本テストは「複数スレッドから書き込みを同時に発行しても、コレクションへの
upsert が直列化されている」ことを保証する。critical section 内に同時に
2 スレッド以上が入らないことを spy カウンタで観測する方式を取る。
"""

from __future__ import annotations

import threading
import time

import pytest

from backend.repositories.chroma.store import ChromaStore


@pytest.fixture
def chroma_store(tmp_path):
    """テスト用の一時 ChromaStore を返す（embedding は default）。"""
    return ChromaStore(db_path=str(tmp_path))


class TestChromaStoreWriteSerialization:
    """ChromaStore の書き込み系メソッドが互いに直列化されることの検証。

    本クラスは「2 スレッドが同時に書き込みを発行しても upsert は重ならない」
    という保証を CI レベルで定着させる。spy カウンタを critical section に仕込み、
    同時実行数の最大値が 1 を超えないことを assert する。
    threading.RLock の acquire / release を直接観測するのではなく、
    実際の collection.upsert が並行実行されないことを観測する点が肝。
    """

    def _run_concurrent_writes(self, chroma_store, monkeypatch, write_callables: list):
        """write_callables を別スレッドで同時実行し、critical section の最大同時実行数を返す。

        ``collection.upsert`` を 50ms スリープ入りでラップし、その実行中に
        同時にどれだけのスレッドが居たかを spy カウンタで観測する。
        ロックが効いていれば 1、無ければ呼び出し数まで増える。
        """
        in_critical = {"count": 0, "max_seen": 0}
        counter_lock = threading.Lock()

        real_get_collection = chroma_store._get_collection
        real_get_chat = chroma_store._get_chat_collection

        def _wrap(get_fn):
            """指定の get 関数で取得した collection の upsert を spy 化したラッパを返す。"""

            def patched(cid: str):
                col = get_fn(cid)
                real_upsert = col.upsert

                def slow_upsert(**kwargs):
                    """upsert を50ms遅延させ、同時実行数をカウンタで観測する。"""
                    with counter_lock:
                        in_critical["count"] += 1
                        if in_critical["count"] > in_critical["max_seen"]:
                            in_critical["max_seen"] = in_critical["count"]
                    try:
                        # 並行性が顕在化するための猶予。ロックが無ければ
                        # 別スレッドの upsert が確実にこの間に入り込む。
                        time.sleep(0.05)
                        return real_upsert(**kwargs)
                    finally:
                        with counter_lock:
                            in_critical["count"] -= 1

                col.upsert = slow_upsert
                return col

            return patched

        monkeypatch.setattr(chroma_store, "_get_collection", _wrap(real_get_collection))
        monkeypatch.setattr(chroma_store, "_get_chat_collection", _wrap(real_get_chat))

        # 並行スレッド起動
        threads = [threading.Thread(target=fn) for fn in write_callables]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return in_critical["max_seen"]

    def test_concurrent_add_memory_is_serialized(self, chroma_store, monkeypatch):
        """5 スレッドからの add_memory が直列化されること（同時実行数 1 を超えない）。"""
        char_id = "char-concurrent-add"

        def make_worker(i: int):
            """1 件の add_memory を発行する worker を返す。"""
            def worker():
                chroma_store.add_memory(
                    memory_id=f"m-{i}",
                    content=f"並行書き込み {i}",
                    character_id=char_id,
                    metadata={"category": "contextual"},
                )
            return worker

        max_concurrent = self._run_concurrent_writes(
            chroma_store, monkeypatch,
            [make_worker(i) for i in range(5)],
        )

        # 全件が登録され、HNSW count が一致すること
        col = chroma_store._get_collection(char_id)
        assert col.count() == 5

        # critical section 内の同時実行は常に 1 以下であること（直列化の証明）
        assert max_concurrent == 1, (
            f"add_memory が並行実行されています（max_concurrent={max_concurrent}）。"
            "ChromaStore._write_lock が機能していない可能性があります。"
        )

    def test_concurrent_add_memory_and_chat_turn_are_serialized(
        self, chroma_store, monkeypatch
    ):
        """異種書き込み（記憶 × チャット）も同一ロックで直列化されること。

        chat_indexer と inscriber が同時走するシナリオを再現する。両者は別の
        コレクション（char_xxx と chat_xxx）を触るが、HNSW バイナリ整合性を
        プロセス全体で守るため同じロックで排他する設計。
        """
        char_id = "char-mixed-write"

        def add_memory_worker(i: int):
            def worker():
                chroma_store.add_memory(
                    memory_id=f"mem-{i}",
                    content=f"記憶 {i}",
                    character_id=char_id,
                    metadata={"category": "contextual"},
                )
            return worker

        def add_chat_turn_worker(i: int):
            def worker():
                chroma_store.add_chat_turn(
                    message_id=f"msg-{i}",
                    content=f"chat: ターン {i}",
                    character_id=char_id,
                    metadata={"role": "user"},
                )
            return worker

        callables = [add_memory_worker(i) for i in range(3)] + [
            add_chat_turn_worker(i) for i in range(3)
        ]
        max_concurrent = self._run_concurrent_writes(
            chroma_store, monkeypatch, callables
        )

        # 両方のコレクションに正しく登録されていること
        assert chroma_store._get_collection(char_id).count() == 3
        assert chroma_store._get_chat_collection(char_id).count() == 3

        # 異種書き込み同士でも直列化されていること
        assert max_concurrent == 1, (
            f"add_memory × add_chat_turn が並行実行されています "
            f"（max_concurrent={max_concurrent}）。"
        )
