"""embedding 再インデックスサービス（``services.memory.migration_service``）の
**再インデックス対象に漏れがない** ことを保証するテスト。

過去 ``migrate_embeddings`` は ``char_{id}`` の記憶コレクションのみを
再インデックスしていたため、embedding model 変更後に ``chat_{id}``
（チャット履歴）と ``char_definitions``（別れ機能用グローバル）の旧次元
コレクションが残存し、新しいチャットメッセージの indexer 書き込みが
``Collection expecting embedding with dimension of 3072, got 768`` で
連続失敗する不具合があった。

このテストでは embedding fn を「ベクトル次元の異なる関数」に差し替えて
再インデックスを走らせ、3 種すべてのコレクションが新次元で動くことを確認する。

テスト対象:
  - ``_reindex_chat_collections``  : 全セッションの全メッセージが participant 全員の
                                     chat コレクションに upsert されること
  - ``_reindex_definition_collection``: 全キャラの定義テキストが char_definitions に
                                     再投入され、relationship_status が維持されること
  - ``migrate_embeddings`` 経由の統合: ``_do_migrate`` 末尾で上記2関数が
                                     呼び出されること（再インデックス漏れ防止のガード）
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Optional, Sequence
from unittest.mock import patch

import pytest

from backend.repositories.chroma.store import ChromaStore
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.memory import migration_service
from backend.services.memory.migration_service import (
    _reindex_chat_collections,
    _reindex_definition_collection,
)


# ---------------------------------------------------------------------------
# 偽 EmbeddingFunction — 任意の次元を返すスタブ
# ---------------------------------------------------------------------------


class _FixedDimEmbedding:
    """指定した次元のゼロベクトルだけを返すスタブ embedding 関数。

    ChromaDB の ``EmbeddingFunction`` プロトコルに従い、文字列リストを受け取り
    同数のベクトルリストを返す。テスト目的のため意味のある値は返さない
    （次元一致のみが検証対象）。

    Args:
        dim: 返却ベクトルの次元数。
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def __call__(self, input):  # type: ignore[no-untyped-def]
        # ChromaDB の API は ``input`` 引数名で文字列リストを渡してくる
        if isinstance(input, str):
            input = [input]
        return [[0.01 * (i + 1)] * self._dim for i in range(len(input))]

    # ChromaDB 0.5+ がコレクション作成時に呼ぶシグネチャ取得用
    def name(self) -> str:
        return f"_FixedDimEmbedding({self._dim})"


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_store_tmp():
    """テスト用の一時 SQLite ストアを返す（テスト後に自動クリーンアップ）。"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(path)
    yield store
    store.engine.dispose()
    try:
        os.remove(path)
    except PermissionError:
        pass


@pytest.fixture
def chroma_with_dim(tmp_path):
    """指定次元の embedding fn で初期化された ChromaStore を返すファクトリ。

    Returns:
        ``factory(dim) -> ChromaStore`` 形式の関数。
        同一 ``tmp_path`` を共有し、後段で ``_embedding_fn`` を差し替える運用と
        合わせやすい構成にしている。
    """

    def _factory(dim: int) -> ChromaStore:
        store = ChromaStore(db_path=str(tmp_path))
        # 内部の embedding 関数をスタブに差し替える（既存テスト test_chroma_store と同様）
        store._embedding_fn = _FixedDimEmbedding(dim)
        return store

    return _factory


# ---------------------------------------------------------------------------
# ヘルパー: 試験データのセットアップ
# ---------------------------------------------------------------------------


def _seed_characters_and_chat(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    *,
    chars: Sequence[tuple[str, str, str, str]],
    sessions: Sequence[tuple[str, str]],
    messages: Sequence[tuple[str, str, str, Optional[str]]],
) -> None:
    """テスト用にキャラ・セッション・メッセージ・定義を一括投入する。

    Args:
        sqlite: SQLite ストア。
        chroma: ChromaStore（``upsert_character_definition`` を呼ぶため）。
        chars: ``(char_id, name, definition_text, status)`` のリスト。
        sessions: ``(session_id, model_id)`` のリスト（model_id は ``"{char_name}@preset"`` 形式）。
        messages: ``(message_id, session_id, role, content)`` のリスト。
                  role が ``"character"`` の場合の character_name は session の char_name から推定する。
    """
    # キャラ作成 + char_definitions に登録
    for char_id, name, definition, status in chars:
        sqlite.create_character(
            character_id=char_id,
            name=name,
            system_prompt_block1=definition,
            relationship_status=status,
        )
        if definition:
            chroma.upsert_character_definition(char_id, definition, status=status)

    # セッション作成
    for sid, model_id in sessions:
        sqlite.create_chat_session(session_id=sid, model_id=model_id)

    # メッセージ作成
    for mid, sid, role, content in messages:
        # role==character の場合のキャラ名はセッションの model_id から取り出す
        sess = sqlite.get_chat_session(sid)
        char_name = sess.model_id.split("@", 1)[0]
        sqlite.create_chat_message(
            message_id=mid,
            session_id=sid,
            role=role,
            content=content,
            character_name=char_name if role != "user" else None,
        )


# ---------------------------------------------------------------------------
# _reindex_chat_collections
# ---------------------------------------------------------------------------


class TestReindexChatCollections:
    """``_reindex_chat_collections`` が **全セッション × 全 participant** のメッセージを
    新次元で chat コレクションへ再投入することを検証するテストクラス。

    検証ポイント:
      - 旧次元の chat_{id} を削除し、新次元で作り直していること
      - is_system_message が立っているメッセージは除外されること
      - participant 解決ロジック（``get_participant_char_ids``）と整合していること
      - 1 メッセージが複数キャラの chat コレクションに投入されないこと
        （※ 1on1 では participant は 1 名のみ）
    """

    def test_chat_collection_recreated_with_new_dimension(
        self, sqlite_store_tmp, chroma_with_dim
    ):
        """旧次元 (4) で投入した chat コレクションを、新次元 (8) で再インデックスする。

        再インデックス後は 8 次元の embedding で query / upsert ができ、
        全メッセージが新コレクションに保持されていることを確認する。
        """
        sqlite = sqlite_store_tmp

        # 旧次元 4 の ChromaStore に投入
        chroma_old = chroma_with_dim(dim=4)
        _seed_characters_and_chat(
            sqlite,
            chroma_old,
            chars=[("char-haru-id", "haru", "はるの定義", "active")],
            sessions=[("sess-1", "haru@default")],
            messages=[
                ("msg-1", "sess-1", "user", "こんにちは"),
                ("msg-2", "sess-1", "character", "やっほー"),
                ("msg-3", "sess-1", "user", "今日いい天気だね"),
            ],
        )
        # 通常の indexer 経由で旧次元コレクションに 1 件入れておく（残留検証用）
        chroma_old.add_chat_turn(
            message_id="msg-1",
            content="ユーザ: こんにちは",
            character_id="char-haru-id",
            metadata={"session_id": "sess-1", "role": "user"},
        )
        # この時点で chat_char_haru_id コレクションが次元 4 で存在する

        # 新次元 8 に embedding fn を差し替えてから再インデックスを走らせる
        chroma_old._embedding_fn = _FixedDimEmbedding(dim=8)
        _reindex_chat_collections(sqlite, chroma_old)

        # 再インデックス後、コレクションは 8 次元で動作するはず
        collection = chroma_old._get_chat_collection("char-haru-id")
        # 全 3 件が入っている（user 2 件 + character 1 件）
        assert collection.count() == 3, (
            f"期待: 3 件、実際: {collection.count()}。"
            "is_system_message=False の全メッセージが upsert されているはず"
        )
        # 8 次元のクエリベクトルで query が動くこと（次元不整合エラーが出ないこと）
        result = collection.query(
            query_embeddings=[[0.1] * 8],
            n_results=1,
            include=["documents"],
        )
        assert result["ids"] and result["ids"][0], "新次元での query が結果を返すべき"

    def test_system_messages_are_excluded(self, sqlite_store_tmp, chroma_with_dim):
        """``is_system_message=1`` のメッセージは再インデックス対象から外れることを確認する。

        index_message_sync の挙動と一致させるための仕様（指定なら build_chat_doc_and_metadata
        が None を返す）。
        """
        sqlite = sqlite_store_tmp
        chroma = chroma_with_dim(dim=4)

        sqlite.create_character(
            character_id="char-x", name="X", system_prompt_block1="x"
        )
        sqlite.create_chat_session(session_id="sess-x", model_id="X@default")
        sqlite.create_chat_message(
            message_id="msg-normal",
            session_id="sess-x",
            role="user",
            content="通常メッセージ",
        )
        # システムメッセージは除外されるべき
        sqlite.create_chat_message(
            message_id="msg-system",
            session_id="sess-x",
            role="character",
            content="（退席通知）",
            character_name="X",
            is_system_message=True,
        )

        _reindex_chat_collections(sqlite, chroma)

        collection = chroma._get_chat_collection("char-x")
        assert collection.count() == 1
        ids = collection.get(include=[])["ids"]
        assert "msg-normal" in ids and "msg-system" not in ids

    def test_no_sessions_logs_and_returns(self, sqlite_store_tmp, chroma_with_dim):
        """セッションが存在しない場合は早期 return して例外を出さないことを確認する。

        新規プロジェクトや初回起動など、まだ会話が無い状態でも安全に動作するため。
        """
        sqlite = sqlite_store_tmp
        chroma = chroma_with_dim(dim=4)
        # 例外が出なければOK
        _reindex_chat_collections(sqlite, chroma)


# ---------------------------------------------------------------------------
# _reindex_definition_collection
# ---------------------------------------------------------------------------


class TestReindexDefinitionCollection:
    """``_reindex_definition_collection`` が全キャラの定義テキストを
    新次元で ``char_definitions`` に再投入することを検証するテストクラス。

    relationship_status が維持されることが特に重要（estranged キャラの
    「同一定義による再作成」検出機能が壊れないため）。
    """

    def test_all_characters_reindexed_with_status_preserved(
        self, sqlite_store_tmp, chroma_with_dim
    ):
        """全キャラが char_definitions に再登録され、status が維持されることを確認する。"""
        sqlite = sqlite_store_tmp
        chroma_old = chroma_with_dim(dim=4)

        _seed_characters_and_chat(
            sqlite,
            chroma_old,
            chars=[
                ("c-active", "Alice", "Alice の定義", "active"),
                ("c-estranged", "Bob", "Bob の定義", "estranged"),
                ("c-empty", "NoDef", "", "active"),  # 定義なしはスキップされるべき
            ],
            sessions=[],
            messages=[],
        )

        # 新次元に差し替えて再インデックス
        chroma_old._embedding_fn = _FixedDimEmbedding(dim=8)
        characters = sqlite.list_characters()
        _reindex_definition_collection(sqlite, chroma_old, characters)

        collection = chroma_old._get_definition_collection()
        # 定義あり 2 件 + 定義なし 0 件 = 2 件
        assert collection.count() == 2

        # 各キャラの metadata.status が維持されていることを確認
        all_data = collection.get(include=["metadatas"])
        id_to_status = {
            doc_id: meta.get("status")
            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"])
        }
        assert id_to_status.get("c-active") == "active"
        assert id_to_status.get("c-estranged") == "estranged"
        assert "c-empty" not in id_to_status

    def test_definition_collection_works_with_new_dimension(
        self, sqlite_store_tmp, chroma_with_dim
    ):
        """再インデックス後、新次元で query が成功すること（次元不整合エラーが出ないこと）を確認する。"""
        sqlite = sqlite_store_tmp
        chroma_old = chroma_with_dim(dim=4)

        sqlite.create_character(
            character_id="c1", name="Alpha", system_prompt_block1="アルファの定義"
        )
        chroma_old.upsert_character_definition("c1", "アルファの定義", status="active")

        # 新次元 16 に差し替えて再インデックス
        chroma_old._embedding_fn = _FixedDimEmbedding(dim=16)
        _reindex_definition_collection(sqlite, chroma_old, sqlite.list_characters())

        collection = chroma_old._get_definition_collection()
        # 16 次元のクエリで query が通れば次元不整合は解消されている
        result = collection.query(
            query_embeddings=[[0.1] * 16],
            n_results=1,
            include=["documents"],
        )
        assert result["ids"] and result["ids"][0]


# ---------------------------------------------------------------------------
# migrate_embeddings 経由の統合: 漏れ防止ガード
# ---------------------------------------------------------------------------


class TestMigrateEmbeddingsCoverage:
    """``migrate_embeddings`` が ``_reindex_chat_collections`` と
    ``_reindex_definition_collection`` を **必ず呼ぶ** ことを保証するガードテスト。

    将来このフックが消えたり呼び出し順が変わったりした場合に CI で気付けるようにする。
    """

    @pytest.mark.asyncio
    async def test_migrate_embeddings_invokes_chat_and_definitions_reindex(
        self, sqlite_store_tmp, chroma_with_dim, monkeypatch
    ):
        """``migrate_embeddings`` が記憶以外の 2 系統の再インデックス関数を呼ぶこと。

        実際の embedding API 呼び出しを避けるため、``get_embedding_function`` と
        2 つの再インデックス関数をモックでスパイする。
        """
        sqlite = sqlite_store_tmp
        chroma = chroma_with_dim(dim=4)

        # 検証用キャラを 1 件だけ作成（_do_migrate 内で characters を反復するため）
        sqlite.create_character(
            character_id="c-only", name="Solo", system_prompt_block1="solo"
        )

        called: dict[str, int] = {"chat": 0, "def": 0}

        def _spy_chat(_sqlite, _chroma):
            called["chat"] += 1

        def _spy_def(_sqlite, _chroma, _characters):
            called["def"] += 1

        # embedding 関数差し替えをスタブで済ませる
        def _fake_get_embedding_function(*args, **kwargs):
            return _FixedDimEmbedding(dim=8)

        monkeypatch.setattr(
            "backend.repositories.chroma.store.get_embedding_function",
            _fake_get_embedding_function,
        )
        monkeypatch.setattr(migration_service, "_reindex_chat_collections", _spy_chat)
        monkeypatch.setattr(
            migration_service, "_reindex_definition_collection", _spy_def
        )

        await migration_service.migrate_embeddings(
            sqlite=sqlite,
            old_chroma=chroma,
            chroma_db_path="(unused)",
            drift_manager=None,
            new_provider="default",
            new_model="",
            new_api_key="",
        )

        assert called["chat"] == 1, (
            "_reindex_chat_collections が呼ばれていない。chat_{id} の再インデックス漏れ。"
        )
        assert called["def"] == 1, (
            "_reindex_definition_collection が呼ばれていない。char_definitions の再インデックス漏れ。"
        )

    @pytest.mark.asyncio
    async def test_migrate_embeddings_acquires_write_lock(
        self, sqlite_store_tmp, chroma_with_dim, monkeypatch
    ):
        """``migrate_embeddings`` が ChromaStore._write_lock を取得して migration 全体を保護することを保証する。

        write_lock を取らないと、delete_collection と get_or_create_collection の間に
        他経路の書き込み（chat indexer / 同期 inscribe / リトライワーカー）が割り込んで
        HNSW 初期化が破損するシナリオが残る（欠陥 E）。
        ``with old_chroma._write_lock:`` が消えたり外に出たりしたら CI で即気付くための
        ガードテスト。RLock の生のオブジェクトを spy ラッパで包み、
        ``__enter__`` が migration 中に少なくとも 1 回呼ばれることを観測する。
        """
        sqlite = sqlite_store_tmp
        chroma = chroma_with_dim(dim=4)

        class _SpyLock:
            """RLock を委譲しながら ``__enter__`` 呼び出しを観測する spy。

            ChromaStore の write メソッドからは ``with chroma._write_lock:`` の形で
            使われるため、``__enter__`` / ``__exit__`` だけ実装すれば足りる。
            """

            def __init__(self, real):
                """委譲先の本物のロックを保持する。"""
                self._real = real
                self.enter_count = 0
                self.exit_count = 0

            def __enter__(self):
                self.enter_count += 1
                return self._real.__enter__()

            def __exit__(self, *args):
                self.exit_count += 1
                return self._real.__exit__(*args)

        spy_lock = _SpyLock(chroma._write_lock)
        chroma._write_lock = spy_lock  # type: ignore[assignment]

        # 実 embedding API を呼ばないようにスタブ化
        def _fake_get_embedding_function(*args, **kwargs):
            return _FixedDimEmbedding(dim=4)

        monkeypatch.setattr(
            "backend.repositories.chroma.store.get_embedding_function",
            _fake_get_embedding_function,
        )

        # 内部の chat / definitions 再インデックスは欠陥 E と直交するため空 spy で済ませる
        monkeypatch.setattr(migration_service, "_reindex_chat_collections", lambda *_a, **_kw: None)
        monkeypatch.setattr(migration_service, "_reindex_definition_collection", lambda *_a, **_kw: None)

        await migration_service.migrate_embeddings(
            sqlite=sqlite,
            old_chroma=chroma,
            chroma_db_path="(unused)",
            drift_manager=None,
            new_provider="default",
            new_model="",
            new_api_key="",
        )

        assert spy_lock.enter_count >= 1, (
            "migrate_embeddings 中に ChromaStore._write_lock が取得されていません。"
            "migration の atomicity が保証されていない可能性（欠陥 E 再発）。"
        )
        # acquire したロックは必ず release されること（exit が enter と対になっていること）
        assert spy_lock.exit_count == spy_lock.enter_count, (
            f"write_lock の enter/exit 数が一致しない: "
            f"enter={spy_lock.enter_count} exit={spy_lock.exit_count}"
        )
