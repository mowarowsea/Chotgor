"""LanceDB によるベクトル永続化層 — ChromaStore の置き換え。

# 設計概要

ChromaDB の HNSW バイナリは multi-writer-unsafe で、書き込み中にプロセスが落ちると
インデックスが破損する持病があった（過去に「はる」のコレクション全消失事故あり）。
LanceDB は以下の特性により、この種の破損を構造的に回避する。

  - Lance フォーマット（列指向、追記ベース）+ アトミックなマニフェストコミット
  - 書き込みコンフリクト時は楽観ロックで失敗→リトライ可（黙って壊れない）
  - HNSW のような外部バイナリインデックスは持たず、ベクトルもデータファイルに同居

# テーブル構成（単一テーブル方式）

ChromaDB がキャラクターごとにコレクションを作っていたのと違い、LanceStore は
3用途それぞれを **単一テーブル + character_id カラムでフィルタ** する。
キャラクター数の増加でテーブル数が爆発せず、横断統計も1クエリで取れる。

  1. ``memories``    — 記憶コレクション（旧 ``char_{character_id}``）
  2. ``chat_turns``  — チャット履歴（旧 ``chat_{character_id}``）
  3. ``definitions`` — キャラクター定義（旧 ``char_definitions``）

# 互換性方針

ChromaStore のパブリックメソッドシグネチャを 1:1 で踏襲する。
``MemoryManager`` / ``inscriber`` / ``chat/indexer`` 等の呼び出し側に変更を強いない。
内部メソッド ``_get_collection`` 等を直接触っていた ``migration_service`` は
別途書き換える（呼び出し側修正は Phase 5 で実施）。

# Embedding 戦略

ChromaDB は ``EmbeddingFunction`` をコレクションに紐付けて自動 embed していたが、
LanceStore は **アプリ側で明示的に embed して vector カラムへ詰める**。
既存の ``InfinityEmbeddingFunction`` / ``GeminiEmbeddingFunction`` を流用する。

# Vector 次元の決定

テーブル作成時に vector の次元を固定する必要がある。LanceStore は
**初回 embed 時に dummy text を 1 回投げて次元を決定** し、テーブルを作成する。
既存テーブルを open する場合は schema から次元を読み取る。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import lancedb
import pyarrow as pa

from backend.repositories.embeddings import (
    GeminiEmbeddingFunction,
    InfinityEmbeddingFunction,
    get_embedding_function,
)


logger = logging.getLogger(__name__)


# テーブル名定数。単一テーブル方式のため固定。
_TABLE_MEMORIES = "memories"
_TABLE_CHAT_TURNS = "chat_turns"
_TABLE_DEFINITIONS = "definitions"
_TABLE_WM_THREADS = "wm_threads"


def _where_dict_to_sql(where: dict) -> str:
    """ChromaDB 形式の where dict を LanceDB の SQL 文字列に変換する。

    ChromaDB の where フィルタは ``{"category": "identity"}`` のような単純比較と、
    ``{"category": {"$ne": "identity"}}`` のような演算子付き比較をサポートする。
    LanceStore はこれを SQL 文字列（``category = 'identity'`` / ``category != 'identity'``）に
    変換して ``LanceQueryBuilder.where()`` に渡す。

    対応演算子: ``$eq`` / ``$ne`` / ``$gt`` / ``$gte`` / ``$lt`` / ``$lte`` / ``$in`` / ``$nin``。
    複数キーは AND 結合。

    Args:
        where: ChromaDB 形式の where 辞書。

    Returns:
        LanceDB の where 句に渡せる SQL 文字列。空辞書なら空文字列。
    """
    if not where:
        return ""

    op_map = {
        "$eq": "=",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
    }

    def _quote(v) -> str:
        """SQL リテラルとして安全にクオートする。"""
        if isinstance(v, str):
            # シングルクォートは2重化でエスケープ
            escaped = v.replace("'", "''")
            return f"'{escaped}'"
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if v is None:
            return "NULL"
        return str(v)

    clauses: list[str] = []
    for key, val in where.items():
        if isinstance(val, dict):
            # 演算子付き比較
            for op, operand in val.items():
                if op in op_map:
                    clauses.append(f"{key} {op_map[op]} {_quote(operand)}")
                elif op == "$in":
                    if not isinstance(operand, (list, tuple)) or not operand:
                        # 空集合は常に偽
                        clauses.append("1 = 0")
                    else:
                        items = ", ".join(_quote(x) for x in operand)
                        clauses.append(f"{key} IN ({items})")
                elif op == "$nin":
                    if not isinstance(operand, (list, tuple)) or not operand:
                        # 空集合の補集合は常に真
                        clauses.append("1 = 1")
                    else:
                        items = ", ".join(_quote(x) for x in operand)
                        clauses.append(f"{key} NOT IN ({items})")
                else:
                    raise ValueError(f"未対応の where 演算子: {op}")
        else:
            # 単純等価比較
            if val is None:
                clauses.append(f"{key} IS NULL")
            else:
                clauses.append(f"{key} = {_quote(val)}")

    return " AND ".join(clauses)


def _quote_id(value: str) -> str:
    """ID 文字列を SQL リテラルとして安全にクオートする。"""
    return "'" + value.replace("'", "''") + "'"


class LanceStore:
    """LanceDB 永続化ストア — ChromaStore のドロップイン置換。

    パブリックメソッドのシグネチャは ChromaStore と互換。内部実装のみ LanceDB で再構築している。
    Lance フォーマットの書き込み原子性により、ChromaStore で必要だった
    ``_safe_get_or_create_collection`` / ``rebuild_memory_collection`` 等の
    破損対応コードは不要になる。
    """

    def __init__(
        self,
        db_path: str,
        embedding_provider: str = "infinity",
        embedding_model: str = "",
        api_key: str = "",
        base_url: str = "http://localhost:7997",
        embedding_fn=None,
    ):
        """LanceDB クライアントを初期化する。

        テーブル作成は遅延（初回 add 時）。embedding fn の dim 決定が必要なため、
        コンストラクタ時点では embedding 解決のみ行う。

        Args:
            db_path: LanceDB データディレクトリのパス。
            embedding_provider: ``"infinity"`` / ``"google"``。
                ChromaDB 組み込みの ``"default"`` は LanceStore では非サポート
                （sentence-transformers 依存を持ち込まない方針のため）。
            embedding_model: embedding モデル ID。空ならプロバイダーのデフォルト。
            api_key: プロバイダーの API キー（infinity の場合は未使用）。
            base_url: infinity サーバーの BaseURL。
            embedding_fn: テスト用裏口。明示的に指定された場合は ``embedding_provider`` 等を
                無視してこのインスタンスを使う。本番コードでは指定しない。
        """
        os.makedirs(db_path, exist_ok=True)
        self._db_path = db_path
        self._db = lancedb.connect(db_path)

        if embedding_fn is not None:
            self._embedding_fn = embedding_fn
        else:
            self._embedding_fn = get_embedding_function(
                embedding_provider, embedding_model, api_key, base_url
            )
        if self._embedding_fn is None:
            raise ValueError(
                f"LanceStore は embedding_provider='{embedding_provider}' を非サポート。"
                "infinity または google を指定すること。"
            )

        # vector 次元は遅延決定する。既存テーブルがあれば open 時に schema から読む。
        self._vector_dim: Optional[int] = None

        # ChromaStore と互換のロック。LanceDB は書き込み原子性を保証するため
        # 本来は不要だが、テーブル作成時の dim 決定 race condition 回避と、
        # migration_service が ``_write_lock`` を直接参照しているため互換維持目的で残す。
        self._write_lock = threading.RLock()

    # ─── Embedding ヘルパ ─────────────────────────────────────────────

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文書としてのプレフィックスで embedding する。

        Infinity の ruri-v3 系は文書には ``文章: `` プレフィックスが必要。
        他プロバイダーは ``__call__`` がそのまま動く。
        """
        return list(self._embedding_fn(texts))

    def _embed_query(self, text: str) -> list[float]:
        """クエリとしてのプレフィックスで embedding する。

        Infinity の場合は ``クエリ: `` プレフィックスを使い、検索精度を上げる。
        他プロバイダーは文書 embedding と同じ。
        """
        if isinstance(self._embedding_fn, InfinityEmbeddingFunction):
            return list(self._embedding_fn.embed_query([text])[0])
        return list(self._embedding_fn([text])[0])

    def _ensure_vector_dim(self) -> int:
        """vector 次元を決定する（未決定なら dummy embed を1回叩く）。"""
        if self._vector_dim is not None:
            return self._vector_dim
        with self._write_lock:
            if self._vector_dim is None:
                vec = self._embed_documents(["__lancedb_init__"])[0]
                self._vector_dim = len(vec)
                logger.info("LanceStore: vector_dim 決定 dim=%d", self._vector_dim)
            return self._vector_dim

    def _list_table_names(self) -> list[str]:
        """LanceDB の ``list_tables()`` を文字列リストに正規化する。

        LanceDB のバージョンにより戻り値が ``list[str]`` だったり
        ``ListTablesResponse(tables=[...], page_token=...)`` だったりする。
        本ヘルパで吸収して、常に文字列リストを返す。
        """
        resp = self._db.list_tables()
        if hasattr(resp, "tables"):
            return list(resp.tables)
        return list(resp)

    # ─── スキーマ生成 ────────────────────────────────────────────────

    def _schema_memories(self, dim: int) -> pa.Schema:
        """記憶テーブルのスキーマ。"""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("character_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("category", pa.string()),
            pa.field("contextual_importance", pa.float32()),
            pa.field("semantic_importance", pa.float32()),
            pa.field("identity_importance", pa.float32()),
            pa.field("user_importance", pa.float32()),
        ])

    def _schema_chat_turns(self, dim: int) -> pa.Schema:
        """チャット履歴テーブルのスキーマ。"""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("character_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("session_id", pa.string()),
            pa.field("role", pa.string()),
            pa.field("speaker_name", pa.string()),
            pa.field("created_at", pa.string()),
        ])

    def _schema_definitions(self, dim: int) -> pa.Schema:
        """キャラクター定義テーブルのスキーマ。"""
        return pa.schema([
            pa.field("id", pa.string()),  # = character_id（一意制約代わり）
            pa.field("character_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("status", pa.string()),
        ])

    def _schema_wm_threads(self, dim: int) -> pa.Schema:
        """ワーキングメモリスレッドテーブルのスキーマ。

        ``content`` には embedding の素材（summary + 最新ポスト本文の結合）を格納する。
        heat 想起時に type / is_open でフィルタするため、両者をカラムに持つ。
        """
        return pa.schema([
            pa.field("id", pa.string()),  # = WorkingMemoryThread.id
            pa.field("character_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("type", pa.string()),
            pa.field("importance", pa.float32()),
            pa.field("is_open", pa.int32()),  # 1=Open, 0=Archived
        ])

    def _ensure_table(self, name: str):
        """指定テーブルを open する。存在しなければ schema を決めて新規作成する。

        既存テーブルを open した場合は、その vector 次元を _vector_dim にキャッシュする。
        embedding model が変わった等で次元ミスマッチが起きた場合は warning を出すが、
        破壊操作は行わない（ユーザが明示的に migration を走らせるべき）。
        """
        existing = self._list_table_names()
        if name in existing:
            tbl = self._db.open_table(name)
            # スキーマから vector 次元を読み取り、未確定ならキャッシュする
            if self._vector_dim is None:
                vfield = tbl.schema.field("vector")
                if pa.types.is_fixed_size_list(vfield.type):
                    self._vector_dim = vfield.type.list_size
            return tbl

        # 新規作成（dim を決める必要がある）
        with self._write_lock:
            # ロック取得後に再チェック（他スレッドが先に作っていたら open）
            if name in self._list_table_names():
                return self._db.open_table(name)
            dim = self._ensure_vector_dim()
            if name == _TABLE_MEMORIES:
                schema = self._schema_memories(dim)
            elif name == _TABLE_CHAT_TURNS:
                schema = self._schema_chat_turns(dim)
            elif name == _TABLE_DEFINITIONS:
                schema = self._schema_definitions(dim)
            elif name == _TABLE_WM_THREADS:
                schema = self._schema_wm_threads(dim)
            else:
                raise ValueError(f"未知のテーブル名: {name}")
            tbl = self._db.create_table(name, schema=schema)
            logger.info("LanceStore: テーブル作成 name=%s dim=%d", name, dim)
            return tbl

    # ─── 記憶コレクション ────────────────────────────────────────────

    def add_memory(
        self,
        memory_id: str,
        content: str,
        character_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """記憶を ``memories`` テーブルに upsert する（merge_insert）。

        ChromaStore.add_memory と互換シグネチャ。同一 ID が存在すれば更新、なければ挿入。

        Args:
            memory_id: 記憶の一意 ID（SQLite の Memory.id と同一）。
            content: embedding するテキスト本文。
            character_id: キャラクター ID。
            metadata: 追加メタデータ。``category`` / ``*_importance`` を読み取る。
        """
        meta = metadata or {}
        with self._write_lock:
            tbl = self._ensure_table(_TABLE_MEMORIES)
            vec = self._embed_documents([content])[0]
            row = {
                "id": memory_id,
                "character_id": character_id,
                "content": content,
                "vector": vec,
                "category": str(meta.get("category", "")),
                "contextual_importance": float(meta.get("contextual_importance", 0.0)),
                "semantic_importance": float(meta.get("semantic_importance", 0.0)),
                "identity_importance": float(meta.get("identity_importance", 0.0)),
                "user_importance": float(meta.get("user_importance", 0.0)),
            }
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )

    def recall_memory(
        self,
        query: str,
        character_id: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """類似度検索で記憶を取得する。

        ChromaStore.recall_memory と互換戻り値（``id`` / ``content`` / ``distance`` / ``metadata``）。
        距離は cosine（0=同一、2=対極）で揃える（ChromaDB の hnsw:space=cosine と等価）。

        Args:
            query: 検索クエリテキスト。
            character_id: キャラクター ID（必ず character_id でフィルタする）。
            top_k: 取得する最大件数。
            where: 追加 where 辞書（カテゴリ絞り込み等）。

        Returns:
            id / content / distance / metadata のリスト。
        """
        # テーブルが存在しないなら結果は空（embed すら呼ばない）
        if _TABLE_MEMORIES not in self._list_table_names():
            return []
        tbl = self._db.open_table(_TABLE_MEMORIES)
        if tbl.count_rows() == 0:
            return []

        vec = self._embed_query(query)
        sql_clauses = [f"character_id = {_quote_id(character_id)}"]
        if where:
            extra = _where_dict_to_sql(where)
            if extra:
                sql_clauses.append(f"({extra})")
        full_where = " AND ".join(sql_clauses)

        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where(full_where, prefilter=True)
                .limit(top_k)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning(
                "LanceStore.recall_memory 失敗 char=%s where=%s error=%s",
                character_id, where, e,
            )
            return []

        memories = []
        for r in results:
            memories.append({
                "id": r["id"],
                "content": r["content"],
                "distance": float(r.get("_distance", 0.0)),
                "metadata": {
                    "character_id": r["character_id"],
                    "category": r.get("category", ""),
                    "contextual_importance": r.get("contextual_importance"),
                    "semantic_importance": r.get("semantic_importance"),
                    "identity_importance": r.get("identity_importance"),
                    "user_importance": r.get("user_importance"),
                },
            })
        return memories

    def find_similar_in_category(
        self,
        content: str,
        character_id: str,
        category: str,
        threshold: float = 0.15,
    ) -> Optional[str]:
        """同一キャラクター・カテゴリ内で類似する記憶 ID を返す（重複排除用）。

        find_similar 系は「文書同士の比較」なので、検索クエリも文書プレフィックスで embed する
        （ChromaStore の as_document=True と等価な扱い）。

        Args:
            content: 検索クエリとなる新しい記憶テキスト。
            character_id: キャラクター ID。
            category: 検索対象のカテゴリ。
            threshold: 更新判定のコサイン距離しきい値。

        Returns:
            類似記憶が見つかった場合はその memory_id、見つからなければ None。
        """
        if _TABLE_MEMORIES not in self._list_table_names():
            return None
        tbl = self._db.open_table(_TABLE_MEMORIES)
        if tbl.count_rows() == 0:
            return None

        # 文書対文書比較なので _embed_documents を使う
        vec = self._embed_documents([content])[0]
        full_where = f"character_id = {_quote_id(character_id)} AND category = {_quote_id(category)}"
        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where(full_where, prefilter=True)
                .limit(1)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning(
                "LanceStore.find_similar_in_category 失敗 char=%s category=%s error=%s",
                character_id, category, e,
            )
            return None

        if not results:
            return None
        if float(results[0].get("_distance", 1.0)) < threshold:
            return results[0]["id"]
        return None

    def delete_memory(self, memory_id: str, character_id: str) -> None:
        """指定 ID の記憶を物理削除する。

        ChromaStore.delete_memory と互換。character_id は API 互換のため受け取るが、
        単一テーブルなので id だけで一意に特定できる。

        Args:
            memory_id: 削除する記憶 ID。
            character_id: キャラクター ID（互換シグネチャ用）。
        """
        if _TABLE_MEMORIES not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_MEMORIES)
            tbl.delete(f"id = {_quote_id(memory_id)}")

    def delete_all_memories(self, character_id: str) -> None:
        """指定キャラクターの全記憶を削除する（キャラクター削除時に使用）。

        ChromaStore はコレクションごと消していたが、LanceStore は単一テーブル方式なので
        ``character_id`` でフィルタした行を削除する。テーブル自体は維持する。
        """
        if _TABLE_MEMORIES not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_MEMORIES)
            tbl.delete(f"character_id = {_quote_id(character_id)}")

    def rebuild_memory_collection(self, character_id: str) -> int:
        """互換性のためのスタブ。LanceDB は HNSW 破損が起きないため実体不要。

        ChromaStore では HNSW 破損時の緊急再構築用だったが、LanceDB の Lance フォーマットは
        書き込みが追記＋アトミックコミットで完結するため、この種の再構築は不要。
        呼び出されてもログだけ出して該当キャラの行数を返す。
        """
        if _TABLE_MEMORIES not in self._list_table_names():
            return 0
        tbl = self._db.open_table(_TABLE_MEMORIES)
        try:
            n = len(
                tbl.search().where(
                    f"character_id = {_quote_id(character_id)}", prefilter=True
                ).limit(10**9).to_arrow()
            )
        except Exception:
            n = 0
        logger.info("LanceStore.rebuild_memory_collection スキップ（不要） char=%s count=%d", character_id, n)
        return n

    # ─── チャット履歴コレクション ────────────────────────────────────

    def add_chat_turn(
        self,
        message_id: str,
        content: str,
        character_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """チャット履歴ターンを ``chat_turns`` テーブルに upsert する。

        Args:
            message_id: SQLite の chat_messages.id と同一。
            content: ``"{speaker_name}: {content}"`` 形式のテキスト。
            character_id: 発言を所属させるキャラクター ID。
            metadata: ``session_id`` / ``role`` / ``speaker_name`` / ``created_at`` を読む。
        """
        meta = metadata or {}
        with self._write_lock:
            tbl = self._ensure_table(_TABLE_CHAT_TURNS)
            vec = self._embed_documents([content])[0]
            row = {
                "id": message_id,
                "character_id": character_id,
                "content": content,
                "vector": vec,
                "session_id": str(meta.get("session_id", "")),
                "role": str(meta.get("role", "")),
                "speaker_name": str(meta.get("speaker_name", "")),
                "created_at": str(meta.get("created_at", "")),
            }
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )

    def recall_chat_turns(
        self,
        query: str,
        character_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """類似度検索でチャット履歴ターンを取得する（PowerRecall 用）。

        ChromaStore.recall_chat_turns と互換戻り値。

        Args:
            query: 検索クエリテキスト。
            character_id: キャラクター ID。
            top_k: 取得する最大件数。

        Returns:
            id / content / distance / metadata のリスト。
        """
        if _TABLE_CHAT_TURNS not in self._list_table_names():
            return []
        tbl = self._db.open_table(_TABLE_CHAT_TURNS)
        if tbl.count_rows() == 0:
            return []

        vec = self._embed_query(query)
        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where(f"character_id = {_quote_id(character_id)}", prefilter=True)
                .limit(top_k)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning("LanceStore.recall_chat_turns 失敗 char=%s error=%s", character_id, e)
            return []

        turns = []
        for r in results:
            turns.append({
                "id": r["id"],
                "content": r["content"],
                "distance": float(r.get("_distance", 0.0)),
                "metadata": {
                    "character_id": r["character_id"],
                    "session_id": r.get("session_id", ""),
                    "role": r.get("role", ""),
                    "speaker_name": r.get("speaker_name", ""),
                    "created_at": r.get("created_at", ""),
                },
            })
        return turns

    # ─── キャラクター定義 ────────────────────────────────────────────

    def upsert_character_definition(
        self,
        character_id: str,
        definition_text: str,
        status: str = "active",
    ) -> str:
        """キャラクター定義を ``definitions`` テーブルに upsert する。

        Args:
            character_id: キャラクター ID（doc ID として使用）。
            definition_text: キャラクター定義テキスト（system_prompt_block1）。
            status: ``"active"`` または ``"estranged"``。

        Returns:
            upsert した doc ID（character_id と同一）。
        """
        with self._write_lock:
            tbl = self._ensure_table(_TABLE_DEFINITIONS)
            vec = self._embed_documents([definition_text])[0]
            row = {
                "id": character_id,
                "character_id": character_id,
                "content": definition_text,
                "vector": vec,
                "status": status,
            }
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )
        return character_id

    def find_similar_definition(
        self,
        definition_text: str,
        exclude_character_id: str = "",
        threshold: float = 0.1,
    ) -> list[dict]:
        """estranged キャラクターの定義と類似するものを返す。

        キャラクター再作成による「なかったことにする」防止のため、status='estranged' に
        絞り込んで類似検索する。

        Args:
            definition_text: 比較対象の定義テキスト。
            exclude_character_id: 除外するキャラクター ID（更新時に自分自身を除外）。
            threshold: コサイン距離しきい値。

        Returns:
            類似する estranged キャラクターの情報リスト（character_id, distance）。
        """
        if _TABLE_DEFINITIONS not in self._list_table_names():
            return []
        tbl = self._db.open_table(_TABLE_DEFINITIONS)
        if tbl.count_rows() == 0:
            return []

        vec = self._embed_documents([definition_text])[0]
        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where("status = 'estranged'", prefilter=True)
                .limit(5)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning("LanceStore.find_similar_definition 失敗 error=%s", e)
            return []

        similar = []
        for r in results:
            if r["character_id"] == exclude_character_id:
                continue
            dist = float(r.get("_distance", 1.0))
            if dist < threshold:
                similar.append({
                    "character_id": r["character_id"],
                    "distance": dist,
                })
        return similar

    def mark_definition_estranged(self, character_id: str) -> None:
        """キャラクター定義の status を ``estranged`` に更新する。"""
        if _TABLE_DEFINITIONS not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_DEFINITIONS)
            try:
                tbl.update(
                    where=f"character_id = {_quote_id(character_id)}",
                    values={"status": "estranged"},
                )
            except Exception as e:
                logger.warning(
                    "LanceStore.mark_definition_estranged 更新失敗 char=%s error=%s",
                    character_id, e,
                )

    # ─── ワーキングメモリスレッド ────────────────────────────────────

    def upsert_wm_thread(
        self,
        thread_id: str,
        index_text: str,
        character_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """ワーキングメモリスレッドを ``wm_threads`` テーブルに upsert する。

        ``index_text`` は embedding の素材。スレッドの summary 単独では中身に
        強く関連するが summary に出てこない語の想起精度が落ちるため、
        ``summary + 最新ポスト本文`` を結合したテキストを渡すこと。
        summary 更新時・ポスト追加時のどちらでも呼ばれる。

        Args:
            thread_id: スレッドの一意 ID（SQLite の WorkingMemoryThread.id と同一）。
            index_text: embedding する素材テキスト（summary + 最新ポスト本文）。
            character_id: キャラクター ID。
            metadata: ``type`` / ``importance`` / ``is_open`` を読み取る。
        """
        meta = metadata or {}
        with self._write_lock:
            tbl = self._ensure_table(_TABLE_WM_THREADS)
            vec = self._embed_documents([index_text])[0]
            row = {
                "id": thread_id,
                "character_id": character_id,
                "content": index_text,
                "vector": vec,
                "type": str(meta.get("type", "")),
                "importance": float(meta.get("importance", 0.5)),
                "is_open": int(meta.get("is_open", 1)),
            }
            (
                tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )

    def recall_wm_threads(
        self,
        query: str,
        character_id: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """類似度検索でワーキングメモリスレッドを取得する（heat 想起用）。

        戻り値の ``distance`` は cosine 距離。呼び出し側（WorkingMemoryManager）が
        ``relevance`` に変換し、importance × 時間減衰と乗じて heat を算出する。

        Args:
            query: 検索クエリテキスト（直近のユーザー発言など）。
            character_id: キャラクター ID。
            top_k: 取得する最大件数。
            where: 追加 where 辞書（``{"type": {"$in": [...]}, "is_open": 1}`` 等）。

        Returns:
            id / content / distance / metadata（type / importance / is_open）のリスト。
        """
        if _TABLE_WM_THREADS not in self._list_table_names():
            return []
        tbl = self._db.open_table(_TABLE_WM_THREADS)
        if tbl.count_rows() == 0:
            return []

        vec = self._embed_query(query)
        sql_clauses = [f"character_id = {_quote_id(character_id)}"]
        if where:
            extra = _where_dict_to_sql(where)
            if extra:
                sql_clauses.append(f"({extra})")
        full_where = " AND ".join(sql_clauses)

        try:
            results = (
                tbl.search(vec)
                .metric("cosine")
                .where(full_where, prefilter=True)
                .limit(top_k)
                .to_arrow()
                .to_pylist()
            )
        except Exception as e:
            logger.warning(
                "LanceStore.recall_wm_threads 失敗 char=%s where=%s error=%s",
                character_id, where, e,
            )
            return []

        threads = []
        for r in results:
            threads.append({
                "id": r["id"],
                "content": r["content"],
                "distance": float(r.get("_distance", 0.0)),
                "metadata": {
                    "character_id": r["character_id"],
                    "type": r.get("type", ""),
                    "importance": r.get("importance"),
                    "is_open": r.get("is_open"),
                },
            })
        return threads

    def delete_wm_thread(self, thread_id: str) -> None:
        """指定 ID のワーキングメモリスレッドを物理削除する（スレッド削除時に使用）。"""
        if _TABLE_WM_THREADS not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_WM_THREADS)
            tbl.delete(f"id = {_quote_id(thread_id)}")

    def delete_all_wm_threads(self, character_id: str) -> None:
        """指定キャラクターの全スレッドを削除する（キャラクター削除時に使用）。"""
        if _TABLE_WM_THREADS not in self._list_table_names():
            return
        with self._write_lock:
            tbl = self._db.open_table(_TABLE_WM_THREADS)
            tbl.delete(f"character_id = {_quote_id(character_id)}")

    # ─── 全件再インデックス（embedding model 変更時） ─────────────

    def reindex_all(
        self,
        new_embedding_fn,
        sqlite,
    ) -> dict:
        """embedding model 変更時に全テーブルを drop → 新 embedding で再構築する。

        旧 ChromaStore 時代の ``migrate_embeddings`` を、LanceStore の単純な構造に合わせて
        再実装したもの。SQLite を source of truth として全データを読み直し、
        新 embedding で再 embed して LanceStore に流し込む。

        - memories       : SQLite ``Memory`` テーブルから全アクティブ記憶を取得して再 embed
        - chat_turns     : SQLite ``chat_messages`` から全メッセージを取得して再 embed
        - definitions    : SQLite ``characters`` から system_prompt_block1 を取得して再 embed
        - wm_threads     : SQLite ``wm_threads`` から全スレッドを取得し summary + 最新ポストで再 embed

        Args:
            new_embedding_fn: 新しい embedding function（``__call__`` を持つ）。
            sqlite: SQLiteStore インスタンス。

        Returns:
            ``{"memories": int, "chat_turns": int, "definitions": int, "wm_threads": int}`` の件数辞書。
        """
        from backend.services.chat.indexer import build_chat_doc_and_metadata, get_participant_char_ids

        with self._write_lock:
            # 1. 既存テーブルを全 drop（vector 次元が変わる可能性があるため）
            for name in (_TABLE_MEMORIES, _TABLE_CHAT_TURNS, _TABLE_DEFINITIONS, _TABLE_WM_THREADS):
                if name in self._list_table_names():
                    self._db.drop_table(name)
                    logger.info("LanceStore.reindex_all: drop %s", name)

            # 2. embedding fn 差し替え + 次元再決定
            self._embedding_fn = new_embedding_fn
            self._vector_dim = None  # 次元を再決定させる

            counts = {"memories": 0, "chat_turns": 0, "definitions": 0, "wm_threads": 0}

            # 3. memories 再構築
            characters = sqlite.list_characters()
            for char in characters:
                memories = sqlite.get_all_active_memories(char.id)
                for m in memories:
                    self.add_memory(
                        memory_id=m.id,
                        content=m.content,
                        character_id=char.id,
                        metadata={
                            "category": m.memory_category or "",
                            "contextual_importance": m.contextual_importance,
                            "semantic_importance": m.semantic_importance,
                            "identity_importance": m.identity_importance,
                            "user_importance": m.user_importance,
                        },
                    )
                    counts["memories"] += 1
                if memories:
                    logger.info("reindex_all(memories): char=%s count=%d", char.name, len(memories))

            # 4. chat_turns 再構築
            sessions = sqlite.list_chat_sessions(limit=1_000_000)
            for session in sessions:
                participant_ids = get_participant_char_ids(session, sqlite)
                if not participant_ids:
                    continue
                msgs = sqlite.list_chat_messages(session.id)
                for msg in msgs:
                    built = build_chat_doc_and_metadata(msg)
                    if built is None:
                        continue
                    doc, meta = built
                    for char_id in participant_ids:
                        self.add_chat_turn(
                            message_id=msg.id,
                            content=doc,
                            character_id=char_id,
                            metadata=meta,
                        )
                        counts["chat_turns"] += 1

            # 5. definitions 再構築
            for char in characters:
                text = (getattr(char, "system_prompt_block1", "") or "").strip()
                if not text:
                    continue
                status = getattr(char, "relationship_status", "active") or "active"
                self.upsert_character_definition(char.id, text, status=status)
                counts["definitions"] += 1

            # 6. wm_threads 再構築（summary + 最新ポスト本文を embedding 素材とする）
            for char in characters:
                threads = sqlite.list_wm_threads(char.id)
                for t in threads:
                    latest = sqlite.get_latest_wm_post(t.id)
                    index_text = (t.summary or "").strip()
                    if latest and latest.content:
                        index_text = (index_text + "\n" + latest.content).strip()
                    if not index_text:
                        continue
                    self.upsert_wm_thread(
                        thread_id=t.id,
                        index_text=index_text,
                        character_id=char.id,
                        metadata={
                            "type": t.type,
                            "importance": t.importance,
                            "is_open": t.is_open,
                        },
                    )
                    counts["wm_threads"] += 1

            logger.info("LanceStore.reindex_all 完了 counts=%s", counts)
            return counts
