"""LanceDB によるベクトル永続化層。

# 設計概要

Lance フォーマット（列指向、追記ベース）+ アトミックなマニフェストコミットにより、
multi-writer 環境でも書き込みが原子的で、HNSW のような外部バイナリインデックスを
持たない構造的に堅牢なベクトルストア。

# テーブル構成（単一テーブル方式）

用途別に **単一テーブル + character_id カラムでフィルタ** する。
キャラクター数の増加でテーブル数が爆発せず、横断統計も1クエリで取れる。

  1. ``inscribed_memories``     — 保存記憶コレクション
  2. ``chat_turns``             — チャット履歴
  3. ``definitions``            — キャラクター定義
  4. ``working_memory_threads`` — ワーキングメモリスレッドの index 用ベクトル

# Embedding 戦略

アプリ側で明示的に embed して vector カラムへ詰める。
``InfinityEmbeddingFunction`` / ``GeminiEmbeddingFunction`` のいずれかを使う。

# Vector 次元の決定

テーブル作成時に vector の次元を固定する必要がある。LanceStore は
**初回 embed 時に dummy text を 1 回投げて次元を決定** し、テーブルを作成する。
既存テーブルを open する場合は schema から次元を読み取る。
"""

from __future__ import annotations

import logging
import os
import threading
import lancedb
import pyarrow as pa

from backend.repositories.embeddings import (
    GeminiEmbeddingFunction,
    InfinityEmbeddingFunction,
    get_embedding_function,
)


logger = logging.getLogger(__name__)


# テーブル名定数。単一テーブル方式のため固定。
_TABLE_INSCRIBED_MEMORIES = "inscribed_memories"
_TABLE_CHAT_TURNS = "chat_turns"
_TABLE_DEFINITIONS = "definitions"
_TABLE_WORKING_MEMORY_THREADS = "working_memory_threads"

# reindex_all で一度に embedding サーバへ送るテキスト件数。
# infinity (ruri-v3 等) は 64〜128件程度のバッチでも問題なく処理できる。
# 大きくしすぎると単一リクエストのレイテンシが増えて進捗ログの粒度が粗くなるため、64 に設定。
_REINDEX_BATCH_SIZE = 64


class EmbeddingError(Exception):
    """embedding（ベクトル化）処理の失敗を表す例外。

    infinity サーバ未起動・接続失敗・タイムアウト・HTTPエラーなど、
    クエリや文書のベクトル化に失敗したケースを「想起そのものの失敗」と区別するために使う。
    上位（service 層）はこの例外型を見て、UI へ embedding 起因のエラーであることを伝える。
    """


def _where_dict_to_sql(where: dict) -> str:
    """where dict を LanceDB の SQL 文字列に変換する。

    where フィルタは ``{"category": "identity"}`` のような単純比較と、
    ``{"category": {"$ne": "identity"}}`` のような演算子付き比較をサポートする。
    SQL 文字列（``category = 'identity'`` / ``category != 'identity'``）に
    変換して ``LanceQueryBuilder.where()`` に渡す。

    対応演算子: ``$eq`` / ``$ne`` / ``$gt`` / ``$gte`` / ``$lt`` / ``$lte`` / ``$in`` / ``$nin``。
    複数キーは AND 結合。

    Args:
        where: where 辞書（``$eq`` 等の演算子付きを含む）。

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


class LanceStoreBase:
    """LanceDB 永続化ストアの基盤クラス。

    DB接続・embedding 関数・テーブルスキーマ定義・テーブル生成を担う。
    各テーブルへの CRUD 操作は ops ミックスイン（inscribed_ops.py ほか）が提供し、
    store.py の LanceStore が全てを合成する。

    Lance フォーマットの書き込み原子性により、書き込み中にプロセスが落ちても
    インデックスが破損しない。バックグラウンドのリトライキューや破損対応の
    再構築機構は不要。
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
                sentence-transformers 系の組み込み ``"default"`` プロバイダーは
                依存を増やさない方針のため非サポート。
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
        self._vector_dim: int | None = None

        # テーブル作成時の dim 決定 race condition 回避用のロック。
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

        embedding サーバへの接続失敗等は ``EmbeddingError`` に変換して送出する。
        想起呼び出し元（service 層）がこの型を見て、UI に embedding 起因のエラーを伝えるため。
        """
        try:
            if isinstance(self._embedding_fn, InfinityEmbeddingFunction):
                return list(self._embedding_fn.embed_query([text])[0])
            return list(self._embedding_fn([text])[0])
        except Exception as e:
            raise EmbeddingError(str(e)) from e

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

    def _schema_inscribed_memories(self, dim: int) -> pa.Schema:
        """保存記憶テーブルのスキーマ。"""
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

    def _schema_working_memory_threads(self, dim: int) -> pa.Schema:
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
        破壊操作は行わない（ユーザが明示的に再インデックスを走らせるべき）。
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
            if name == _TABLE_INSCRIBED_MEMORIES:
                schema = self._schema_inscribed_memories(dim)
            elif name == _TABLE_CHAT_TURNS:
                schema = self._schema_chat_turns(dim)
            elif name == _TABLE_DEFINITIONS:
                schema = self._schema_definitions(dim)
            elif name == _TABLE_WORKING_MEMORY_THREADS:
                schema = self._schema_working_memory_threads(dim)
            else:
                raise ValueError(f"未知のテーブル名: {name}")
            tbl = self._db.create_table(name, schema=schema)
            logger.info("LanceStore: テーブル作成 name=%s dim=%d", name, dim)
            return tbl

    # ─── 保存記憶コレクション ────────────────────────────────────────────

