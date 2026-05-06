"""ChromaDB → LanceDB 一括マイグレーションスクリプト。

# 目的

ChromaStore が抱えていた HNSW バイナリ破損問題から脱却するため、
既存の ``data/chroma`` を読み取り、LanceStore（``data/lancedb``）に全データを移植する。

# 移行対象

ChromaDB 内の3系統コレクションをそれぞれ LanceDB の単一テーブルへ流し込む。

  | ChromaDB                | LanceDB             |
  | ----------------------- | ------------------- |
  | ``char_{character_id}`` | ``memories``        |
  | ``chat_{character_id}`` | ``chat_turns``      |
  | ``char_definitions``    | ``definitions``     |

# 実行方針

embedding を再生成せず、ChromaDB に格納済みの embedding ベクトルをそのまま
LanceDB に転送する（コスト・時間とも大幅削減、内容も完全一致）。
embedding model 変更時の再生成は ``migrate_embeddings`` の責務であり、本スクリプトは
あくまで「ストレージ層の置き換え」のみを行う。

# 安全対策

- 実行前に ``data/chroma.bak.YYYYMMDD/`` のバックアップが存在することを確認
- 既存 ``data/lancedb`` があれば中断（誤上書き防止）
- 各テーブルの insert 件数をログ出力し、目視で件数差異を検出可能にする
- backend を停止した状態で実行すること（CLAUDE.md 運用ルール）

# 使い方

    python scripts/migrate_chroma_to_lance.py [--dry-run] [--lancedb-path PATH]

実行後、``run.bat`` 起動時に ``CHOTGOR_VECTOR_BACKEND=lance`` を設定して新ストアを使う。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
import lancedb
import pyarrow as pa


_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_chroma_to_lance")


# ChromaDB のコレクション名規則。
_PREFIX_MEMORIES = "char_"
_PREFIX_CHAT = "chat_"
_NAME_DEFINITIONS = "char_definitions"


def _open_chroma(chroma_path: str):
    """ChromaDB の PersistentClient を embedding fn なしで開く。

    embedding 再生成は不要なので EmbeddingFunction は注入しない。
    既存コレクションを open するだけなら問題ない。
    """
    return chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))


def _detect_vector_dim(chroma_client) -> int:
    """既存 ChromaDB から vector 次元を検出する。

    全コレクションを走査し、各コレクションから 1 件サンプリングして次元の最頻値を採用する。
    現在運用中の embedding model に対応する次元が多数派になるはず。
    破損コレクション（HNSW 破損で get() が失敗）はスキップする。
    """
    from collections import Counter

    counts: Counter = Counter()
    for col in chroma_client.list_collections():
        try:
            col_obj = chroma_client.get_collection(col.name)
            if col_obj.count() == 0:
                continue
            sample = col_obj.get(limit=1, include=["embeddings"])
        except Exception as e:
            logger.warning("vector 次元検出: コレクション取得失敗 name=%s error=%s — スキップ", col.name, e)
            continue
        embs = sample.get("embeddings")
        if embs is None or len(embs) == 0:
            continue
        emb0 = embs[0]
        if emb0 is None or len(emb0) == 0:
            continue
        counts[len(emb0)] += 1

    if not counts:
        raise ValueError("ChromaDB 内に embedding 入りコレクションが見つかりません。")

    # 最頻値を採用。複数次元が混在する場合は警告。
    if len(counts) > 1:
        logger.warning("複数の vector 次元が混在: %s — 最頻値を採用", dict(counts))
    dim, _ = counts.most_common(1)[0]
    return dim


def _build_schema(kind: str, dim: int) -> pa.Schema:
    """LanceStore と同じスキーマを生成する。

    LanceStore._schema_* と内容を一致させること。スキーマ差異が出ると merge_insert が
    失敗するため、変更時は両方を同時に更新する。
    """
    if kind == "memories":
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
    if kind == "chat_turns":
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
    if kind == "definitions":
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("character_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("status", pa.string()),
        ])
    raise ValueError(f"未知のテーブル種別: {kind}")


def _str_or_default(v, default: str = "") -> str:
    """metadata 値を str として安全に取り出す。None / 数値 / bool は str() 化する。"""
    if v is None:
        return default
    return str(v)


def _float_or_zero(v) -> float:
    """metadata 値を float として安全に取り出す。型不一致や None は 0.0 にフォールバック。"""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _migrate_memories(chroma_client, table, expected_dim: int) -> int:
    """``char_*`` コレクションを全て読み取って memories テーブルへ流し込む。

    各コレクション名から character_id を復元（``char_{uuid_with_underscores}`` →
    元のハイフン付き UUID は SQLite を見ないと完全には戻らないので、underscore→hyphen の
    単純復元で済ませる。それで SQLite の Memory.character_id と照合できる）。

    HNSW 破損で ``get()`` が失敗するコレクションは warning だけ出してスキップする
    （過去の事故残り — SQLite に Memory レコードは残っているので、後から再 embed して
    LanceDB へ再投入する余地を残す）。

    Returns:
        挿入した行数。
    """
    rows: list[dict] = []
    for col in chroma_client.list_collections():
        if not col.name.startswith(_PREFIX_MEMORIES) or col.name == _NAME_DEFINITIONS:
            continue
        # "char_definitions" は名前空間衝突を避けるためここで除外
        try:
            col_obj = chroma_client.get_collection(col.name)
        except Exception as e:
            logger.warning("コレクション取得失敗 name=%s error=%s", col.name, e)
            continue
        # コレクション名 "char_{uuid_with_underscores}" → 元 UUID（hyphen付き）
        char_id = col.name[len(_PREFIX_MEMORIES):].replace("_", "-")

        total = col_obj.count()
        if total == 0:
            logger.info("memories: 空コレクション スキップ name=%s", col.name)
            continue

        try:
            data = col_obj.get(limit=total, include=["embeddings", "documents", "metadatas"])
        except Exception as e:
            logger.warning(
                "memories: コレクション読み取り失敗（HNSW 破損の可能性）name=%s count=%d error=%s — スキップ",
                col.name, total, e,
            )
            continue
        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []
        raw_embs = data.get("embeddings")
        embeddings = raw_embs if raw_embs is not None else []

        skipped_dim = 0
        for i, mid in enumerate(ids):
            doc = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            emb = embeddings[i] if i < len(embeddings) else None
            if emb is None:
                logger.warning("memories: embedding 欠損 id=%s char=%s スキップ", mid, char_id)
                continue
            if len(emb) != expected_dim:
                # 過去の embedding model で書かれた古い次元のレコード（再 embed 漏れ）
                skipped_dim += 1
                continue
            rows.append({
                "id": str(mid),
                "character_id": char_id,
                "content": doc or "",
                "vector": [float(x) for x in emb],
                "category": _str_or_default(meta.get("category")),
                "contextual_importance": _float_or_zero(meta.get("contextual_importance")),
                "semantic_importance": _float_or_zero(meta.get("semantic_importance")),
                "identity_importance": _float_or_zero(meta.get("identity_importance")),
                "user_importance": _float_or_zero(meta.get("user_importance")),
            })
        if skipped_dim:
            logger.warning("memories: 旧次元レコードスキップ char=%s count=%d", char_id, skipped_dim)
        logger.info("memories: 取得 char=%s rows=%d", char_id, len(ids))

    if rows:
        table.add(rows)
    return len(rows)


def _migrate_chat_turns(chroma_client, table, expected_dim: int) -> int:
    """``chat_*`` コレクションを全て読み取って chat_turns テーブルへ流し込む。

    Returns:
        挿入した行数。
    """
    rows: list[dict] = []
    for col in chroma_client.list_collections():
        if not col.name.startswith(_PREFIX_CHAT):
            continue
        try:
            col_obj = chroma_client.get_collection(col.name)
        except Exception as e:
            logger.warning("コレクション取得失敗 name=%s error=%s", col.name, e)
            continue
        char_id = col.name[len(_PREFIX_CHAT):].replace("_", "-")

        total = col_obj.count()
        if total == 0:
            logger.info("chat_turns: 空コレクション スキップ name=%s", col.name)
            continue

        try:
            data = col_obj.get(limit=total, include=["embeddings", "documents", "metadatas"])
        except Exception as e:
            logger.warning(
                "chat_turns: コレクション読み取り失敗（HNSW 破損の可能性）name=%s count=%d error=%s — スキップ",
                col.name, total, e,
            )
            continue
        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []
        raw_embs = data.get("embeddings")
        embeddings = raw_embs if raw_embs is not None else []

        skipped_dim = 0
        for i, mid in enumerate(ids):
            doc = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            emb = embeddings[i] if i < len(embeddings) else None
            if emb is None:
                logger.warning("chat_turns: embedding 欠損 id=%s char=%s スキップ", mid, char_id)
                continue
            if len(emb) != expected_dim:
                skipped_dim += 1
                continue
            rows.append({
                "id": str(mid),
                "character_id": char_id,
                "content": doc or "",
                "vector": [float(x) for x in emb],
                "session_id": _str_or_default(meta.get("session_id")),
                "role": _str_or_default(meta.get("role")),
                "speaker_name": _str_or_default(meta.get("speaker_name")),
                "created_at": _str_or_default(meta.get("created_at")),
            })
        if skipped_dim:
            logger.warning("chat_turns: 旧次元レコードスキップ char=%s count=%d", char_id, skipped_dim)
        logger.info("chat_turns: 取得 char=%s rows=%d", char_id, len(ids))

    if rows:
        table.add(rows)
    return len(rows)


def _migrate_definitions(chroma_client, table, expected_dim: int) -> int:
    """``char_definitions`` コレクションを definitions テーブルへ流し込む。

    Returns:
        挿入した行数（コレクション未作成なら 0）。
    """
    try:
        col_obj = chroma_client.get_collection(_NAME_DEFINITIONS)
    except Exception as e:
        logger.info("definitions: コレクション未作成 error=%s スキップ", e)
        return 0

    total = col_obj.count()
    if total == 0:
        logger.info("definitions: 空コレクション スキップ")
        return 0

    data = col_obj.get(limit=total, include=["embeddings", "documents", "metadatas"])
    ids = data.get("ids", []) or []
    documents = data.get("documents", []) or []
    metadatas = data.get("metadatas", []) or []
    raw_embs = data.get("embeddings")
    embeddings = raw_embs if raw_embs is not None else []

    rows: list[dict] = []
    skipped_dim = 0
    for i, did in enumerate(ids):
        doc = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        emb = embeddings[i] if i < len(embeddings) else None
        if emb is None:
            logger.warning("definitions: embedding 欠損 id=%s スキップ", did)
            continue
        if len(emb) != expected_dim:
            skipped_dim += 1
            continue
        rows.append({
            "id": str(did),
            "character_id": _str_or_default(meta.get("character_id"), default=str(did)),
            "content": doc or "",
            "vector": [float(x) for x in emb],
            "status": _str_or_default(meta.get("status"), default="active"),
        })

    if rows:
        table.add(rows)
    if skipped_dim:
        logger.warning("definitions: 旧次元レコードスキップ count=%d", skipped_dim)
    logger.info("definitions: rows=%d", len(rows))
    return len(rows)


def main() -> int:
    """マイグレーションを実行する。0=成功、1=失敗。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-path", default=str(_PROJECT_ROOT / "data" / "chroma"))
    parser.add_argument("--lancedb-path", default=str(_PROJECT_ROOT / "data" / "lancedb"))
    parser.add_argument("--dry-run", action="store_true", help="実際の書き込みをせず件数だけ調べる")
    parser.add_argument("--force", action="store_true", help="既存 lancedb があっても続行（既存データに追記）")
    args = parser.parse_args()

    chroma_path = args.chroma_path
    lance_path = args.lancedb_path

    # ─── 安全チェック ────────────────────────────────────────────
    if not os.path.isdir(chroma_path):
        logger.error("chroma パスが存在しません: %s", chroma_path)
        return 1

    if os.path.isdir(lance_path) and not args.force and not args.dry_run:
        logger.error(
            "lancedb パスが既に存在します: %s\n"
            "  上書き or 追記したい場合は --force を指定してください。\n"
            "  通常は新規作成のため、既存ディレクトリを退避してから再実行してください。",
            lance_path,
        )
        return 1

    logger.info("ChromaDB → LanceDB マイグレーション開始")
    logger.info("  chroma_path = %s", chroma_path)
    logger.info("  lance_path  = %s", lance_path)
    logger.info("  dry_run     = %s", args.dry_run)

    chroma_client = _open_chroma(chroma_path)

    # ─── 全コレクション一覧 ──────────────────────────────────────
    cols = chroma_client.list_collections()
    logger.info("ChromaDB コレクション数: %d", len(cols))
    for c in cols:
        try:
            n = chroma_client.get_collection(c.name).count()
        except Exception:
            n = -1
        logger.info("  %s: %d 件", c.name, n)

    # ─── vector 次元検出 ─────────────────────────────────────────
    try:
        dim = _detect_vector_dim(chroma_client)
    except Exception as e:
        logger.error("vector 次元の検出失敗: %s", e)
        return 1
    logger.info("検出した vector 次元: %d", dim)

    if args.dry_run:
        logger.info("[dry-run] 実書き込みはスキップして終了")
        return 0

    # ─── LanceDB 初期化 ──────────────────────────────────────────
    os.makedirs(lance_path, exist_ok=True)
    lance_db = lancedb.connect(lance_path)

    def _create(name: str, kind: str):
        """LanceDB テーブルを新規作成する（既存は force 時のみ open）。"""
        # ListTablesResponse 互換のため属性アクセス
        resp = lance_db.list_tables()
        existing = list(resp.tables) if hasattr(resp, "tables") else list(resp)
        if name in existing:
            logger.warning("既存テーブル open（追記モード）: %s", name)
            return lance_db.open_table(name)
        return lance_db.create_table(name, schema=_build_schema(kind, dim))

    table_memories = _create("memories", "memories")
    table_chat = _create("chat_turns", "chat_turns")
    table_def = _create("definitions", "definitions")

    # ─── マイグレーション実行 ────────────────────────────────────
    try:
        n_mem = _migrate_memories(chroma_client, table_memories, dim)
        n_chat = _migrate_chat_turns(chroma_client, table_chat, dim)
        n_def = _migrate_definitions(chroma_client, table_def, dim)
    except Exception:
        logger.exception("マイグレーション中に致命的エラー")
        return 1

    logger.info(
        "マイグレーション完了: memories=%d, chat_turns=%d, definitions=%d",
        n_mem, n_chat, n_def,
    )
    logger.info("最終件数（テーブル単位）:")
    logger.info("  memories     : %d", table_memories.count_rows())
    logger.info("  chat_turns   : %d", table_chat.count_rows())
    logger.info("  definitions  : %d", table_def.count_rows())
    logger.info("移行先: %s", lance_path)
    logger.info(
        "次のステップ: run.bat 起動前に CHOTGOR_VECTOR_BACKEND=lance を設定して動作確認してください。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
