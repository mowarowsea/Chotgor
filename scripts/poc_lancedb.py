"""LanceDB PoC — Windowsホスト上での動作確認スクリプト。

Phase 0 として、ChromaDB置き換え候補のLanceDBが以下の機能で動くことを検証する。
失敗した場合はLanceDB移行計画自体を見直す必要がある。

検証項目:
1. lancedb.connect() がWindowsローカルパスで動作する
2. create_table（pyarrow schema指定、exist_ok=True）
3. add（バッチ insert）
4. merge_insert（upsert 相当）
5. search（vector similarity）
6. where filter（SQL文字列）
7. delete（条件指定）
8. drop_table
9. プロセスを再openしてもデータが永続化されている
"""

import shutil
import os
import sys
from pathlib import Path

import lancedb
import pyarrow as pa


# 検証用一時ディレクトリ（プロジェクトルート相対）
POC_DIR = Path(__file__).resolve().parent.parent / "data" / "lancedb_poc"


def cleanup() -> None:
    """前回のPoC残骸を削除する。"""
    if POC_DIR.exists():
        shutil.rmtree(POC_DIR)
    POC_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    """セクション見出しを出力する。"""
    print(f"\n=== {title} ===")


def main() -> int:
    """PoCを実行し、全項目pass で 0、失敗で 1 を返す。"""
    cleanup()
    print(f"PoCディレクトリ: {POC_DIR}")
    print(f"lancedb version: {lancedb.__version__}")
    print(f"pyarrow version: {pa.__version__}")

    # ─── 1. connect ───────────────────────────────────────────────
    section("1. lancedb.connect()")
    db = lancedb.connect(str(POC_DIR))
    print(f"  ok: db type = {type(db).__name__}")

    # ─── 2. create_table（schema指定） ────────────────────────────
    section("2. create_table with pyarrow schema")
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("character_id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 4)),  # 4次元の小さなベクトルでテスト
        pa.field("category", pa.string()),
        pa.field("importance", pa.float32()),
        pa.field("deleted_at", pa.timestamp("us"), nullable=True),
    ])
    table = db.create_table("memories", schema=schema, exist_ok=True)
    print(f"  ok: table = {table.name}")

    # ─── 3. add（バッチinsert） ────────────────────────────────────
    section("3. add (batch insert)")
    rows = [
        {
            "id": "m1", "character_id": "char_a", "content": "コーヒーが好き",
            "vector": [0.1, 0.2, 0.3, 0.4], "category": "user", "importance": 0.5,
            "deleted_at": None,
        },
        {
            "id": "m2", "character_id": "char_a", "content": "犬を飼っている",
            "vector": [0.9, 0.8, 0.1, 0.0], "category": "user", "importance": 0.7,
            "deleted_at": None,
        },
        {
            "id": "m3", "character_id": "char_b", "content": "私は穏やかな性格",
            "vector": [0.1, 0.21, 0.3, 0.41], "category": "identity", "importance": 0.9,
            "deleted_at": None,
        },
    ]
    table.add(rows)
    print(f"  ok: row count = {table.count_rows()}")

    # ─── 4. merge_insert (upsert) ─────────────────────────────────
    section("4. merge_insert (upsert)")
    update_rows = [
        {
            "id": "m1", "character_id": "char_a", "content": "カフェラテが大好き（更新）",
            "vector": [0.15, 0.25, 0.35, 0.45], "category": "user", "importance": 0.8,
            "deleted_at": None,
        },
        {
            "id": "m4", "character_id": "char_a", "content": "新規追加レコード",
            "vector": [0.5, 0.5, 0.5, 0.5], "category": "contextual", "importance": 0.4,
            "deleted_at": None,
        },
    ]
    (
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(update_rows)
    )
    after = table.to_arrow().to_pylist()
    after_by_id = {r["id"]: r for r in after}
    print(f"  ok: row count = {table.count_rows()}")
    print(f"  m1 content = {after_by_id['m1']['content']!r}")
    print(f"  m4 exists = {'m4' in after_by_id}")

    # ─── 5. search（vector similarity） ──────────────────────────
    section("5. vector search")
    results = (
        table.search([0.1, 0.2, 0.3, 0.4])
        .limit(3)
        .to_arrow()
        .to_pylist()
    )
    print(f"  ok: result count = {len(results)}")
    for r in results:
        print(f"  id={r['id']} char={r['character_id']} dist={r['_distance']:.4f} content={r['content']!r}")

    # ─── 6. where filter (SQL文字列) ─────────────────────────────
    section("6. where filter")
    filtered = (
        table.search([0.1, 0.2, 0.3, 0.4])
        .where("character_id = 'char_a' AND category = 'user'")
        .limit(5)
        .to_arrow()
        .to_pylist()
    )
    print(f"  ok: filtered count = {len(filtered)}")
    for r in filtered:
        print(f"  id={r['id']} category={r['category']}")

    # ─── 7. delete（条件指定） ────────────────────────────────────
    section("7. delete by condition")
    table.delete("id = 'm4'")
    print(f"  ok: row count after delete = {table.count_rows()}")

    # ─── 8. プロセス再open ───────────────────────────────────────
    section("8. reconnect (永続化確認)")
    del db, table
    db2 = lancedb.connect(str(POC_DIR))
    table2 = db2.open_table("memories")
    print(f"  ok: reopened row count = {table2.count_rows()}")
    names = db2.list_tables()
    print(f"  table names: {names}")

    # ─── 9. drop_table ───────────────────────────────────────────
    section("9. drop_table")
    db2.drop_table("memories")
    names_after = db2.list_tables()
    print(f"  ok: tables after drop = {names_after}")

    # ─── 10. 後始末 ───────────────────────────────────────────────
    section("10. cleanup")
    del db2, table2
    try:
        shutil.rmtree(POC_DIR)
        print(f"  ok: removed {POC_DIR}")
    except Exception as e:
        print(f"  warn: rmtree failed (Windows file lock?): {e}")

    print("\n[PoC: 全項目 PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
