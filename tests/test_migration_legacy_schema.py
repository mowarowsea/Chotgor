"""
旧スキーマからのマイグレーション互換性テスト。

以下の問題への回帰テストを含む:
  - characters テーブルに旧カラム（meta_instructions NOT NULL DEFAULT なし）が残っている場合、
    SQLiteStore 初期化時のマイグレーションで現行スキーマに揃えられること。
  - マイグレーション後、create_character が NOT NULL 制約エラーなく成功すること。
  - 旧カラムに保存されていた既存キャラクターのデータが inner_narrative に引き継がれること。
"""

import os
import sqlite3
import tempfile
import uuid

import pytest
from sqlalchemy import text

from backend.core.memory.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# フィクスチャ: 旧スキーマ DB を生成するヘルパー
# ---------------------------------------------------------------------------

def _create_legacy_db(path: str) -> None:
    """meta_instructions を持つ旧スキーマの characters テーブルを含む DB を作成する。

    この DDL は meta_instructions TEXT NOT NULL で DEFAULT を持たない。
    そのため ORM 経由の INSERT は NOT NULL 制約違反となる（修正前の再現スキーマ）。
    """
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE characters (
            id VARCHAR NOT NULL PRIMARY KEY,
            name VARCHAR NOT NULL,
            system_prompt_block1 TEXT NOT NULL,
            meta_instructions TEXT NOT NULL,
            cleanup_config JSON NOT NULL,
            created_at DATETIME,
            updated_at DATETIME,
            provider TEXT NOT NULL DEFAULT 'claude_cli',
            model TEXT NOT NULL DEFAULT '',
            enabled_providers TEXT NOT NULL DEFAULT '{}',
            image_data TEXT,
            ghost_model TEXT,
            switch_angle_enabled INTEGER NOT NULL DEFAULT 0,
            inner_narrative TEXT NOT NULL DEFAULT '',
            afterglow_default INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE memories (
            id VARCHAR NOT NULL PRIMARY KEY,
            character_id VARCHAR NOT NULL,
            content TEXT NOT NULL,
            memory_category VARCHAR NOT NULL DEFAULT 'general',
            contextual_importance FLOAT DEFAULT 0.5,
            semantic_importance FLOAT DEFAULT 0.5,
            identity_importance FLOAT DEFAULT 0.5,
            user_importance FLOAT DEFAULT 0.5,
            created_at DATETIME,
            last_accessed_at DATETIME,
            access_count INTEGER DEFAULT 0,
            is_deleted INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE llm_model_presets (
            id VARCHAR NOT NULL PRIMARY KEY,
            name VARCHAR NOT NULL,
            provider VARCHAR NOT NULL,
            model_id VARCHAR NOT NULL DEFAULT '',
            created_at DATETIME
        )
    """)
    conn.execute("""
        CREATE TABLE global_settings (
            key VARCHAR NOT NULL PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()


@pytest.fixture
def legacy_store():
    """旧スキーマ DB を持つ SQLiteStore を提供するフィクスチャ。

    SQLiteStore の __init__ 内でマイグレーションが実行されるため、
    このフィクスチャが yield した時点ではマイグレーション済みとなる。
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    _create_legacy_db(path)
    store = SQLiteStore(path)
    yield store
    store.engine.dispose()
    try:
        os.remove(path)
    except PermissionError:
        pass


# ---------------------------------------------------------------------------
# マイグレーション後スキーマ検証
# ---------------------------------------------------------------------------

class TestLegacySchemaMigration:
    """旧スキーマ DB に対するマイグレーションの正確性を検証するテストスイート。

    修正前は meta_instructions NOT NULL DEFAULT なし のカラムが残るため
    create_character が IntegrityError を引き起こしていた。
    このクラスでは修正後に問題が再発しないことを担保する。
    """

    def test_meta_instructions_column_removed_after_migration(self, legacy_store):
        """マイグレーション後、meta_instructions カラムが characters テーブルに存在しないこと。

        meta_instructions が残っていると ORM INSERT 時に NOT NULL 制約エラーになる。
        """
        with legacy_store.engine.connect() as conn:
            result = conn.execute(
                text("SELECT count(*) FROM pragma_table_info('characters') WHERE name='meta_instructions'")
            )
            count = result.fetchone()[0]
        assert count == 0, "meta_instructions カラムはマイグレーション後に除去されていなければならない"

    def test_create_character_succeeds_after_migration(self, legacy_store):
        """マイグレーション後、create_character が例外なく成功すること。

        修正前は NOT NULL 制約違反（IntegrityError）が発生していた。
        """
        char_id = str(uuid.uuid4())
        char = legacy_store.create_character(character_id=char_id, name="新キャラ")
        assert char.id == char_id
        assert char.name == "新キャラ"

    def test_create_multiple_characters_succeed(self, legacy_store):
        """連続して複数のキャラクターを作成できること。"""
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for i, cid in enumerate(ids):
            legacy_store.create_character(character_id=cid, name=f"キャラ{i}")

        for cid in ids:
            char = legacy_store.get_character(cid)
            assert char is not None

    def test_existing_character_data_preserved_after_migration(self):
        """マイグレーション後、旧カラムに保存されていた既存キャラクターのデータが保持されること。

        meta_instructions の内容は inner_narrative に引き継がれる。
        既に inner_narrative に値がある場合はそちらを優先する。
        """
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        _create_legacy_db(path)

        # 旧スキーマでキャラクターを直接挿入する
        conn = sqlite3.connect(path)
        conn.execute(
            "INSERT INTO characters VALUES (?,?,?,?,?,NULL,NULL,'claude_cli','','{}',NULL,NULL,0,'',0)",
            ("id-meta-only", "旧キャラ（meta_instructions のみ）",
             "block1 content", "古いメタ指示", "{}")
        )
        conn.execute(
            "INSERT INTO characters VALUES (?,?,?,?,?,NULL,NULL,'claude_cli','','{}',NULL,NULL,0,'既存 inner_narrative',0)",
            ("id-both", "旧キャラ（両方あり）",
             "block1 content", "古いメタ指示（上書きされない）", "{}")
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(path)

        # meta_instructions のみのキャラ: inner_narrative に meta_instructions の内容が入る
        char_meta = store.get_character("id-meta-only")
        assert char_meta is not None
        assert char_meta.inner_narrative == "古いメタ指示"

        # inner_narrative が既にある場合はそちらを優先
        char_both = store.get_character("id-both")
        assert char_both is not None
        assert char_both.inner_narrative == "既存 inner_narrative"

        store.engine.dispose()
        try:
            os.remove(path)
        except PermissionError:
            pass

    def test_new_db_is_unaffected(self, sqlite_store):
        """新規 DB（旧スキーマなし）では create_character が問題なく動作すること。

        マイグレーション処理が正常な DB に悪影響を与えないことを確認する。
        """
        char_id = str(uuid.uuid4())
        char = sqlite_store.create_character(character_id=char_id, name="普通のキャラ")
        assert char.id == char_id
