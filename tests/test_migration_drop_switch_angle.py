"""`_migrate_drop_switch_angle_enabled` のテスト。

switch_angle 機能撤去（docs/planned/switch_angle_removal_plan.md）の残置物だった
`characters.switch_angle_enabled` 列を落とすマイグレーションを検証する。

手順: 新スキーマの DB に SQL 直叩きで旧列を追加し直して「撤去前の旧DB」を再現し、
その DB を SQLiteStore で開き直す（起動時マイグレーション経路）か、マイグレーションを
直接呼ぶ。検証する観点:
    - 旧列が DROP される（起動時経路・直接呼び出しの両方）
    - 列以外のデータ（キャラクター行）は保たれる
    - 冪等: 列が無い状態で呼んでもエラーにならない
"""

import os
import tempfile

from backend.repositories.sqlite.store import SQLiteStore


def _columns(store, table: str) -> set[str]:
    with store.engine.begin() as conn:
        return {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()}


def _recreate_legacy_column(store) -> None:
    """旧 switch_angle_enabled 列を追加して「撤去前の旧DB」を再現する。"""
    with store.engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE characters ADD COLUMN switch_angle_enabled INTEGER NOT NULL DEFAULT 0"
        )


class TestDropSwitchAngleEnabled:
    def test_fresh_db_has_no_column(self, sqlite_store):
        """新規 DB（ORM 定義どおり）には列が無いこと（モデルからも撤去済み）。"""
        assert "switch_angle_enabled" not in _columns(sqlite_store, "characters")

    def test_direct_call_drops_legacy_column(self, sqlite_store):
        """旧列を持つ DB でマイグレーションを呼ぶと列が落ち、行は残ること。"""
        sqlite_store.create_character("char-1", "テスト")
        _recreate_legacy_column(sqlite_store)
        assert "switch_angle_enabled" in _columns(sqlite_store, "characters")

        sqlite_store._migrate_drop_switch_angle_enabled()

        assert "switch_angle_enabled" not in _columns(sqlite_store, "characters")
        assert sqlite_store.get_character("char-1").name == "テスト"

    def test_idempotent_when_column_absent(self, sqlite_store):
        """列が無い状態で呼んでもエラーにならないこと。"""
        sqlite_store._migrate_drop_switch_angle_enabled()
        sqlite_store._migrate_drop_switch_angle_enabled()
        assert "switch_angle_enabled" not in _columns(sqlite_store, "characters")

    def test_startup_path_drops_legacy_column(self):
        """旧列を持つ DB を SQLiteStore で開き直すと、起動時マイグレーションで列が落ちること。"""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            first = SQLiteStore(path)
            first.create_character("char-1", "テスト")
            _recreate_legacy_column(first)
            first.engine.dispose()

            reopened = SQLiteStore(path)
            try:
                assert "switch_angle_enabled" not in _columns(reopened, "characters")
                assert reopened.get_character("char-1").name == "テスト"
            finally:
                reopened.engine.dispose()
        finally:
            try:
                os.remove(path)
            except PermissionError:
                pass
