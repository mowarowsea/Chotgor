"""
SQLiteStore の SessionDrift CRUD テストおよび DriftManager テスト。

fixtures の sqlite_store は conftest.py が提供するインメモリ一時DBを使用する。
各テストは独立した一時DBで動作するため、テスト間の干渉はない。

SessionDrift は ChatSession に FK があるため、
drift を追加する前に必ずセッションを作成すること。
"""

import uuid

import pytest

from backend.core.memory.drift_manager import DriftManager


# --- TestSessionDriftCRUD ---

class TestSessionDriftCRUD:
    """SQLiteStore の SessionDrift CRUD を網羅的に検証するテストスイート。

    - add_session_drift / list_session_drifts / list_active_session_drifts の正常系
    - toggle_session_drift / reset_session_drifts の正常系・異常系
    - 上限3件ルール（キャラクターごとに独立）
    - セッション削除時のカスケード削除
    - キャラクター間の分離
    """

    def _make_session(self, store, model_id: str = "alice@gemini") -> str:
        """テスト用セッションを作成してIDを返すヘルパー。

        SessionDrift は session_id に FK があるため、drift 追加前に必要。
        """
        sid = str(uuid.uuid4())
        store.create_chat_session(session_id=sid, model_id=model_id)
        return sid

    def test_add_and_list(self, sqlite_store):
        """drift 追加後に list_session_drifts で取得できること。"""
        sid = self._make_session(sqlite_store)
        char_id = "char-1"

        drift = sqlite_store.add_session_drift(sid, char_id, "もっとクールに話す")

        assert drift.session_id == sid
        assert drift.character_id == char_id
        assert drift.content == "もっとクールに話す"

        listed = sqlite_store.list_session_drifts(sid)
        assert len(listed) == 1
        assert listed[0].id == drift.id

    def test_enabled_default_is_true(self, sqlite_store):
        """drift 追加直後は enabled=1 であること。"""
        sid = self._make_session(sqlite_store)
        drift = sqlite_store.add_session_drift(sid, "char-1", "指針A")

        assert drift.enabled == 1

    def test_toggle_enabled(self, sqlite_store):
        """toggle_session_drift で enabled が 1→0 に反転すること。"""
        sid = self._make_session(sqlite_store)
        drift = sqlite_store.add_session_drift(sid, "char-1", "指針A")
        assert drift.enabled == 1

        toggled = sqlite_store.toggle_session_drift(drift.id)

        assert toggled is not None
        assert toggled.enabled == 0

    def test_toggle_twice(self, sqlite_store):
        """2回 toggle すると元の enabled 値（1）に戻ること。"""
        sid = self._make_session(sqlite_store)
        drift = sqlite_store.add_session_drift(sid, "char-1", "指針A")

        sqlite_store.toggle_session_drift(drift.id)
        toggled_twice = sqlite_store.toggle_session_drift(drift.id)

        assert toggled_twice is not None
        assert toggled_twice.enabled == 1

    def test_toggle_nonexistent(self, sqlite_store):
        """存在しない drift ID を toggle した場合に None が返ること。"""
        result = sqlite_store.toggle_session_drift("no-such-drift-id")

        assert result is None

    def test_reset(self, sqlite_store):
        """reset_session_drifts で対象キャラの全 drift が物理削除されること。"""
        sid = self._make_session(sqlite_store)
        char_id = "char-1"

        sqlite_store.add_session_drift(sid, char_id, "指針A")
        sqlite_store.add_session_drift(sid, char_id, "指針B")
        assert len(sqlite_store.list_session_drifts(sid)) == 2

        deleted_count = sqlite_store.reset_session_drifts(sid, char_id)

        assert deleted_count == 2
        assert sqlite_store.list_session_drifts(sid) == []

    def test_reset_only_target_character(self, sqlite_store):
        """reset_session_drifts は指定キャラのdriftのみ削除し、他キャラのdriftは残すこと。

        マルチキャラクターのグループセッションを想定した重要な分離テスト。
        """
        sid = self._make_session(sqlite_store)
        char_a = "char-alice"
        char_b = "char-bob"

        sqlite_store.add_session_drift(sid, char_a, "Aliceの指針")
        sqlite_store.add_session_drift(sid, char_b, "Bobの指針")

        # char_a のみリセット
        sqlite_store.reset_session_drifts(sid, char_a)

        remaining = sqlite_store.list_session_drifts(sid)
        assert len(remaining) == 1
        assert remaining[0].character_id == char_b

    def test_list_active_only_returns_enabled(self, sqlite_store):
        """list_active_session_drifts は enabled=1 のdriftのみ返すこと。"""
        sid = self._make_session(sqlite_store)
        char_id = "char-1"

        drift_on = sqlite_store.add_session_drift(sid, char_id, "有効な指針")
        drift_off = sqlite_store.add_session_drift(sid, char_id, "無効な指針")
        # drift_off を無効化する
        sqlite_store.toggle_session_drift(drift_off.id)

        active = sqlite_store.list_active_session_drifts(sid, char_id)

        assert len(active) == 1
        assert active[0] == "有効な指針"
        assert "無効な指針" not in active

    def test_list_active_by_character(self, sqlite_store):
        """list_active_session_drifts は character_id でフィルタされること。"""
        sid = self._make_session(sqlite_store)
        char_a = "char-alice"
        char_b = "char-bob"

        sqlite_store.add_session_drift(sid, char_a, "Aliceの指針")
        sqlite_store.add_session_drift(sid, char_b, "Bobの指針")

        alice_active = sqlite_store.list_active_session_drifts(sid, char_a)
        bob_active = sqlite_store.list_active_session_drifts(sid, char_b)

        assert alice_active == ["Aliceの指針"]
        assert bob_active == ["Bobの指針"]

    def test_max_3_enforced(self, sqlite_store):
        """4件 add すると上限3件が保持され、最古のdriftが自動削除されること。"""
        sid = self._make_session(sqlite_store)
        char_id = "char-1"

        for i in range(4):
            sqlite_store.add_session_drift(sid, char_id, f"指針{i}")

        drifts = sqlite_store.list_session_drifts(sid)

        # 上限3件に抑えられること
        assert len(drifts) == 3
        # 最も古い「指針0」は削除されていること
        contents = [d.content for d in drifts]
        assert "指針0" not in contents
        assert "指針1" in contents
        assert "指針2" in contents
        assert "指針3" in contents

    def test_max_3_per_character(self, sqlite_store):
        """上限3件ルールはキャラクターごとに独立して適用されること。

        char_a と char_b それぞれが3件ずつ保持できること（合計6件）を検証する。
        """
        sid = self._make_session(sqlite_store)
        char_a = "char-alice"
        char_b = "char-bob"

        for i in range(3):
            sqlite_store.add_session_drift(sid, char_a, f"Aliceの指針{i}")
            sqlite_store.add_session_drift(sid, char_b, f"Bobの指針{i}")

        all_drifts = sqlite_store.list_session_drifts(sid)

        # 両キャラ合計6件が保持されること
        assert len(all_drifts) == 6
        alice_drifts = [d for d in all_drifts if d.character_id == char_a]
        bob_drifts = [d for d in all_drifts if d.character_id == char_b]
        assert len(alice_drifts) == 3
        assert len(bob_drifts) == 3

    def test_created_at_order(self, sqlite_store):
        """list_session_drifts は作成日時の昇順（古い順）で返すこと。"""
        sid = self._make_session(sqlite_store)
        char_id = "char-1"

        # 3件連続追加（同一ミリ秒になる可能性を考慮してIDで追跡）
        d1 = sqlite_store.add_session_drift(sid, char_id, "最初の指針")
        d2 = sqlite_store.add_session_drift(sid, char_id, "2番目の指針")
        d3 = sqlite_store.add_session_drift(sid, char_id, "3番目の指針")

        drifts = sqlite_store.list_session_drifts(sid)

        # 内容の順序で昇順になっていることを確認（挿入順序に依存するため内容で検証）
        contents = [d.content for d in drifts]
        assert contents.index("最初の指針") < contents.index("3番目の指針")

    def test_session_delete_cascades(self, sqlite_store):
        """セッションを削除すると紐づく drift も一緒に削除されること。"""
        sid = self._make_session(sqlite_store)
        char_id = "char-1"

        sqlite_store.add_session_drift(sid, char_id, "指針A")
        sqlite_store.add_session_drift(sid, char_id, "指針B")
        assert len(sqlite_store.list_session_drifts(sid)) == 2

        sqlite_store.delete_chat_session(sid)

        # セッション削除後は drift も存在しないこと
        assert sqlite_store.list_session_drifts(sid) == []


# --- TestDriftManager ---

class TestDriftManager:
    """DriftManager の統合テストスイート。

    DriftManager は SQLiteStore の薄いラッパーだが、
    add_drift / list_active_drifts / reset_drifts / toggle_drift の
    各メソッドが SQLiteStore に正しく委譲されていることを検証する。
    """

    def _make_store_and_manager(self, sqlite_store):
        """テスト用の DriftManager インスタンスを作成するヘルパー。"""
        return DriftManager(sqlite=sqlite_store)

    def _make_session(self, store, model_id: str = "alice@gemini") -> str:
        """テスト用セッションを作成してIDを返すヘルパー。"""
        sid = str(uuid.uuid4())
        store.create_chat_session(session_id=sid, model_id=model_id)
        return sid

    def test_add_and_list_active(self, sqlite_store):
        """add_drift で追加した drift が list_active_drifts で取得できること。"""
        manager = self._make_store_and_manager(sqlite_store)
        sid = self._make_session(sqlite_store)
        char_id = "char-1"

        manager.add_drift(sid, char_id, "テスト指針")

        active = manager.list_active_drifts(sid, char_id)
        assert len(active) == 1
        assert active[0] == "テスト指針"

    def test_reset_by_character(self, sqlite_store):
        """reset_drifts で指定キャラのdriftのみ削除され、他キャラのdriftは残ること。"""
        manager = self._make_store_and_manager(sqlite_store)
        sid = self._make_session(sqlite_store)
        char_a = "char-alice"
        char_b = "char-bob"

        manager.add_drift(sid, char_a, "Aliceの指針")
        manager.add_drift(sid, char_b, "Bobの指針")

        # char_a のみリセット
        deleted = manager.reset_drifts(sid, char_a)

        assert deleted == 1
        assert manager.list_active_drifts(sid, char_a) == []
        assert manager.list_active_drifts(sid, char_b) == ["Bobの指針"]

    def test_toggle_via_manager(self, sqlite_store):
        """toggle_drift で enabled フラグが反転すること。"""
        manager = self._make_store_and_manager(sqlite_store)
        sid = self._make_session(sqlite_store)
        char_id = "char-1"

        drift = manager.add_drift(sid, char_id, "指針A")
        assert drift.enabled == 1

        toggled = manager.toggle_drift(drift.id)
        assert toggled is not None
        assert toggled.enabled == 0

        # list_active_drifts では無効化されたものが除外されること
        active = manager.list_active_drifts(sid, char_id)
        assert active == []
