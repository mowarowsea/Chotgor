"""シナリオチャット SQLite 永続化 — あらすじカラム＆PC Slot マイグレーションのテスト。

検証する観点:
    - あらすじカラム CRUD（synopsis_auto / synopsis_manual / last_turn_index）
    - `_migrate_unify_user_alias_to_pc_slot` のバックフィルと冪等性
    - `resolve_user_speaker_name`（ユーザPC名の統一解決）
"""

import uuid

import pytest

from tests._scenario_sqlite_helpers import (
    _make_npc,
    _make_scenario,
    _make_session,
    _make_turn,
)

# ─── あらすじカラム CRUD（記憶捏造対策） ─────────────────────────────────────


class TestScenarioSessionSynopsis:
    """`scenario_sessions.synopsis_auto` / `synopsis_manual` / `synopsis_last_turn_index` の検証。

    記憶捏造対策として導入された「セッション単位のあらすじ」機構。
    主な観点:
        - 新規セッションでは 3 カラムとも空・初期値（-1）であること
        - get_scenario_session_synopsis が dict で返ること
        - update_scenario_session_synopsis が部分更新（None の引数は触らない）
        - auto のみ更新で manual が破壊されないこと（逆も同様）
        - last_turn_index の永続化
        - 存在しないセッションへの更新は None
    """

    def test_new_session_has_empty_synopsis(self, sqlite_store):
        """新規セッションは synopsis_auto / manual ともに空、last_turn_index は -1。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        assert synopsis is not None
        assert synopsis["auto"] == ""
        assert synopsis["manual"] == ""
        assert synopsis["last_turn_index"] == -1

    def test_get_nonexistent_session_returns_none(self, sqlite_store):
        """存在しないセッション ID は None を返すこと。"""
        assert sqlite_store.get_scenario_session_synopsis("nope") is None

    def test_update_auto_only_preserves_manual(self, sqlite_store):
        """auto だけ更新したら manual は破壊されないこと（記憶捏造対策の核心）。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        # まず manual を先に設定
        sqlite_store.update_scenario_session_synopsis(
            session.id, manual="プレイヤーが手書きした重要な経緯"
        )
        # 次に auto だけ更新（manual は None で渡す）
        sqlite_store.update_scenario_session_synopsis(
            session.id, auto="自動生成された要約", last_turn_index=10
        )
        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        assert synopsis["auto"] == "自動生成された要約"
        assert synopsis["manual"] == "プレイヤーが手書きした重要な経緯"
        assert synopsis["last_turn_index"] == 10

    def test_update_manual_only_preserves_auto(self, sqlite_store):
        """manual だけ更新したら auto は破壊されないこと。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        sqlite_store.update_scenario_session_synopsis(
            session.id, auto="既存の自動要約", last_turn_index=5
        )
        sqlite_store.update_scenario_session_synopsis(
            session.id, manual="後から追加したメモ"
        )
        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        assert synopsis["auto"] == "既存の自動要約"
        assert synopsis["manual"] == "後から追加したメモ"
        # last_turn_index は触っていないので前の値を保持
        assert synopsis["last_turn_index"] == 5

    def test_update_can_set_empty_string(self, sqlite_store):
        """空文字列での更新は許容される（ユーザが auto を全部削除するケース）。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        sqlite_store.update_scenario_session_synopsis(session.id, auto="aaa", manual="bbb")
        # 空文字列で上書き（None ではない）
        sqlite_store.update_scenario_session_synopsis(session.id, auto="")
        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        assert synopsis["auto"] == ""
        assert synopsis["manual"] == "bbb"

    def test_update_nonexistent_session_returns_none(self, sqlite_store):
        """存在しないセッション ID への更新は None を返すこと。"""
        result = sqlite_store.update_scenario_session_synopsis(
            "no-such-session", auto="x"
        )
        assert result is None

    def test_update_returns_latest_state(self, sqlite_store):
        """update の戻り値が更新後の最新状態を反映していること。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        result = sqlite_store.update_scenario_session_synopsis(
            session.id, auto="A", manual="M", last_turn_index=3
        )
        assert result == {"auto": "A", "manual": "M", "last_turn_index": 3}

    def test_synopsis_persists_across_session_lookup(self, sqlite_store):
        """SQLite に永続化され、ORM 経由でも読めること。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        sqlite_store.update_scenario_session_synopsis(
            session.id, auto="X", manual="Y", last_turn_index=7
        )
        # get_scenario_session（ORM）からも読める
        sess = sqlite_store.get_scenario_session(session.id)
        assert sess.synopsis_auto == "X"
        assert sess.synopsis_manual == "Y"
        assert sess.synopsis_last_turn_index == 7


class TestUserAliasToPcSlotMigration:
    """`_migrate_unify_user_alias_to_pc_slot` のバックフィルと冪等性を検証する。

    旧スキーマ（scenarios.user_alias 列あり）を再現し、移行が
      - ユーザPC枠を pc_slots に生成すること
      - セッションへ player_type="user" 割当を補完すること
      - user_alias 列を削除すること
      - 既に user 割当を持つ ensemble_pc データには余計な枠を作らないこと
      - 二度実行しても安全（冪等）であること
    を網羅的に確認する。破壊的スキーマ変更（DROP COLUMN）を含むため、
    旧データからの移行が無損失であることをテストで保証する意義は大きい。
    """

    @staticmethod
    def _readd_user_alias_and_set(store, scenario_id, alias):
        """移行テスト用に user_alias 列を旧スキーマ相当で復活させ、値を設定する。

        新スキーマでは ORM に user_alias が無いため、移行前状態を再現するには
        生 SQL で列を足してから値を入れる必要がある。
        """
        with store.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenarios)"
                ).fetchall()
            }
            if "user_alias" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE scenarios ADD COLUMN user_alias TEXT"
                )
            conn.exec_driver_sql(
                "UPDATE scenarios SET user_alias = ? WHERE id = ?",
                (alias, scenario_id),
            )

    @staticmethod
    def _user_alias_column_exists(store) -> bool:
        """scenarios テーブルに user_alias 列が残っているかを返す。"""
        with store.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql(
                    "PRAGMA table_info(scenarios)"
                ).fetchall()
            }
        return "user_alias" in cols

    def test_ensemble_backfill_and_drop(self, sqlite_store):
        """ensemble 旧データから user 枠を生成・割当補完し、user_alias 列を削除すること。

        旧 ensemble シナリオ（pc_slots NULL）とセッション（pc_assignments NULL）に対し、
        user_alias("勇者") を pc_slots 先頭の user 枠へ昇格し、セッションへ user 割当を補い、
        最後に user_alias 列が消えていることを確認する。
        """
        store = sqlite_store
        scenario = _make_scenario(store, pc_slots=None)  # 旧 ensemble（pc_slots NULL）
        session = _make_session(store, scenario.id)       # pc_assignments NULL
        self._readd_user_alias_and_set(store, scenario.id, "勇者")

        store._migrate_unify_user_alias_to_pc_slot()

        fetched = store.get_scenario(scenario.id)
        slots = fetched.pc_slots
        assert slots and slots[0]["name"] == "勇者"
        user_slot_id = slots[0]["slot_id"]
        sess = store.get_scenario_session(session.id)
        assert any(
            a.get("slot_id") == user_slot_id and a.get("player_type") == "user"
            for a in (sess.pc_assignments or [])
        )
        assert not self._user_alias_column_exists(store)

    def test_idempotent_second_run(self, sqlite_store):
        """移行を二度走らせても例外が出ず結果が壊れないこと（列削除後は早期 return）。"""
        store = sqlite_store
        scenario = _make_scenario(store, pc_slots=None)
        _make_session(store, scenario.id)
        self._readd_user_alias_and_set(store, scenario.id, "旅人")

        store._migrate_unify_user_alias_to_pc_slot()
        # 2 回目: user_alias 列はもう無い → 早期 return（例外が出ないこと）
        store._migrate_unify_user_alias_to_pc_slot()

        fetched = store.get_scenario(scenario.id)
        assert fetched.pc_slots[0]["name"] == "旅人"

    def test_ensemble_pc_existing_user_not_duplicated(self, sqlite_store):
        """既に user 割当を持つ ensemble_pc データには余計な user 枠を作らないこと。"""
        store = sqlite_store
        cast = [
            {"slot_id": "pc1", "name": "アリス", "description": ""},
            {"slot_id": "pc2", "name": "ボブ", "description": ""},
        ]
        scenario = _make_scenario(store, pc_slots=cast)
        store.create_scenario_session(
            session_id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            title="pc play",
            gm_preset_id="preset-test",
            synopsis_preset_id="preset-test",
            engine_type="ensemble_pc",
            pc_assignments=[{"slot_id": "pc1", "player_type": "user"}],
        )
        self._readd_user_alias_and_set(store, scenario.id, "vestige")

        store._migrate_unify_user_alias_to_pc_slot()

        fetched = store.get_scenario(scenario.id)
        names = [s["name"] for s in fetched.pc_slots]
        # vestige という余計な枠が増えず、既存 cast のままであること
        assert "vestige" not in names
        assert names == ["アリス", "ボブ"]

    def test_legacy_custom_prompt_unified(self, sqlite_store):
        """既存 custom_system_prompt の旧「主役（プレイヤー）」行が移行で除去されること。"""
        store = sqlite_store
        legacy = (
            "# 既知の話者\n"
            "@{user_alias}   ← この物語の主役（プレイヤー）。"
            "あなたは絶対に代弁しない。\n"
            "@{narrator_name}       ← 情景・状況描写。\n"
        )
        scenario = _make_scenario(
            store, pc_slots=None, custom_system_prompt=legacy
        )
        self._readd_user_alias_and_set(store, scenario.id, "勇者")

        store._migrate_unify_user_alias_to_pc_slot()

        fetched = store.get_scenario(scenario.id)
        # 主役行は除去され、Narrator 行は残る
        assert "主役（プレイヤー）" not in fetched.custom_system_prompt
        assert "@{narrator_name}" in fetched.custom_system_prompt


class TestResolveUserSpeakerName:
    """`resolve_user_speaker_name`（ユーザPC名の統一解決）を検証する。

    旧 user_alias 廃止後、ユーザの @タグ名は engine_type に依存せず
    pc_assignments の player_type="user" スロットの name から解決される。
    user 割当が無い異常時は default を返すことも確認する。
    """

    def test_returns_user_assignment_slot_name(self, sqlite_store):
        """user 割当スロットの name を返すこと。"""
        from backend.services.scenario_chat.service import resolve_user_speaker_name

        store = sqlite_store
        scenario = _make_scenario(
            store, pc_slots=[{"slot_id": "pc1", "name": "アリス", "description": ""}]
        )
        session = store.create_scenario_session(
            session_id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            title="t",
            gm_preset_id="preset-test",
            synopsis_preset_id="preset-test",
            engine_type="ensemble_pc",
            pc_assignments=[{"slot_id": "pc1", "player_type": "user"}],
        )
        assert resolve_user_speaker_name(scenario, session, store) == "アリス"

    def test_returns_default_when_no_user_assignment(self, sqlite_store):
        """user 割当が無ければフォールバック名を返すこと。"""
        from backend.services.scenario_chat.service import resolve_user_speaker_name

        store = sqlite_store
        scenario = _make_scenario(
            store, pc_slots=[{"slot_id": "pc1", "name": "アリス", "description": ""}]
        )
        # character_id が存在しない割当のみ → normalize で除外され user なしになる
        session = store.create_scenario_session(
            session_id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            title="t",
            gm_preset_id="preset-test",
            synopsis_preset_id="preset-test",
            engine_type="ensemble_pc",
            pc_assignments=[
                {"slot_id": "pc1", "player_type": "character", "character_id": "missing"}
            ],
        )
        assert (
            resolve_user_speaker_name(scenario, session, store, default="プレイヤー")
            == "プレイヤー"
        )
