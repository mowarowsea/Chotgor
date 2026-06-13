"""うつつ（Usual Days）— SQLite データモデル層のテスト（Phase 1）。

検証する観点:
    - scenarios への owner_character_id / usual_config 列の永続化（create / update）
    - 汎用シナリオ一覧（list_scenarios）が owner 付き（うつつ）シナリオを除外すること
    - include_usual=True で うつつ も含めて取得できること
    - get_usual_scenario / list_usual_scenarios によるオーナー別取得
    - _migrate_add_usual_days の冪等性（二重実行しても安全・列が存在し続ける）

うつつは「キャラ固有の生活世界」であり 1 キャラ 1 世界。汎用シナリオ一覧には並べず、
owner_character_id を除外判定キーに使う設計をここで担保する。
"""

from tests._scenario_sqlite_helpers import _make_scenario


class TestUsualScenarioPersistence:
    """うつつ専用カラム（owner_character_id / usual_config）の永続化を検証する。"""

    def test_create_usual_scenario_persists_fields(self, sqlite_store):
        """owner_character_id と usual_config を指定して作成すると、両方が永続化されること。"""
        cfg = {
            "enabled": True,
            "slots": ["10:00", "13:00", "17:00"],
            "event_probability": 0.3,
            "max_turns_per_scene": 8,
        }
        scenario = _make_scenario(
            sqlite_store,
            title="はるの日常",
            owner_character_id="char-haru",
            usual_config=cfg,
        )
        # 取得し直しても保持されていること（JSON はラウンドトリップで dict のまま戻る）
        fetched = sqlite_store.get_scenario(scenario.id)
        assert fetched.owner_character_id == "char-haru"
        assert fetched.usual_config == cfg

    def test_generic_scenario_defaults_owner_to_none(self, sqlite_store):
        """通常シナリオ（owner 未指定）は owner_character_id が NULL であること。"""
        scenario = _make_scenario(sqlite_store, title="汎用シナリオ")
        fetched = sqlite_store.get_scenario(scenario.id)
        assert fetched.owner_character_id is None
        assert fetched.usual_config is None

    def test_update_usual_config(self, sqlite_store):
        """update_scenario で usual_config（有効化トグル等）を部分更新できること。"""
        scenario = _make_scenario(
            sqlite_store,
            owner_character_id="char-haru",
            usual_config={"enabled": False, "slots": []},
        )
        sqlite_store.update_scenario(
            scenario.id, usual_config={"enabled": True, "slots": ["09:00"]}
        )
        fetched = sqlite_store.get_scenario(scenario.id)
        assert fetched.usual_config == {"enabled": True, "slots": ["09:00"]}


class TestUsualScenarioListing:
    """汎用一覧からの除外と、うつつ専用取得メソッドを検証する。"""

    def test_list_scenarios_excludes_usual(self, sqlite_store):
        """汎用シナリオ一覧は owner 付き（うつつ）シナリオを除外すること。"""
        generic = _make_scenario(sqlite_store, title="汎用")
        _make_scenario(sqlite_store, title="うつつ", owner_character_id="char-haru")

        listed_ids = {s.id for s in sqlite_store.list_scenarios()}
        assert generic.id in listed_ids
        # うつつ世界は汎用一覧に出ない
        assert all(s.owner_character_id is None for s in sqlite_store.list_scenarios())

    def test_list_scenarios_include_usual(self, sqlite_store):
        """include_usual=True のときは うつつ も含めて返すこと。"""
        _make_scenario(sqlite_store, title="汎用")
        usual = _make_scenario(sqlite_store, title="うつつ", owner_character_id="char-haru")

        listed_ids = {s.id for s in sqlite_store.list_scenarios(include_usual=True)}
        assert usual.id in listed_ids

    def test_get_usual_scenario_by_owner(self, sqlite_store):
        """get_usual_scenario が指定キャラのうつつ世界を返すこと（無ければ None）。"""
        usual = _make_scenario(
            sqlite_store, title="はるの日常", owner_character_id="char-haru"
        )
        assert sqlite_store.get_usual_scenario("char-haru").id == usual.id
        assert sqlite_store.get_usual_scenario("char-unknown") is None

    def test_list_usual_scenarios(self, sqlite_store):
        """list_usual_scenarios が owner 付きシナリオだけを返すこと（スケジューラ用）。"""
        _make_scenario(sqlite_store, title="汎用")
        u1 = _make_scenario(sqlite_store, title="うつつA", owner_character_id="char-a")
        u2 = _make_scenario(sqlite_store, title="うつつB", owner_character_id="char-b")

        ids = {s.id for s in sqlite_store.list_usual_scenarios()}
        assert ids == {u1.id, u2.id}


class TestUsualMigrationIdempotency:
    """_migrate_add_usual_days の冪等性を検証する。"""

    def test_migration_is_idempotent(self, sqlite_store):
        """マイグレーションを再実行しても例外を出さず、列が存在し続けること。

        SQLiteStore.__init__ で 1 度実行済み。ここでさらに 2 回呼んでも
        ALTER TABLE が二重発行されない（列存在チェックで弾く）ことを確認する。
        """
        # 二重・三重に呼んでも安全
        sqlite_store._migrate_add_usual_days()
        sqlite_store._migrate_add_usual_days()

        with sqlite_store.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql("PRAGMA table_info(scenarios)").fetchall()
            }
        assert "owner_character_id" in cols
        assert "usual_config" in cols
