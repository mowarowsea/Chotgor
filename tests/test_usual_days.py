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

import asyncio

import backend.services.scenario_chat.pc_runner as pc_runner_mod
import backend.services.scenario_chat.service as svc

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


# ---------------------------------------------------------------------------
# Phase 2 — 無人ループ（headless / run_scenario_turn・run_usual_days_scene）
# ---------------------------------------------------------------------------


def _build_usual_session(store, max_turns: int = 8):
    """うつつ（engine_type="usual_days"）の最小セッションを組み立てて返す。

    主人公キャラ 1 体・PC枠 1 つ・ユーザ枠ゼロのうつつ世界を作る。
    GM ターンと PC ターンの LLM 呼び出しはテスト側でモックする前提なので、
    プリセットはダミーで構わない（ただし _resolve_pc_preset_id が実在を要求するため
    実レコードとして作成する）。

    Returns:
        (session_id, character_id) のタプル。
    """
    cid = "char-haru"
    pid = "preset-usual"
    store.create_character(cid, "はる")
    store.create_model_preset(pid, "うつつ用", "anthropic", "claude-x")
    scenario = _make_scenario(
        store,
        title="はるの日常",
        owner_character_id=cid,
        pc_slots=[{"slot_id": "pc1", "name": "はる", "description": "主人公。会社員。"}],
        usual_config={"max_turns_per_scene": max_turns},
    )
    sid = "sess-usual"
    store.create_scenario_session(
        session_id=sid,
        scenario_id=scenario.id,
        title="うつつ",
        gm_preset_id=pid,
        synopsis_preset_id=pid,
        engine_type="usual_days",
        pc_assignments=[{
            "slot_id": "pc1",
            "player_type": "character",
            "character_id": cid,
            "preset_id": pid,
        }],
    )
    return sid, cid


def _install_mocks(monkeypatch, gm_script: list[str], pc_calls: list[dict]):
    """_run_gm_turn / stream_pc_response をモック差し替えするヘルパー。

    - fake_gm: gm_script から 1 件ずつ取り出した文字列を Narrator ターンとして保存する
      （raw_response に格納。ルーティングは @はる 等のメンションや [SCENE_CLOSE] で制御）。
      script を使い切ったら無害なデフォルト文を返す。
    - fake_pc: pc_done を 1 回 yield する。呼び出し時の kwargs（特に default_origin）を
      pc_calls へ記録し、検証に使う。PC 本文はメンションを含めない（→ headless は GM へ継続）。
    """
    async def fake_gm(**kwargs):
        text = gm_script.pop(0) if gm_script else "Narrator: しずかな時間が流れた。@はる"
        svc._save_turn(
            sqlite=kwargs["sqlite"],
            session_id=kwargs["session_id"],
            speaker_type="narrator",
            speaker_name="Narrator",
            content=text,
            raw_response=text,
        )
        return
        yield  # 到達しないが async generator にするためのダミー

    async def fake_pc(**kwargs):
        pc_calls.append(dict(kwargs))
        name = kwargs["pc"].name
        cid = kwargs["pc"].character_id
        yield ("pc_done", {
            "character": name,
            "character_id": cid,
            "full_text": "（はるは黙々と仕事を続けた）",
            "anticipation": None,
        })

    monkeypatch.setattr(svc, "_run_gm_turn", fake_gm)
    monkeypatch.setattr(pc_runner_mod, "stream_pc_response", fake_pc)
    # あらすじ蒸留は LLM を叩くためモックで無効化する。
    monkeypatch.setattr(svc, "compute_synopsis_progress", lambda *a, **k: None)


class TestHeadlessLoop:
    """うつつ無人ループ（run_scenario_turn headless / run_usual_days_scene）を検証する。"""

    def test_headless_runs_until_cap(self, sqlite_store, monkeypatch):
        """メンションが続く限り GM↔PC を連鎖し、max_turns_per_scene で停止すること。

        GM は毎ターン @はる を呼び、PC はメンションを返さない。headless は PC 後も
        ユーザに戻さず GM へ継続するため、上限（4）に達するまで GM,PC,GM,PC と回る。
        """
        sid, _ = _build_usual_session(sqlite_store, max_turns=4)
        pc_calls: list[dict] = []
        # GM は常に @はる を呼ぶ（script を空にしてデフォルト文 "...@はる" を使わせる）
        _install_mocks(monkeypatch, gm_script=[], pc_calls=pc_calls)

        result = asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))

        turns = sqlite_store.list_scenario_turns(sid)
        # GM2 + PC2 = 4 ターン保存され、上限で停止
        assert len(turns) == 4
        assert result["scene_closed"] is False
        # PC は 2 回呼ばれている
        assert len(pc_calls) == 2

    def test_headless_stops_on_scene_close(self, sqlite_store, monkeypatch):
        """GM が [SCENE_CLOSE] を宣言したら上限前でも停止すること。"""
        sid, _ = _build_usual_session(sqlite_store, max_turns=20)
        pc_calls: list[dict] = []
        # 1回目GM=@はる呼び、2回目GM=幕引き宣言
        gm_script = [
            "Narrator: 朝。@はる、出勤の時間だ。",
            "Narrator: 一日が終わり、はるは眠りについた。[SCENE_CLOSE]",
        ]
        _install_mocks(monkeypatch, gm_script=gm_script, pc_calls=pc_calls)

        result = asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))

        turns = sqlite_store.list_scenario_turns(sid)
        # GM1(@はる) → PC1 → GM2([SCENE_CLOSE]) で停止 = 3 ターン（上限20には遠い）
        assert len(turns) == 3
        assert result["scene_closed"] is True

    def test_headless_passes_origin_usual(self, sqlite_store, monkeypatch):
        """PC ターンに default_origin="usual" が渡されること（記憶の由来タグ）。"""
        sid, _ = _build_usual_session(sqlite_store, max_turns=2)
        pc_calls: list[dict] = []
        _install_mocks(monkeypatch, gm_script=[], pc_calls=pc_calls)

        asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))

        assert pc_calls, "PC ターンが 1 度も実行されていない"
        assert all(c.get("default_origin") == "usual" for c in pc_calls)
