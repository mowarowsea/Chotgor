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
import json
import random
import types
from datetime import datetime, timedelta

import backend.main as mainmod
import backend.services.scenario_chat.pc_runner as pc_runner_mod
import backend.services.scenario_chat.service as svc
from backend.api.ui.characters import _parse_usual_form, _usual_time_grid_text
from backend.lib.log_context import current_log_feature
from backend.services.chat.request_builder import build_system_prompt
from backend.lib.time_awareness import (
    format_time_context,
    japanese_season,
    japanese_time_of_day,
    japanese_weekday,
)
from backend.services.scenario_chat.prompt_builder import build_gm_system_prompt

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

    def test_migration_alters_legacy_schema(self, sqlite_store):
        """列の無い旧スキーマ（レガシーDB）に対して ALTER TABLE が実際に走ること。

        インメモリDBは ORM create_all で最初から列があるため、通常テストでは
        ALTER 分岐が走らない。ここでは列を一旦 DROP して「旧DB」を再現し、
        マイグレーションが列を追加し、既存行を壊さず NULL 既定で補うことを検証する。
        ユーザの実DB（移行前）に再起動時に走るのと同じ経路。
        """
        # 既存シナリオを 1 件作っておく（移行で消えないことの確認用）
        existing = _make_scenario(sqlite_store, title="移行前からある汎用シナリオ")

        # 旧スキーマ再現: 新カラムを物理 DROP（SQLite 3.35+ の DROP COLUMN）
        with sqlite_store.engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE scenarios DROP COLUMN owner_character_id")
            conn.exec_driver_sql("ALTER TABLE scenarios DROP COLUMN usual_config")
            cols_before = {
                r[1] for r in conn.exec_driver_sql("PRAGMA table_info(scenarios)").fetchall()
            }
        assert "owner_character_id" not in cols_before
        assert "usual_config" not in cols_before

        # マイグレーション実行（ALTER パスが走る）
        sqlite_store._migrate_add_usual_days()

        with sqlite_store.engine.begin() as conn:
            cols_after = {
                r[1] for r in conn.exec_driver_sql("PRAGMA table_info(scenarios)").fetchall()
            }
        assert "owner_character_id" in cols_after
        assert "usual_config" in cols_after

        # 既存行は壊れず、新カラムは NULL 既定で補われている
        migrated = sqlite_store.get_scenario(existing.id)
        assert migrated.title == "移行前からある汎用シナリオ"
        assert migrated.owner_character_id is None
        assert migrated.usual_config is None


# ---------------------------------------------------------------------------
# Phase 2 — 無人ループ（headless / run_scenario_turn・run_usual_days_scene）
# ---------------------------------------------------------------------------


def _build_usual_session(store, max_turns: int = 8, user_label: str | None = None):
    """うつつ（engine_type="usual_days"）の最小セッションを組み立てて返す。

    主人公キャラ 1 体・PC枠 1 つのうつつ世界を作る。``user_label`` を渡すと、
    「不在のユーザPC」枠（slot_id="user"）とその割当（player_type="user"）も併せて作る
    （ユーザがターンを取らないことの検証用）。
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
    pc_slots = [{"slot_id": "pc1", "name": "はる", "description": "主人公。会社員。"}]
    pc_assignments = [{
        "slot_id": "pc1",
        "player_type": "character",
        "character_id": cid,
        "preset_id": pid,
    }]
    if user_label:
        pc_slots.append({
            "slot_id": "user",
            "name": user_label,
            "description": "【この場面に不在・姿/言動を描かない】主任。はるの直属の上司。",
        })
        pc_assignments.append({"slot_id": "user", "player_type": "user"})
    scenario = _make_scenario(
        store,
        title="はるの日常",
        owner_character_id=cid,
        pc_slots=pc_slots,
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
        pc_assignments=pc_assignments,
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

    # run_usual_days_scene はシーン完走後にあらすじ自動蒸留を呼ぶ。蒸留本体は
    # プリセット読込・プロバイダー呼び出しを伴うため、ここでは no-op に差し替える
    # （蒸留が「正しい引数で呼ばれるか」は専用テストで別途検証する）。
    async def fake_synopsis(*a, **k):
        return None

    monkeypatch.setattr(svc, "maybe_update_auto_synopsis", fake_synopsis)


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
        # マーカーは表示用 content から除去されている（raw_response には残る）
        last_gm = [t for t in turns if t.speaker_type == "narrator"][-1]
        assert "[SCENE_CLOSE]" not in (last_gm.content or "")
        assert "[SCENE_CLOSE]" in (last_gm.raw_response or "")

    def test_headless_suppresses_early_scene_close(self, sqlite_store, monkeypatch):
        """GM が初手で [SCENE_CLOSE] を出しても、主人公が未発話なら幕引きを抑止すること。

        うつつは「キャラが生きる」のが目的なので、本人が一言も発さないまま GM だけで
        シーンが終わるのは不具合。GM が 1 ターン目から場面を畳もうとしても、主人公
        （キャラ PC）へ一度ターンを回し、その後の GM ターンで改めて幕引きを成立させる。

        台本: GM1=即幕引き → (抑止) → PC1(主人公) → GM2=幕引き → (受理・停止)。
        期待: GM, PC, GM の 3 ターン。PC は最低 1 回呼ばれる。最初の GM の幕引きでは
        終わらない。表示用 content からはマーカーが除去される。
        """
        sid, _ = _build_usual_session(sqlite_store, max_turns=20)
        pc_calls: list[dict] = []
        gm_script = [
            # 1ターン目からいきなり完結した小景を書いて幕引き（主人公はまだ未発話）。
            "Narrator: 朝が来て、一日が静かに終わった。[SCENE_CLOSE]",
            # 主人公が一度発話した後の幕引きは正規に受理される。
            "Narrator: そうして夜は更けていった。[SCENE_CLOSE]",
        ]
        _install_mocks(monkeypatch, gm_script=gm_script, pc_calls=pc_calls)

        result = asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))

        turns = sqlite_store.list_scenario_turns(sid)
        # GM1(抑止) → PC1 → GM2(受理) = 3 ターン。GM だけの 1 ターンで終わっていない。
        assert len(turns) == 3
        assert turns[0].speaker_type == "narrator"
        assert turns[1].speaker_type == "pc"
        assert turns[2].speaker_type == "narrator"
        # 主人公（キャラ PC）が最低 1 回ターンを取っている。
        assert len(pc_calls) == 1
        # 最終的には 2 度目の GM 幕引きでシーンは閉じる。
        assert result["scene_closed"] is True
        # 抑止された 1 ターン目の GM 発話も、表示用 content からマーカーが除去されている。
        assert "[SCENE_CLOSE]" not in (turns[0].content or "")

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

    def test_scene_completion_triggers_synopsis(self, sqlite_store, monkeypatch):
        """シーン完走後に、あらすじ自動蒸留が force=False・セッションのプリセットで呼ばれること。

        うつつは無人ゆえフロントから蒸留を起動できない。代わりに run_usual_days_scene が
        シーン完走を蒸留のチェックポイントにして maybe_update_auto_synopsis を呼ぶ。
        ここでは「正しい引数（force=False、session の synopsis_preset_id）で 1 度だけ
        呼ばれること」を検証する（蒸留本体の中身は別ユニットの責務）。
        """
        sid, _ = _build_usual_session(sqlite_store, max_turns=4)
        pc_calls: list[dict] = []
        _install_mocks(monkeypatch, gm_script=[], pc_calls=pc_calls)

        # _install_mocks の no-op を、引数を記録する版へ差し替える。
        synopsis_calls: list[dict] = []

        async def rec_synopsis(**kwargs):
            synopsis_calls.append(kwargs)
            return None

        monkeypatch.setattr(svc, "maybe_update_auto_synopsis", rec_synopsis)

        asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))

        assert len(synopsis_calls) == 1
        assert synopsis_calls[0]["force"] is False
        assert synopsis_calls[0]["session_id"] == sid
        # _build_usual_session は synopsis_preset_id="preset-usual" を記録している
        assert synopsis_calls[0]["synopsis_preset_id"] == "preset-usual"


class TestUsualUserPc:
    """うつつの「不在のユーザPC」枠を検証する。

    うつつ＝ユーザがそばにいない時間。ユーザは「ターンを取らない不在の PC」として
    GM の pc_summary に並べ、GM の「PC を代弁しない」保護を効かせる一方、無人ループでは
    ルーティング候補から除外してユーザにターンが回らないようにする。
    """

    def test_parse_form_builds_absent_user_slot(self):
        """フォームに呼称が入っていれば、不在マーカー付きのユーザPC枠が組み立てられること。"""
        from backend.api.ui import characters as ui_chars

        form = {"usual_user_label": "太郎", "usual_user_position": "主任。直属の上司。"}
        scenario_kwargs, _ = ui_chars._parse_usual_form(form, "はる")
        slots = scenario_kwargs["pc_slots"]
        user_slot = next(s for s in slots if s["slot_id"] == "user")
        assert user_slot["name"] == "太郎"
        # 最重要の「不在・描かない」が先頭に来る（format_pc_summary の 80 字切り詰め対策）。
        assert user_slot["description"].startswith(ui_chars._USUAL_USER_ABSENT_PREFIX)
        assert "主任" in user_slot["description"]
        # 不在マーカーを除けば位置づけが復元できる（編集画面の再描画用）。
        assert ui_chars._split_usual_user_description(user_slot["description"]) == "主任。直属の上司。"

    def test_parse_form_omits_user_slot_when_label_blank(self):
        """呼称が空ならユーザPC枠は作らない（任意設定）。"""
        from backend.api.ui import characters as ui_chars

        scenario_kwargs, _ = ui_chars._parse_usual_form({}, "はる")
        assert all(s["slot_id"] != "user" for s in scenario_kwargs["pc_slots"])

    def test_ensure_session_adds_user_assignment(self, sqlite_store):
        """ユーザPC枠が定義されたシナリオから、user 割当付きセッションが作られること。"""
        cid, pid = "char-haru", "preset-usual"
        sqlite_store.create_character(cid, "はる")
        sqlite_store.create_model_preset(pid, "うつつ用", "anthropic", "claude-x")
        scenario = _make_scenario(
            sqlite_store,
            title="はるの日常",
            owner_character_id=cid,
            pc_slots=[
                {"slot_id": "pc1", "name": "はる", "description": "主人公。"},
                {"slot_id": "user", "name": "太郎",
                 "description": "【この場面に不在・姿/言動を描かない】主任。"},
            ],
            usual_config={"gm_preset_id": pid, "max_turns_per_scene": 4},
        )
        session = svc.ensure_usual_session(sqlite_store, scenario)
        assert session is not None
        ptypes = [a.get("player_type") for a in session.pc_assignments]
        assert "character" in ptypes
        assert "user" in ptypes

    def test_ensure_session_reconciles_existing(self, sqlite_store):
        """ユーザ枠が後から定義された既存セッションに、user 割当が冪等に補われること。"""
        cid, pid = "char-haru", "preset-usual"
        sqlite_store.create_character(cid, "はる")
        sqlite_store.create_model_preset(pid, "うつつ用", "anthropic", "claude-x")
        # 先に user 枠なしで作る
        scenario = _make_scenario(
            sqlite_store, title="はるの日常", owner_character_id=cid,
            pc_slots=[{"slot_id": "pc1", "name": "はる", "description": "主人公。"}],
            usual_config={"gm_preset_id": pid},
        )
        s1 = svc.ensure_usual_session(sqlite_store, scenario)
        assert all(a.get("player_type") != "user" for a in s1.pc_assignments)

        # 後から user 枠を足す
        sqlite_store.update_scenario(scenario.id, pc_slots=[
            {"slot_id": "pc1", "name": "はる", "description": "主人公。"},
            {"slot_id": "user", "name": "太郎",
             "description": "【この場面に不在・姿/言動を描かない】主任。"},
        ])
        s2 = svc.ensure_usual_session(sqlite_store, sqlite_store.get_scenario(scenario.id))
        assert s2.id == s1.id  # 永続1本（同一セッション）
        assert "user" in [a.get("player_type") for a in s2.pc_assignments]
        # 冪等: もう一度呼んでも user 割当は1つだけ
        s3 = svc.ensure_usual_session(sqlite_store, sqlite_store.get_scenario(scenario.id))
        user_count = sum(1 for a in s3.pc_assignments if a.get("player_type") == "user")
        assert user_count == 1

    def test_user_pc_never_takes_turn(self, sqlite_store, monkeypatch):
        """GM が毎ターン @ユーザ を呼んでも、ユーザPCにはターンが回らないこと。

        ルーティング候補からユーザPCを除外するため、@太郎（ユーザ）指名は GM へ巻き戻り、
        GM↔キャラPC（はる）で回り続ける（早期 break もしない）。
        """
        sid, _ = _build_usual_session(sqlite_store, max_turns=4, user_label="太郎")
        pc_calls: list[dict] = []
        gm_script = [
            "Narrator: 朝。@太郎 はもう出かけた。",
            "Narrator: 昼下がり、静かなオフィス。@太郎",
        ]
        _install_mocks(monkeypatch, gm_script=gm_script, pc_calls=pc_calls)

        asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))

        turns = sqlite_store.list_scenario_turns(sid)
        # GM2 + PC2 = 4 ターン（上限4で停止）。@太郎 指名で早期終了していない。
        assert len(turns) == 4
        # PC ターンは 2 回ともキャラ（はる）。ユーザ（太郎）は 1 度も実行されない。
        assert len(pc_calls) == 2
        assert all(c["pc"].name == "はる" for c in pc_calls)
        assert all(not c["pc"].is_user for c in pc_calls)

    def test_gm_sees_absent_user_in_pc_summary(self, sqlite_store, monkeypatch):
        """GM の pc_summary に、不在ユーザPCが「不在」表記付きで載ること。"""
        sid, _ = _build_usual_session(sqlite_store, max_turns=2, user_label="太郎")
        _install_mocks(monkeypatch, gm_script=[], pc_calls=[])

        # _install_mocks の fake_gm を、pc_summary を記録する版に差し替える。
        gm_kwargs: list[dict] = []

        async def rec_gm(**kwargs):
            gm_kwargs.append(kwargs)
            svc._save_turn(
                sqlite=kwargs["sqlite"], session_id=kwargs["session_id"],
                speaker_type="narrator", speaker_name="Narrator",
                content="Narrator: 静かな朝。@はる", raw_response="Narrator: 静かな朝。@はる",
            )
            return
            yield

        monkeypatch.setattr(svc, "_run_gm_turn", rec_gm)

        asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))

        assert gm_kwargs, "GM ターンが実行されていない"
        summary = gm_kwargs[0]["pc_summary"]
        assert "太郎" in summary       # 不在ユーザが PC ロスターに載る
        assert "不在" in summary       # 不在マーカーが GM へ届いている


# ---------------------------------------------------------------------------
# Phase 3 — GMプロンプト拡張（時間文脈・偶発イベント・SCENE_CLOSE・ソフト収束）
# ---------------------------------------------------------------------------


class TestTimeContext:
    """日本語の時間文脈算出（曜日・時間帯・季節）を検証する。"""

    def test_weekday(self):
        """曜日が月=月〜日=日で算出されること。2026-06-14 は日曜。"""
        assert japanese_weekday(datetime(2026, 6, 14)) == "日"
        assert japanese_weekday(datetime(2026, 6, 15)) == "月"

    def test_season_boundaries(self):
        """季節が月境界で春/夏/秋/冬に切り替わること。"""
        assert japanese_season(datetime(2026, 3, 1)) == "春"
        assert japanese_season(datetime(2026, 5, 31)) == "春"
        assert japanese_season(datetime(2026, 6, 1)) == "夏"
        assert japanese_season(datetime(2026, 9, 1)) == "秋"
        assert japanese_season(datetime(2026, 12, 1)) == "冬"
        assert japanese_season(datetime(2026, 2, 28)) == "冬"

    def test_time_of_day_boundaries(self):
        """時間帯が時刻境界で正しく切り替わること。"""
        assert japanese_time_of_day(datetime(2026, 6, 14, 6)) == "早朝"
        assert japanese_time_of_day(datetime(2026, 6, 14, 9)) == "朝"
        assert japanese_time_of_day(datetime(2026, 6, 14, 12)) == "昼"
        assert japanese_time_of_day(datetime(2026, 6, 14, 15)) == "午後"
        assert japanese_time_of_day(datetime(2026, 6, 14, 18)) == "夕方"
        assert japanese_time_of_day(datetime(2026, 6, 14, 21)) == "夜"
        assert japanese_time_of_day(datetime(2026, 6, 14, 2)) == "深夜"

    def test_format_time_context_contains_all(self):
        """1 文に日付・曜日・時間帯・季節がすべて含まれること。"""
        ctx = format_time_context(datetime(2026, 6, 14, 18))
        assert "2026年6月14日" in ctx
        assert "（日）" in ctx
        assert "夕方" in ctx
        assert "夏" in ctx


class TestUsualEventRoll:
    """偶発イベント抽選（混合方式: 確率で機械抽選・中身はカテゴリのみ）を検証する。"""

    def test_zero_probability_never_fires(self):
        """event_probability=0 では決して発生しないこと。"""
        cfg = {"event_probability": 0.0, "event_categories": ["来客", "残業"]}
        assert svc.roll_usual_event(cfg, rng=random.Random(1)) == ""

    def test_full_probability_always_fires_with_category(self):
        """event_probability=1 では必ず発生し、カテゴリ名を含むこと。"""
        cfg = {"event_probability": 1.0, "event_categories": ["来客"]}
        hint = svc.roll_usual_event(cfg, rng=random.Random(1))
        assert "来客" in hint
        assert "[OOC]" in hint

    def test_no_categories_returns_empty(self):
        """カテゴリ候補が無ければ確率に関わらず空文字列。"""
        assert svc.roll_usual_event({"event_probability": 1.0}, rng=random.Random(1)) == ""

    def test_dict_categories_flattened(self):
        """event_categories が dict（時間帯/季節別）なら全 value を平坦化して候補にすること。"""
        cfg = {
            "event_probability": 1.0,
            "event_categories": {"朝": ["寝坊"], "夜": ["飲み会"]},
        }
        hint = svc.roll_usual_event(cfg, rng=random.Random(0))
        assert ("寝坊" in hint) or ("飲み会" in hint)

    def test_probability_distribution(self):
        """確率 0.3 で多数試行したときの発生率がおおむね 0.3 付近に収まること。"""
        cfg = {"event_probability": 0.3, "event_categories": ["x"]}
        rng = random.Random(12345)
        fired = sum(1 for _ in range(2000) if svc.roll_usual_event(cfg, rng=rng))
        assert 0.25 < fired / 2000 < 0.35


class TestSceneCloseAndSoftHint:
    """[SCENE_CLOSE] 検出と、終盤のソフト収束ヒント注入を検証する。"""

    def test_has_scene_close(self):
        """マーカー検出が大文字小文字を無視して効くこと。"""
        assert svc._has_scene_close("一日が終わった。[SCENE_CLOSE]") is True
        assert svc._has_scene_close("つづく……") is False
        assert svc._has_scene_close("...[scene_close]") is True

    def test_standing_framing_always_present(self):
        """毎ターン、うつつの常設フレーミング（本人の日常・GMは外的フレーム・[SCENE_CLOSE]）が入ること。

        これが無いと GM は普通の TRPG と誤認し主たる停止機構が働かないため、
        中盤の GM ターンでも必ず含まれることを担保する。

        Chotgor 哲学に沿い、メタ的な「ユーザ不在」「人間のプレイヤーがいない」表現は
        使わず、その人物の日常そのものを描く前提を GM に渡している点も検証する
        （文言を変えるたびに哲学が逆戻りしないようガードする目的）。
        """

        class _Scn:
            usual_config = {"event_probability": 0.0}

        scn = _Scn()
        mid = svc._build_usual_gm_appendix(scn, fired_turns=2, max_turns=8, is_first_gm=False)
        # うつつの中心概念（本人の日常をそのまま描く）が必ず含まれる
        assert "日常" in mid
        # 中盤でもマーカーの存在を GM に伝える（主たる停止機構）
        assert "[SCENE_CLOSE]" in mid
        # GM の役回り（外的フレームのみ、内心は代弁しない）が明示されている
        assert "外的な状況" in mid
        # AI 扱い・メタ表現が混入していないこと（哲学ガード）。
        # 本文中で「誰かを演じるための時間ではない」のように「演じる」を否定的に
        # 引用するのは許容するため、ここでは「演じる」自体の含有はチェックしない。
        assert "プレイヤーはいません" not in mid
        assert "ユーザ不在" not in mid

    def test_soft_hint_only_near_end(self):
        """終盤の念押し（ソフト収束ヒント）は残りターンが閾値以下のときだけ加わること。"""

        class _Scn:
            usual_config = {"event_probability": 0.0}

        scn = _Scn()
        # 残り多数 → 常設フレーミングはあるが「畳む頃合い」の念押しは無い
        early = svc._build_usual_gm_appendix(scn, fired_turns=0, max_turns=8, is_first_gm=False)
        assert "畳む頃合い" not in early
        # 残り僅少 → 念押しが加わる
        late = svc._build_usual_gm_appendix(scn, fired_turns=7, max_turns=8, is_first_gm=False)
        assert "畳む頃合い" in late

    def test_event_hint_only_on_first_gm(self):
        """偶発イベントはシーン最初の GM ターンでのみ抽選されること。"""

        class _Scn:
            usual_config = {"event_probability": 1.0, "event_categories": ["来客"]}

        scn = _Scn()
        # is_first_gm=True → イベント抽選される
        first = svc._build_usual_gm_appendix(
            scn, fired_turns=0, max_turns=8, is_first_gm=True, rng=random.Random(1),
        )
        assert "来客" in first
        # is_first_gm=False（残りも多い）→ 常設フレーミングはあるがイベントは出ない
        mid = svc._build_usual_gm_appendix(
            scn, fired_turns=2, max_turns=8, is_first_gm=False, rng=random.Random(1),
        )
        assert "来客" not in mid

    def test_extract_scene_close(self):
        """[SCENE_CLOSE] を本文から除去しつつ検出フラグを返すこと（揺れ対応）。"""
        clean, found = svc.extract_scene_close("一日が終わった。\n[SCENE_CLOSE]")
        assert found is True
        assert "[SCENE_CLOSE]" not in clean
        assert "一日が終わった。" in clean
        # 区切り・大小の揺れ
        _c, f2 = svc.extract_scene_close("はるは眠った。 [ scene close ]")
        assert f2 is True
        # マーカー無しは素通し
        assert svc.extract_scene_close("つづく") == ("つづく", False)


class TestGmPromptWiring:
    """time_context / gm_ooc_appendix が GM system prompt に確実に届くことを検証する。"""

    def test_time_context_and_ooc_appended(self):
        """テンプレに {time_context} が無くても、時間文脈と OOC が末尾に補われること。"""

        class _Scn:
            scenario = "静かな住宅街。"
            custom_system_prompt = "# 役割\nあなたは GM。\n{history_block}"

        prompt = build_gm_system_prompt(
            scenario=_Scn(),
            npcs=[],
            history_text="",
            time_context="いまは 2026年6月14日（日）の夕方。季節は夏。",
            gm_ooc_appendix="[OOC] 今日は「来客」が起きてよい。",
        )
        assert "2026年6月14日" in prompt
        assert "来客" in prompt


# ---------------------------------------------------------------------------
# Phase 4 — スケジューラ（スロット到来判定・冪等性・経過時間・セッション解決）
# ---------------------------------------------------------------------------


def _build_usual_world(store, slots, enabled=True, max_turns=4):
    """セッション未作成のうつつ世界（キャラ＋プリセット＋うつつシナリオ）を組む。

    ensure_usual_session に session を作らせる経路を試すため、ここでは session を作らない。
    Returns:
        scenario オブジェクト。
    """
    cid = "char-haru"
    pid = "preset-usual"
    store.create_character(cid, "はる")
    store.create_model_preset(pid, "うつつ用", "anthropic", "claude-x")
    return _make_scenario(
        store,
        title="はるの日常",
        owner_character_id=cid,
        pc_slots=[{"slot_id": "pc1", "name": "はる", "description": "主人公"}],
        usual_config={
            "enabled": enabled,
            "slots": slots,
            "max_turns_per_scene": max_turns,
            "gm_preset_id": pid,
            "pc_preset_id": pid,
        },
    )


class TestSlotParsing:
    """スロット時刻のパースと到来判定を検証する。"""

    def test_parse_valid_slot(self):
        """"HH:MM" が今日のその時刻の datetime になること。"""
        now = datetime(2026, 6, 14, 18, 30)
        dt = mainmod._parse_slot_time(now, "10:00")
        assert dt == datetime(2026, 6, 14, 10, 0)
        # 10:00 は 18:30 より前 → 到来済み
        assert now >= dt

    def test_parse_invalid_slot(self):
        """不正なスロット文字列は None。"""
        now = datetime(2026, 6, 14, 18, 30)
        assert mainmod._parse_slot_time(now, "なし") is None
        assert mainmod._parse_slot_time(now, "25:00") is None
        assert mainmod._parse_slot_time(now, "") is None


class TestEnsureUsualSession:
    """うつつ永続セッションの find-or-create を検証する。"""

    def test_creates_then_reuses(self, sqlite_store):
        """初回は新規作成、2回目は同じ active セッションを返すこと（永続1本）。"""
        scenario = _build_usual_world(sqlite_store, slots=["10:00"])
        s1 = svc.ensure_usual_session(sqlite_store, scenario)
        assert s1 is not None
        assert s1.engine_type == "usual_days"
        s2 = svc.ensure_usual_session(sqlite_store, scenario)
        assert s2.id == s1.id  # 新規に増やさず再利用

    def test_returns_none_without_preset(self, sqlite_store):
        """GM プリセット未設定なら None（起動不能）。"""
        cid = "char-x"
        sqlite_store.create_character(cid, "X")
        scenario = _make_scenario(
            sqlite_store,
            owner_character_id=cid,
            pc_slots=[{"slot_id": "pc1", "name": "X", "description": ""}],
            usual_config={"enabled": True, "slots": ["10:00"]},  # gm_preset_id なし
        )
        assert svc.ensure_usual_session(sqlite_store, scenario) is None

    def _make_world_with_presets(
        self, store, *, ghost_model=None, pc_preset_id=None,
    ):
        """PC プリセット解決の検証用に、owner キャラ＋各プリセット＋うつつ世界を組む。

        gm/ghost/pc 各プリセットを実レコードとして作り、owner キャラの ghost_model と
        usual_config.pc_preset_id を引数で切り替えられるようにする。

        Returns:
            (scenario, gm_pid) のタプル。
        """
        cid = "char-haru"
        gm_pid = "preset-gm"
        store.create_model_preset(gm_pid, "GM用", "anthropic", "claude-x")
        store.create_model_preset("preset-ghost", "Ghost用", "anthropic", "claude-haiku")
        store.create_model_preset("preset-pc", "PC用", "anthropic", "claude-y")
        store.create_character(cid, "はる", ghost_model=ghost_model)
        cfg = {"enabled": True, "slots": ["10:00"], "gm_preset_id": gm_pid}
        if pc_preset_id is not None:
            cfg["pc_preset_id"] = pc_preset_id
        scenario = _make_scenario(
            store,
            owner_character_id=cid,
            pc_slots=[{"slot_id": "pc1", "name": "はる", "description": ""}],
            usual_config=cfg,
        )
        return scenario, gm_pid

    def test_pc_preset_defaults_to_ghost_model(self, sqlite_store):
        """PC プリセット未指定なら owner キャラの Ghost Model が PC 配役に使われること。"""
        scenario, _ = self._make_world_with_presets(
            sqlite_store, ghost_model="preset-ghost", pc_preset_id=None,
        )
        session = svc.ensure_usual_session(sqlite_store, scenario)
        assert session is not None
        assert session.pc_assignments[0]["preset_id"] == "preset-ghost"

    def test_explicit_pc_preset_overrides_ghost_model(self, sqlite_store):
        """PC プリセットが明示指定されていれば Ghost Model より優先されること。"""
        scenario, _ = self._make_world_with_presets(
            sqlite_store, ghost_model="preset-ghost", pc_preset_id="preset-pc",
        )
        session = svc.ensure_usual_session(sqlite_store, scenario)
        assert session.pc_assignments[0]["preset_id"] == "preset-pc"

    def test_pc_preset_falls_back_to_gm_without_ghost_model(self, sqlite_store):
        """PC プリセットも Ghost Model も無ければ GM プリセットにフォールバックすること。"""
        scenario, gm_pid = self._make_world_with_presets(
            sqlite_store, ghost_model=None, pc_preset_id=None,
        )
        session = svc.ensure_usual_session(sqlite_store, scenario)
        assert session.pc_assignments[0]["preset_id"] == gm_pid


class TestUsualElapsedNote:
    """前回シーンからの経過時間メモを検証する。"""

    def test_empty_history_returns_blank(self, sqlite_store):
        """履歴が無ければ空文字列。"""
        scenario = _build_usual_world(sqlite_store, slots=["10:00"])
        session = svc.ensure_usual_session(sqlite_store, scenario)
        assert svc.usual_elapsed_note(sqlite_store, session.id) == ""

    def test_with_prior_turn(self, sqlite_store):
        """直近ターンがあれば経過時間メモを返すこと。"""
        scenario = _build_usual_world(sqlite_store, slots=["10:00"])
        session = svc.ensure_usual_session(sqlite_store, scenario)
        sqlite_store.create_scenario_turn(
            turn_id="t1", session_id=session.id, turn_index=0,
            speaker_type="narrator", speaker_name="Narrator", content="朝。",
        )
        # 3 時間後を now として与える
        note = svc.usual_elapsed_note(
            sqlite_store, session.id, now=datetime.now() + timedelta(hours=3),
        )
        assert note.startswith("[OOC]")
        assert "経過" in note


class TestUsualScheduler:
    """スケジューラ本体（_run_due_usual_scenes）の発火・冪等性・コストガードを検証する。"""

    def _fake_app(self, sqlite_store):
        """app.state.sqlite / chat_service を持つ最小 app スタブ。"""
        return types.SimpleNamespace(
            state=types.SimpleNamespace(
                sqlite=sqlite_store, chat_service=object(),
            )
        )

    def test_fires_once_per_slot_idempotent(self, sqlite_store, monkeypatch):
        """到来済みスロットは 1 度だけ発火し、同日中の再実行では発火しないこと。"""
        _build_usual_world(sqlite_store, slots=["00:00"])  # 必ず到来済み
        calls: list[str] = []

        async def fake_scene(**kwargs):
            calls.append(kwargs["session_id"])
            return {"saved_turn_ids": [], "fired_turns": 0, "scene_closed": True, "error": None}

        monkeypatch.setattr(svc, "run_usual_days_scene", fake_scene)
        app = self._fake_app(sqlite_store)

        asyncio.run(mainmod._run_due_usual_scenes(app))
        asyncio.run(mainmod._run_due_usual_scenes(app))  # 2 回目は冪等で発火しない

        assert len(calls) == 1

    def test_disabled_world_does_not_fire(self, sqlite_store, monkeypatch):
        """enabled=False のうつつ世界は発火しないこと。"""
        _build_usual_world(sqlite_store, slots=["00:00"], enabled=False)
        calls: list[str] = []

        async def fake_scene(**kwargs):
            calls.append(kwargs["session_id"])
            return {"saved_turn_ids": [], "fired_turns": 0, "scene_closed": True, "error": None}

        monkeypatch.setattr(svc, "run_usual_days_scene", fake_scene)
        asyncio.run(mainmod._run_due_usual_scenes(self._fake_app(sqlite_store)))
        assert calls == []

    def test_daily_cap_blocks(self, sqlite_store, monkeypatch):
        """日次上限に達していると発火しないこと（コストガード）。"""
        _build_usual_world(sqlite_store, slots=["00:00"])
        today = datetime.now().date().isoformat()
        sqlite_store.set_setting("usual_days_daily_cap", "1")
        sqlite_store.set_setting(f"usual_days_scene_count_{today}", "1")  # 既に上限
        calls: list[str] = []

        async def fake_scene(**kwargs):
            calls.append(kwargs["session_id"])
            return {"saved_turn_ids": [], "fired_turns": 0, "scene_closed": True, "error": None}

        monkeypatch.setattr(svc, "run_usual_days_scene", fake_scene)
        asyncio.run(mainmod._run_due_usual_scenes(self._fake_app(sqlite_store)))
        assert calls == []

    def test_future_slot_not_due(self, sqlite_store, monkeypatch):
        """まだ到来していないスロット（未来時刻）は発火しないこと。"""
        _build_usual_world(sqlite_store, slots=["23:59"])
        calls: list[str] = []

        async def fake_scene(**kwargs):
            calls.append(kwargs["session_id"])
            return {"saved_turn_ids": [], "fired_turns": 0, "scene_closed": True, "error": None}

        monkeypatch.setattr(svc, "run_usual_days_scene", fake_scene)
        asyncio.run(mainmod._run_due_usual_scenes(self._fake_app(sqlite_store)))
        # 23:59 が現在より未来ならスキップ（テスト実行が 23:59 以降だと発火しうるが稀）
        if datetime.now() < datetime.now().replace(hour=23, minute=59, second=0, microsecond=0):
            assert calls == []


# ---------------------------------------------------------------------------
# Phase 5 — 管理UI（うつつ設定フォームのパース）
# ---------------------------------------------------------------------------


class TestUsualFormParsing:
    """キャラ編集UIの「うつつ」フォーム解釈（_parse_usual_form）を検証する。"""

    def test_full_form(self):
        """各フィールドが usual_config / scenario_kwargs に正しく変換されること。"""
        form = {
            "usual_enabled": "1",
            "usual_world": "都内の制作会社。",
            "usual_pc_description": "3年目デザイナー。",
            "usual_slots": "10:00, 13:00\n17:00",
            "usual_gm_preset_id": "gm-p",
            "usual_pc_preset_id": "pc-p",
            "usual_event_categories": "来客\n残業\n\n雑談",
            "usual_event_probability": "0.3",
            "usual_max_turns": "6",
            "usual_history_max_turns": "40",
            "usual_history_max_chars": "30000",
            "usual_time_grid": '{"平日朝": "通勤"}',
        }
        scn_kwargs, cfg = _parse_usual_form(form, "はる")

        assert cfg["enabled"] is True
        # スロットはカンマ・改行どちらの区切りも配列化
        assert cfg["slots"] == ["10:00", "13:00", "17:00"]
        # 空行は除去
        assert cfg["event_categories"] == ["来客", "残業", "雑談"]
        assert cfg["event_probability"] == 0.3
        assert cfg["max_turns_per_scene"] == 6
        assert cfg["gm_preset_id"] == "gm-p"
        assert cfg["pc_preset_id"] == "pc-p"
        assert cfg["time_grid"] == {"平日朝": "通勤"}
        # 履歴上限（あらすじ起稿タイミングのノブ）は scenario 列へ保存される
        assert scn_kwargs["history_max_turns"] == 40
        assert scn_kwargs["history_max_chars"] == 30000
        # 主人公 PC 枠はキャラ名で 1 枠
        assert scn_kwargs["pc_slots"][0]["name"] == "はる"
        assert scn_kwargs["pc_slots"][0]["description"] == "3年目デザイナー。"
        assert scn_kwargs["scenario"] == "都内の制作会社。"
        assert scn_kwargs["usual_config"] is cfg

    def test_disabled_and_empty_defaults(self):
        """未チェック・空欄では enabled=False と安全な既定値になること。"""
        scn_kwargs, cfg = _parse_usual_form({}, "X")
        assert cfg["enabled"] is False
        assert cfg["slots"] == []
        assert cfg["event_categories"] == []
        assert cfg["event_probability"] == 0.0
        assert cfg["max_turns_per_scene"] == 8
        assert cfg["time_grid"] == {}
        assert scn_kwargs["scenario"] is None
        # 履歴上限は空欄 → None（設定既定に委ねる）で保存される
        assert scn_kwargs["history_max_turns"] is None
        assert scn_kwargs["history_max_chars"] is None

    def test_invalid_time_grid_json_ignored(self):
        """time_grid の JSON が不正なら空 dict にフォールバックすること。"""
        _scn, cfg = _parse_usual_form({"usual_time_grid": "{壊れた"}, "X")
        assert cfg["time_grid"] == {}

    def test_invalid_numbers_fallback(self):
        """確率・上限ターンが不正値なら既定値にフォールバックすること。"""
        _scn, cfg = _parse_usual_form(
            {"usual_event_probability": "abc", "usual_max_turns": ""}, "X",
        )
        assert cfg["event_probability"] == 0.0
        assert cfg["max_turns_per_scene"] == 8


class TestUsualTimeGridDisplay:
    """時間グリッドの編集画面表示用テキスト整形（_usual_time_grid_text）を検証する。

    Jinja の ``| tojson`` は ``ensure_ascii=True`` で日本語を ``\\uXXXX`` に
    エスケープしてしまい、テキストエリアで読めなくなる。ビュー側ヘルパーが
    日本語をそのまま読める JSON 文字列に整形できることを担保する。
    """

    def test_japanese_not_escaped(self):
        """日本語が ``\\uXXXX`` にエスケープされず、そのまま JSON 文字列になること。"""
        text = _usual_time_grid_text({"time_grid": {"朝": "通勤"}})
        # 生の日本語が含まれ、エスケープ表記が含まれないこと
        assert "朝" in text
        assert "通勤" in text
        assert "\\u" not in text
        # 出力は再パース可能な有効 JSON であること
        assert json.loads(text) == {"朝": "通勤"}

    def test_empty_or_missing_returns_blank(self):
        """time_grid が空・未設定・None なら空文字列を返すこと（プレースホルダ表示用）。"""
        assert _usual_time_grid_text({"time_grid": {}}) == ""
        assert _usual_time_grid_text({}) == ""
        assert _usual_time_grid_text(None) == ""


# ---------------------------------------------------------------------------
# Phase 6 — 1on1 システムプロンプト注釈（うつつ有効時のみ）
# ---------------------------------------------------------------------------


class TestUsualSystemPromptNotice:
    """うつつ有効/無効で 1on1 システムプロンプトの注釈ブロック有無が切り替わることを検証する。"""

    _MARKER = "ユーザが知らない"

    def test_notice_present_when_enabled(self):
        """usual_days_enabled=True で日常生活の注釈が挿入されること。"""
        prompt = build_system_prompt(
            character_system_prompt="あなたは はる。",
            usual_days_enabled=True,
        )
        assert self._MARKER in prompt
        assert "あなたの日常について" in prompt

    def test_notice_absent_when_disabled(self):
        """usual_days_enabled=False（既定）では注釈が出ないこと。"""
        prompt = build_system_prompt(
            character_system_prompt="あなたは はる。",
            usual_days_enabled=False,
        )
        assert self._MARKER not in prompt
        assert "あなたの日常について" not in prompt


# ---------------------------------------------------------------------------
# Phase 7 — 可視性（セッション一覧除外）・ログ feature ラベル
# ---------------------------------------------------------------------------


class TestUsualVisibility:
    """うつつセッションが通常のセッション一覧から隠れることを検証する。"""

    def test_list_scenario_sessions_excludes_usual(self, sqlite_store):
        """list_scenario_sessions は既定で usual_days セッションを除外すること。"""
        # 通常 ensemble シナリオ＋セッション
        normal_scn = _make_scenario(sqlite_store, title="通常")
        sqlite_store.create_scenario_session(
            session_id="sess-normal", scenario_id=normal_scn.id, title="通常",
            gm_preset_id="p", synopsis_preset_id="p", engine_type="ensemble",
        )
        # うつつ世界＋うつつセッション
        scenario = _build_usual_world(sqlite_store, slots=["10:00"])
        usual_session = svc.ensure_usual_session(sqlite_store, scenario)

        default_ids = {s.id for s in sqlite_store.list_scenario_sessions()}
        assert "sess-normal" in default_ids
        assert usual_session.id not in default_ids  # うつつは隠れる

        # include_usual=True なら含む
        all_ids = {s.id for s in sqlite_store.list_scenario_sessions(include_usual=True)}
        assert usual_session.id in all_ids


class TestUsualLogFeature:
    """うつつ無人シーンが /ui/logs 用に feature="usual_days" でログされることを検証する。"""

    def test_headless_sets_feature_label(self, sqlite_store, monkeypatch):
        """headless シーン進行中、GM ターンの feature が "usual_days" であること。

        ContextVar は asyncio.run の外へ伝播しないため、GM ターン実行中の値を捕捉する。
        """
        sid, _ = _build_usual_session(sqlite_store, max_turns=2)
        seen_features: list[str] = []

        async def fake_gm(**kwargs):
            seen_features.append(current_log_feature.get())
            svc._save_turn(
                sqlite=kwargs["sqlite"], session_id=kwargs["session_id"],
                speaker_type="narrator", speaker_name="Narrator",
                content="しずかな夜。", raw_response="しずかな夜。[SCENE_CLOSE]",
            )
            return
            yield

        monkeypatch.setattr(svc, "_run_gm_turn", fake_gm)
        monkeypatch.setattr(svc, "compute_synopsis_progress", lambda *a, **k: None)

        asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))
        assert seen_features == ["usual_days"]


# ---------------------------------------------------------------------------
# 追加対応 — フロント可視化（デバッグ時）・ダイス抑止・エラー時の挙動
# ---------------------------------------------------------------------------


class TestSessionListDebugGate:
    """/api/scenario_chat/sessions がデバッグモード時のみ うつつ を含めることを検証する。"""

    def _setup(self, sqlite_store):
        """通常セッション1件＋うつつセッション1件を用意する。"""
        normal = _make_scenario(sqlite_store, title="通常")
        sqlite_store.create_scenario_session(
            session_id="sess-normal", scenario_id=normal.id, title="通常",
            gm_preset_id="p", synopsis_preset_id="p", engine_type="ensemble",
        )
        scenario = _build_usual_world(sqlite_store, slots=["10:00"])
        usual = svc.ensure_usual_session(sqlite_store, scenario)
        return usual.id

    def test_hidden_when_not_debug(self, sqlite_store, monkeypatch):
        """非デバッグ時は うつつ セッションが一覧に出ないこと。"""
        from fastapi.testclient import TestClient

        import backend.lib.debug_logger as dbg
        from tests._scenario_api_helpers import _build_app

        usual_id = self._setup(sqlite_store)
        monkeypatch.setattr(dbg.logger, "is_debug_enabled", lambda: False)
        client = TestClient(_build_app(sqlite_store))
        ids = {s["id"] for s in client.get("/api/scenario_chat/sessions").json()}
        assert "sess-normal" in ids
        assert usual_id not in ids

    def test_shown_when_debug(self, sqlite_store, monkeypatch):
        """デバッグ時は うつつ セッションも一覧に含まれること（動作監視のため）。"""
        from fastapi.testclient import TestClient

        import backend.lib.debug_logger as dbg
        from tests._scenario_api_helpers import _build_app

        usual_id = self._setup(sqlite_store)
        monkeypatch.setattr(dbg.logger, "is_debug_enabled", lambda: True)
        client = TestClient(_build_app(sqlite_store))
        sessions = client.get("/api/scenario_chat/sessions").json()
        ids = {s["id"] for s in sessions}
        assert usual_id in ids
        # フロントが識別できるよう engine_type が含まれること
        usual = next(s for s in sessions if s["id"] == usual_id)
        assert usual["engine_type"] == "usual_days"


class TestHeadlessDiceSuppressed:
    """うつつ（headless）では GM にダイスプールを渡さないことを検証する。"""

    def test_no_dice_pool_in_headless(self, sqlite_store, monkeypatch):
        """日常シーンに TRPG 用ダイスが注入されないこと（dice_pool が空文字）。"""
        sid, _ = _build_usual_session(sqlite_store, max_turns=2)
        seen_dice: list[str] = []

        async def fake_gm(**kwargs):
            seen_dice.append(kwargs.get("dice_pool", "<absent>"))
            svc._save_turn(
                sqlite=kwargs["sqlite"], session_id=kwargs["session_id"],
                speaker_type="narrator", speaker_name="Narrator",
                content="朝。", raw_response="朝。[SCENE_CLOSE]",
            )
            return
            yield

        monkeypatch.setattr(svc, "_run_gm_turn", fake_gm)
        monkeypatch.setattr(svc, "compute_synopsis_progress", lambda *a, **k: None)

        asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))
        assert seen_dice == [""]  # headless はダイスを渡さない


class TestUsualErrorHandling:
    """GM／PC のエラー時に無人シーンがどう振る舞うかを検証する。"""

    def test_pc_error_surfaced_in_result(self, sqlite_store, monkeypatch):
        """PC（キャラ）応答が例外を投げたら、シーンは打ち切られ result.error に畳まれること。"""
        sid, _ = _build_usual_session(sqlite_store, max_turns=8)

        async def fake_gm(**kwargs):
            svc._save_turn(
                sqlite=kwargs["sqlite"], session_id=kwargs["session_id"],
                speaker_type="narrator", speaker_name="Narrator",
                content="朝。@はる", raw_response="朝。@はる",
            )
            return
            yield

        async def boom_pc(**kwargs):
            raise RuntimeError("provider down")
            yield  # 到達しないが async generator にする

        monkeypatch.setattr(svc, "_run_gm_turn", fake_gm)
        monkeypatch.setattr(pc_runner_mod, "stream_pc_response", boom_pc)
        monkeypatch.setattr(svc, "compute_synopsis_progress", lambda *a, **k: None)

        result = asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))
        # GM ターンは保存され、PC エラーで打ち切り。error に PC エラーが畳まれている
        assert result["error"] is not None
        assert "PC応答エラー" in result["error"]

    def test_gm_exception_surfaced_in_result(self, sqlite_store, monkeypatch):
        """GM ターンが例外を投げたら、シーンは中断し result.error に記録されること。"""
        sid, _ = _build_usual_session(sqlite_store, max_turns=8)

        async def boom_gm(**kwargs):
            raise RuntimeError("gm provider down")
            yield

        monkeypatch.setattr(svc, "_run_gm_turn", boom_gm)
        monkeypatch.setattr(svc, "compute_synopsis_progress", lambda *a, **k: None)

        result = asyncio.run(svc.run_usual_days_scene(
            session_id=sid, sqlite=sqlite_store, settings={}, chat_service=object(),
        ))
        assert result["error"] is not None
        assert "gm provider down" in result["error"]
