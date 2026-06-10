"""シナリオチャット SQLite 永続化 — テンプレート＆NPC のテスト。

検証する観点:
    - シナリオ CRUD（作成・取得・一覧・更新・削除）
    - シナリオ削除時のカスケード（NPC / セッション / ターンすべて消える）
    - シナリオ「更新」時はセッションが残る（重要）
    - NPC CRUD（テンプレに紐づくこと、シナリオ内一意制約）
    - NPC 削除時の発話履歴保持
"""

import uuid

import pytest

from tests._scenario_sqlite_helpers import (
    _make_npc,
    _make_scenario,
    _make_session,
    _make_turn,
)

# ─── Scenario テンプレ CRUD ──────────────────────────────────────────────


class TestScenarioCRUD:
    """シナリオテンプレートの基本 CRUD を検証する。

    Scenario は何度でも遊べる「設定の塊」。
    フィールドが正しく永続化され、updated_at が自動更新されることを確認する。
    """

    def test_create_minimum_fields(self, sqlite_store):
        """必須フィールドだけでシナリオを作成できること。デフォルト値が適切。"""
        scenario = _make_scenario(sqlite_store)
        assert scenario.id
        assert scenario.title == "テストシナリオ"
        # 旧 user_alias は廃止。ユーザPCは pc_slots の先頭枠として保持される。
        assert scenario.pc_slots[0]["name"] == "プレイヤー"
        # 省略フィールドは None
        assert scenario.scenario is None
        assert scenario.history_max_turns is None
        # GM プリセットはテンプレート側には保持されない（セッション単位の設定に移行）。
        assert not hasattr(scenario, "gm_preset_id")

    def test_create_full_fields(self, sqlite_store):
        """全フィールド指定で作成し、すべて永続化されること。"""
        scenario = _make_scenario(
            sqlite_store,
            title="廃墟探索",
            user_alias="ヴァン",
            scenario="魔導書を探す。古城。詩的に語る。slow テンポ。",
            history_max_turns=50,
            history_max_chars=40000,
        )
        fetched = sqlite_store.get_scenario(scenario.id)
        assert fetched is not None
        assert "魔導書" in fetched.scenario
        assert fetched.history_max_turns == 50
        assert fetched.history_max_chars == 40000

    def test_get_nonexistent_returns_none(self, sqlite_store):
        """存在しないシナリオ ID で取得した場合 None。"""
        assert sqlite_store.get_scenario("does-not-exist") is None

    def test_list_returns_newest_first(self, sqlite_store):
        """list_scenarios は updated_at の新しい順で返すこと。"""
        s1 = _make_scenario(sqlite_store, title="A")
        s2 = _make_scenario(sqlite_store, title="B")
        s3 = _make_scenario(sqlite_store, title="C")
        sqlite_store.update_scenario(s1.id, title="A-updated")
        result = sqlite_store.list_scenarios()
        # s1 が直近で更新されたので先頭
        assert result[0].id == s1.id
        remaining_ids = {r.id for r in result[1:]}
        assert s2.id in remaining_ids
        assert s3.id in remaining_ids

    def test_list_respects_limit(self, sqlite_store):
        """limit 引数より多くの結果は返さないこと。"""
        for i in range(5):
            _make_scenario(sqlite_store, title=f"S{i}")
        result = sqlite_store.list_scenarios(limit=3)
        assert len(result) == 3

    def test_update_fields(self, sqlite_store):
        """update_scenario で任意フィールドを更新でき、updated_at が変化すること。"""
        scenario = _make_scenario(sqlite_store)
        original_updated = scenario.updated_at
        updated = sqlite_store.update_scenario(
            scenario.id,
            title="更新後タイトル",
            scenario="新しいシナリオ本文",
        )
        assert updated.title == "更新後タイトル"
        assert updated.scenario == "新しいシナリオ本文"
        assert updated.updated_at >= original_updated

    def test_update_ignores_unknown_kwargs(self, sqlite_store):
        """ORM にないキーは silently 無視されること（hasattr チェック）。"""
        scenario = _make_scenario(sqlite_store)
        updated = sqlite_store.update_scenario(scenario.id, nonexistent_field="x")
        assert updated is not None

    def test_update_nonexistent_returns_none(self, sqlite_store):
        """存在しないシナリオの更新は None。"""
        assert sqlite_store.update_scenario("does-not-exist", title="X") is None


# ─── シナリオ削除のカスケード ────────────────────────────────────────────────


class TestScenarioCascadeDelete:
    """シナリオ削除時に、紐づく NPC・セッション・ターンも一括削除されることを確認する。

    一方、シナリオ「更新」時はセッションが残ることも確認する（重要）。
    """

    def test_delete_removes_npcs(self, sqlite_store):
        """シナリオ削除で NPC も一括削除される。"""
        scenario = _make_scenario(sqlite_store)
        npc1 = _make_npc(sqlite_store, scenario.id, name="レイカ")
        npc2 = _make_npc(sqlite_store, scenario.id, name="トウコ")
        assert sqlite_store.delete_scenario(scenario.id) is True
        assert sqlite_store.get_scenario_npc(npc1.id) is None
        assert sqlite_store.get_scenario_npc(npc2.id) is None

    def test_delete_removes_sessions(self, sqlite_store):
        """シナリオ削除でセッションも一括削除される。"""
        scenario = _make_scenario(sqlite_store)
        s1 = _make_session(sqlite_store, scenario.id)
        s2 = _make_session(sqlite_store, scenario.id, title="別プレイ")
        sqlite_store.delete_scenario(scenario.id)
        assert sqlite_store.get_scenario_session(s1.id) is None
        assert sqlite_store.get_scenario_session(s2.id) is None

    def test_delete_removes_turns(self, sqlite_store):
        """シナリオ削除でセッションのターンも一括削除される。"""
        scenario = _make_scenario(sqlite_store)
        s1 = _make_session(sqlite_store, scenario.id)
        _make_turn(sqlite_store, s1.id, content="残らない")
        _make_turn(sqlite_store, s1.id, content="これも消える")
        sqlite_store.delete_scenario(scenario.id)
        assert sqlite_store.list_scenario_turns(s1.id) == []

    def test_delete_returns_bool(self, sqlite_store):
        """削除成功時 True、存在しない場合 False。"""
        scenario = _make_scenario(sqlite_store)
        assert sqlite_store.delete_scenario(scenario.id) is True
        assert sqlite_store.delete_scenario(scenario.id) is False

    def test_update_keeps_sessions(self, sqlite_store):
        """シナリオ「更新」ではセッションが残ること（仕様の核）。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        _make_turn(sqlite_store, session.id, content="残るべき発話")
        sqlite_store.update_scenario(scenario.id, title="編集後")
        assert sqlite_store.get_scenario_session(session.id) is not None
        turns = sqlite_store.list_scenario_turns(session.id)
        assert len(turns) == 1

    def test_delete_does_not_touch_other_scenarios(self, sqlite_store):
        """シナリオ A 削除でシナリオ B の子レコードに影響がないこと。"""
        sa = _make_scenario(sqlite_store, title="A")
        sb = _make_scenario(sqlite_store, title="B")
        _make_npc(sqlite_store, sa.id, name="A-N")
        npc_b = _make_npc(sqlite_store, sb.id, name="B-N")
        session_b = _make_session(sqlite_store, sb.id)
        sqlite_store.delete_scenario(sa.id)
        assert sqlite_store.get_scenario_npc(npc_b.id) is not None
        assert sqlite_store.get_scenario_session(session_b.id) is not None


# ─── ScenarioNpc CRUD ─────────────────────────────────────────────────────────────


class TestScenarioNpcCRUD:
    """シナリオに紐づく NPC の CRUD と一意制約を検証する。"""

    def test_create_minimum_fields(self, sqlite_store):
        """必須フィールドだけで NPC を作成できること。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="レイカ")
        assert npc.scenario_id == scenario.id
        assert npc.name == "レイカ"
        assert npc.description is None
        assert npc.image_data is None
        assert npc.promoted_character_id is None

    def test_create_full_fields(self, sqlite_store):
        """全フィールド指定で作成。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(
            sqlite_store,
            scenario.id,
            name="トウコ",
            description="無口な少女。敬語が多い。",
            image_data="data:image/png;base64,iVBOR...",
        )
        fetched = sqlite_store.get_scenario_npc(npc.id)
        assert fetched.description == "無口な少女。敬語が多い。"
        assert fetched.image_data and fetched.image_data.startswith("data:image/")

    def test_list_sorted_by_created_at(self, sqlite_store):
        """list_scenario_npcs は created_at 昇順で返す（追加順）。"""
        scenario = _make_scenario(sqlite_store)
        _make_npc(sqlite_store, scenario.id, name="first")
        _make_npc(sqlite_store, scenario.id, name="second")
        _make_npc(sqlite_store, scenario.id, name="third")
        names = [n.name for n in sqlite_store.list_scenario_npcs(scenario.id)]
        assert names == ["first", "second", "third"]

    def test_list_filters_by_scenario(self, sqlite_store):
        """指定シナリオの NPC のみを返すこと。"""
        sa = _make_scenario(sqlite_store, title="A")
        sb = _make_scenario(sqlite_store, title="B")
        _make_npc(sqlite_store, sa.id, name="A-N1")
        _make_npc(sqlite_store, sa.id, name="A-N2")
        _make_npc(sqlite_store, sb.id, name="B-N1")
        assert {n.name for n in sqlite_store.list_scenario_npcs(sa.id)} == {"A-N1", "A-N2"}
        assert {n.name for n in sqlite_store.list_scenario_npcs(sb.id)} == {"B-N1"}

    def test_update_fields(self, sqlite_store):
        """部分更新が反映されること。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="X")
        updated = sqlite_store.update_scenario_npc(
            npc.id,
            description="新プロフィール",
            image_data="新口調",
        )
        assert updated.description == "新プロフィール"
        assert updated.image_data == "新口調"

    def test_update_nonexistent_returns_none(self, sqlite_store):
        """存在しない NPC 更新は None。"""
        assert sqlite_store.update_scenario_npc("does-not-exist", description="X") is None

    def test_delete_returns_bool(self, sqlite_store):
        """削除成功時 True、存在しない場合 False。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="N")
        assert sqlite_store.delete_scenario_npc(npc.id) is True
        assert sqlite_store.delete_scenario_npc(npc.id) is False

    def test_delete_keeps_turn_history(self, sqlite_store):
        """NPC 削除後も、その NPC が過去にしゃべった発話は履歴に残ること。

        speaker_id は元 NPC を参照しているが、NPC レコードはないので参照不能になる。
        speaker_name はスナップショットされているため表示は問題ない。
        """
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        npc = _make_npc(sqlite_store, scenario.id, name="消えるNPC")
        turn = _make_turn(
            sqlite_store,
            session.id,
            speaker_type="npc",
            speaker_id=npc.id,
            speaker_name="消えるNPC",
            content="残る発言",
        )
        sqlite_store.delete_scenario_npc(npc.id)
        turns = sqlite_store.list_scenario_turns(session.id)
        assert len(turns) == 1
        assert turns[0].id == turn.id
        assert turns[0].speaker_id == npc.id  # 参照不能だが値は残る
        assert turns[0].speaker_name == "消えるNPC"
        assert sqlite_store.get_scenario_npc(npc.id) is None

    def test_unique_name_per_scenario(self, sqlite_store):
        """同一シナリオ内で同名 NPC を 2 件作成すると例外。"""
        scenario = _make_scenario(sqlite_store)
        _make_npc(sqlite_store, scenario.id, name="ダブり")
        with pytest.raises(Exception):
            _make_npc(sqlite_store, scenario.id, name="ダブり")

    def test_same_name_across_scenarios_ok(self, sqlite_store):
        """別シナリオであれば同名 NPC を作成できる。"""
        sa = _make_scenario(sqlite_store, title="A")
        sb = _make_scenario(sqlite_store, title="B")
        n1 = _make_npc(sqlite_store, sa.id, name="同名")
        n2 = _make_npc(sqlite_store, sb.id, name="同名")
        assert n1.id != n2.id


