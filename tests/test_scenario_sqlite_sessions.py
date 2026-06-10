"""シナリオチャット SQLite 永続化 — プレイセッション＆ターンのテスト。

検証する観点:
    - セッション CRUD（テンプレ参照、削除時はターンも消える）
    - セッション削除がテンプレに影響しないこと
    - ターン CRUD（next_turn_index、各種多態 speaker_type）
    - エッジケース（存在しない ID）
"""

import uuid

import pytest

from tests._scenario_sqlite_helpers import (
    _make_npc,
    _make_scenario,
    _make_session,
    _make_turn,
)

# ─── ScenarioSession CRUD ────────────────────────────────────────────────────────


class TestScenarioSessionCRUD:
    """プレイインスタンス CRUD を検証する。"""

    def test_create(self, sqlite_store):
        """シナリオからセッションを起動できること。gm_preset_id がセッションに紐づくこと。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(
            sqlite_store,
            scenario.id,
            title="プレイ #1",
            gm_preset_id="preset-session-A",
        )
        assert session.id
        assert session.scenario_id == scenario.id
        assert session.title == "プレイ #1"
        assert session.status == "active"
        assert session.engine_type == "ensemble"
        assert session.gm_preset_id == "preset-session-A"

    def test_create_two_sessions_can_have_different_presets(self, sqlite_store):
        """同一シナリオから別の GM プリセットでセッションを起動できること。

        Scenario からセッション単位へ gm_preset_id を移した目的のコア要件。
        """
        scenario = _make_scenario(sqlite_store)
        s_a = _make_session(sqlite_store, scenario.id, gm_preset_id="preset-A")
        s_b = _make_session(sqlite_store, scenario.id, gm_preset_id="preset-B")
        assert s_a.gm_preset_id == "preset-A"
        assert s_b.gm_preset_id == "preset-B"

    def test_update_session_gm_preset(self, sqlite_store):
        """セッションの gm_preset_id を後から変更できること（ヘッダーUI のモデル切替に対応）。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id, gm_preset_id="preset-X")
        updated = sqlite_store.update_scenario_session(
            session.id, gm_preset_id="preset-Y"
        )
        assert updated.gm_preset_id == "preset-Y"

    def test_get_nonexistent_returns_none(self, sqlite_store):
        assert sqlite_store.get_scenario_session("does-not-exist") is None

    def test_list_returns_newest_first(self, sqlite_store):
        """セッション一覧が updated_at の新しい順で返ること。"""
        scenario = _make_scenario(sqlite_store)
        s1 = _make_session(sqlite_store, scenario.id, title="A")
        s2 = _make_session(sqlite_store, scenario.id, title="B")
        sqlite_store.update_scenario_session(s1.id, title="A-updated")
        result = sqlite_store.list_scenario_sessions()
        assert result[0].id == s1.id
        assert {r.id for r in result} == {s1.id, s2.id}

    def test_list_by_scenario(self, sqlite_store):
        """list_scenario_sessions_by_scenario は指定シナリオのセッションのみ返す。"""
        sa = _make_scenario(sqlite_store, title="A")
        sb = _make_scenario(sqlite_store, title="B")
        _make_session(sqlite_store, sa.id, title="A1")
        _make_session(sqlite_store, sa.id, title="A2")
        _make_session(sqlite_store, sb.id, title="B1")
        a_titles = {s.title for s in sqlite_store.list_scenario_sessions_by_scenario(sa.id)}
        b_titles = {s.title for s in sqlite_store.list_scenario_sessions_by_scenario(sb.id)}
        assert a_titles == {"A1", "A2"}
        assert b_titles == {"B1"}

    def test_update_session(self, sqlite_store):
        """セッションのタイトル / status を更新できる。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        updated = sqlite_store.update_scenario_session(
            session.id, title="リネーム", status="ended"
        )
        assert updated.title == "リネーム"
        assert updated.status == "ended"

    def test_update_nonexistent_returns_none(self, sqlite_store):
        assert sqlite_store.update_scenario_session("nope", title="x") is None

    def test_delete_session_removes_turns(self, sqlite_store):
        """セッション削除でターンも消える。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        _make_turn(sqlite_store, session.id, content="a")
        _make_turn(sqlite_store, session.id, content="b")
        assert sqlite_store.delete_scenario_session(session.id) is True
        assert sqlite_store.list_scenario_turns(session.id) == []
        assert sqlite_store.get_scenario_session(session.id) is None

    def test_delete_session_keeps_scenario_and_npcs(self, sqlite_store):
        """セッション削除はシナリオ・NPC に影響しないこと。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="N")
        session = _make_session(sqlite_store, scenario.id)
        sqlite_store.delete_scenario_session(session.id)
        assert sqlite_store.get_scenario(scenario.id) is not None
        assert sqlite_store.get_scenario_npc(npc.id) is not None

    def test_delete_nonexistent_returns_false(self, sqlite_store):
        assert sqlite_store.delete_scenario_session("nope") is False


# ─── ScenarioTurn CRUD ────────────────────────────────────────────────────────────


class TestScenarioTurnCRUD:
    """発話ターンの CRUD と多態 speaker_type を検証する。"""

    def test_create_user_turn(self, sqlite_store):
        """user 種別のターンを作成できる。speaker_id は NULL でよい。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        turn = _make_turn(
            sqlite_store,
            session.id,
            speaker_type="user",
            speaker_name="プレイヤー",
            content="話しかける",
        )
        assert turn.speaker_type == "user"
        assert turn.speaker_id is None
        assert turn.turn_index == 0

    def test_create_turn_with_anticipation(self, sqlite_store):
        """anticipation（GM の予想）を指定してターンを作成すると保存・取得できること。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        sqlite_store.create_scenario_turn(
            turn_id=str(uuid.uuid4()),
            session_id=session.id,
            turn_index=0,
            speaker_type="npc",
            speaker_name="レイカ",
            content="……来たんだ",
            anticipation="このあとプレイヤーは戸惑うと予想",
        )
        turns = sqlite_store.list_scenario_turns(session.id)
        assert turns[0].anticipation == "このあとプレイヤーは戸惑うと予想"

    def test_create_turn_anticipation_defaults_none(self, sqlite_store):
        """anticipation を指定しない場合は None として保存されること。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        _make_turn(sqlite_store, session.id, content="やあ")
        turns = sqlite_store.list_scenario_turns(session.id)
        assert turns[0].anticipation is None

    def test_create_narrator_turn(self, sqlite_store):
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        turn = _make_turn(
            sqlite_store,
            session.id,
            speaker_type="narrator",
            speaker_name="Narrator",
            content="雨が降っている",
        )
        assert turn.speaker_type == "narrator"
        assert turn.speaker_id is None

    def test_create_npc_turn_known(self, sqlite_store):
        """既知 NPC の発話ターンを作成でき、speaker_id が NPC.id を参照する。"""
        scenario = _make_scenario(sqlite_store)
        npc = _make_npc(sqlite_store, scenario.id, name="レイカ")
        session = _make_session(sqlite_store, scenario.id)
        turn = _make_turn(
            sqlite_store,
            session.id,
            speaker_type="npc",
            speaker_id=npc.id,
            speaker_name="レイカ",
            content="……来たんだ",
        )
        assert turn.speaker_id == npc.id

    def test_create_npc_turn_unknown(self, sqlite_store):
        """未知 NPC（ephemeral）のターンは speaker_id=None で速記名のみ保存。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        turn = _make_turn(
            sqlite_store,
            session.id,
            speaker_type="npc",
            speaker_id=None,
            speaker_name="モブの店主",
            content="お買い物ですかい？",
        )
        assert turn.speaker_id is None
        assert turn.speaker_name == "モブの店主"

    def test_raw_response_stored(self, sqlite_store):
        """raw_response にデバッグ用の生 LLM 出力を保存できる。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        raw = "@Narrator: 雨\n@レイカ: 来た"
        turn = _make_turn(
            sqlite_store,
            session.id,
            speaker_type="narrator",
            speaker_name="Narrator",
            content="雨",
            raw_response=raw,
        )
        fetched = sqlite_store.list_scenario_turns(session.id)[0]
        assert fetched.raw_response == raw

    def test_list_turns_sorted_by_turn_index(self, sqlite_store):
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        _make_turn(sqlite_store, session.id, content="2nd", turn_index=1)
        _make_turn(sqlite_store, session.id, content="3rd", turn_index=2)
        _make_turn(sqlite_store, session.id, content="1st", turn_index=0)
        contents = [t.content for t in sqlite_store.list_scenario_turns(session.id)]
        assert contents == ["1st", "2nd", "3rd"]

    def test_list_turns_empty_session(self, sqlite_store):
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        assert sqlite_store.list_scenario_turns(session.id) == []

    def test_next_turn_index_initial(self, sqlite_store):
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        assert sqlite_store.get_next_scenario_turn_index(session.id) == 0

    def test_next_turn_index_after_creates(self, sqlite_store):
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        _make_turn(sqlite_store, session.id)
        _make_turn(sqlite_store, session.id)
        _make_turn(sqlite_store, session.id)
        assert sqlite_store.get_next_scenario_turn_index(session.id) == 3

    def test_delete_turns_from_pivot(self, sqlite_store):
        """delete_scenario_turns_from は指定ターン以降（自身含む）を一括削除する。

        ユーザ発話の編集 / GM ターン再生成の前処理として使う。
        pivot より前のターンは残り、pivot 以降は全部消える。
        """
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        t0 = _make_turn(sqlite_store, session.id, content="0")
        t1 = _make_turn(sqlite_store, session.id, content="1")  # pivot
        _make_turn(sqlite_store, session.id, content="2")
        _make_turn(sqlite_store, session.id, content="3")
        assert sqlite_store.delete_scenario_turns_from(session.id, t1.id) is True
        contents = [t.content for t in sqlite_store.list_scenario_turns(session.id)]
        assert contents == ["0"]

    def test_delete_turns_from_unknown_returns_false(self, sqlite_store):
        """存在しない turn_id を渡すと False（影響なし）。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        _make_turn(sqlite_store, session.id, content="残す")
        assert sqlite_store.delete_scenario_turns_from(session.id, "nope") is False
        assert len(sqlite_store.list_scenario_turns(session.id)) == 1

    def test_delete_turns_from_other_session_returns_false(self, sqlite_store):
        """別セッションの turn_id を渡しても削除されない。"""
        scenario = _make_scenario(sqlite_store)
        sa = _make_session(sqlite_store, scenario.id, title="A")
        sb = _make_session(sqlite_store, scenario.id, title="B")
        t_a = _make_turn(sqlite_store, sa.id, content="A0")
        _make_turn(sqlite_store, sb.id, content="B0")
        assert sqlite_store.delete_scenario_turns_from(sb.id, t_a.id) is False
        assert len(sqlite_store.list_scenario_turns(sa.id)) == 1
        assert len(sqlite_store.list_scenario_turns(sb.id)) == 1

    def test_delete_turns_from_clamps_synopsis_last_turn_index(self, sqlite_store):
        """ターン削除時、`synopsis_last_turn_index` が削除域に踏み込んでいたら
        `pivot.turn_index - 1` までクランプされること。

        これは「過去のあるタイミングで synopsis が turn_index=125 まで蒸留済み
        だったが、その後ユーザがターン編集ロールバックで turn 5 以降を削除した」
        という具体的シナリオへの根治対応。クランプを怠ると、以降の自動蒸留が
        `new_dropped = []` 判定で永久にスキップされ、あらすじが二度と
        再生成されなくなる（旧ユーザ報告：「あらすじ作成が効いていない」現象）。
        """
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        t0 = _make_turn(sqlite_store, session.id, content="0")
        _make_turn(sqlite_store, session.id, content="1")
        t2 = _make_turn(sqlite_store, session.id, content="2")  # pivot (turn_index=2)
        _make_turn(sqlite_store, session.id, content="3")
        # 過去に蒸留が進んでいた状態を模す（last_turn_index=3 まで処理済）
        sqlite_store.update_scenario_session_synopsis(
            session.id, auto="蒸留済みあらすじ", last_turn_index=3,
        )
        # turn_index=2 以降を削除
        assert sqlite_store.delete_scenario_turns_from(session.id, t2.id) is True
        # クランプ確認: 削除済み境界より小さい値（pivot - 1 = 1）に丸まる
        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        assert synopsis["last_turn_index"] == 1
        # auto 本文は触らない（ユーザ編集の保護とは別軸）
        assert synopsis["auto"] == "蒸留済みあらすじ"

    def test_delete_turns_from_preserves_lower_synopsis_index(self, sqlite_store):
        """既に `synopsis_last_turn_index` が削除域より低い場合は据え置きであること。

        ロールバック範囲が「まだ蒸留されていないターン群」だけのケース。
        この場合は last_turn_index を巻き戻す必要がないし、巻き戻すと
        蒸留済みあらすじとポインタが食い違うので触らないのが正しい。
        """
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        _make_turn(sqlite_store, session.id, content="0")
        _make_turn(sqlite_store, session.id, content="1")
        _make_turn(sqlite_store, session.id, content="2")
        t3 = _make_turn(sqlite_store, session.id, content="3")  # pivot (turn_index=3)
        # last_turn_index=1（pivot=3 より小さい）
        sqlite_store.update_scenario_session_synopsis(
            session.id, auto="あらすじ", last_turn_index=1,
        )
        assert sqlite_store.delete_scenario_turns_from(session.id, t3.id) is True
        # クランプは作用しない（pivot - 1 = 2 > 既存値 1 なので据え置き）
        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        assert synopsis["last_turn_index"] == 1

    def test_delete_turns_from_first_turn_resets_synopsis_index(self, sqlite_store):
        """先頭ターン（turn_index=0）から削除した場合、`last_turn_index` は -1 まで戻ること。

        セッションを実質的に「白紙巻き戻し」した状態。蒸留進捗ポインタも
        新規セッション同等（-1）に戻すのが妥当。
        """
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        t0 = _make_turn(sqlite_store, session.id, content="0")
        _make_turn(sqlite_store, session.id, content="1")
        sqlite_store.update_scenario_session_synopsis(
            session.id, auto="蒸留済み", last_turn_index=1,
        )
        assert sqlite_store.delete_scenario_turns_from(session.id, t0.id) is True
        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        # pivot.turn_index = 0 → クランプ先は -1
        assert synopsis["last_turn_index"] == -1

    def test_next_turn_index_per_session(self, sqlite_store):
        """next_turn_index はセッションごとに独立してカウントされる。"""
        scenario = _make_scenario(sqlite_store)
        sa = _make_session(sqlite_store, scenario.id, title="A")
        sb = _make_session(sqlite_store, scenario.id, title="B")
        _make_turn(sqlite_store, sa.id)
        _make_turn(sqlite_store, sa.id)
        _make_turn(sqlite_store, sb.id)
        assert sqlite_store.get_next_scenario_turn_index(sa.id) == 2
        assert sqlite_store.get_next_scenario_turn_index(sb.id) == 1


