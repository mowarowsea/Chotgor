"""シナリオチャット SQLite 永続化レイヤーのテスト。

ScenarioChatStoreMixin が提供する以下 4 テーブルへの CRUD 操作を網羅的に検証する:
  - scenarios : シナリオテンプレート
  - scenario_npcs      : テンプレに紐づく NPC
  - scenario_sessions  : テンプレから起動されたプレイインスタンス
  - scenario_turns     : セッションの発話履歴

検証する観点:
    - シナリオ CRUD（作成・取得・一覧・更新・削除）
    - シナリオ削除時のカスケード（NPC / セッション / ターンすべて消える）
    - シナリオ「更新」時はセッションが残る（重要）
    - NPC CRUD（テンプレに紐づくこと、シナリオ内一意制約）
    - NPC 削除時の発話履歴保持
    - セッション CRUD（テンプレ参照、削除時はターンも消える）
    - セッション削除がテンプレに影響しないこと
    - ターン CRUD（next_turn_index、各種多態 speaker_type）
    - エッジケース（存在しない ID）
"""

import uuid

import pytest


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


def _make_scenario(
    store,
    title: str = "テストシナリオ",
    user_alias: str = "プレイヤー",
    gm_preset_id: str = "preset-test",
    **kwargs,
):
    """シナリオテンプレートを 1 件作成して返すユーティリティ。"""
    scenario_id = str(uuid.uuid4())
    return store.create_scenario(
        scenario_id=scenario_id,
        title=title,
        user_alias=user_alias,
        gm_preset_id=gm_preset_id,
        **kwargs,
    )


def _make_npc(store, scenario_id: str, name: str = "レイカ", **kwargs):
    """シナリオ内 NPC を 1 件作成して返すユーティリティ。"""
    npc_id = str(uuid.uuid4())
    return store.create_scenario_npc(
        npc_id=npc_id,
        scenario_id=scenario_id,
        name=name,
        **kwargs,
    )


def _make_session(
    store,
    scenario_id: str,
    title: str = "プレイ #1",
    engine_type: str = "ensemble",
):
    """プレイセッションを 1 件作成して返すユーティリティ。"""
    session_id = str(uuid.uuid4())
    return store.create_scenario_session(
        session_id=session_id,
        scenario_id=scenario_id,
        title=title,
        engine_type=engine_type,
    )


def _make_turn(
    store,
    session_id: str,
    speaker_type: str = "user",
    speaker_name: str = "プレイヤー",
    content: str = "こんにちは",
    speaker_id: str = None,
    raw_response: str = None,
    turn_index: int = None,
):
    """発話ターンを 1 件作成して返すユーティリティ。"""
    turn_id = str(uuid.uuid4())
    if turn_index is None:
        turn_index = store.get_next_scenario_turn_index(session_id)
    return store.create_scenario_turn(
        turn_id=turn_id,
        session_id=session_id,
        turn_index=turn_index,
        speaker_type=speaker_type,
        speaker_name=speaker_name,
        content=content,
        speaker_id=speaker_id,
        raw_response=raw_response,
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
        assert scenario.user_alias == "プレイヤー"
        assert scenario.gm_preset_id == "preset-test"
        # 省略フィールドは None
        assert scenario.scenario is None
        assert scenario.history_max_turns is None

    def test_create_full_fields(self, sqlite_store):
        """全フィールド指定で作成し、すべて永続化されること。"""
        scenario = _make_scenario(
            sqlite_store,
            title="廃墟探索",
            user_alias="ヴァン",
            gm_preset_id="preset-X",
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


# ─── ScenarioSession CRUD ────────────────────────────────────────────────────────


class TestScenarioSessionCRUD:
    """プレイインスタンス CRUD を検証する。"""

    def test_create(self, sqlite_store):
        """シナリオからセッションを起動できること。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id, title="プレイ #1")
        assert session.id
        assert session.scenario_id == scenario.id
        assert session.title == "プレイ #1"
        assert session.status == "active"
        assert session.engine_type == "ensemble"

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
