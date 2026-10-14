"""シナリオログの枝分かれ（レスポンスガチャ）と手動書き換えのテスト。

設計は docs/planned/scenario_turn_variants_plan.md を参照。

検証する観点:
    - 非活性化した枝が履歴（本線）から消え、DB には残ること
    - 兄弟枝の判定（同一 branch_point_index の generation 同士）と枝番号の採番
    - 枝の切替で下流が巻き戻ること・下流は復元しないこと
    - turn_index が飛んでも会話順が保たれること
    - 封筒（タイムライン）の retract と再活性化時の append
    - 1 レスポンス全体の手動書き換え
    - ユーザ発話の編集（物理削除）が枝ごと一掃すること
    - あらすじ境界（synopsis_last_turn_index）のクランプ
"""

import uuid

import pytest

from backend.lib.log_context import (
    current_branch_point_index,
    current_generation_id,
)
from tests._scenario_sqlite_helpers import _make_scenario, _make_session, _make_turn


def _make_generation(
    store,
    session_id: str,
    generation_id: str,
    branch_point_index: int,
    contents: list[str],
    speaker_type: str = "narrator",
):
    """1 リクエスト分の応答群（＝1 つの枝）を保存するヘルパ。

    実運用では stream エンドポイントが ContextVar へ枝情報を積むので、
    テストでも同じ経路（ContextVar）を通して create_scenario_turn を呼ぶ。

    Args:
        contents: 保存する発話本文のリスト（同一 generation の話者ブロック群）。

    Returns:
        保存した ScenarioTurn のリスト。
    """
    gen_token = current_generation_id.set(generation_id)
    branch_token = current_branch_point_index.set(branch_point_index)
    try:
        return [
            _make_turn(
                store,
                session_id,
                speaker_type=speaker_type,
                speaker_name="Narrator",
                content=c,
                raw_response=generation_id,
            )
            for c in contents
        ]
    finally:
        current_generation_id.reset(gen_token)
        current_branch_point_index.reset(branch_token)


@pytest.fixture
def session_with_first_response(sqlite_store):
    """「reqA → resA」まで進んだセッションを用意する。

    resA は generation "genA1"（分岐点 = reqA の turn_index）として保存される。
    レスポンスガチャの各テストはここから枝を生やす。
    """
    scenario = _make_scenario(sqlite_store)
    session = _make_session(sqlite_store, scenario.id)
    req_a = _make_turn(sqlite_store, session.id, content="reqA")
    _make_generation(sqlite_store, session.id, "genA1", req_a.turn_index, ["resA-1"])
    return session, req_a


class TestVariantBasics:
    """枝の生成・非活性化・本線フィルタの基本挙動を検証する。"""

    def test_active_only_in_history(self, sqlite_store, session_with_first_response):
        """非活性化した枝は list_scenario_turns（本線）から消えること。

        履歴を読む経路はほぼ全て list_scenario_turns 経由なので、
        ここが効いていれば prompt_builder / synopsis / chronicle からも枝は見えない。
        """
        session, req_a = session_with_first_response
        turns = sqlite_store.list_scenario_turns(session.id)
        assert [t.content for t in turns] == ["reqA", "resA-1"]

        res_a1 = turns[-1]
        assert sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id) is True

        assert [t.content for t in sqlite_store.list_scenario_turns(session.id)] == ["reqA"]

    def test_deactivated_rows_remain_in_db(
        self, sqlite_store, session_with_first_response
    ):
        """非活性化は物理削除ではなく、行は DB に残り続けること（枝として選び直せる）。"""
        session, _ = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)

        variants = sqlite_store.list_scenario_generation_variants(session.id)
        assert "genA1" in variants  # 本線から外れても枝としては生きている

    def test_generation_id_recorded_for_user_turn_too(self, sqlite_store):
        """ユーザ発話も同じ generation に入ること。

        ユーザ発話を枝から外すと、枝を戻したときにユーザ発話だけが
        非活性のまま取り残されるため、同一リクエストのターンは全部束ねる。
        """
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        turns = _make_generation(
            sqlite_store, session.id, "gen1", -1, ["req", "res"],
        )
        # speaker_type に関わらず generation_id が振られる
        assert all(t.generation_id == "gen1" for t in turns)

    def test_intro_turns_have_no_generation(self, sqlite_store):
        """ContextVar 未設定（stream 外）のターンは枝を持たないこと。

        intro の展開など、枝の対象外の保存経路が誤って枝扱いされないことの確認。
        """
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        turn = _make_turn(sqlite_store, session.id, content="intro")
        assert turn.generation_id is None
        assert turn.branch_point_index == -1


class TestSiblingVariants:
    """兄弟枝の判定と枝番号の採番を検証する。"""

    def test_same_branch_point_becomes_siblings(
        self, sqlite_store, session_with_first_response
    ):
        """同じ分岐点から生えた generation 同士が兄弟枝になること。"""
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        # ガチャ引き直し: resA-1 を巻き戻して同じ分岐点から生やす
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        _make_generation(
            sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"],
        )

        variants = sqlite_store.list_scenario_generation_variants(session.id)
        assert variants["genA1"] == {
            "index": 1, "count": 2, "siblings": ["genA1", "genA2"],
        }
        assert variants["genA2"]["index"] == 2

    def test_different_branch_point_is_not_sibling(
        self, sqlite_store, session_with_first_response
    ):
        """分岐点が違えば兄弟にならないこと。

        ユーザ発話を編集して作り直した応答が、編集前のガチャと同じ枝リストに
        並んでしまわないことの担保でもある。
        """
        session, _ = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        # 別の分岐点（resA-1 の後）から生やす＝会話が先へ進んだ場合
        _make_generation(
            sqlite_store, session.id, "genB1", res_a1.turn_index, ["resB-1"],
        )

        variants = sqlite_store.list_scenario_generation_variants(session.id)
        assert variants["genA1"]["count"] == 1
        assert variants["genB1"]["count"] == 1

    def test_variant_numbering_follows_creation_order(
        self, sqlite_store, session_with_first_response
    ):
        """枝番号が生成順（先頭 turn_index 昇順）で振られること。"""
        session, req_a = session_with_first_response
        for gen in ("genA2", "genA3"):
            tail = sqlite_store.list_scenario_turns(session.id)[-1]
            sqlite_store.deactivate_scenario_turns_from(session.id, tail.id)
            _make_generation(sqlite_store, session.id, gen, req_a.turn_index, [gen])

        variants = sqlite_store.list_scenario_generation_variants(session.id)
        assert [variants[g]["index"] for g in ("genA1", "genA2", "genA3")] == [1, 2, 3]
        assert variants["genA1"]["siblings"] == ["genA1", "genA2", "genA3"]


class TestActivateGeneration:
    """枝の切替（本線の差し替え）を検証する。"""

    def test_switch_back_to_previous_variant(
        self, sqlite_store, session_with_first_response
    ):
        """引き直した枝から元の枝へ戻せること（レスポンスガチャの主目的）。"""
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        _make_generation(sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"])
        assert [t.content for t in sqlite_store.list_scenario_turns(session.id)] == [
            "reqA", "resA-2",
        ]

        assert sqlite_store.activate_scenario_generation(session.id, "genA1") is True
        assert [t.content for t in sqlite_store.list_scenario_turns(session.id)] == [
            "reqA", "resA-1",
        ]

    def test_switching_past_variant_rolls_back_downstream(
        self, sqlite_store, session_with_first_response
    ):
        """過去の枝へ切り替えると、その分岐点より後の本線が巻き戻ること。

        「resB/reqB を捨てて resA を選び直す」操作に相当する。
        """
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        # resA-1 のガチャ違い（genA2）を作り、そこから会話を先へ進める
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        gen_a2 = _make_generation(
            sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"],
        )
        _make_generation(
            sqlite_store, session.id, "genB1", gen_a2[-1].turn_index, ["reqB", "resB"],
        )
        assert len(sqlite_store.list_scenario_turns(session.id)) == 4

        # 過去の枝 genA1 へ戻す → reqB/resB ごと巻き戻る
        assert sqlite_store.activate_scenario_generation(session.id, "genA1") is True
        assert [t.content for t in sqlite_store.list_scenario_turns(session.id)] == [
            "reqA", "resA-1",
        ]

    def test_downstream_is_not_restored(
        self, sqlite_store, session_with_first_response
    ):
        """巻き戻した下流は、元の枝へ戻しても復元されないこと（部分木復元はしない）。

        案2.5（parent_generation_id による部分木復元）を採用しなかったことの明示。
        """
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        gen_a2 = _make_generation(
            sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"],
        )
        _make_generation(
            sqlite_store, session.id, "genB1", gen_a2[-1].turn_index, ["resB"],
        )

        sqlite_store.activate_scenario_generation(session.id, "genA1")
        sqlite_store.activate_scenario_generation(session.id, "genA2")
        # genA2 の続きだった resB は戻ってこない
        assert [t.content for t in sqlite_store.list_scenario_turns(session.id)] == [
            "reqA", "resA-2",
        ]

    def test_conversation_order_survives_index_gaps(
        self, sqlite_store, session_with_first_response
    ):
        """turn_index が飛んでも会話順が保たれること。

        turn_index は非活性行を含む max+1 で採番されるため枝の増減で番号が飛ぶ。
        親が必ず子より先に採番される性質により、活性行の昇順＝会話順になる。
        """
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        gen_a2 = _make_generation(
            sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"],
        )
        # genA1 へ戻してから会話を続ける（新ターンの番号は genA2 より後になる）
        sqlite_store.activate_scenario_generation(session.id, "genA1")
        _make_generation(
            sqlite_store, session.id, "genB1", res_a1.turn_index, ["resB"],
        )

        turns = sqlite_store.list_scenario_turns(session.id)
        assert [t.content for t in turns] == ["reqA", "resA-1", "resB"]
        indices = [t.turn_index for t in turns]
        assert indices == sorted(indices)
        # 非活性の genA2 のぶんだけ番号が飛んでいる（＝詰め直していない）
        assert gen_a2[0].turn_index not in indices

    def test_activate_returns_false_for_unknown_or_current(
        self, sqlite_store, session_with_first_response
    ):
        """存在しない枝・既に本線の枝は False を返すこと。"""
        session, _ = session_with_first_response
        assert sqlite_store.activate_scenario_generation(session.id, "nope") is False
        assert sqlite_store.activate_scenario_generation(session.id, "genA1") is False


class TestFailedRegenerateRestore:
    """引き直しに失敗したときの復元シーケンスを検証する。

    フロント（useScenarioChat）の引き直しは
      1. 起点ユーザ発話以降を非活性化（枝として保持）
      2. 再ストリーム
    という順で走り、2 がエラーで終わった場合に
      3. 失敗した試行で保存された分を物理削除
      4. 元の枝を再活性化
    で巻き戻しを戻す。ガチャを外しただけで直前のレスポンスを失わないための経路で、
    ここではその 1→4 が「元に戻る」ことと、後片付けの安全性を確認する。
    """

    def _setup(self, sqlite_store):
        """「intro → (reqA, resA)」まで進んだセッションを用意する。

        引き直しの起点はユーザ発話なので、ユーザ発話と GM 応答が同じ枝
        （gen1）に入っている実運用と同じ形にしておく。

        Returns:
            (session, intro ターン, gen1 のターン列)
        """
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        intro = _make_turn(sqlite_store, session.id, content="intro")
        gen1 = _make_generation(
            sqlite_store, session.id, "gen1", intro.turn_index, ["reqA", "resA"],
        )
        return session, intro, gen1

    def test_restores_original_response_after_failed_attempt(self, sqlite_store):
        """引き直しが失敗しても、元のユーザ発話とレスポンスが本線へ戻ること。"""
        session, intro, gen1 = self._setup(sqlite_store)

        # 1. 引き直し: 起点ユーザ発話（gen1 の先頭）以降を巻き戻す
        assert sqlite_store.deactivate_scenario_turns_from(session.id, gen1[0].id) is True
        assert [t.content for t in sqlite_store.list_scenario_turns(session.id)] == ["intro"]

        # 2. 再ストリームがユーザ発話だけ保存してエラーで終わった
        failed = _make_generation(
            sqlite_store, session.id, "gen2", intro.turn_index, ["reqA"],
        )

        # 3. 失敗分を物理削除 → 4. 元の枝を再活性化
        assert sqlite_store.delete_scenario_turns_from(session.id, failed[0].id) is True
        assert sqlite_store.activate_scenario_generation(session.id, "gen1") is True

        assert [t.content for t in sqlite_store.list_scenario_turns(session.id)] == [
            "intro", "reqA", "resA",
        ]

    def test_failed_attempt_leaves_no_sibling_branch(self, sqlite_store):
        """失敗した試行が、中身のない兄弟枝として枝ナビに残らないこと。

        非活性のまま放置すると、引き直しを外すたびに「ユーザ発話だけの枝」が
        兄弟として増え、枝ナビ（◀ 1/3 ▶）が実体のない選択肢で埋まる。
        """
        session, intro, gen1 = self._setup(sqlite_store)
        sqlite_store.deactivate_scenario_turns_from(session.id, gen1[0].id)
        failed = _make_generation(
            sqlite_store, session.id, "gen2", intro.turn_index, ["reqA"],
        )

        sqlite_store.delete_scenario_turns_from(session.id, failed[0].id)
        sqlite_store.activate_scenario_generation(session.id, "gen1")

        variants = sqlite_store.list_scenario_generation_variants(session.id)
        assert "gen2" not in variants
        assert variants["gen1"]["count"] == 1

    def test_cleanup_keeps_variants_drawn_before(self, sqlite_store):
        """失敗分の後片付けが、それ以前に引いた兄弟枝を巻き添えにしないこと。

        物理削除は「指定 turn_index 以降を枝ごと」消すため、失敗分が常に最新の
        turn_index を持つ（＝最後に保存された）ことに依存している。ここが崩れると
        過去のガチャ結果まで消える。
        """
        session, intro, gen1 = self._setup(sqlite_store)
        # 1回目の引き直しは成功して gen2 が生えた
        sqlite_store.deactivate_scenario_turns_from(session.id, gen1[0].id)
        gen2 = _make_generation(
            sqlite_store, session.id, "gen2", intro.turn_index, ["reqA", "resA-2"],
        )
        # 2回目の引き直しはユーザ発話だけ保存して失敗した
        sqlite_store.deactivate_scenario_turns_from(session.id, gen2[0].id)
        failed = _make_generation(
            sqlite_store, session.id, "gen3", intro.turn_index, ["reqA"],
        )

        sqlite_store.delete_scenario_turns_from(session.id, failed[0].id)
        sqlite_store.activate_scenario_generation(session.id, "gen2")

        assert [t.content for t in sqlite_store.list_scenario_turns(session.id)] == [
            "intro", "reqA", "resA-2",
        ]
        variants = sqlite_store.list_scenario_generation_variants(session.id)
        assert set(variants) == {"gen1", "gen2"}
        assert variants["gen2"]["count"] == 2


class TestTimelineEnvelopes:
    """枝の出入りとタイムライン封筒（めぐり）の整合を検証する。"""

    def _scene_envelopes(self, sqlite_store, character_id: str) -> list:
        """対象キャラの scene.turn 封筒を全件返す（retract 済みも含む）。"""
        from backend.repositories.sqlite.models import TimelineEvent

        with sqlite_store.get_session() as s:
            return (
                s.query(TimelineEvent)
                .filter(
                    TimelineEvent.character_id == character_id,
                    TimelineEvent.event_type == "scene.turn",
                )
                .all()
            )

    @pytest.fixture
    def usual_session(self, sqlite_store):
        """封筒が作られる条件（うつつ世界＝所有者キャラあり）のセッションを用意する。"""
        char_id = str(uuid.uuid4())
        sqlite_store.create_character(character_id=char_id, name="はる")
        scenario = _make_scenario(sqlite_store, owner_character_id=char_id)
        session = _make_session(
            sqlite_store, scenario.id, engine_type="usual_days",
        )
        return session, char_id

    def test_deactivate_retracts_envelope(self, sqlite_store, usual_session):
        """非活性化した枝の封筒が retracted になること（不可逆性の担保）。"""
        session, char_id = usual_session
        turns = _make_generation(sqlite_store, session.id, "gen1", -1, ["res"])

        sqlite_store.deactivate_scenario_turns_from(session.id, turns[0].id)

        envelopes = self._scene_envelopes(sqlite_store, char_id)
        assert len(envelopes) == 1
        assert envelopes[0].retracted_at is not None

    def test_reactivate_appends_new_envelope(self, sqlite_store, usual_session):
        """再活性化では retract を戻さず、新しい封筒を積むこと。

        retract の取り消しは「なかったことが再びあったことになる」不可逆性の破れ。
        追記型を保つため、本線へ戻った事実を新しい封筒として記録する。
        """
        session, char_id = usual_session
        turns = _make_generation(sqlite_store, session.id, "gen1", -1, ["res"])
        sqlite_store.deactivate_scenario_turns_from(session.id, turns[0].id)
        _make_generation(sqlite_store, session.id, "gen2", -1, ["res2"])

        sqlite_store.activate_scenario_generation(session.id, "gen1")

        envelopes = self._scene_envelopes(sqlite_store, char_id)
        retracted = [e for e in envelopes if e.retracted_at is not None]
        alive = [e for e in envelopes if e.retracted_at is None]
        # gen1(初回) と gen2 が retract 済み、gen1 の再活性化ぶんが 1 通生きている
        assert len(retracted) == 2
        assert len(alive) == 1


    def test_rewrite_reenvelopes_only_changed_speakers(
        self, sqlite_store, usual_session
    ):
        """書き換えでは話者が変わった行・増えた行だけ封筒が積み直され、
        消えた行の封筒は retract されること。本文だけの変更では封筒に触らない。"""
        session, char_id = usual_session
        turns = _make_generation(sqlite_store, session.id, "gen1", -1, ["A", "B", "C"])
        before = len(self._scene_envelopes(sqlite_store, char_id))
        assert before == 3

        sqlite_store.rewrite_scenario_response(
            session.id, turns[0].id,
            _blocks(("Narrator", "A改"), ("レイカ", "B改")),
        )

        envelopes = self._scene_envelopes(sqlite_store, char_id)
        alive = {e.source_id: e for e in envelopes if e.retracted_at is None}
        # 1 行目: 本文のみ変更 → 元の封筒が生きたまま（積み直しなし）
        # 2 行目: Narrator → レイカ → 旧封筒 retract + 新封筒（actor=npc:レイカ）
        # 3 行目: 削除 → retract のみ
        assert set(alive) == {turns[0].id, turns[1].id}
        assert alive[turns[1].id].actor == "npc:レイカ"
        assert len(envelopes) == 4


def _blocks(*pairs):
    """(話者名, 本文) の組から store へ渡すブロック列を作る（全員 Narrator 型以外は NPC 型）。"""
    return [
        {
            "speaker_type": "narrator" if name == "Narrator" else "npc",
            "speaker_id": None,
            "speaker_name": name,
            "content": content,
        }
        for name, content in pairs
    ]


class TestResponseRewrite:
    """1 レスポンス全体の手動書き換え（store 層）を検証する。

    設計は scenario_turn_variants_plan.md「レスポンス単位の書き換え」。
    観点:
        - 起点がどのバブルでも、同一 raw_response の連続区間（1 レスポンス）が対象になること
        - 同数ブロックは id・turn_index を再利用して本文と話者だけ差し替えること
        - ブロックが減ると余りの行が物理削除されること
        - ブロックが増えると後続の turn_index / branch_point_index / 蒸留境界が繰り下がり、
          会話順と兄弟枝の判定が壊れないこと
        - reasoning は先頭行、anticipation は末尾行へ付け替わること
        - raw_response（モデルの生出力記録）は汚さないこと
        - ユーザ発話・存在しないターンは対象外（None）
    """

    @pytest.fixture
    def two_responses(self, sqlite_store):
        """reqA → resA(2 ブロック) → reqB → resB(1 ブロック) まで進んだセッション。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        req_a = _make_turn(sqlite_store, session.id, content="reqA")
        res_a = _make_generation(
            sqlite_store, session.id, "genA", req_a.turn_index, ["A-1", "A-2"],
        )
        token_g = current_generation_id.set("genB")
        token_b = current_branch_point_index.set(res_a[-1].turn_index)
        try:
            req_b = _make_turn(sqlite_store, session.id, content="reqB")
            res_b = _make_turn(
                sqlite_store, session.id, speaker_type="narrator",
                speaker_name="Narrator", content="B-1", raw_response="genB",
            )
        finally:
            current_generation_id.reset(token_g)
            current_branch_point_index.reset(token_b)
        return session, res_a, req_b, res_b

    def test_same_count_reuses_rows(self, sqlite_store, two_responses):
        """同数なら既存行の id / turn_index を保ったまま本文・話者が差し替わること。"""
        session, res_a, _, _ = two_responses
        result = sqlite_store.rewrite_scenario_response(
            session.id, res_a[1].id, _blocks(("Narrator", "新1"), ("レイカ", "新2")),
        )
        assert result["index_shift"] == 0
        assert [t.id for t in result["turns"]] == [t.id for t in res_a]
        assert [(t.speaker_name, t.content) for t in result["turns"]] == [
            ("Narrator", "新1"), ("レイカ", "新2"),
        ]

    def test_raw_response_untouched(self, sqlite_store, two_responses):
        """手編集で raw_response を汚さないこと（増えた行も末尾行の値を引き継ぐ）。"""
        session, res_a, _, _ = two_responses
        result = sqlite_store.rewrite_scenario_response(
            session.id, res_a[0].id,
            _blocks(("Narrator", "x"), ("Narrator", "y"), ("Narrator", "z")),
        )
        assert {t.raw_response for t in result["turns"]} == {"genA"}
        assert {t.generation_id for t in result["turns"]} == {"genA"}

    def test_shrink_deletes_extra_rows(self, sqlite_store, two_responses):
        """ブロックが減ると余りの行が DB から消えること（枝として残さない）。"""
        session, res_a, _, _ = two_responses
        sqlite_store.rewrite_scenario_response(
            session.id, res_a[0].id, _blocks(("Narrator", "ひとつだけ")),
        )
        contents = [t.content for t in sqlite_store.list_scenario_turns(session.id)]
        assert contents == ["reqA", "ひとつだけ", "reqB", "B-1"]
        from backend.repositories.sqlite.store import ScenarioTurn
        with sqlite_store.get_session() as s:
            assert s.get(ScenarioTurn, res_a[1].id) is None

    def test_grow_keeps_order_and_shifts_following(self, sqlite_store, two_responses):
        """ブロックが増えても会話順が保たれ、後続の turn_index が繰り下がること。"""
        session, res_a, req_b, res_b = two_responses
        result = sqlite_store.rewrite_scenario_response(
            session.id, res_a[0].id,
            _blocks(("Narrator", "1"), ("レイカ", "2"), ("Narrator", "3"), ("トウコ", "4")),
        )
        assert result["index_shift"] == 2
        turns = sqlite_store.list_scenario_turns(session.id)
        assert [t.content for t in turns] == ["reqA", "1", "2", "3", "4", "reqB", "B-1"]
        indexes = [t.turn_index for t in turns]
        assert indexes == sorted(indexes) and len(set(indexes)) == len(indexes)
        by_id = {t.id: t for t in turns}
        assert by_id[req_b.id].turn_index == req_b.turn_index + 2
        assert by_id[res_b.id].turn_index == res_b.turn_index + 2

    def test_grow_shifts_branch_point_of_later_generation(
        self, sqlite_store, two_responses
    ):
        """後に生えた枝の分岐点が新しい末尾行へ追従し、兄弟判定が保たれること。"""
        session, res_a, req_b, _ = two_responses
        sqlite_store.rewrite_scenario_response(
            session.id, res_a[0].id,
            _blocks(("Narrator", "1"), ("Narrator", "2"), ("Narrator", "3")),
        )
        turns = sqlite_store.list_scenario_turns(session.id)
        new_tail = [t for t in turns if t.content == "3"][0]
        moved_req_b = [t for t in turns if t.id == req_b.id][0]
        assert moved_req_b.branch_point_index == new_tail.turn_index

    def test_grow_shifts_synopsis_boundary(self, sqlite_store, two_responses):
        """レスポンス全体が蒸留済みなら、境界も新しい末尾行へ繰り下がること。"""
        session, res_a, _, _ = two_responses
        sqlite_store.update_scenario_session_synopsis(
            session.id, last_turn_index=res_a[-1].turn_index,
        )
        result = sqlite_store.rewrite_scenario_response(
            session.id, res_a[0].id,
            _blocks(("Narrator", "1"), ("Narrator", "2"), ("Narrator", "3")),
        )
        assert result["synopsis_last_turn_index"] == res_a[-1].turn_index + 1
        assert (
            sqlite_store.get_scenario_session(session.id).synopsis_last_turn_index
            == result["turns"][-1].turn_index
        )

    def test_reasoning_and_anticipation_move_to_ends(self, sqlite_store, two_responses):
        """スケッチは先頭行、予想は末尾行へ付け替わること（ブロック数が変わっても）。"""
        session, res_a, _, _ = two_responses
        sqlite_store.update_scenario_turn(res_a[0].id, reasoning="スケッチ")
        sqlite_store.update_scenario_turn(res_a[1].id, anticipation="予想")
        result = sqlite_store.rewrite_scenario_response(
            session.id, res_a[0].id,
            _blocks(("Narrator", "1"), ("Narrator", "2"), ("Narrator", "3")),
        )
        assert [t.reasoning for t in result["turns"]] == ["スケッチ", None, None]
        assert [t.anticipation for t in result["turns"]] == [None, None, "予想"]

    def test_user_turn_and_unknown_are_not_targets(self, sqlite_store, two_responses):
        """ユーザ発話・存在しないターンを起点にすると None を返すこと。"""
        session, _, req_b, _ = two_responses
        blocks = _blocks(("Narrator", "x"))
        assert sqlite_store.rewrite_scenario_response(session.id, req_b.id, blocks) is None
        assert sqlite_store.rewrite_scenario_response(session.id, "nope", blocks) is None


class TestUserEditWipesVariants:
    """ユーザ発話の編集（物理削除）が枝ごと一掃することを検証する。"""

    def test_delete_removes_inactive_variants_too(
        self, sqlite_store, session_with_first_response
    ):
        """物理削除は非活性な枝も含めて消すこと。

        発言そのものを書き換える以上、その発言に対して引いた過去のガチャは
        すべて無効。内容の違う発話が同じ枝リストに並ぶのも防ぐ。
        """
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        _make_generation(sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"])

        # reqA から物理削除 → 枝（genA1 / genA2）ごと消える
        assert sqlite_store.delete_scenario_turns_from(session.id, req_a.id) is True
        assert sqlite_store.list_scenario_turns(session.id) == []
        assert sqlite_store.list_scenario_generation_variants(session.id) == {}


class TestSynopsisBoundaryClamp:
    """巻き戻し時のあらすじ境界クランプを検証する。"""

    def test_deactivate_clamps_boundary(
        self, sqlite_store, session_with_first_response
    ):
        """非活性化域に蒸留境界が含まれるとき、境界が巻き戻し点の直前まで下がること。

        クランプを怠ると「巻き戻したターンまで蒸留済み」の誤認識が残り、
        以降あらすじが二度と再生成されなくなる。
        """
        session, _ = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        sqlite_store.update_scenario_session_synopsis(
            session.id, auto="ここまでのあらすじ", last_turn_index=res_a1.turn_index,
        )

        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)

        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        assert synopsis["last_turn_index"] == res_a1.turn_index - 1

    def test_activate_clamps_boundary(
        self, sqlite_store, session_with_first_response
    ):
        """枝の切替でも境界がクランプされること（切替は巻き戻しを伴うため）。"""
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        gen_a2 = _make_generation(
            sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"],
        )
        sqlite_store.update_scenario_session_synopsis(
            session.id, auto="genA2 までのあらすじ",
            last_turn_index=gen_a2[-1].turn_index,
        )

        sqlite_store.activate_scenario_generation(session.id, "genA1")

        synopsis = sqlite_store.get_scenario_session_synopsis(session.id)
        assert synopsis["last_turn_index"] == req_a.turn_index


class TestTurnWindowWithVariants:
    """履歴ウィンドウ取得（limit / before_index）が枝と共存することを検証する。

    UI は直近ウィンドウしか読まないため、ウィンドウの切り出しが「本線」に対して
    行われることが前提になる。非活性の枝行は `turn_index` を消費した状態で DB に
    残り続けるので、以下が崩れると表示がずれる:

    - 末尾 limit 件は**活性行だけ**を数えて取ること（枝を数に含めない）
    - `before_index` は活性行の turn_index を境界に使い、枝を飛ばして遡れること
    - 枝の増減で turn_index が飛んでも、ウィンドウ内の並びが会話順であること

    設計は docs/planned/scenario_history_perf_plan.md を参照。
    """

    def test_tail_window_counts_active_turns_only(
        self, sqlite_store, session_with_first_response
    ):
        """非活性の枝は末尾 N 件の数に入らない。"""
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        # genA1 を捨てて genA2 を本線にする（genA1 は非活性で DB に残る）
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        _make_generation(
            sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"],
        )

        window = sqlite_store.list_scenario_turns(session.id, limit=2)

        assert [t.content for t in window] == ["reqA", "resA-2"]

    def test_before_index_skips_inactive_branch(
        self, sqlite_store, session_with_first_response
    ):
        """遡りの境界に枝の turn_index が挟まっても、本線だけを遡れる。"""
        session, req_a = session_with_first_response
        res_a1 = sqlite_store.list_scenario_turns(session.id)[-1]
        sqlite_store.deactivate_scenario_turns_from(session.id, res_a1.id)
        gen_a2 = _make_generation(
            sqlite_store, session.id, "genA2", req_a.turn_index, ["resA-2"],
        )
        # 本線: reqA → resA-2。resA-2 より手前を遡ると reqA だけが返る。
        older = sqlite_store.list_scenario_turns(
            session.id, limit=5, before_index=gen_a2[-1].turn_index,
        )

        assert [t.content for t in older] == ["reqA"]

    def test_window_keeps_conversation_order(self, sqlite_store):
        """turn_index が飛んでいてもウィンドウ内は会話順（昇順）で返る。"""
        scenario = _make_scenario(sqlite_store)
        session = _make_session(sqlite_store, scenario.id)
        for i in range(3):
            req = _make_turn(sqlite_store, session.id, content=f"req{i}")
            gen = _make_generation(
                sqlite_store, session.id, f"gen{i}a", req.turn_index, [f"res{i}a"],
            )
            # 各ターンで 1 回引き直す → 捨てた枝が turn_index を消費して番号が飛ぶ
            sqlite_store.deactivate_scenario_turns_from(session.id, gen[0].id)
            _make_generation(
                sqlite_store, session.id, f"gen{i}b", req.turn_index, [f"res{i}b"],
            )

        window = sqlite_store.list_scenario_turns(session.id, limit=4)

        assert [t.content for t in window] == ["req1", "res1b", "req2", "res2b"]
        indexes = [t.turn_index for t in window]
        assert indexes == sorted(indexes)
