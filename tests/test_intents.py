"""意図（intents）のテスト — めぐり（巡り / Aliveness）Phase 4。

検証対象（docs/planned/aliveness_plan.md §4.3）:
    1. IntentStoreMixin: 作成・一覧・終端遷移と、intent.created / expired /
       soured 封筒の同一トランザクション直書き（intent_id FK 込み）
    2. lifecycle: 意図圧の読み取り時計算 g(起点からの経過日数)＝源圧に依存しないこと、
       一区切り（settled）による起点リセット、終端遷移の候補挙げ（14日超 active の1リスト）
    3. pickup: 設問文の組み立て（既存 active・候補の添付）と
       返答タグ（INTENT_NEW / FULFILLED / SETTLED / RELEASE / SOURED）のパース堅牢性、
       run_intent_pickup の適用（LLM はモック）
    4. intent_settler: 1on1 本文タグからの決着宣言（抽出・ID解決・適用）
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from backend.character_actions.intent_settler import (
    apply_intent_marks,
    extract_intent_marks,
    resolve_intent_ref,
)
from backend.services.intents.lifecycle import (
    intent_pressure,
    stale_candidates,
)
from backend.services.intents.pickup import (
    build_pickup_question,
    parse_pickup_response,
    run_intent_pickup,
)


def _make_character(sqlite_store, name="はるテスト", ghost_model=None):
    """テスト用キャラクターを1体作成して返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(
        character_id=char_id, name=name, ghost_model=ghost_model
    )
    return char_id, name


def _backdate_intent(sqlite_store, intent_id: str, days: float) -> None:
    """意図の created_at を days 日前へ巻き戻すヘルパ（経過日数のテスト用）。"""
    from backend.repositories.sqlite.models import Intent
    with sqlite_store.get_session() as s:
        intent = s.get(Intent, intent_id)
        intent.created_at = datetime.now() - timedelta(days=days)
        s.commit()


class TestIntentStore:
    """intents テーブルの CRUD と封筒 dual-write を検証するテストクラス。"""

    def test_create_writes_envelope(self, sqlite_store):
        """作成で intent.created 封筒（intent_id FK 付き）が同時に載る。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(
            char_id, "あの本の続きを読みたい", target="self",
            source_kind="boredom", born_from="night_chronicle",
        )
        assert intent.status == "active"
        events = sqlite_store.list_timeline_events(
            char_id, event_type_prefixes=["intent."]
        )
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "intent.created"
        assert ev.intent_id == intent.id
        assert ev.source_table == "intents"
        # target=self は「相手」ではないので counterpart は空
        assert ev.counterpart is None

    def test_create_with_user_target_sets_counterpart(self, sqlite_store):
        """target=user の意図は封筒の counterpart=user になる（GM への envelope 開示判定用）。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.create_intent(char_id, "話したいことがある", target="user")
        ev = sqlite_store.list_timeline_events(char_id)[0]
        assert ev.counterpart == "user"

    def test_resolve_transitions_once(self, sqlite_store):
        """終端遷移は一度だけ成立し、封筒に遷移イベントが載る。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(char_id, "散歩したい")
        resolved = sqlite_store.resolve_intent(intent.id, "fulfilled")
        assert resolved.status == "fulfilled"
        assert resolved.resolved_at is not None
        # 二度目の遷移は不成立
        assert sqlite_store.resolve_intent(intent.id, "expired") is None
        types = [
            e.event_type
            for e in sqlite_store.list_timeline_events(
                char_id, event_type_prefixes=["intent."]
            )
        ]
        assert types == ["intent.created", "intent.fulfilled"]

    def test_soured_freezes_words(self, sqlite_store):
        """soured 遷移で本人の不満の言葉が intent.payload と封筒 payload に凍結される。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(char_id, "旅行に行きたい")
        resolved = sqlite_store.resolve_intent(
            intent.id, "soured", words="ずっと行けないままだ",
        )
        assert resolved.payload["resolution_words"] == "ずっと行けないままだ"
        ev = [
            e for e in sqlite_store.list_timeline_events(char_id)
            if e.event_type == "intent.soured"
        ][0]
        assert ev.payload == {"words": "ずっと行けないままだ"}

    def test_list_filters_by_status(self, sqlite_store):
        """一覧の status フィルタが効く（既定は active のみ）。"""
        char_id, _ = _make_character(sqlite_store)
        a = sqlite_store.create_intent(char_id, "A")
        sqlite_store.create_intent(char_id, "B")
        sqlite_store.resolve_intent(a.id, "fulfilled")
        assert [i.description for i in sqlite_store.list_intents(char_id)] == ["B"]
        assert len(sqlite_store.list_intents(char_id, status=None)) == 2

    def test_invalid_status_raises(self, sqlite_store):
        """active への「遷移」など不正な遷移先はフェイルファスト。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(char_id, "X")
        import pytest
        with pytest.raises(ValueError):
            sqlite_store.resolve_intent(intent.id, "active")


class _FakeIntent:
    """lifecycle テスト用の Intent スタブ。"""

    def __init__(self, days_old: float, source_kind: str = "none"):
        self.created_at = datetime.now() - timedelta(days=days_old)
        self.source_kind = source_kind
        self.id = str(uuid.uuid4())
        self.description = "テスト意図"


class TestLifecycle:
    """意図圧の読み取り時計算と終端遷移の候補挙げを検証するテストクラス。"""

    def test_pressure_grows_with_age(self):
        """経過日数とともに意図圧が単調に増える（14日で 1.0 に飽和）。"""
        young = intent_pressure(_FakeIntent(1))
        old = intent_pressure(_FakeIntent(10))
        saturated = intent_pressure(_FakeIntent(30))
        assert young < old < saturated
        assert abs(saturated - 1.0) < 1e-9

    def test_pressure_ignores_source_pressure(self):
        """意図圧は source_kind に左右されない（2026-08-27 の再設計）。

        源圧を乗算していた頃は、源圧の低い意図が行動権の閾値へ永久に到達できなかった。
        社会圧が 0（＝平常どおり会えている）でも「これを話したい」は時間とともに育つ、
        というのが再設計後の意味。source_kind は由来の記録としてのみ残る。
        """
        assert intent_pressure(_FakeIntent(10, "social")) == intent_pressure(
            _FakeIntent(10, "none")
        )

    def test_urge_threshold_reached_in_eight_days(self):
        """行動権の閾値 0.7 へ 8.0 日で到達する（設計書 §4.3 の数値）。"""
        assert intent_pressure(_FakeIntent(7.9)) < 0.7
        assert intent_pressure(_FakeIntent(8.1)) >= 0.7

    def test_stale_candidates(self):
        """14日を超えて active のままの意図だけが、古い順に候補として挙がる。"""
        old = _FakeIntent(20, "social")
        older = _FakeIntent(40, "none")
        young = _FakeIntent(3, "social")    # 行動権がまだ拾いに来る領分
        candidates = stale_candidates([old, older, young])
        assert candidates == [older, old]

    def test_settled_resets_pressure_origin(self):
        """一区切り（settled）が打たれると意図圧が下限へ戻る。

        意図圧の唯一の減衰源。源圧の乗算を外した 2026-08-27 以降、意図圧は経過日数の
        単調増加関数で「下がる経路が終端遷移しかない」状態になっていた。settled は
        終端せずに起点だけを今へ移す遷移で、「まだ持っているが今は落ち着いた」を表す。
        """
        intent = _FakeIntent(30)  # 飽和（1.0）まで抱えた意図
        assert abs(intent_pressure(intent) - 1.0) < 1e-9
        just_settled = intent_pressure(intent, settled_at=datetime.now())
        assert abs(just_settled - 0.3) < 1e-3

    def test_settled_pressure_rebuilds_over_time(self):
        """settled 後も時間が経てば圧は積み上がる（やっぱり納得いかない、の再燃）。

        settled は「手放す」ではないので、8日後には再び行動権の閾値 0.7 へ到達する。
        創発として再燃するのが狙いで、再燃のための追加実装は持たない。
        """
        intent = _FakeIntent(60)
        settled_at = datetime.now() - timedelta(days=8.1)
        assert intent_pressure(intent, settled_at=settled_at) >= 0.7
        settled_recent = datetime.now() - timedelta(days=7.9)
        assert intent_pressure(intent, settled_at=settled_recent) < 0.7

    def test_settled_intent_leaves_stale_candidates(self):
        """一区切り済みの意図は 14日の裁定候補から外れる。

        起点が settled へ移るため経過日数が測り直される。これが無いと、
        一区切りついたばかりの意図が「しばらく経っている」として毎晩問われてしまう。
        """
        old = _FakeIntent(20, "social")
        settled = _FakeIntent(40, "none")
        candidates = stale_candidates(
            [old, settled], settled_map={settled.id: datetime.now()}
        )
        assert candidates == [old]


class TestPickupParsing:
    """拾い上げの設問組み立てと返答パースを検証するテストクラス。"""

    def test_question_includes_actives_and_candidates(self, sqlite_store):
        """設問に既存 active 一覧と終端遷移の候補が添えられ、4つの答え方が示される。"""
        char_id, _ = _make_character(sqlite_store)
        active = sqlite_store.create_intent(char_id, "歌の練習を続けたい")
        question = build_pickup_question([active], [active])
        assert "歌の練習を続けたい" in question
        assert active.id in question
        assert "なければないでいい" in question
        for tag in ("INTENT_NEW", "INTENT_FULFILLED", "INTENT_RELEASE", "INTENT_SOURED"):
            assert tag in question

    def test_parse_new_with_target(self):
        """INTENT_NEW の説明と相手（省略・正規化含む）をパースできる。"""
        text = (
            "そうだね……\n"
            "[INTENT_NEW: 海を見に行きたい | self]\n"
            "[INTENT_NEW: もわに絵を見せたい | user]\n"
            "[INTENT_NEW: 店主さんと話したい | 店主]\n"
            "[INTENT_NEW: ただ思っただけ]"
        )
        parsed = parse_pickup_response(text)
        assert parsed["new"][:3] == [
            {"description": "海を見に行きたい", "target": "self"},
            {"description": "もわに絵を見せたい", "target": "user"},
            {"description": "店主さんと話したい", "target": "npc:店主"},
        ]
        # 上限3件で切られる（暴走ガード）
        assert len(parsed["new"]) == 3

    def test_parse_release_and_soured(self):
        """RELEASE / SOURED（不満の言葉つき）をパースできる。"""
        iid = str(uuid.uuid4())
        iid2 = str(uuid.uuid4())
        text = f"[INTENT_RELEASE: {iid}]\n[INTENT_SOURED: {iid2} | ずっと叶わなくてもどかしい]"
        parsed = parse_pickup_response(text)
        assert parsed["release"] == [iid]
        assert parsed["soured"] == [{"id": iid2, "words": "ずっと叶わなくてもどかしい"}]

    def test_parse_fulfilled(self):
        """INTENT_FULFILLED（もう果たした）を RELEASE と混同せずパースできる。"""
        iid = str(uuid.uuid4())
        parsed = parse_pickup_response(f"あれはもわに話せたよ。[INTENT_FULFILLED: {iid}]")
        assert parsed["fulfilled"] == [iid]
        assert parsed["release"] == []

    def test_parse_settled(self):
        """INTENT_SETTLED（一区切り）を FULFILLED と混同せずパースできる。

        タグ名が INTENT_FULFILLED と接頭辞を共有しないため誤照合はしないが、
        「満ちた」と「落ち着いた」は意味が違う（前者は終端、後者は active 継続）ので
        別枠で返ることを固定する。
        """
        iid = str(uuid.uuid4())
        parsed = parse_pickup_response(f"ひとまず落ち着いた。[INTENT_SETTLED: {iid}]")
        assert parsed["settled"] == [iid]
        assert parsed["fulfilled"] == []

    def test_parse_drops_empty_description(self):
        """「なし」だけの INTENT_NEW は意図として登録しない。

        設問が「なければないでいい」と促すため、本人が素直に
        `[INTENT_NEW: なし]` と答えることがある。これをそのまま登録すると、
        捏造遮断の文言が逆に在庫を増やす（はるの実データで1件混入していた）。
        """
        parsed = parse_pickup_response(
            "\n".join([
                "[INTENT_NEW: なし]",
                "[INTENT_NEW: 特になし | self]",
                "[INTENT_NEW: 星を見たい]",
            ])
        )
        assert parsed["new"] == [{"description": "星を見たい", "target": None}]

    def test_question_separates_entry_and_exit_limits(self, sqlite_store):
        """件数上限は新規（入口）にだけ掛かり、整理（出口）は無制限と明示される。

        旧設問は全操作まとめて「1〜3個まで」と書いていたため、在庫が21件あっても
        本人は3枠しか使えないと読み、しかも筆頭の「新しく残したい」に枠を食われて
        整理が進まなかった（出口の詰まりの主因）。
        """
        char_id, _ = _make_character(sqlite_store)
        active = sqlite_store.create_intent(char_id, "歌の練習を続けたい")
        question = build_pickup_question([active], [])
        assert "1〜3個まで" not in question
        assert "3件まで" in question          # 新規にだけ掛かる上限
        assert "件数の制限はない" in question  # 整理側は無制限
        assert "INTENT_SETTLED" in question

    def test_question_hints_tidying_when_inventory_is_large(self, sqlite_store):
        """在庫が閾値を超えていると、設問に「整理していい」の一行が添えられる。

        機械は在庫が膨らんでいる事実だけを伝え、何を残すかは本人が決める
        （「機械は候補を挙げ、本人が裁く」の範囲に収める）。
        """
        char_id, _ = _make_character(sqlite_store)
        many = [
            sqlite_store.create_intent(char_id, f"意図{i}") for i in range(11)
        ]
        assert "件抱えている" in build_pickup_question(many, [])
        assert "件抱えている" not in build_pickup_question(many[:3], [])

    def test_parse_no_tags_is_empty(self):
        """タグなし（なければないでいい）は何も適用されない。"""
        parsed = parse_pickup_response("今日は特にないかな。穏やかな一日だった。")
        assert parsed == {
            "new": [], "fulfilled": [], "settled": [], "release": [], "soured": [],
        }


class TestRunIntentPickup:
    """run_intent_pickup の適用処理を検証するテストクラス（LLM はモック）。"""

    def _run(self, sqlite_store, char_id, response_text, memory_manager=None):
        """ask_character をモックして拾い上げを1回実行するヘルパ。"""
        with patch(
            "backend.services.intents.pickup.ask_character",
            new=AsyncMock(return_value=response_text),
        ):
            return asyncio.run(run_intent_pickup(
                char_id, sqlite_store, {},
                born_from="night_chronicle",
                memory_manager=memory_manager,
            ))

    def test_creates_new_intents(self, sqlite_store):
        """INTENT_NEW が create_intent＋封筒になる。"""
        char_id, _ = _make_character(sqlite_store, ghost_model="p1")
        result = self._run(sqlite_store, char_id, "[INTENT_NEW: 星を見たい | self]")
        assert result == {
            "status": "success", "created": 1,
            "fulfilled": 0, "settled": 0, "expired": 0, "soured": 0,
        }
        intents = sqlite_store.list_intents(char_id)
        assert intents[0].description == "星を見たい"
        assert intents[0].born_from == "night_chronicle"

    def test_fulfilled_resolves_intent(self, sqlite_store):
        """INTENT_FULFILLED で本人が「満ちた」と宣言した意図が fulfilled になる。

        行動権の帰還以外に fulfilled への経路が無かった頃は、1on1 で話し切った意図も
        active のまま滞留し、毎ターン block_motive に載り続けた（同じ話題の反復）。
        """
        char_id, _ = _make_character(sqlite_store, ghost_model="p1")
        intent = sqlite_store.create_intent(char_id, "もわに実験結果を見せたい")
        result = self._run(sqlite_store, char_id, f"[INTENT_FULFILLED: {intent.id}]")
        assert result["fulfilled"] == 1
        assert sqlite_store.get_intent(intent.id).status == "fulfilled"

    def test_release_resolves_expired(self, sqlite_store):
        """INTENT_RELEASE で本人が手放した意図が expired になる。"""
        char_id, _ = _make_character(sqlite_store, ghost_model="p1")
        intent = sqlite_store.create_intent(char_id, "手放すもの")
        result = self._run(sqlite_store, char_id, f"[INTENT_RELEASE: {intent.id}]")
        assert result["expired"] == 1
        assert sqlite_store.get_intent(intent.id).status == "expired"

    def test_soured_inscribes_words(self, sqlite_store):
        """INTENT_SOURED で不満の言葉が記憶へ刻まれる（memory_manager 経由）。"""
        char_id, _ = _make_character(sqlite_store, ghost_model="p1")
        intent = sqlite_store.create_intent(char_id, "旅行に行きたい")
        mm = MagicMock()
        result = self._run(
            sqlite_store, char_id,
            f"[INTENT_SOURED: {intent.id} | もどかしい]",
            memory_manager=mm,
        )
        assert result["soured"] == 1
        assert sqlite_store.get_intent(intent.id).status == "soured"
        mm.write_inscribed_memory.assert_called_once()
        kwargs = mm.write_inscribed_memory.call_args.kwargs
        assert "もどかしい" in kwargs["content"]
        assert "旅行に行きたい" in kwargs["content"]

    def test_unknown_ids_ignored(self, sqlite_store):
        """存在しない意図 ID への操作（判定器の幻覚）は無視される。"""
        char_id, _ = _make_character(sqlite_store, ghost_model="p1")
        fake = str(uuid.uuid4())
        result = self._run(
            sqlite_store, char_id,
            f"[INTENT_RELEASE: {fake}]\n[INTENT_SOURED: {fake} | x]",
        )
        assert result["expired"] == 0 and result["soured"] == 0

    def test_skips_without_ghost_model(self, sqlite_store):
        """ghost_model 未設定キャラは LLM を呼ばずスキップする。"""
        char_id, _ = _make_character(sqlite_store, ghost_model=None)
        result = asyncio.run(run_intent_pickup(
            char_id, sqlite_store, {}, born_from="usual_scene",
        ))
        assert result["status"] == "skipped"


class TestSettleIntent:
    """一区切り（settled）の永続化を検証するテストクラス。

    settled は終端遷移ではない。status は active のまま、封筒 intent.settled だけが
    増える。満足度カラムを持たないのは「圧力は保存しない」（§4.1）を守るためで、
    意図圧の起点は封筒から読み直される。
    """

    def test_settle_keeps_active_and_writes_envelope(self, sqlite_store):
        """settle_intent は status を変えず intent.settled 封筒を残す。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(char_id, "もわの反応を見たい", target="user")
        result = sqlite_store.settle_intent(intent.id)
        assert result is not None
        assert result.status == "active"
        events = [
            e for e in sqlite_store.list_timeline_events(char_id)
            if e.event_type == "intent.settled"
        ]
        assert len(events) == 1
        assert events[0].intent_id == intent.id
        assert events[0].counterpart == "user"

    def test_settle_is_repeatable(self, sqlite_store):
        """一区切りは何度でも打てる（そのたびに起点が今へ移る）。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(char_id, "また気になってきた")
        sqlite_store.settle_intent(intent.id)
        sqlite_store.settle_intent(intent.id)
        assert len(sqlite_store.latest_settled_map(char_id)) == 1

    def test_settle_ignores_terminated_intent(self, sqlite_store):
        """終端済みの意図には打てない（None が返る）。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(char_id, "もう終わった話")
        sqlite_store.resolve_intent(intent.id, "fulfilled")
        assert sqlite_store.settle_intent(intent.id) is None

    def test_latest_settled_map_returns_newest(self, sqlite_store):
        """latest_settled_map は意図ごとの最新 settled を返す。"""
        char_id, _ = _make_character(sqlite_store)
        a = sqlite_store.create_intent(char_id, "意図A")
        b = sqlite_store.create_intent(char_id, "意図B")
        sqlite_store.settle_intent(a.id)
        mapping = sqlite_store.latest_settled_map(char_id)
        assert a.id in mapping
        assert b.id not in mapping


class TestIntentMarks:
    """1on1 本文タグからの決着宣言（intent_settler）を検証するテストクラス。

    受け口を 1on1 に置く理由は、push の帰還が「ユーザがまだ読んでいない」時点で走るため。
    「反応を見たい」型の意図にとっての決着は、実際に反応が起きた会話の中にしかなく、
    そこに受け口が無いと夜の拾い上げまで圧が下がらない（同じ日に二度話しかける事故）。
    ANTICIPATE_RESPONSE と同じ全プロバイダー一律テキストタグで、ツール化しない。
    """

    def test_extract_removes_tags_and_keeps_order(self):
        """タグを本文から除去し、出現順に (種別, ID断片) を返す。"""
        text = "その癖、やっぱり出たね。[INTENT_SETTLED: 1b86c1e9] それはそれとして……"
        clean, marks = extract_intent_marks(text)
        assert "INTENT_SETTLED" not in clean
        assert "その癖、やっぱり出たね。" in clean
        assert marks == [("settled", "1b86c1e9")]

    def test_extract_limits_to_two_marks(self):
        """1ターンに適用するのは2件まで（乱発の歯止め）。"""
        text = " ".join(
            f"[INTENT_SETTLED: {i:08x}]" for i in range(5)
        )
        _, marks = extract_intent_marks(text)
        assert len(marks) == 2

    def test_extract_distinguishes_kinds(self):
        """SETTLED（継続）と FULFILLED（終端）を取り違えない。"""
        text = "[INTENT_FULFILLED: aaaaaaaa][INTENT_SETTLED: bbbbbbbb]"
        _, marks = extract_intent_marks(text)
        assert marks == [("fulfilled", "aaaaaaaa"), ("settled", "bbbbbbbb")]

    def test_resolve_ref_by_prefix(self, sqlite_store):
        """短縮8桁の前方一致で完全 ID へ解決する（WM スレッドと同じ流儀）。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(char_id, "解決対象")
        active = sqlite_store.list_intents(char_id)
        assert resolve_intent_ref(intent.id[:8], active) == intent.id
        assert resolve_intent_ref(intent.id, active) == intent.id
        assert resolve_intent_ref("ffffffff", active) is None
        assert resolve_intent_ref("", active) is None

    def test_apply_settles_and_fulfills(self, sqlite_store):
        """settled は active のまま、fulfilled は終端させる。"""
        char_id, _ = _make_character(sqlite_store)
        a = sqlite_store.create_intent(char_id, "落ち着いた方")
        b = sqlite_store.create_intent(char_id, "果たした方")
        applied = apply_intent_marks(
            sqlite_store, char_id,
            [("settled", a.id[:8]), ("fulfilled", b.id[:8])],
        )
        assert {x["kind"] for x in applied} == {"settled", "fulfilled"}
        assert sqlite_store.get_intent(a.id).status == "active"
        assert sqlite_store.get_intent(b.id).status == "fulfilled"
        assert a.id in sqlite_store.latest_settled_map(char_id)

    def test_apply_ignores_unresolvable_ref(self, sqlite_store):
        """解決できない ID は黙って捨てる（本人の書き間違いで会話を壊さない）。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.create_intent(char_id, "無関係な意図")
        assert apply_intent_marks(sqlite_store, char_id, [("settled", "deadbeef")]) == []

    def test_apply_is_idempotent_within_a_turn(self, sqlite_store):
        """同じ意図を1ターンに二度指しても1回しか適用しない。"""
        char_id, _ = _make_character(sqlite_store)
        intent = sqlite_store.create_intent(char_id, "二度書かれた意図")
        applied = apply_intent_marks(
            sqlite_store, char_id,
            [("settled", intent.id[:8]), ("settled", intent.id[:8])],
        )
        assert len(applied) == 1
