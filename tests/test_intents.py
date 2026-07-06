"""意図（intents）のテスト — めぐり（巡り / Aliveness）Phase 4。

検証対象（docs/aliveness_plan.md §4.3）:
    1. IntentStoreMixin: 作成・一覧・終端遷移と、intent.created / expired /
       soured 封筒の同一トランザクション直書き（intent_id FK 込み）
    2. lifecycle: 意図圧の読み取り時計算 g(経過日数, source_kind の現在圧)、
       失効候補（低圧14日）・不満化候補（高圧7日）の候補挙げ
    3. pickup: 設問文の組み立て（既存 active・候補の添付）と
       返答タグ（INTENT_NEW / RELEASE / SOURED）のパース堅牢性、
       run_intent_pickup の適用（LLM はモック）
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.intents.lifecycle import (
    expired_candidates,
    intent_pressure,
    soured_candidates,
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
    """意図圧の読み取り時計算と失効/不満化の候補挙げを検証するテストクラス。"""

    def test_pressure_grows_with_age(self):
        """経過日数とともに意図圧が単調に増える（14日で飽和）。"""
        pressures = {"social": 0.5}
        young = intent_pressure(_FakeIntent(1, "social"), pressures)
        old = intent_pressure(_FakeIntent(10, "social"), pressures)
        saturated = intent_pressure(_FakeIntent(30, "social"), pressures)
        assert young < old <= saturated

    def test_pressure_scales_with_source(self):
        """source_kind の現在圧が高いほど意図圧が押される。"""
        intent = _FakeIntent(7, "social")
        low = intent_pressure(intent, {"social": 0.1})
        high = intent_pressure(intent, {"social": 0.9})
        assert low < high

    def test_expired_candidates(self):
        """低圧のまま14日超の意図だけが失効候補になる。"""
        pressures = {"social": 0.0, "boredom": 0.0, "body": 0.0}
        old_low = _FakeIntent(20, "social")     # 低圧・古い → 候補
        young_low = _FakeIntent(3, "social")    # 低圧・若い → 対象外
        candidates = expired_candidates([old_low, young_low], pressures)
        assert candidates == [old_low]

    def test_soured_candidates(self):
        """高圧なのに7日超遷移できない意図だけが不満化候補になる。"""
        pressures = {"social": 1.0}
        stuck = _FakeIntent(10, "social")   # 高圧・7日超 → 候補
        fresh = _FakeIntent(2, "social")    # 高圧・若い → まだ行動権の領分
        candidates = soured_candidates([stuck, fresh], pressures)
        assert candidates == [stuck]


class TestPickupParsing:
    """拾い上げの設問組み立てと返答パースを検証するテストクラス。"""

    def test_question_includes_actives_and_candidates(self, sqlite_store):
        """設問に既存 active 一覧・失効候補・不満化候補が添えられる。"""
        char_id, _ = _make_character(sqlite_store)
        active = sqlite_store.create_intent(char_id, "歌の練習を続けたい")
        question = build_pickup_question([active], [active], [])
        assert "歌の練習を続けたい" in question
        assert active.id in question
        assert "なければないでいい" in question

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

    def test_parse_no_tags_is_empty(self):
        """タグなし（なければないでいい）は何も適用されない。"""
        parsed = parse_pickup_response("今日は特にないかな。穏やかな一日だった。")
        assert parsed == {"new": [], "release": [], "soured": []}


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
        assert result == {"status": "success", "created": 1, "expired": 0, "soured": 0}
        intents = sqlite_store.list_intents(char_id)
        assert intents[0].description == "星を見たい"
        assert intents[0].born_from == "night_chronicle"

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
