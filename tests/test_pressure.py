"""圧力（pressure）のテスト — めぐり（巡り / Aliveness）Phase 3。

検証対象（docs/aliveness_plan.md §4.1〜4.2）:
    1. 純関数性: 圧力は保存されず、封筒＋体質＋時刻から毎回同じ値が出る
    2. 社会圧: 接触からの経過で単調増加・相手別重み（関係の厚み×体質の鋭さ）で
       減衰量が変わる・同日の連続発言は1接触に丸められる
    3. 退屈圧: イベント密度・多様性の低さで上がり、賑やかなタイムラインで下がる
    4. 体調圧: 疲労成分（イベント密度の減衰積分）＋リズム成分（決定論の波）。
       夢（night.chronicle）が回復項になる
    5. 淡白な一行: 閾値を超えた圧だけが物理の報告として言語化される
    6. 体質インタビュー: 返答パースの堅牢性・ルーブリックの決定論写像
    7. 動機ブロック: 圧力の一行＋意図＋話題権の明文化がプロンプトに載る
"""

import uuid
from datetime import datetime, timedelta

from backend.services.chat.request_builder import build_system_prompt, build_turn_annotation
from backend.services.pressure.engine import (
    DEFAULT_PROFILE,
    compute_boredom,
    compute_body,
    compute_pressures,
    compute_social,
    merge_profile,
    pressure_plain_lines,
    record_pressure_meters,
    rhythm_component,
)
from backend.services.pressure.interview import (
    answers_to_profile,
    parse_interview_answers,
)


def _make_character(sqlite_store, name="はるテスト"):
    """テスト用キャラクターを1体作成して返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name)
    return char_id, name


class _FakeEvent:
    """圧力計算に渡す封筒のスタブ（純関数テスト用の最小属性セット）。"""

    def __init__(self, event_type, occurred_at, actor=None, counterpart=None, origin="real"):
        self.event_type = event_type
        self.occurred_at = occurred_at
        self.actor = actor
        self.counterpart = counterpart
        self.origin = origin


_NOW = datetime(2026, 7, 6, 12, 0, 0)


def _chat_event(days_ago: float):
    """days_ago 日前の chat.message 封筒スタブを作るヘルパ。"""
    return _FakeEvent("chat.message", _NOW - timedelta(days=days_ago), actor="user")


class TestSocialPressure:
    """社会圧の計算を検証するテストクラス。

    接触なし→最大、直近接触→低下、経過で単調増加、体質の鋭さと関係の重みで
    減衰量が変わること、同日の連続発言が1接触に丸められることを確認する。
    """

    def test_no_contact_is_max(self):
        """対人イベントゼロなら社会圧は 1.0。"""
        profile = merge_profile(None)
        assert compute_social([], _NOW, profile, lambda p: 0.5) == 1.0

    def test_recent_contact_lowers_pressure(self):
        """直近の接触があるほど圧が低く、時間経過で単調に上がる。"""
        profile = merge_profile(None)
        w = lambda p: 0.8  # noqa: E731
        fresh = compute_social([_chat_event(0.1)], _NOW, profile, w)
        stale = compute_social([_chat_event(3.0)], _NOW, profile, w)
        old = compute_social([_chat_event(10.0)], _NOW, profile, w)
        assert fresh < stale < old

    def test_same_day_messages_count_once(self):
        """同じ相手との同日の発言10件は1接触ぶんの安らぎにしかならない。"""
        profile = merge_profile(None)
        w = lambda p: 0.8  # noqa: E731
        one = compute_social([_chat_event(0.5)], _NOW, profile, w)
        many = compute_social(
            [_chat_event(0.5) for _ in range(10)], _NOW, profile, w
        )
        assert abs(one - many) < 1e-9

    def test_sharpness_discounts_light_relations(self):
        """鋭さが高い体質では、軽い関係（重み小）の接触がほぼ安らぎにならない。"""
        events = [_FakeEvent(
            "scene.turn", _NOW - timedelta(hours=6), actor="npc:通りすがり",
        )]
        light = lambda p: 0.35  # noqa: E731
        soft = merge_profile({"social": {"sharpness": 0.0}})
        sharp = merge_profile({"social": {"sharpness": 1.0}})
        p_soft = compute_social(events, _NOW, soft, light)
        p_sharp = compute_social(events, _NOW, sharp, light)
        # 誰でもいい派は通りすがりでも安らぐが、特定の人派はほぼ安らがない
        assert p_soft < p_sharp
        assert p_sharp > 0.95

    def test_heavy_relation_relieves_even_sharp(self):
        """鋭い体質でも、厚い関係（重み大）の接触は安らぎになる。"""
        events = [_FakeEvent(
            "scene.turn", _NOW - timedelta(hours=6), actor="npc:親友",
        )]
        sharp = merge_profile({"social": {"sharpness": 1.0}})
        p = compute_social(events, _NOW, sharp, lambda p: 0.9)
        assert p < 0.5


class TestBoredomPressure:
    """退屈圧の計算を検証するテストクラス。"""

    def test_empty_timeline_is_bored(self):
        """イベントゼロの1週間は退屈圧が最大近く。"""
        profile = merge_profile(None)
        assert compute_boredom([], _NOW, profile) == 1.0

    def test_busy_diverse_week_is_not_bored(self):
        """密度も多様性も高い1週間は退屈圧が低い。"""
        profile = merge_profile(None)
        events = []
        for day in range(7):
            for i in range(14):
                events.append(_FakeEvent(
                    ["chat.message", "scene.turn", "memory.inscribed", "scene.closed"][i % 4],
                    _NOW - timedelta(days=day, hours=i),
                    actor=["user", "character", "npc:店主", "narrator"][i % 4],
                    origin=["real", "usual"][i % 2],
                ))
        assert compute_boredom(events, _NOW, profile) < 0.2

    def test_sensitivity_scales(self):
        """感度が高い体質ほど同じタイムラインでも退屈圧が高い。"""
        events = [_chat_event(d) for d in range(3)]
        bored_prone = merge_profile({"boredom": {"sensitivity": 1.4}})
        calm = merge_profile({"boredom": {"sensitivity": 0.6}})
        assert (
            compute_boredom(events, _NOW, bored_prone)
            > compute_boredom(events, _NOW, calm)
        )


class TestBodyPressure:
    """体調圧（疲労＋リズム）の計算を検証するテストクラス。"""

    def test_rhythm_is_deterministic(self):
        """リズム成分は同じキャラ・同じ時刻なら常に同じ値（乱数は世界に置く）。"""
        a = rhythm_component("char-123", _NOW)
        b = rhythm_component("char-123", _NOW)
        assert a == b
        # 別キャラは（ほぼ確実に）別の波
        c = rhythm_component("char-456", _NOW)
        assert 0.0 <= a <= 0.25 and 0.0 <= c <= 0.25

    def test_fatigue_accumulates_with_activity(self):
        """直近の活動イベントが多いほど疲労が積み上がる。"""
        profile = merge_profile(None)
        quiet = compute_body([], _NOW, profile, "c1")
        busy_events = [_chat_event(i / 24) for i in range(48)]  # 直近2日で48発言
        busy = compute_body(busy_events, _NOW, profile, "c1")
        assert busy > quiet

    def test_chronicle_recovers_fatigue(self):
        """夢（night.chronicle）は疲労の回復項になる。"""
        profile = merge_profile(None)
        activity = [_chat_event(i / 24) for i in range(24)]
        without = compute_body(activity, _NOW, profile, "c1")
        with_dream = compute_body(
            activity + [_FakeEvent("night.chronicle", _NOW - timedelta(hours=8))],
            _NOW, profile, "c1",
        )
        assert with_dream < without


class TestPlainLines:
    """淡白な一行（プロンプト注入用の物理報告）を検証するテストクラス。"""

    def test_low_pressures_say_nothing(self):
        """全部低圧なら何も言わない（沈黙も情報）。"""
        assert pressure_plain_lines({"social": 0.2, "boredom": 0.3, "body": 0.1}) == []

    def test_high_pressures_reported_plainly(self):
        """閾値超えの圧だけが淡白に言語化される。"""
        lines = pressure_plain_lines({"social": 0.7, "boredom": 0.2, "body": 0.9})
        assert any("体" in line for line in lines)
        assert any("人と" in line for line in lines)
        assert not any("単調" in line for line in lines)


class TestComputePressuresIntegration:
    """実 SQLiteStore（封筒 dual-write）に対する統合計算を検証するテストクラス。"""

    def test_compute_from_real_store(self, sqlite_store):
        """封筒ゼロのキャラは社会圧・退屈圧が最大、計算は保存を伴わない。"""
        char_id, _ = _make_character(sqlite_store)
        now = datetime.now()
        p = compute_pressures(sqlite_store, char_id, now=now)
        assert p["social"] == 1.0
        assert p["boredom"] == 1.0
        assert 0.0 <= p["body"] <= 1.0
        # 純関数: 同じ基準時刻なら2回目も完全に同じ（保存された状態がない）
        assert compute_pressures(sqlite_store, char_id, now=now) == p

    def test_relation_weight_from_wm_thread(self, sqlite_store):
        """WM relation スレッドの importance が相手別重みとして効く。"""
        char_id, char_name = _make_character(sqlite_store)
        sqlite_store.update_character(char_id, user_label="もわ")
        # 厚い関係スレッドを立てる
        sqlite_store.add_working_memory_thread(
            thread_id=str(uuid.uuid4()), character_id=char_id,
            type="relation", summary="もわとの関係", importance=0.9,
            relation_target="もわ",
        )
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        sqlite_store.create_chat_message(
            message_id=str(uuid.uuid4()), session_id=sid, role="user", content="やあ",
        )
        p = compute_pressures(sqlite_store, char_id)
        # 直近接触＋厚い関係 → 社会圧は大きく下がる
        assert p["social"] < 0.6

    def test_record_pressure_meters(self, sqlite_store):
        """日次スナップショットが3変数×キャラ数ぶん記録される。"""
        _make_character(sqlite_store)
        count = record_pressure_meters(sqlite_store)
        assert count == 3
        rows = sqlite_store.list_meter_snapshots(meter_id="pressure_social")
        assert len(rows) == 1


class TestInterviewRubric:
    """体質インタビューのパース・ルーブリック写像を検証するテストクラス。

    LLM は呼ばず、返答テキストの揺れに対するパーサの堅牢性と、
    選択肢→係数の決定論写像だけを確認する。
    """

    def test_parse_standard_format(self):
        """「1: a」形式の標準回答をパースできる。"""
        text = "1: a\n2: c\n3: b\n4: a\n\n一人は苦手かな。"
        assert parse_interview_answers(text) == {1: "a", 2: "c", 3: "b", 4: "a"}

    def test_parse_variants(self):
        """「1. a」「2）b」全角などの揺れも許容する。"""
        text = "うーん…… 1. a で、2）ｂ かなあ。3:c。4 a だね"
        parsed = parse_interview_answers(text)
        assert parsed == {1: "a", 2: "b", 3: "c", 4: "a"}

    def test_parse_garbage_returns_partial(self):
        """選択肢が読み取れない返答は空 dict（欠損に寛容）。"""
        assert parse_interview_answers("難しい質問だね。どれも違う気がする。") == {}

    def test_rubric_mapping(self):
        """選択肢が係数へ決定論写像される。"""
        profile = answers_to_profile({1: "a", 2: "c", 3: "c", 4: "c"})
        assert profile["social"]["tau_days"] == 1.0       # すぐ人恋しい
        assert profile["social"]["sharpness"] == 0.9      # 特定の人派
        assert profile["boredom"]["sensitivity"] == 0.6   # 穏やか好き
        assert profile["body"]["fatigue_sensitivity"] == 1.4  # 引きずる

    def test_missing_answers_keep_defaults(self):
        """答えの無い設問は標準値のまま。"""
        profile = answers_to_profile({1: "b"})
        assert profile["social"]["tau_days"] == 2.5
        assert profile["social"]["sharpness"] == DEFAULT_PROFILE["social"]["sharpness"]


class TestMotiveBlock:
    """動機ブロック（話題権）のターン注釈への注入を検証するテストクラス。

    動機ブロック（圧力の一行＋active な意図）は毎ターン変動しうる情報のため、
    プロンプトキャッシュ対応（docs/prompt_cache_plan.md A案）でシステムプロンプト
    からターン注釈（build_turn_annotation）へ移設された。文言・見出しは移設前と
    同一であること（キャラクターに見える中身は変えない）も含めて検証する。
    """

    def test_block_appears_with_lines(self):
        """圧力の一行があれば動機ブロックと話題権の明文化が注釈に載る。"""
        annotation = build_turn_annotation(
            motive_lines=["ここ数日、体は重め。"],
        )
        assert "いまのあなた（体と意図）" in annotation
        assert "ここ数日、体は重め。" in annotation
        assert "話題に乗る義務はありません" in annotation

    def test_block_with_intents(self):
        """active な意図も動機ブロックに載る（Phase 4 接続点）。"""
        annotation = build_turn_annotation(
            motive_lines=[],
            active_intents=[{"description": "あの本の続きを読みたい", "target": "self"}],
        )
        assert "あの本の続きを読みたい" in annotation

    def test_block_absent_when_empty(self):
        """素材ゼロならブロック自体が出ない（毎ターンのノイズにしない）。"""
        annotation = build_turn_annotation(
            motive_lines=[],
        )
        assert "いまのあなた（体と意図）" not in annotation

    def test_block_not_in_system_prompt(self):
        """動機ブロックがシステムプロンプト側に紛れ込まないこと（移設の回帰防止）。

        build_system_prompt が motive_lines を受け取らないこと（TypeError）で、
        変動情報の混入によるキャッシュプレフィックス破壊を構造的に防ぐ。
        """
        import pytest
        with pytest.raises(TypeError):
            build_system_prompt(
                character_system_prompt="あなたは「はる」です。",
                motive_lines=["ここ数日、体は重め。"],
            )
