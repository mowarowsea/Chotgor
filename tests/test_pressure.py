"""圧力（pressure）のテスト — めぐり（巡り / Aliveness）Phase 3。

検証対象（docs/planned/aliveness_plan.md §4.1〜4.2）:
    1. 純関数性: 圧力は保存されず、封筒＋体質＋時刻から毎回同じ値が出る
    2. 社会圧: 接触からの経過で単調増加・相手別重み（関係の厚み×体質の鋭さ）で
       減衰量が変わる・同日の接触件数と対面かどうかが頭打ち付きで安らぎに反映される
    3. 退屈圧: イベント密度・多様性の低さで上がり、賑やかなタイムラインで下がる
    4. 体調圧: 疲労成分（イベント密度の減衰積分）＋リズム成分（決定論の波）。
       正規化はキャラ自身の平常活動量から動的に導き、回復は静かな日の指数減衰が担う
       （夜間バッチ night.* は活動でも回復でもなく無視される）
    5. 淡白な一行: 閾値を超えた圧だけが物理の報告として言語化される
    6. 体質インタビュー: 返答パースの堅牢性・ルーブリックの決定論写像
    7. 動機ブロック: 圧力の一行＋意図＋話題権の明文化がプロンプトに載る
"""

import uuid
from datetime import datetime, timedelta

from backend.services.chat.request_builder import build_system_prompt, build_turn_annotation
from backend.services.pressure.engine import (
    _BODY_GOOD_LINES,
    _BODY_HIGH_LINES,
    _BODY_MID_LINES,
    _BOREDOM_GOOD_LINES,
    _BOREDOM_HIGH_LINES,
    _BOREDOM_MID_LINES,
    _FATIGUE_DEFAULT_RATE,
    _FATIGUE_MIN_RATE,
    _SOCIAL_GOOD_LINES,
    _SOCIAL_HIGH_LINES,
    _SOCIAL_MID_LINES,
    DEFAULT_PROFILE,
    _baseline_activity_rate,
    _canonical_label_map,
    compute_speech_thresholds,
    _partner_of,
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

    def __init__(
        self, event_type, occurred_at, actor=None, counterpart=None, origin="real",
        modality=None,
    ):
        self.event_type = event_type
        self.occurred_at = occurred_at
        self.actor = actor
        self.counterpart = counterpart
        self.origin = origin
        self.modality = modality


_NOW = datetime(2026, 7, 6, 12, 0, 0)


def _chat_event(days_ago: float, modality: str | None = None):
    """days_ago 日前の chat.message 封筒スタブを作るヘルパ。"""
    return _FakeEvent(
        "chat.message", _NOW - timedelta(days=days_ago), actor="user", modality=modality,
    )


def _events_on_day(days_ago: int, count: int):
    """days_ago 日前の「同一日内」に count 件の chat.message を作るヘルパ。

    分単位のオフセットで同じ日付内に収める（時間の広がりで日付を跨がせない）。
    平常件数/日 の算出テストで、日ごとの件数を厳密に制御するために使う。
    """
    base = _NOW - timedelta(days=days_ago)
    return [
        _FakeEvent("chat.message", base + timedelta(minutes=m), actor="user")
        for m in range(count)
    ]


class TestSocialPressure:
    """社会圧の計算を検証するテストクラス。

    接触なし→最大、直近接触→低下、経過で単調増加、体質の鋭さと関係の重みで
    減衰量が変わること、同日の接触件数・対面かどうかが頭打ち付きで安らぎに
    反映されることを確認する。
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

    def test_same_day_messages_give_diminishing_boost(self):
        """同日の発言が多いほど安らぐが、頭打ちがありスパムで無限には積み上がらない。"""
        profile = merge_profile(None)
        w = lambda p: 0.8  # noqa: E731
        one = compute_social([_chat_event(0.5)], _NOW, profile, w)
        few = compute_social([_chat_event(0.5) for _ in range(3)], _NOW, profile, w)
        many = compute_social([_chat_event(0.5) for _ in range(10)], _NOW, profile, w)
        capped = compute_social([_chat_event(0.5) for _ in range(6)], _NOW, profile, w)
        # 発言が多い日ほど安らぎが増え、圧は下がる
        assert many < few < one
        # ただし上限件数（6件）を超えると頭打ちで同じ結果になる
        assert abs(many - capped) < 1e-9

    def test_face_to_face_relieves_more_than_text(self):
        """同じ回数の接触でも、対面がある日はテキストのみの日より安らぐ。"""
        profile = merge_profile(None)
        w = lambda p: 0.8  # noqa: E731
        text_only = compute_social(
            [_chat_event(0.5) for _ in range(3)], _NOW, profile, w,
        )
        with_face = compute_social(
            [_chat_event(0.5, modality="face")] + [_chat_event(0.5) for _ in range(2)],
            _NOW, profile, w,
        )
        assert with_face < text_only

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


class TestPartnerIdentification:
    """対人接触の「相手」判定を検証するテストクラス。

    うつつの封筒は GM が `@名前:` に書いた文字列がそのまま actor に入るため、
    次の2つの汚染が構造的に起きる。どちらも社会圧を誤らせるので明示的に守る:

    1. **キャラクター本人の混入** — GM は PC 本人の発話も `@<本人名>:` で書く。
       素通しすると「自分と会話して人恋しさが癒える」ことになる
       （はるの実測で30日に89件混入していた）。
    2. **同一人物の表記揺れ** — 「ひろこ」と「田中ひろこ」が別人として扱われ、
       relation スレッドと突合できずコールドスタート既定値へ落ちる。
       正規名の辞書は存在しない（未知 NPC を通す設計）ため、封筒に現れた
       ラベル集合だけから寄せ先を決める必要がある。
    """

    def test_self_is_not_a_partner(self):
        """本人名の npc: 発話は対人接触として数えない。"""
        ev = _FakeEvent("scene.turn", _NOW, actor="npc:はる")
        assert _partner_of(ev, self_name="はる") is None
        # 本人名を渡さなければ従来どおり相手として拾う（後方互換）
        assert _partner_of(ev) == "はる"

    def test_other_npc_is_still_a_partner(self):
        """本人以外の npc: 発話はこれまでどおり相手として拾う。"""
        ev = _FakeEvent("scene.turn", _NOW, actor="npc:田中ひろこ")
        assert _partner_of(ev, self_name="はる") == "田中ひろこ"

    def test_self_contact_does_not_relieve_social_pressure(self):
        """自分との会話ばかりの週は、社会圧が下がらない。"""
        events = [
            _FakeEvent("scene.turn", _NOW - timedelta(hours=h), actor="npc:はる")
            for h in range(1, 10)
        ]
        p = compute_social(events, _NOW, merge_profile(None), lambda p: 0.9, self_name="はる")
        assert p == 1.0

    def test_label_variant_folds_into_longer_form(self):
        """短いラベルは、それを含む一意な長いラベルへ寄る。"""
        m = _canonical_label_map({"ひろこ", "田中ひろこ", "菊地寛", "user"})
        assert m == {"ひろこ": "田中ひろこ"}

    def test_ambiguous_label_is_left_alone(self):
        """寄せ先の候補が複数あるラベルは、取り違えを避けて寄せない。"""
        m = _canonical_label_map({"佐藤", "佐藤彰", "佐藤花子"})
        assert "佐藤" not in m

    def test_variants_share_one_relation_weight(self):
        """表記揺れした接触は同一人物として集約され、関係の重みが引ける。

        「ひろこ」名義の接触も「田中ひろこ」の重み（厚い関係）で安らぐこと。
        寄せが効かないと既定値（薄い関係）に落ちて社会圧が下がらない。
        """
        events = [
            _FakeEvent("scene.turn", _NOW - timedelta(hours=2), actor="npc:ひろこ"),
            _FakeEvent("scene.turn", _NOW - timedelta(hours=1), actor="npc:田中ひろこ"),
        ]
        weights = {"田中ひろこ": 0.9}
        sharp = merge_profile({"social": {"sharpness": 1.0}})
        p = compute_social(events, _NOW, sharp, lambda t: weights.get(t, 0.05))
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

    def test_night_batch_is_ignored(self):
        """夜間バッチ（night.*）は活動でも回復項でもなく、疲労に影響しない。

        回復は静かな日の指数減衰が担う設計になったため、夢（night.chronicle）を
        足しても疲労は変わらない（成否が不安定な夜間バッチに回復を依存させない）。
        """
        profile = merge_profile(None)
        activity = [_chat_event(i / 24) for i in range(24)]
        without = compute_body(activity, _NOW, profile, "c1")
        with_dream = compute_body(
            activity + [_FakeEvent("night.chronicle", _NOW - timedelta(hours=8))],
            _NOW, profile, "c1",
        )
        assert with_dream == without

    def test_baseline_rate_cold_start_uses_default(self):
        """平常が測れない間（データ無し・当日のみ）はコールドスタート既定値を返す。

        当日ぶんの活動は平常の分母に入れない（今日の忙しさで基準が甘くならない）。
        """
        assert _baseline_activity_rate([], _NOW) == _FATIGUE_DEFAULT_RATE
        today_only = _events_on_day(0, 30)  # 今日だけ30件
        assert _baseline_activity_rate(today_only, _NOW) == _FATIGUE_DEFAULT_RATE

    def test_baseline_rate_from_history(self):
        """十分な履歴があれば「活動件数 ÷ 実データ日数」で平常件数/日を導く。"""
        events = []
        for d in range(1, 11):  # 昨日〜10日前、毎日20件
            events += _events_on_day(d, 20)
        # 200件 ÷ 10日 = 20.0 件/日
        assert _baseline_activity_rate(events, _NOW) == 20.0

    def test_baseline_rate_short_history_falls_back(self):
        """データ日数が最小要件未満なら既定値（過渡のブレを避ける）。"""
        events = _events_on_day(1, 50) + _events_on_day(2, 50) + _events_on_day(3, 50)
        assert _baseline_activity_rate(events, _NOW) == _FATIGUE_DEFAULT_RATE

    def test_baseline_rate_floored(self):
        """寡動キャラでも下限でクランプし、極端に小さい NORM にしない。"""
        events = []
        for d in range(1, 11):  # 昨日〜10日前、毎日1件（=1.0件/日）
            events += _events_on_day(d, 1)
        assert _baseline_activity_rate(events, _NOW) == _FATIGUE_MIN_RATE


class TestPlainLines:
    """淡白な一行（プロンプト注入用の物理報告）を検証するテストクラス。

    語彙は口癖化を避けるためプールから乱数選択されるので、個々の文言では
    なく「どのプール（高圧/中圧/好調/ニュートラル）から選ばれたか」を検証する。
    """

    def test_neutral_pressures_say_nothing(self):
        """全部ニュートラル（中間域）なら何も言わない（沈黙も情報）。"""
        assert pressure_plain_lines({"social": 0.4, "boredom": 0.3, "body": 0.5}) == []

    def test_high_pressures_reported_plainly(self):
        """閾値超えの圧だけが淡白に言語化される。"""
        lines = pressure_plain_lines({"social": 0.7, "boredom": 0.3, "body": 0.9})
        assert any(line in _BODY_HIGH_LINES for line in lines)
        assert any(line in _SOCIAL_MID_LINES for line in lines)
        assert not any(line in _BOREDOM_HIGH_LINES or line in _BOREDOM_MID_LINES for line in lines)

    def test_good_pressures_reported_plainly(self):
        """きわめて低圧（好調）のときも一行報告される。"""
        lines = pressure_plain_lines({"social": 0.1, "boredom": 0.2, "body": 0.05})
        assert any(line in _BODY_GOOD_LINES for line in lines)
        assert any(line in _SOCIAL_GOOD_LINES for line in lines)
        assert any(line in _BOREDOM_GOOD_LINES for line in lines)


class TestSpeechThresholds:
    """発話閾値の分位点化を検証するテストクラス。

    絶対値の閾値は「その定式化のときたまたま噛み合っていた数字」でしかなく、
    物理量を触るたびに言いすぎ／言わなすぎへ倒れる。そこで閾値をキャラ自身の
    メーター履歴の分布から取る。ここで守るのは3点:

    1. 十分な履歴と分散があれば分位点が引けること（p90/p75/p20）。
    2. **沈黙ガード** — 分布が平坦な圧は黙ること。分位点は分布がどれだけ
       平坦でも必ず上位10%を作るため、これが無いと毎日どれかの圧が喋る。
    3. **ウォームアップ** — 履歴が足りない間は絶対値へフォールバックし、
       導入直後に無言にならないこと。
    """

    @staticmethod
    def _neutral(**overrides):
        """全圧を絶対値閾値の沈黙域(0.4)に置き、指定した圧だけ差し替えるヘルパ。

        1つの圧だけを dict に入れると、残りの圧が 0.0 として「好調」と判定され、
        検証したい圧以外の一行が混ざる。
        """
        p = {"body": 0.4, "social": 0.4, "boredom": 0.4}
        p.update(overrides)
        return p

    def _seed_meters(self, sqlite_store, char_id, meter_id, values):
        """メーター履歴を日次1点ずつ過去へ遡って仕込むヘルパ。"""
        base = datetime.now()
        for i, v in enumerate(values):
            sqlite_store.record_meter(
                meter_id, v, character_id=char_id,
                occurred_at=base - timedelta(days=i),
            )

    def test_quantiles_from_history(self, sqlite_store):
        """十分な履歴と分散があれば p90/p75/p20 が閾値になる。"""
        char_id, _ = _make_character(sqlite_store)
        self._seed_meters(
            sqlite_store, char_id, "pressure_body", [i / 40.0 for i in range(41)]
        )
        th = compute_speech_thresholds(sqlite_store, char_id)
        high, mid, good = th["body"]
        assert good < mid < high
        assert 0.85 <= high <= 0.95
        assert 0.15 <= good <= 0.25

    def test_flat_distribution_is_silenced(self, sqlite_store):
        """ほぼ動かない圧は沈黙する（IQR 下限ガード）。"""
        char_id, _ = _make_character(sqlite_store)
        self._seed_meters(
            sqlite_store, char_id, "pressure_boredom", [0.45 + (i % 3) * 0.005 for i in range(30)]
        )
        th = compute_speech_thresholds(sqlite_store, char_id)
        assert th["boredom"] is None
        # 沈黙指定は実際に一行を落とす
        assert pressure_plain_lines(self._neutral(boredom=0.46), th) == []

    def test_warmup_falls_back_to_absolute(self, sqlite_store):
        """履歴が足りない間は絶対値閾値へフォールバックする。"""
        char_id, _ = _make_character(sqlite_store)
        self._seed_meters(sqlite_store, char_id, "pressure_body", [0.1, 0.9, 0.5])
        th = compute_speech_thresholds(sqlite_store, char_id)
        assert "body" not in th  # キーを置かない＝絶対値へ
        lines = pressure_plain_lines(self._neutral(body=0.85), th)
        assert len(lines) == 1 and lines[0] in _BODY_HIGH_LINES

    def test_thresholds_shift_what_gets_said(self, sqlite_store):
        """同じ圧の値でも、その子の分布次第で言うことが変わる。

        分位点化の眼目そのもの: 平均的に体調圧の高い子にとっての 0.5 は
        「普通の日」であり、低い子にとっては「重い日」になる。
        """
        heavy = {"body": (0.9, 0.75, 0.4)}   # 高めに分布している子
        light = {"body": (0.4, 0.3, 0.1)}    # 低めに分布している子
        assert pressure_plain_lines(self._neutral(body=0.5), heavy) == []
        assert pressure_plain_lines(self._neutral(body=0.5), light)[0] in _BODY_HIGH_LINES


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
    プロンプトキャッシュ対応（docs/planned/prompt_cache_plan.md A案）でシステムプロンプト
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
