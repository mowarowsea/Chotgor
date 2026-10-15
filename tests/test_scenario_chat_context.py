"""シナリオチャット用 履歴整形・切り出しユーティリティのテスト。

backend.services.scenario_chat.context モジュールの各関数を検証する。

検証する観点:
    - resolve_history_limits の優先順（レスポンス数=シナリオ > settings > 既定値、
      文字数=GM プリセット > settings > 既定値）
    - group_responses / count_responses のレスポンス単位の束ね
    - slice_history のレスポンス数上限・文字数上限の **AND** 動作と、レスポンス途中で切らないこと
    - 切り出し結果が時系列昇順を維持すること
    - format_turn_for_gm が話者種別ごとに正しいタグを使うこと
    - format_history_for_gm が改行区切りで連結すること
    - エッジケース: 空履歴・上限ゼロ・content が None など

context.py は ORM オブジェクトの構造に依存しないよう
duck typing で書かれている。テストも軽量な dataclass で代替する。
"""

from dataclasses import dataclass

from backend.services.scenario_chat.context import (
    DEFAULT_HISTORY_MAX_CHARS,
    DEFAULT_HISTORY_MAX_TURNS,
    TurnView,
    count_responses,
    dropped_history,
    group_responses,
    format_history_for_gm,
    format_turn_for_gm,
    resolve_history_limits,
    slice_history,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


@dataclass
class FakeSession:
    """Scenario 風のダミーオブジェクト（レスポンス数上限だけを持つ）。テスト用。"""

    history_max_turns: object = None


@dataclass
class FakePreset:
    """LLMModelPreset 風のダミーオブジェクト（コンテキスト長だけを持つ）。テスト用。"""

    context_window_tokens: object = None


@dataclass
class RawTurn:
    """raw_response を持つ ScenarioTurn 風ダミー。レスポンス単位の束ねを検証する。"""

    content: str
    raw_response: object = None


def _turn(speaker_type: str = "user", speaker_name: str = "プレイヤー", content: str = "x") -> TurnView:
    """簡易 TurnView ファクトリ（raw_response を持たない＝1 行 1 レスポンス）。"""
    return TurnView(speaker_type=speaker_type, speaker_name=speaker_name, content=content)


# ─── resolve_history_limits ──────────────────────────────────────────────────


class TestResolveHistoryLimits:
    """履歴上限の解決優先順を検証する。

    優先順:
        レスポンス数: scenario.history_max_turns → settings["scenario_chat_default_history_max_turns"]
            → DEFAULT_HISTORY_MAX_TURNS
        文字数: GM プリセットの context_window_tokens × 1/2（履歴に使う分）× 6/10（生ログ分）
            → settings["scenario_chat_default_history_max_chars"] → DEFAULT_HISTORY_MAX_CHARS
    プリセットはコンテキスト長（モデルの性質）だけを持ち、生ログ・あらすじへの振り分けは
    シナリオ側の定数で行う（2026-10-15 にシナリオ側 history_max_chars を廃止）。
    """

    def test_all_none_falls_back_to_defaults(self):
        """シナリオ・プリセット・settings いずれも未設定なら既定値が返ること。"""
        session = FakeSession()
        turns, chars = resolve_history_limits(session, settings={})
        assert turns == DEFAULT_HISTORY_MAX_TURNS
        assert chars == DEFAULT_HISTORY_MAX_CHARS

    def test_scenario_turns_and_preset_window_take_priority(self):
        """シナリオのレスポンス数・プリセットのコンテキスト長が settings より優先されること。

        200,000 トークン → 履歴総量 100,000 字 → 生ログ 60,000 字（1 トークン ≒ 1 文字）。
        """
        session = FakeSession(history_max_turns=10)
        settings = {
            "scenario_chat_default_history_max_turns": 999,
            "scenario_chat_default_history_max_chars": 999999,
        }
        turns, chars = resolve_history_limits(
            session, settings=settings, gm_preset=FakePreset(context_window_tokens=200000),
        )
        assert turns == 10
        assert chars == 60000

    def test_settings_used_when_scenario_and_preset_none(self):
        """シナリオ・プリセットが未設定なら settings が使われること。"""
        session = FakeSession()
        settings = {
            "scenario_chat_default_history_max_turns": 50,
            "scenario_chat_default_history_max_chars": 12345,
        }
        turns, chars = resolve_history_limits(
            session, settings=settings, gm_preset=FakePreset(),
        )
        assert turns == 50
        assert chars == 12345

    def test_scenario_chars_attribute_is_ignored(self):
        """シナリオ側に旧 history_max_chars 属性が残っていても参照しないこと（廃止済み）。"""

        @dataclass
        class LegacyScenario:
            history_max_turns: object = None
            history_max_chars: object = 500

        _, chars = resolve_history_limits(LegacyScenario(), settings={})
        assert chars == DEFAULT_HISTORY_MAX_CHARS

    def test_invalid_int_in_settings_fallback(self):
        """settings の値が int 変換できないなら既定値に落ちること。"""
        session = FakeSession()
        settings = {
            "scenario_chat_default_history_max_turns": "abc",
            "scenario_chat_default_history_max_chars": None,
        }
        turns, chars = resolve_history_limits(session, settings=settings)
        assert turns == DEFAULT_HISTORY_MAX_TURNS
        assert chars == DEFAULT_HISTORY_MAX_CHARS

    def test_settings_string_int_accepted(self):
        """settings の数値文字列は int として受け入れること。"""
        session = FakeSession()
        settings = {
            "scenario_chat_default_history_max_turns": "25",
            "scenario_chat_default_history_max_chars": "4000",
        }
        turns, chars = resolve_history_limits(session, settings=settings)
        assert turns == 25
        assert chars == 4000

    def test_no_settings_argument(self):
        """settings 引数が None でも既定値が返ること。"""
        session = FakeSession()
        turns, chars = resolve_history_limits(session, settings=None)
        assert turns == DEFAULT_HISTORY_MAX_TURNS
        assert chars == DEFAULT_HISTORY_MAX_CHARS


# ─── slice_history ────────────────────────────────────────────────────────────


class TestGroupResponses:
    """ターン列をレスポンス単位へ束ねる挙動を検証する。

    レスポンス = 1 回の LLM 呼出で生まれたターン群。raw_response が同じで連続する行を
    1 つに束ね、raw_response が None / 空の行（ユーザ発話・intro）は 1 行で 1 レスポンス。
    """

    def test_same_raw_consecutive_rows_grouped(self):
        """同じ raw_response の連続行が 1 グループになること。"""
        turns = [
            RawTurn("u", None),
            RawTurn("g1", "R1"), RawTurn("g2", "R1"), RawTurn("g3", "R1"),
            RawTurn("u2", None),
            RawTurn("g4", "R2"),
        ]
        groups = group_responses(turns)
        assert [[t.content for t in g] for g in groups] == [
            ["u"], ["g1", "g2", "g3"], ["u2"], ["g4"],
        ]
        assert count_responses(turns) == 4

    def test_none_and_empty_raw_are_single_rows(self):
        """raw_response が None / 空文字の行は連続していても束ねないこと。"""
        turns = [RawTurn("a", None), RawTurn("b", None), RawTurn("c", ""), RawTurn("d", "")]
        assert count_responses(turns) == 4

    def test_same_raw_not_consecutive_split(self):
        """同じ raw_response でも間に別行が挟まれば別グループになること。"""
        turns = [RawTurn("a", "R"), RawTurn("u", None), RawTurn("b", "R")]
        assert count_responses(turns) == 3

    def test_turns_without_raw_attribute(self):
        """raw_response 属性を持たない行（TurnView 等）は 1 行 1 レスポンスとして扱うこと。"""
        assert count_responses([_turn(), _turn()]) == 2


class TestSliceHistory:
    """履歴切り出し（直近 N レスポンス × T 文字 AND）の挙動を検証する。

    切り出し方針:
        - 直近側からレスポンス単位で逆順に詰める（レスポンスの途中では切らない）
        - max_turns（レスポンス数）に達するか max_chars を超えそうなら停止
        - 結果は時系列昇順で返す
    """

    def test_empty_history(self):
        """空履歴では空リストが返ること。"""
        assert slice_history([], 10, 1000) == []

    def test_within_limits_all_kept(self):
        """ターン数・文字数いずれも上限に達しないなら全件返ること。"""
        turns = [_turn(content="a"), _turn(content="b"), _turn(content="c")]
        result = slice_history(turns, max_turns=10, max_chars=1000)
        assert [t.content for t in result] == ["a", "b", "c"]

    def test_max_turns_limits(self):
        """max_turns で件数が制限され、直近側が残ること。"""
        turns = [_turn(content=str(i)) for i in range(10)]
        result = slice_history(turns, max_turns=3, max_chars=10000)
        # 直近 3 件（7, 8, 9）が時系列昇順で残る
        assert [t.content for t in result] == ["7", "8", "9"]

    def test_max_chars_limits(self):
        """max_chars で累計文字数が制限されること。"""
        # 各 turn の content 長は 10 文字
        turns = [_turn(content="x" * 10) for _ in range(5)]
        # 25 文字制限 → 2 件分の 20 文字までしか入らない（3 件目で 30 文字超え）
        result = slice_history(turns, max_turns=99, max_chars=25)
        assert len(result) == 2

    def test_zero_turns(self):
        """max_turns=0 なら空リスト。"""
        turns = [_turn(content="a"), _turn(content="b")]
        assert slice_history(turns, max_turns=0, max_chars=1000) == []

    def test_zero_chars(self):
        """max_chars=0 なら空リスト。"""
        turns = [_turn(content="a"), _turn(content="b")]
        assert slice_history(turns, max_turns=10, max_chars=0) == []

    def test_order_preserved(self):
        """切り出し後も時系列昇順が維持されること。"""
        turns = [_turn(content=f"#{i}") for i in range(5)]
        result = slice_history(turns, max_turns=3, max_chars=10000)
        contents = [t.content for t in result]
        assert contents == sorted(contents, key=lambda s: int(s.lstrip("#")))

    def test_content_none_safe(self):
        """content が None でも例外を出さず、長さ 0 として扱うこと。"""
        @dataclass
        class T:
            content: object = None

        result = slice_history([T(content=None), T(content="ok")], max_turns=10, max_chars=10)
        assert len(result) == 2

    def test_max_turns_counts_responses_not_rows(self):
        """max_turns がバブル行数ではなくレスポンス数で効くこと。"""
        turns = [
            RawTurn("u1", None),
            RawTurn("a1", "A"), RawTurn("a2", "A"), RawTurn("a3", "A"),
            RawTurn("u2", None),
            RawTurn("b1", "B"), RawTurn("b2", "B"),
        ]
        result = slice_history(turns, max_turns=3, max_chars=10000)
        # 直近 3 レスポンス = A(3行) / u2 / B(2行) → 6 行
        assert [t.content for t in result] == ["a1", "a2", "a3", "u2", "b1", "b2"]

    def test_chars_limit_does_not_split_response(self):
        """文字数上限でもレスポンスの途中から始まらないこと（グループごと落とす）。"""
        turns = [
            RawTurn("x" * 10, "A"), RawTurn("x" * 10, "A"),
            RawTurn("y" * 10, "B"),
        ]
        # 25 文字: B(10) は入るが A(20) を足すと 30 で超過 → A は丸ごと落ちる
        result = slice_history(turns, max_turns=99, max_chars=25)
        assert [t.content for t in result] == ["y" * 10]

    def test_first_response_exceeds_chars_still_included_whole(self):
        """直近 1 レスポンスが単体で max_chars を超えても、そのレスポンスは丸ごと入ること。"""
        turns = [RawTurn("x" * 50, "A"), RawTurn("x" * 50, "A")]
        result = slice_history(turns, max_turns=5, max_chars=10)
        assert len(result) == 2

    def test_first_item_exceeds_chars_still_included(self):
        """直近 1 件が単体で max_chars を超える場合でも 1 件は入ること（極端な上限対策）。"""
        turns = [_turn(content="x" * 100)]
        result = slice_history(turns, max_turns=5, max_chars=10)
        assert len(result) == 1


# ─── format_turn_for_gm ──────────────────────────────────────────────────────


class TestFormatTurnForGm:
    """発話 1 件の `@話者: 本文` 整形を検証する。

    GM の出力規則・ScenarioChatParser と同じ `@名前: 本文` 規約に揃えてある。
    speaker_type ごとに使う名前が変わる:
        - user      → @{user_alias}
        - narrator  → @Narrator
        - npc       → @{speaker_name}
        - character → @{speaker_name}（speaker_name 必須）
    """

    def test_user_uses_alias_tag(self):
        """user 発話は user_alias を `@名前:` に使うこと。"""
        t = _turn(speaker_type="user", speaker_name="生の名前", content="やぁ")
        out = format_turn_for_gm(t, user_alias="勇者")
        assert out == "@勇者:\nやぁ"

    def test_narrator_uses_narrator_tag(self):
        """narrator は Narrator 名を使う。speaker_name の値は無視される。"""
        t = _turn(speaker_type="narrator", speaker_name="その他", content="雨")
        out = format_turn_for_gm(t, user_alias="勇者")
        assert out == "@Narrator:\n雨"

    def test_narrator_custom_name(self):
        """narrator_name 引数を変えれば違う名前を使えること。"""
        t = _turn(speaker_type="narrator", speaker_name="X", content="夜")
        out = format_turn_for_gm(t, user_alias="勇者", narrator_name="語り部")
        assert "@語り部:\n夜" == out

    def test_npc_uses_speaker_name(self):
        """npc は speaker_name をそのまま `@名前:` に使うこと。"""
        t = _turn(speaker_type="npc", speaker_name="レイカ", content="来た")
        out = format_turn_for_gm(t, user_alias="勇者")
        assert out == "@レイカ:\n来た"

    def test_character_uses_speaker_name(self):
        """character（P2 予約）も speaker_name を使う。"""
        t = _turn(speaker_type="character", speaker_name="ハル", content="…")
        out = format_turn_for_gm(t, user_alias="勇者")
        assert out == "@ハル:\n…"

    def test_empty_content(self):
        """空本文も例外を出さず `@名前: ` になること。"""
        t = _turn(speaker_type="user", content="")
        out = format_turn_for_gm(t, user_alias="勇者")
        assert out == "@勇者:\n"

    def test_unknown_speaker_type_falls_to_name(self):
        """予期しない speaker_type は speaker_name にフォールバックすること。"""
        t = _turn(speaker_type="???", speaker_name="モブ", content="ｺﾞﾜｺﾞﾜ")
        out = format_turn_for_gm(t, user_alias="勇者")
        assert out == "@モブ:\nｺﾞﾜｺﾞﾜ"


# ─── format_history_for_gm ───────────────────────────────────────────────────


class TestFormatHistoryForGm:
    """履歴全体の整形を検証する。空行区切りで連結されること、空履歴で空文字列。"""

    def test_empty_history(self):
        """空履歴は空文字列を返すこと。"""
        assert format_history_for_gm([], user_alias="勇者") == ""

    def test_multiple_turns_joined_by_blank_line(self):
        """複数ターンが空行（\\n\\n）で連結されること。

        `@名前:` は閉じタグを持たないため、複数行本文でもターン境界が
        明確になるよう空行で区切る。
        """
        turns = [
            _turn(speaker_type="user", content="やぁ"),
            _turn(speaker_type="narrator", content="雨"),
            _turn(speaker_type="npc", speaker_name="レイカ", content="来た"),
        ]
        out = format_history_for_gm(turns, user_alias="勇者")
        expected = "@勇者:\nやぁ\n\n@Narrator:\n雨\n\n@レイカ:\n来た"
        assert out == expected

    def test_multiline_content_keeps_turn_boundary(self):
        """本文に改行が含まれても、ターン間は空行で区切られて境界が保たれること。"""
        turns = [
            _turn(speaker_type="user", content="こんにちは。\n今日はいい天気だね。"),
            _turn(speaker_type="npc", speaker_name="レイカ", content="そうね。"),
        ]
        out = format_history_for_gm(turns, user_alias="勇者")
        expected = (
            "@勇者:\nこんにちは。\n今日はいい天気だね。\n\n@レイカ:\nそうね。"
        )
        assert out == expected

    def test_iterable_input(self):
        """ジェネレータでも動作すること。"""
        def gen():
            yield _turn(speaker_type="user", content="a")
            yield _turn(speaker_type="narrator", content="b")
        out = format_history_for_gm(gen(), user_alias="X")
        assert "@X:\na\n\n@Narrator:\nb" == out


# ─── dropped_history ─────────────────────────────────────────────────────────


class TestDroppedHistory:
    """`slice_history` で送信対象から **外れる** 古いターン側を返す関数のテスト。

    あらすじ自動要約で「これから LLM に渡らなくなるターン」を集約する補助関数。
    観点:
        - slice_history と相補的に動作すること（slice の残り = dropped）
        - 履歴が上限内に収まっているなら dropped は空であること
        - ターン数上限と文字数上限のどちらが先に来ても正しく境界を判定すること
        - 入力が空のときは空リストを返すこと
    """

    def test_no_dropped_when_within_limits(self):
        """履歴が上限内なら dropped は空であること。"""
        turns = [_turn(content="a"), _turn(content="b"), _turn(content="c")]
        result = dropped_history(turns, max_turns=10, max_chars=1000)
        assert result == []

    def test_dropped_when_exceeds_turn_limit(self):
        """ターン数上限を超えた古い側だけを返すこと。"""
        turns = [_turn(content=f"t{i}") for i in range(5)]
        result = dropped_history(turns, max_turns=2, max_chars=10000)
        # 直近 2 件が残り、古い 3 件が dropped
        assert [t.content for t in result] == ["t0", "t1", "t2"]

    def test_dropped_when_exceeds_char_limit(self):
        """文字数上限を超えた古い側を返すこと。"""
        # 各ターン 10 文字。max_chars=15 なら直近 1 件だけ残る。
        turns = [_turn(content="x" * 10) for _ in range(4)]
        result = dropped_history(turns, max_turns=100, max_chars=15)
        # 直近 1 件が残り、古い 3 件が dropped
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        """入力が空なら dropped も空。"""
        assert dropped_history([], max_turns=10, max_chars=100) == []

    def test_dropped_preserves_chronological_order(self):
        """戻り値は時系列昇順を維持すること（slice_history の前置 prefix）。"""
        turns = [_turn(content=str(i)) for i in range(6)]
        result = dropped_history(turns, max_turns=3, max_chars=10000)
        # 古い 3 件 = ["0", "1", "2"]
        assert [t.content for t in result] == ["0", "1", "2"]

    def test_dropped_plus_sliced_equals_all(self):
        """`slice_history` の結果と `dropped_history` の結果を合わせると全件になる不変条件。"""
        turns = [_turn(content=f"c{i}") for i in range(8)]
        kept = slice_history(turns, max_turns=3, max_chars=10000)
        dropped = dropped_history(turns, max_turns=3, max_chars=10000)
        assert len(kept) + len(dropped) == len(turns)
        # 連結すれば元と同じ順序
        assert [t.content for t in (list(dropped) + list(kept))] == [
            t.content for t in turns
        ]

    def test_zero_limits_return_all_as_dropped(self):
        """上限が 0 なら全件が dropped（slice が空になるため）。"""
        turns = [_turn(content="a"), _turn(content="b")]
        result = dropped_history(turns, max_turns=0, max_chars=100)
        assert len(result) == 2
