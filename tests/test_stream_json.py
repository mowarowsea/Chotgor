"""Tests for backend.lib.stream_json — stream-json イベント走査ヘルパー。"""

import json

from backend.lib.stream_json import iter_stream_json_events


class TestIterStreamJsonEvents:
    """iter_stream_json_events の挙動を検証する。

    Claude CLI の生 stdout（1行1イベントの NDJSON）と、debug_logger が
    JSON 文字列値内の \\n を実際の改行に展開したログテキストの両方を
    入力として受けるため、次の3系統を網羅する：
    - 正常系: 1行1イベントの NDJSON を順に取り出せる
    - 複数行またがり: 改行展開で分断されたイベントを積み上げパースで復元できる
    - 汚染系: 非 JSON 行（CLI 警告テキスト等）が混ざっても後続イベントを
      呑み込まない（行単体パースへのフォールバックで回復する）
    """

    def test_plain_ndjson_yields_each_event(self):
        """1行1イベントの NDJSON から全イベントが順に取り出せること。"""
        events = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": []}},
            {"type": "result", "usage": {"input_tokens": 10}},
        ]
        text = "\n".join(json.dumps(e, ensure_ascii=False) for e in events)

        assert list(iter_stream_json_events(text)) == events

    def test_blank_lines_are_skipped(self):
        """空行・空白のみの行はイベントとして扱われず読み飛ばされること。"""
        text = '\n\n{"type": "result"}\n   \n'

        assert list(iter_stream_json_events(text)) == [{"type": "result"}]

    def test_multiline_event_is_reassembled(self):
        """文字列値内の実改行で複数行に分断されたイベントが復元されること。

        debug_logger._unescape_text() は JSON 文字列値内の \\n を実際の改行に
        展開するため、ログ経由のテキストでは1イベントが複数行にまたがる。
        """
        event = {"type": "assistant", "message": {"content": "1行目\n2行目\n3行目"}}
        raw = json.dumps(event, ensure_ascii=False).replace("\\n", "\n")
        assert "\n" in raw  # 前提: 複数行にまたがっている

        assert list(iter_stream_json_events(raw)) == [event]

    def test_garbage_line_does_not_swallow_following_events(self):
        """非 JSON 行の後に続く正常イベントが取りこぼされないこと（Bug修正確認）。

        単純な積み上げパースだと、非 JSON 行が混ざった時点で積み上げが
        永遠にパース不能になり、以降の正常イベントを全て呑み込んでしまう。
        行単体パースへのフォールバックで回復することを確認する。
        """
        text = (
            "not a json line\n"
            "\n"
            '{"type": "result", "usage": {"input_tokens": 5}}'
        )

        events = list(iter_stream_json_events(text))

        assert events == [{"type": "result", "usage": {"input_tokens": 5}}]

    def test_garbage_between_events_keeps_both_sides(self):
        """イベントの間に非 JSON 行が挟まっても前後両方のイベントが取れること。"""
        text = (
            '{"type": "system"}\n'
            "WARNING: some cli notice\n"
            '{"type": "result"}'
        )

        events = list(iter_stream_json_events(text))

        assert events == [{"type": "system"}, {"type": "result"}]

    def test_non_dict_toplevel_json_is_dropped(self):
        """トップレベルが dict 以外の JSON（配列・数値等）はイベントとして返さないこと。"""
        text = '[1, 2]\n123\n{"type": "result"}'

        events = list(iter_stream_json_events(text))

        assert events == [{"type": "result"}]

    def test_numeric_line_inside_multiline_value_does_not_break_event(self):
        """文字列値内の改行展開で「数値だけの行」ができてもイベントが分断されないこと。

        行単体フォールバックは dict が得られた場合のみ採用するため、
        `123` のような行は積み上げを汚染せず、外側のイベントが正しく復元される。
        """
        event = {"type": "assistant", "message": {"content": "1行目\n123\n3行目"}}
        raw = json.dumps(event, ensure_ascii=False).replace("\\n", "\n")

        assert list(iter_stream_json_events(raw)) == [event]

    def test_empty_text_yields_nothing(self):
        """空文字列からは何も取り出されないこと。"""
        assert list(iter_stream_json_events("")) == []
