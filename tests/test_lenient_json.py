"""backend.lib.lenient_json の寛容 JSON パースのテスト。

LLM は JSON 出力を指示しても、日本語文中の強調で裸のダブルクォートを混ぜるなどして
`json.loads` を壊すことがある。このモジュールはそれを json-repair で救う層だが、
**戻り値 3 種の切り分け**こそが本質的な契約であり、本テストの主眼はそこにある。

  - dict : パース成功（素直に通った / 修復で救えた）
  - None : JSON としては読めたが dict ではない = LLM が明示的に「変更なし」を返した
  - {}   : JSON として読めず、修復しても dict にならなかった = パース失敗

None と {} を取り違えると実害が出る。chronicle は None を「変更なし＝正常終了」と
見なして当日会話に chronicled_at を打つため、パース失敗が None で返ると
**その日の会話が二度と棚卸しされないまま失われる**。json-repair が救えない入力に
対して例外ではなく空文字列を返す（"これはJSONではありません" → ""）ことが、
実際にこの取り違えを生んでいた。以下のテストはその回帰防止を含む。
"""

from backend.lib.lenient_json import parse_lenient_json


class TestParseSuccess:
    """パースが成功して dict が返るケース。"""

    def test_plain_json(self):
        """素直な JSON はそのまま dict になる。"""
        assert parse_lenient_json('{"a": 1}') == {"a": 1}

    def test_code_fenced_json(self):
        """コードフェンスや前後の散文があっても、最初の { 〜 最後の } を拾う。"""
        text = 'はい、整理しました。\n```json\n{"thread_updates": []}\n```\nどうぞ。'
        assert parse_lenient_json(text) == {"thread_updates": []}

    def test_repairs_bare_double_quotes(self):
        """文字列値に裸のダブルクォートが混じっても修復して読む。

        実例（2026-08-09〜11 の chronicle）: 強調のつもりで書かれた `"型"` が
        JSON を壊していた。json-repair のフォールバックが働くことを固定する。
        """
        text = '{"post": "感情表現の"型"の方が強い"}'
        result = parse_lenient_json(text, feature_label="chronicle")
        assert isinstance(result, dict)
        assert "post" in result


class TestNoChangeIsNone:
    """「変更なし」を意味する None が返るケース（正常系）。"""

    def test_null_response(self):
        """LLM が null だけを返したら「変更なし」。"""
        assert parse_lenient_json("null") is None

    def test_empty_and_none_input(self):
        """空文字・空白のみ・None も「変更なし」として扱う。"""
        assert parse_lenient_json("") is None
        assert parse_lenient_json("   ") is None
        assert parse_lenient_json(None) is None

    def test_valid_json_but_not_dict(self):
        """JSON として読めるが dict ではない（リスト等）も「変更なし」。

        json.loads が通っている以上、LLM は壊れた出力をしたのではなく、
        単に dict 以外を返しただけ。パース失敗とは区別する。
        """
        assert parse_lenient_json("[1, 2, 3]") is None


class TestParseFailureIsEmptyDict:
    """パース失敗を示す {} が返るケース（呼び出し側がエラー扱いする）。"""

    def test_prose_that_is_not_json_at_all(self):
        """JSON でない散文は {}（None ではない）。

        json-repair はこの入力に対して例外を投げず空文字列を返すため、
        素朴に isinstance(result, dict) だけで判定すると None に落ちて
        「変更なし」と誤認される。その回帰防止。
        """
        assert parse_lenient_json("これはJSONではありません") == {}

    def test_truncated_json_without_recoverable_object(self):
        """途中で切れて object を成さない出力も {}。"""
        assert parse_lenient_json("申し訳ありません、出力できませんでした。") == {}
