"""backend.character_actions.anticipator モジュールのユニットテスト。

[ANTICIPATE_RESPONSE:...] タグの抽出・除去を検証する。
キャラクターが本文末尾に書いた「次の展開への予想（期待）」を取り出す
extract_anticipation の挙動と、全プロバイダー一律で使うガイド定数の健全性を網羅する。
inscriber / carver と同様にタグ抽出の純関数だが、こちらは記憶ストアに一切
書き込まない（抽出のみ）点が異なる。
"""

from backend.character_actions.anticipator import (
    extract_anticipation,
    ANTICIPATE_RESPONSE_TAG_NAME,
    ANTICIPATE_RESPONSE_TAG_GUIDE,
)


# ─── extract_anticipation: 基本動作 ────────────────────────────────────────────


def test_extract_basic():
    """[ANTICIPATE_RESPONSE:...] の本文が抽出され、clean からタグが除去されること。"""
    text = "こんにちは。元気だよ。[ANTICIPATE_RESPONSE:次は相手が笑うと思う]"
    clean, ant = extract_anticipation(text)

    assert ant == "次は相手が笑うと思う"
    assert "[ANTICIPATE_RESPONSE:" not in clean
    assert "こんにちは。元気だよ。" in clean


def test_extract_no_tag_returns_empty_and_original():
    """タグが無いときは予想が空文字列で、本文はそのまま返ること。"""
    text = "タグのない普通の発言です。"
    clean, ant = extract_anticipation(text)

    assert ant == ""
    assert clean == text


def test_extract_multiple_uses_last():
    """連続した複数タグがある場合は最後の予想が採用されること（本文末尾に1つ書かせる運用の保険）。"""
    text = "本文[ANTICIPATE_RESPONSE:予想A][ANTICIPATE_RESPONSE:予想B]"
    clean, ant = extract_anticipation(text)

    assert ant == "予想B"
    assert "[ANTICIPATE_RESPONSE:" not in clean


def test_extract_strips_whitespace():
    """予想本文の前後空白が除去されること。"""
    text = "本文[ANTICIPATE_RESPONSE:  余白つき予想  ]"
    _, ant = extract_anticipation(text)

    assert ant == "余白つき予想"


def test_extract_multiline_anticipation():
    """予想が複数行にまたがっても抽出され、本文から除去されること。

    うつつの GM が「NPC それぞれの思惑を併記」した結果、予想タグが改行を含み、
    タグごと本文に残ってしまった不具合の再現ケース。
    ログUI（tag_extract）だけが改行に対応していた非対称も、この統一で解消された。
    """
    text = (
        "田中はそう言って席に戻った。\n"
        "[ANTICIPATE_RESPONSE:\n"
        "はるは key_holder_id を外部キーとして設計するだろう。\n"
        "15時の新装データ到着に向けて午後のデスクワークが進んでいく。\n"
        "]"
    )
    clean, ant = extract_anticipation(text)

    assert "key_holder_id" in ant
    assert "15時の新装データ到着" in ant
    assert "[ANTICIPATE_RESPONSE:" not in clean
    assert clean == "田中はそう言って席に戻った。"


def test_extract_empty_text():
    """空文字列を渡しても例外なく空の結果を返すこと。"""
    clean, ant = extract_anticipation("")

    assert ant == ""
    assert clean == ""


# ─── エクスポートされた定数のサニティチェック ─────────────────────────────────


def test_tag_name_constant():
    """タグ名定数が想定どおり ANTICIPATE_RESPONSE であること。"""
    assert ANTICIPATE_RESPONSE_TAG_NAME == "ANTICIPATE_RESPONSE"


def test_guide_contains_tag_name():
    """ガイド文に [ANTICIPATE_RESPONSE:...] タグ形式が含まれること。"""
    assert "[ANTICIPATE_RESPONSE:" in ANTICIPATE_RESPONSE_TAG_GUIDE
