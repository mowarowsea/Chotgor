"""backend.core.tag_parser モジュールのユニットテスト。

parse_tags() の文字単位パーサーを網羅的に検証する。
特に正規表現ベースの旧実装では対処できなかったエッジケースを重点的にテストする。
"""

from backend.core.tag_parser import TagMatch, parse_tags


# ─── 基本動作 ──────────────────────────────────────────────────────────────────


def test_no_tags_returns_original_text():
    """タグが存在しない場合、clean_text は元のテキストと同一で matches は空リストであること。"""
    text = "普通のテキストです。タグはありません。"
    clean, matches = parse_tags(text, ["MEMORY", "DRIFT"])

    assert clean == text
    assert matches["MEMORY"] == []
    assert matches["DRIFT"] == []


def test_empty_string():
    """空文字列を渡した場合、clean_text は空で matches は空リストであること。"""
    clean, matches = parse_tags("", ["MEMORY"])

    assert clean == ""
    assert matches["MEMORY"] == []


def test_single_content_tag():
    """単純なコンテンツタグが正しく抽出されること。"""
    text = "こんにちは！[MEMORY:user|0.8|ユーザは猫が好き]よろしく。"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 1
    m = matches["MEMORY"][0]
    assert m.tag_name == "MEMORY"
    assert m.body == "user|0.8|ユーザは猫が好き"
    assert "[MEMORY:" not in clean
    assert "こんにちは！" in clean
    assert "よろしく。" in clean


def test_fixed_marker():
    """固定マーカー [TAG] が正しく抽出されること。"""
    text = "全部リセットします。[DRIFT_RESET]これからよろしく。"
    clean, matches = parse_tags(text, ["DRIFT_RESET", "DRIFT"])

    assert len(matches["DRIFT_RESET"]) == 1
    assert matches["DRIFT_RESET"][0].body == ""
    assert "[DRIFT_RESET]" not in clean
    assert "全部リセットします。" in clean


def test_multiple_tags_same_type():
    """同一タイプのタグが複数ある場合、すべて抽出されること。"""
    text = "[MEMORY:fact|1.0|事実A]テキスト[MEMORY:user|0.5|ユーザB]"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 2
    bodies = [m.body for m in matches["MEMORY"]]
    assert "fact|1.0|事実A" in bodies
    assert "user|0.5|ユーザB" in bodies
    assert "[MEMORY:" not in clean
    assert "テキスト" in clean


def test_multiple_tag_types():
    """複数タイプのタグが混在する場合、それぞれ正しく抽出されること。"""
    text = "[MEMORY:user|1.0|ユーザ情報][DRIFT:クールに話す]普通のテキスト"
    clean, matches = parse_tags(text, ["MEMORY", "DRIFT"])

    assert len(matches["MEMORY"]) == 1
    assert len(matches["DRIFT"]) == 1
    assert matches["MEMORY"][0].body == "user|1.0|ユーザ情報"
    assert matches["DRIFT"][0].body == "クールに話す"
    assert "[MEMORY:" not in clean
    assert "[DRIFT:" not in clean
    assert "普通のテキスト" in clean


# ─── ネストした角括弧（バグ修正の核心） ────────────────────────────────────────


def test_nested_bracket_in_content():
    """コンテンツ内にネストした角括弧が含まれる場合でも正しく抽出されること。

    Issue #49 の直接再現ケース:
    [MEMORY:fact|1.2|[MEMORY:]タグのパースバグで...] という形式で
    コンテンツ内に [MEMORY:] が含まれるときに失敗していた。
    """
    text = "[MEMORY:fact|1.2|[MEMORY:]タグのパースバグで、はるの発言末尾に記憶内容が漏れ出す事象が発生。]"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 1
    m = matches["MEMORY"][0]
    assert m.body.startswith("fact|1.2|")
    content = m.body.split("|", 2)[2]
    assert "[MEMORY:]" in content
    assert "タグのパースバグ" in content
    # クリーンテキストにマーカーが残らないこと
    assert clean == ""


def test_nested_bracket_tag_body_preserved():
    """ネストした括弧がコンテンツに含まれる場合、外側タグの body が完全に取得されること。"""
    text = "[DRIFT:[重要] これを守ること]普通のテキスト"
    clean, matches = parse_tags(text, ["DRIFT"])

    assert len(matches["DRIFT"]) == 1
    assert "[重要] これを守ること" in matches["DRIFT"][0].body
    assert "普通のテキスト" in clean


def test_deeply_nested_brackets():
    """3段階以上のネストした角括弧でも正しく処理されること。"""
    text = "[MEMORY:fact|1.0|[A:[B:[C]]]外側の内容]テキスト"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 1
    body = matches["MEMORY"][0].body
    # 全体のbodyが正しく取れること
    assert "[A:[B:[C]]]外側の内容" in body
    assert "テキスト" in clean


def test_multiple_tags_with_nested_content():
    """複数のタグそれぞれにネストした括弧が含まれる場合も正しく処理されること。"""
    text = "[MEMORY:fact|1.0|[DRIFT:]の内容A][MEMORY:user|0.5|[他の括弧]B]テキスト"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 2
    bodies = [m.body for m in matches["MEMORY"]]
    assert any("[DRIFT:]の内容A" in b for b in bodies)
    assert any("[他の括弧]B" in b for b in bodies)


# ─── バッククォート処理 ────────────────────────────────────────────────────────


def test_inline_code_not_parsed():
    """バッククォートインラインコード内のタグ形式テキストはスキップされること。"""
    text = "これは `[MEMORY:fact|1.0|コード内]` のテキストです。"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert matches["MEMORY"] == []
    # バッククォートごとテキストが残ること
    assert "`[MEMORY:fact|1.0|コード内]`" in clean


def test_fenced_code_block_not_parsed():
    """コードフェンス (```) 内のタグ形式テキストはスキップされること。"""
    text = "説明:\n```\n[MEMORY:fact|1.0|コードブロック内]\n```\n本文"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert matches["MEMORY"] == []
    assert "本文" in clean


def test_tag_outside_code_after_inline_code():
    """インラインコードの外側にあるタグは正しく抽出されること。"""
    text = "コード `例示` の後 [MEMORY:user|1.0|実際の内容] です。"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 1
    assert matches["MEMORY"][0].body == "user|1.0|実際の内容"
    assert "[MEMORY:" not in clean


# ─── TagMatch の位置情報 ────────────────────────────────────────────────────────


def test_tagmatch_position_fields():
    """TagMatch の start/end フィールドがテキスト内の正確な位置を示すこと。"""
    text = "前置き[MEMORY:fact|1.0|内容]後置き"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 1
    m = matches["MEMORY"][0]
    # raw が元テキストの start:end と一致すること
    assert text[m.start : m.end] == m.raw
    assert m.raw == "[MEMORY:fact|1.0|内容]"


# ─── タグ名の自動ソート（列挙順に依存しない設計） ─────────────────────────────


def test_prefix_collision_resolved_by_auto_sort():
    """[DRIFT_RESET] が [DRIFT:...] として誤抽出されないこと。

    呼び出し側が列挙順を意識しなくても、内部の長さ降順ソートで正しく照合されること。
    どちらの順序で渡しても結果が同じであることを両方向で検証する。
    """
    text = "[DRIFT_RESET][DRIFT:新しい指針]"

    # DRIFT を先に渡しても正しく処理されること
    clean1, matches1 = parse_tags(text, ["DRIFT", "DRIFT_RESET"])
    assert len(matches1["DRIFT_RESET"]) == 1
    assert len(matches1["DRIFT"]) == 1
    assert matches1["DRIFT"][0].body == "新しい指針"

    # DRIFT_RESET を先に渡しても結果が同じであること
    clean2, matches2 = parse_tags(text, ["DRIFT_RESET", "DRIFT"])
    assert len(matches2["DRIFT_RESET"]) == 1
    assert len(matches2["DRIFT"]) == 1


def test_auto_sort_with_longer_shared_prefix():
    """プレフィックスが3文字以上共通するタグ名でも、列挙順によらず正しく照合されること。"""
    text = "[SEARCH_RESET][SEARCH:クエリ内容]"

    # どちらの順でも同じ結果になること
    _, matches_a = parse_tags(text, ["SEARCH", "SEARCH_RESET"])
    _, matches_b = parse_tags(text, ["SEARCH_RESET", "SEARCH"])

    assert len(matches_a["SEARCH_RESET"]) == 1
    assert len(matches_a["SEARCH"]) == 1
    assert matches_a["SEARCH"][0].body == "クエリ内容"

    assert len(matches_b["SEARCH_RESET"]) == len(matches_a["SEARCH_RESET"])
    assert len(matches_b["SEARCH"]) == len(matches_a["SEARCH"])


# ─── マルチライン ──────────────────────────────────────────────────────────────


def test_multiline_content():
    """タグコンテンツが複数行にわたる場合でも正しく抽出されること。"""
    text = "[MEMORY:identity|0.9|1行目\n2行目\n3行目]後のテキスト"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 1
    body = matches["MEMORY"][0].body
    assert "1行目" in body
    assert "2行目" in body
    assert "3行目" in body
    assert "後のテキスト" in clean


# ─── 未閉じタグ（マルフォーム） ───────────────────────────────────────────────


def test_unclosed_tag_is_not_extracted():
    """閉じ括弧がないマルフォームタグは抽出されず、テキストはそのまま残ること。"""
    text = "テキスト[MEMORY:fact|1.0|閉じていない"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert matches["MEMORY"] == []
    # 元テキストがそのまま残ること
    assert "テキスト[MEMORY:fact|1.0|閉じていない" in clean


# ─── 空白トリム ────────────────────────────────────────────────────────────────


def test_clean_text_is_stripped():
    """clean_text の前後に余分な空白がないこと。"""
    text = "  [MEMORY:fact|1.0|内容]  "
    clean, _ = parse_tags(text, ["MEMORY"])

    assert clean == ""


def test_clean_text_preserves_internal_whitespace():
    """clean_text の内部の空白は保持されること。"""
    text = "前のテキスト  [MEMORY:fact|1.0|内容]  後のテキスト"
    clean, _ = parse_tags(text, ["MEMORY"])

    assert "前のテキスト" in clean
    assert "後のテキスト" in clean


# ─── 任意タグ（将来追加されるタグの汎用性検証） ────────────────────────────────
#
# 以下のテストは MEMORY / DRIFT など現在定義済みのタグを一切使わない。
# parse_tags() がタグ名に依存せず任意の名前で動作することを証明する。


def test_arbitrary_content_tag():
    """任意のタグ名 SEARCH で parse_tags() が正しく動作すること。"""
    text = "結果: [SEARCH:最新ニュース]以上です。"
    clean, matches = parse_tags(text, ["SEARCH"])

    assert len(matches["SEARCH"]) == 1
    assert matches["SEARCH"][0].body == "最新ニュース"
    assert "[SEARCH:" not in clean
    assert "結果:" in clean
    assert "以上です。" in clean


def test_arbitrary_fixed_marker():
    """任意の固定マーカー CLEAR_CONTEXT が正しく抽出・除去されること。"""
    text = "コンテキストをリセットします。[CLEAR_CONTEXT]よろしく。"
    clean, matches = parse_tags(text, ["CLEAR_CONTEXT"])

    assert len(matches["CLEAR_CONTEXT"]) == 1
    assert matches["CLEAR_CONTEXT"][0].body == ""
    assert "[CLEAR_CONTEXT]" not in clean
    assert "よろしく。" in clean


def test_arbitrary_tag_with_nested_brackets():
    """任意タグ TOOL_CALL のコンテンツにネストした括弧が含まれても正しく処理されること。"""
    text = "[TOOL_CALL:web_search|{\"query\": \"[検索用] キーワード\"}]テキスト続く"
    clean, matches = parse_tags(text, ["TOOL_CALL"])

    assert len(matches["TOOL_CALL"]) == 1
    body = matches["TOOL_CALL"][0].body
    assert "web_search" in body
    assert "[検索用]" in body
    assert "テキスト続く" in clean


def test_arbitrary_multiple_tag_types_coexist():
    """将来追加される複数の任意タグが混在しても互いに干渉しないこと。"""
    text = "[SEARCH:クエリA][WEBHOOK:https://example.com][SEARCH:クエリB]テキスト"
    clean, matches = parse_tags(text, ["SEARCH", "WEBHOOK"])

    assert len(matches["SEARCH"]) == 2
    assert len(matches["WEBHOOK"]) == 1
    search_bodies = [m.body for m in matches["SEARCH"]]
    assert "クエリA" in search_bodies
    assert "クエリB" in search_bodies
    assert matches["WEBHOOK"][0].body == "https://example.com"
    assert "テキスト" in clean


def test_arbitrary_tag_with_reset_variant():
    """任意タグ FOO とそのリセット変種 FOO_RESET が共存しても正しく照合されること。

    現在の実装（長さ降順ソート）により、列挙順によらず FOO_RESET が FOO に
    誤照合されないことを確認する。
    """
    text = "[FOO_RESET][FOO:コンテンツ]"

    # 短い名前を先に渡すケース（旧実装では誤照合していた）
    _, matches = parse_tags(text, ["FOO", "FOO_RESET"])

    assert len(matches["FOO_RESET"]) == 1
    assert len(matches["FOO"]) == 1
    assert matches["FOO"][0].body == "コンテンツ"


def test_arbitrary_tag_unknown_names_not_in_result():
    """指定していないタグ名はマッチ結果に含まれず、テキストにも残ること。"""
    text = "[KNOWN:内容][UNKNOWN:除去しない]"
    clean, matches = parse_tags(text, ["KNOWN"])

    assert "UNKNOWN" not in matches
    assert len(matches["KNOWN"]) == 1
    # 指定外タグはそのまま clean_text に残ること
    assert "[UNKNOWN:除去しない]" in clean


def test_arbitrary_tag_body_format_is_parser_agnostic():
    """parse_tags() はタグ body の内部フォーマットを解釈しない。

    body の分割（パイプ区切りなど）は呼び出し側の責務であり、
    parse_tags() は body を生文字列のまま返すことを確認する。
    """
    text = "[ACTION:move|north|3steps|fast]"
    _, matches = parse_tags(text, ["ACTION"])

    assert len(matches["ACTION"]) == 1
    # body はパースされず生文字列のまま返ること
    assert matches["ACTION"][0].body == "move|north|3steps|fast"
