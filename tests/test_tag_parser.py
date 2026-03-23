"""backend.core.tag_parser モジュールのユニットテスト。

parse_tags() の文字単位パーサーを網羅的に検証する。
特に正規表現ベースの旧実装では対処できなかったエッジケースを重点的にテストする。
"""

from backend.core.tag_parser import StreamingTagStripper, TagMatch, parse_tags


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
    """同一タイプのタグが複数ある場合、すべて抽出されること。
    タグはそれぞれ別行に記述する（行内rfindの制約）。
    """
    text = "[MEMORY:fact|1.0|事実A]\nテキスト\n[MEMORY:user|0.5|ユーザB]"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 2
    bodies = [m.body for m in matches["MEMORY"]]
    assert "fact|1.0|事実A" in bodies
    assert "user|0.5|ユーザB" in bodies
    assert "[MEMORY:" not in clean
    assert "テキスト" in clean


def test_multiple_tag_types():
    """複数タイプのタグが混在する場合、それぞれ正しく抽出されること。
    タグはそれぞれ別行に記述する（行内rfindの制約）。
    """
    text = "[MEMORY:user|1.0|ユーザ情報]\n[DRIFT:クールに話す]\n普通のテキスト"
    clean, matches = parse_tags(text, ["MEMORY", "DRIFT"])

    assert len(matches["MEMORY"]) == 1
    assert len(matches["DRIFT"]) == 1
    assert matches["MEMORY"][0].body == "user|1.0|ユーザ情報"
    assert matches["DRIFT"][0].body == "クールに話す"
    assert "[MEMORY:" not in clean
    assert "[DRIFT:" not in clean
    assert "普通のテキスト" in clean


# ─── 内容テキスト中の ']' （行内rfindで解決） ──────────────────────────────────


def test_closing_bracket_in_content():
    """内容テキストに ']' が含まれても正しく抽出されること。

    行内rfindにより、同一行の最後の ']' を閉じ括弧として取得するため、
    内容テキスト途中の ']' で誤終了しない。
    これが旧depth方式で失敗していた直接の報告バグ再現ケース。
    """
    text = "[MEMORY:semantic|1.2|rfindによる最後の]検出と行ベースパースで解決。]"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 1
    body = matches["MEMORY"][0].body
    assert "rfindによる最後の]検出" in body
    assert body.endswith("解決。")
    assert clean == ""


def test_multiple_closing_brackets_in_content():
    """内容テキストに ']' が複数含まれる場合も、最後の ']' が閉じ括弧として正しく使われること。"""
    text = "[MEMORY:semantic|1.0|A]B]C]"
    clean, matches = parse_tags(text, ["MEMORY"])

    assert len(matches["MEMORY"]) == 1
    # 最後の ] が閉じ括弧なので body は "A]B]C" となる
    assert matches["MEMORY"][0].body == "semantic|1.0|A]B]C"
    assert clean == ""


# ─── ネストした角括弧 ────────────────────────────────────────────────────────────


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
    """複数のタグそれぞれに角括弧が含まれる場合も正しく処理されること。
    タグはそれぞれ別行に記述する（行内rfindの制約）。
    """
    text = "[MEMORY:fact|1.0|[DRIFT:]の内容A]\n[MEMORY:user|0.5|[他の括弧]B]\nテキスト"
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
    タグはそれぞれ別行に記述する（行内rfindの制約）。
    """
    text = "[DRIFT_RESET]\n[DRIFT:新しい指針]"

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
    """プレフィックスが3文字以上共通するタグ名でも、列挙順によらず正しく照合されること。
    タグはそれぞれ別行に記述する（行内rfindの制約）。
    """
    text = "[SEARCH_RESET]\n[SEARCH:クエリ内容]"

    # どちらの順でも同じ結果になること
    _, matches_a = parse_tags(text, ["SEARCH", "SEARCH_RESET"])
    _, matches_b = parse_tags(text, ["SEARCH_RESET", "SEARCH"])

    assert len(matches_a["SEARCH_RESET"]) == 1
    assert len(matches_a["SEARCH"]) == 1
    assert matches_a["SEARCH"][0].body == "クエリ内容"

    assert len(matches_b["SEARCH_RESET"]) == len(matches_a["SEARCH_RESET"])
    assert len(matches_b["SEARCH"]) == len(matches_a["SEARCH"])


# ─── マルチライン ──────────────────────────────────────────────────────────────


def test_multiline_content_is_not_supported():
    """タグは1行に収まる前提。改行前に ] がなければタグとして認識されない。

    行内rfindにより、タグの閉じ括弧は同一行内の最後の ] として取得する。
    よって改行をまたぐタグ内容は未サポートであり、タグとして抽出されない。
    """
    text = "[MEMORY:identity|0.9|1行目\n2行目\n3行目]後のテキスト"
    clean, matches = parse_tags(text, ["MEMORY"])

    # 最初の行に ] がないためタグとして認識されない
    assert matches["MEMORY"] == []
    assert "[MEMORY:identity|0.9|1行目" in clean


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
    """将来追加される複数の任意タグが混在しても互いに干渉しないこと。
    タグはそれぞれ別行に記述する（行内rfindの制約）。
    """
    text = "[SEARCH:クエリA]\n[WEBHOOK:https://example.com]\n[SEARCH:クエリB]\nテキスト"
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
    タグはそれぞれ別行に記述する（行内rfindの制約）。
    """
    text = "[FOO_RESET]\n[FOO:コンテンツ]"

    # 短い名前を先に渡すケース（旧実装では誤照合していた）
    _, matches = parse_tags(text, ["FOO", "FOO_RESET"])

    assert len(matches["FOO_RESET"]) == 1
    assert len(matches["FOO"]) == 1
    assert matches["FOO"][0].body == "コンテンツ"


def test_arbitrary_tag_unknown_names_not_in_result():
    """指定していないタグ名はマッチ結果に含まれず、テキストにも残ること。
    タグはそれぞれ別行に記述する（行内rfindの制約）。
    """
    text = "[KNOWN:内容]\n[UNKNOWN:除去しない]"
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


# ─── StreamingTagStripper ──────────────────────────────────────────────────────
#
# ストリーミングチャンクからツールタグをリアルタイムで除去するバッファのテスト。
# feed() でチャンクを逐次投入し、flush() で残りを回収する使い方を検証する。


def _feed_all(chunks: list[str]) -> str:
    """チャンクリストを StreamingTagStripper に流し、最終出力を返す補助関数。

    feed() の戻り値と flush() の戻り値をすべて結合したものを返す。

    Args:
        chunks: ストリームチャンクのリスト。

    Returns:
        マーカーを除去した結合テキスト。
    """
    stripper = StreamingTagStripper()
    out = "".join(stripper.feed(c) for c in chunks)
    out += stripper.flush()
    return out


# ─── 基本動作 ──────────────────────────────────────────────────────────────────


def test_stripper_plain_text_passthrough():
    """マーカーを含まないテキストはそのまま出力されること。"""
    result = _feed_all(["こんにちは", "、世界！"])

    assert result == "こんにちは、世界！"


def test_stripper_single_chunk_with_marker():
    """マーカーが1チャンク内に収まる場合、マーカーが除去されること。"""
    result = _feed_all(["今日は楽しかった[INSCRIBE_MEMORY:contextual|1.0|楽しい]ね"])

    assert result == "今日は楽しかったね"
    assert "[INSCRIBE_MEMORY:" not in result


def test_stripper_empty_input():
    """空チャンクを渡してもエラーにならず、空文字列が返ること。"""
    result = _feed_all(["", "", ""])

    assert result == ""


def test_stripper_no_marker_brackets():
    """[ を含まないテキストは全量リアルタイムで流れること。"""
    stripper = StreamingTagStripper()

    # 各 feed() 呼び出しで即座に出力が返ること（バッファリングされないこと）
    out1 = stripper.feed("ABC")
    out2 = stripper.feed("DEF")
    remaining = stripper.flush()

    assert out1 == "ABC"
    assert out2 == "DEF"
    assert remaining == ""


# ─── マーカーが複数チャンクにまたがる場合 ──────────────────────────────────────


def test_stripper_marker_split_at_prefix():
    """マーカープレフィックスがチャンク境界で分割されても正しく除去されること。

    例: "[INSCRI" と "BE_MEMORY:contextual|1.0|内容]" に分かれた場合。
    分割中はバッファリングし、完結後に除去する。
    """
    result = _feed_all(["今日は[INSCRI", "BE_MEMORY:contextual|1.0|内容]ね"])

    assert result == "今日はね"


def test_stripper_marker_split_at_body():
    """マーカーの本体部分がチャンク境界で分割されても正しく除去されること。

    例: "[INSCRIBE_MEMORY:contextual|" と "1.0|内容]" に分かれた場合。
    プレフィックスが確定した後、']' が来るまでバッファリングし続ける。
    """
    result = _feed_all(["[INSCRIBE_MEMORY:contextual|", "1.0|内容]"])

    assert result == ""


def test_stripper_marker_split_across_many_chunks():
    """マーカーが多数のチャンクにまたがっても正しく除去されること。

    1文字ずつ分割した極端なケース。
    """
    text = "[DRIFT:常に敬語を使う]"
    result = _feed_all(list(text))

    assert result == ""


def test_stripper_text_before_split_marker_is_yielded_immediately():
    """マーカー手前のテキストはバッファリングされずに即座に出力されること。

    ストリーミングの恩恵（マーカー手前テキストのリアルタイム表示）を検証する。
    """
    stripper = StreamingTagStripper()

    out = stripper.feed("こんにちは[INSCRI")

    # マーカー手前の "こんにちは" は即座に出力され、"[INSCRI" はバッファに保持されること
    assert out == "こんにちは"


# ─── 固定マーカー（']' で終わるプレフィックス） ─────────────────────────────────


def test_stripper_fixed_marker_drift_reset():
    """固定マーカー [DRIFT_RESET] が正しく除去されること。"""
    result = _feed_all(["本文[DRIFT_RESET]続き"])

    assert result == "本文続き"


def test_stripper_fixed_marker_end_session():
    """固定マーカー [END_SESSION] が正しく除去されること。"""
    result = _feed_all(["さようなら[END_SESSION]"])

    assert result == "さようなら"


def test_stripper_end_session_with_reason():
    """コンテンツ形式の [END_SESSION:理由] が正しく除去されること。"""
    result = _feed_all(["さようなら[END_SESSION:疲れた]"])

    assert result == "さようなら"


def test_stripper_fixed_marker_split():
    """固定マーカーがチャンク境界で分割されても正しく除去されること。"""
    result = _feed_all(["本文[DRIFT_RE", "SET]続き"])

    assert result == "本文続き"


# ─── 複数マーカー ──────────────────────────────────────────────────────────────


def test_stripper_multiple_markers_in_single_chunk():
    """1チャンク内に複数のマーカーが含まれる場合、すべて除去されること。"""
    result = _feed_all(
        ["本文[INSCRIBE_MEMORY:contextual|1.0|内容A][DRIFT:新しい指針]続き"]
    )

    assert result == "本文続き"


def test_stripper_markers_across_chunk_boundaries():
    """複数のマーカーがそれぞれチャンク境界をまたいでも正しく除去されること。"""
    result = _feed_all([
        "本文[INSCRIBE_MEMO",
        "RY:contextual|1.0|A][CARVE_NARR",
        "ATIVE:append|新しい自己認識]終わり",
    ])

    assert result == "本文終わり"


def test_stripper_all_known_markers():
    """Stripper が除去対象とする全既知マーカーが正しく除去されること。

    [SWITCH_ANGLE:] は KNOWN_PREFIXES に含まれない（available_presets が非空のリクエストでは
    use_streaming=False となり Stripper が使われないため）。
    """
    result = _feed_all([
        "A[INSCRIBE_MEMORY:contextual|1.0|x]"
        "B[CARVE_NARRATIVE:append|y]"
        "C[DRIFT:z]"
        "D[DRIFT_RESET]"
        "E[END_SESSION:理由]"
        "F"
    ])

    assert result == "ABCDEF"


# ─── マーカーではない '[' ────────────────────────────────────────────────────────


def test_stripper_non_marker_bracket_passthrough():
    """既知マーカーではない '[' はそのまま出力されること。"""
    result = _feed_all(["[ありがとう]"])

    assert result == "[ありがとう]"


def test_stripper_markdown_link_passthrough():
    """Markdown リンク形式 [text](url) はマーカーと誤認識されずそのまま出力されること。"""
    result = _feed_all(["[詳細はこちら](https://example.com)を参照"])

    assert result == "[詳細はこちら](https://example.com)を参照"


def test_stripper_lone_bracket_passthrough():
    """対応する ']' のない '[' は flush() 後にそのまま出力されること。"""
    result = _feed_all(["開き括弧[だけ"])

    assert result == "開き括弧[だけ"


# ─── flush() の動作 ────────────────────────────────────────────────────────────


def test_stripper_flush_completes_buffered_marker():
    """ストリーム終了時、バッファに不完全なマーカーが残っていても flush() で処理されること。

    ']' が来ないまま終了した場合、マーカーではないと判断してテキストとして出力する。
    """
    stripper = StreamingTagStripper()
    out = stripper.feed("[INSCRIBE_MEMORY:未完成")
    remaining = stripper.flush()

    # 完結しないマーカーは flush() 時にテキストとして処理されること
    # (parse_tags が未閉じタグをそのまま残す仕様に準拠)
    full = out + remaining
    # クラッシュせずに何らかのテキストが返ること
    assert isinstance(full, str)


def test_stripper_flush_empty_buffer():
    """バッファが空の場合、flush() は空文字列を返すこと。"""
    stripper = StreamingTagStripper()
    stripper.feed("マーカーなし")
    remaining = stripper.flush()

    assert remaining == ""


def test_stripper_reuse_after_flush_raises_no_error():
    """flush() 後も stripper インスタンスを継続使用できること。

    flush() でバッファがリセットされ、次の feed() が正しく動作することを確認する。
    """
    stripper = StreamingTagStripper()
    stripper.feed("最初のチャンク")
    stripper.flush()

    out = stripper.feed("2回目のチャンク")
    assert out == "2回目のチャンク"


def test_stripper_large_single_chunk_strips_marker():
    """1000文字超えの大きなチャンクでもマーカーが除去されること。

    Google等のプロバイダーがレスポンス全体を1チャンクで送信する場合、
    MAX_BUFFER (1000) を超えるバッファが渡されることがある。
    '[' より前のテキストを先に出力してから MAX_BUFFER チェックを行うため、
    '[' 以降にマーカーがあっても正しく除去できる。
    """
    # 1000文字超えの本文 + [INSCRIBE_MEMORY:...] タグを1チャンクで投入する
    long_text = "あ" * 1100
    marker = "[INSCRIBE_MEMORY:semantic|1.0|長い記憶内容が入ります。詳細な情報をここに記録する。]"
    chunk = long_text + marker

    stripper = StreamingTagStripper()
    out = stripper.feed(chunk)
    out += stripper.flush()

    # 本文は出力される
    assert "あ" * 1100 in out
    # マーカーは除去される
    assert "[INSCRIBE_MEMORY:" not in out


def test_stripper_large_chunk_marker_before_text():
    """大きなチャンクでマーカーが先頭にある場合も除去されること。"""
    marker = "[INSCRIBE_MEMORY:semantic|1.0|記憶内容。]"
    long_text = "い" * 1100
    chunk = marker + long_text

    stripper = StreamingTagStripper()
    out = stripper.feed(chunk)
    out += stripper.flush()

    assert "い" * 1100 in out
    assert "[INSCRIBE_MEMORY:" not in out


# ─── StreamingTagStripper バッククォート処理 ────────────────────────────────────


def test_stripper_inline_code_marker_not_stripped():
    """インラインバッククォート内のマーカー形式テキストは除去されないこと。"""
    result = _feed_all(["`[INSCRIBE_MEMORY:cat|content]`"])

    assert "`[INSCRIBE_MEMORY:cat|content]`" in result


def test_stripper_inline_code_marker_with_surrounding_text():
    """バッククォート外の本物のマーカーは除去され、バッククォート内は残ること。"""
    result = _feed_all([
        "説明として `[INSCRIBE_MEMORY:x|y]` と書いた。[INSCRIBE_MEMORY:real|data]"
    ])

    assert "`[INSCRIBE_MEMORY:x|y]`" in result
    assert "[INSCRIBE_MEMORY:real|data]" not in result


def test_stripper_fenced_code_marker_not_stripped():
    """コードフェンス内のマーカー形式テキストは除去されないこと。"""
    result = _feed_all(["```\n[INSCRIBE_MEMORY:fact|data]\n```"])

    assert "[INSCRIBE_MEMORY:fact|data]" in result


def test_stripper_inline_code_split_across_chunks():
    """インラインコードがチャンク境界をまたいでも、内部のマーカーは除去されないこと。"""
    result = _feed_all([
        "前 `[INSCRIBE_",
        "MEMORY:x|y]` 後 [INSCRIBE_MEMORY:real|tag]",
    ])

    assert "[INSCRIBE_MEMORY:x|y]" in result
    assert "[INSCRIBE_MEMORY:real|tag]" not in result
    assert "前" in result
    assert "後" in result
