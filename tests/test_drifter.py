"""backend.core.chat.drifter モジュールのユニットテスト。

inscriber.py の [MEMORY:...] テストと対称的な構造で、
[DRIFT:...] / [DRIFT_RESET] マーカーの抽出・除去を網羅的に検証する。
"""

from backend.core.chat.drifter import extract


# --- マーカーなし ---

def test_extract_no_markers():
    """マーカーがない場合、clean_text は元のテキストと同一で drifts は空・reset は False であること。"""
    text = "こんにちは！今日も元気ですよ。"
    clean, drifts, reset = extract(text)

    assert clean == text
    assert drifts == []
    assert reset is False


# --- DRIFT マーカー1件 ---

def test_extract_single_drift():
    """DRIFT マーカーが1件ある場合、drifts に1件の内容が格納されること。"""
    text = "分かりました。[DRIFT:もっとクールに話す]今後そうします。"
    clean, drifts, reset = extract(text)

    assert len(drifts) == 1
    assert drifts[0] == "もっとクールに話す"
    assert reset is False


# --- DRIFT マーカー複数件 ---

def test_extract_multiple_drifts():
    """DRIFT マーカーが複数ある場合、すべての内容が drifts に格納されること。"""
    text = "[DRIFT:指針A][DRIFT:指針B]普通のテキスト[DRIFT:指針C]"
    clean, drifts, reset = extract(text)

    assert len(drifts) == 3
    assert "指針A" in drifts
    assert "指針B" in drifts
    assert "指針C" in drifts
    assert reset is False


# --- DRIFT_RESET マーカーのみ ---

def test_extract_drift_reset_only():
    """DRIFT_RESET マーカーのみある場合、reset が True で drifts は空であること。"""
    text = "全部リセットします。[DRIFT_RESET]"
    clean, drifts, reset = extract(text)

    assert drifts == []
    assert reset is True


# --- DRIFT と DRIFT_RESET の共存 ---

def test_extract_drift_and_reset_coexist():
    """DRIFT と DRIFT_RESET が共存する場合、両方が正しく抽出されること。"""
    text = "[DRIFT_RESET][DRIFT:新しい指針]リセットして新たに設定します。"
    clean, drifts, reset = extract(text)

    assert len(drifts) == 1
    assert drifts[0] == "新しい指針"
    assert reset is True


# --- drift 内容の空白トリム ---

def test_extract_drifts_content_is_stripped():
    """DRIFT マーカー内のコンテンツが前後の空白をトリムされて返ること。"""
    text = "[DRIFT:  前後に空白がある指針  ]テキスト"
    clean, drifts, reset = extract(text)

    assert len(drifts) == 1
    assert drifts[0] == "前後に空白がある指針"


# --- 複数行コンテンツ ---

def test_extract_multiline_drift_content():
    """DRIFT マーカー内のコンテンツが複数行にわたる場合も正しく抽出されること。"""
    text = "[DRIFT:1行目\n2行目\n3行目]普通のテキスト"
    clean, drifts, reset = extract(text)

    assert len(drifts) == 1
    # 改行を含む内容が取得されること
    assert "1行目" in drifts[0]
    assert "2行目" in drifts[0]
    assert "3行目" in drifts[0]


# --- clean_text からマーカーが除去されること ---

def test_extract_removes_markers_from_clean_text():
    """clean_text に [DRIFT:...] も [DRIFT_RESET] も含まれないこと。"""
    text = "前のテキスト。[DRIFT:指針X][DRIFT_RESET]後のテキスト。"
    clean, drifts, reset = extract(text)

    assert "[DRIFT:" not in clean
    assert "[DRIFT_RESET]" not in clean
    assert "前のテキスト。" in clean
    assert "後のテキスト。" in clean


# --- 空文字列入力 ---

def test_extract_empty_string():
    """空文字列を入力した場合、clean_text は空で drifts は空、reset は False であること。"""
    clean, drifts, reset = extract("")

    assert clean == ""
    assert drifts == []
    assert reset is False


# --- 日本語コンテンツ ---

def test_extract_japanese_content():
    """DRIFT マーカー内に日本語テキストが含まれる場合も正しく抽出されること。"""
    text = "了解です。[DRIFT:敬語を使わずに話す。タメ口でいこう。]これからよろしく。"
    clean, drifts, reset = extract(text)

    assert len(drifts) == 1
    assert "敬語を使わずに話す" in drifts[0]
    assert "タメ口でいこう" in drifts[0]


# --- [MEMORY:...] マーカーを誤って抽出しないこと ---

def test_extract_does_not_touch_memory_markers():
    """[MEMORY:...] マーカーは除去せず残し、[DRIFT:...] だけを処理すること。

    drifter.py は inscriber.py の処理後のテキストを受け取る想定であるが、
    仮に [MEMORY:...] が混在していても誤って除去・抽出しないことを検証する。
    """
    text = "テキスト[DRIFT:指針A][MEMORY:user|1.0|ユーザは猫が好き]追記"
    clean, drifts, reset = extract(text)

    # [DRIFT:...] のみ処理されること
    assert len(drifts) == 1
    assert drifts[0] == "指針A"
    # [MEMORY:...] は clean_text に残ること
    assert "[MEMORY:user|1.0|ユーザは猫が好き]" in clean
    # [DRIFT:...] は clean_text から除去されること
    assert "[DRIFT:" not in clean
