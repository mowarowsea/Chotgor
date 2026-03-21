"""backend.core.chat.drifter モジュールのユニットテスト。

Drifter クラスの タグ方式（drift_from_text）・ツール呼び出し方式（drift / drift_reset）
の両方を網羅的に検証する。

タグ抽出テスト（drift_from_text）:
    inscriber.py の [INSCRIBE_MEMORY:...] テストと対称的な構造で、
    [DRIFT:...] / [DRIFT_RESET] マーカーの抽出・除去・DBへの反映を検証する。

ツール呼び出しテスト（drift / drift_reset）:
    DriftManager をモックして、drift_manager.add_drift / reset_drifts の呼び出しを検証する。
"""

from unittest.mock import MagicMock

from backend.core.chat.drifter import Drifter


def _make_drifter(session_id="sess-1", character_id="char-1", drift_manager=None) -> Drifter:
    """テスト用 Drifter インスタンスを生成するヘルパー。

    drift_manager が省略された場合は MagicMock を使用する。

    Args:
        session_id: セッションID（デフォルト: "sess-1"）。
        character_id: キャラクターID（デフォルト: "char-1"）。
        drift_manager: 差し替え用 DriftManager（省略時は MagicMock）。

    Returns:
        テスト用 Drifter インスタンス。
    """
    if drift_manager is None:
        drift_manager = MagicMock()
    return Drifter(session_id=session_id, character_id=character_id, drift_manager=drift_manager)


# ===== タグ方式: drift_from_text のテスト =====

class TestDrifterFromText:
    """Drifter.drift_from_text() のタグ抽出・クリーンテキスト生成・DB反映を検証する。"""

    def test_no_markers_returns_original_text(self):
        """マーカーがない場合、clean_text は元のテキストと同一であること。"""
        drifter = _make_drifter()
        text = "こんにちは！今日も元気ですよ。"
        clean = drifter.drift_from_text(text)
        assert clean == text

    def test_no_markers_does_not_call_drift_manager(self):
        """マーカーがない場合、drift_manager のメソッドが呼ばれないこと。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        drifter.drift_from_text("普通のテキスト。")
        dm.add_drift.assert_not_called()
        dm.reset_drifts.assert_not_called()

    def test_single_drift_calls_add_drift(self):
        """DRIFT マーカーが1件ある場合、drift_manager.add_drift が1回呼ばれること。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        drifter.drift_from_text("分かりました。[DRIFT:もっとクールに話す]今後そうします。")
        dm.add_drift.assert_called_once_with("sess-1", "char-1", "もっとクールに話す")

    def test_multiple_drifts_calls_add_drift_multiple_times(self):
        """DRIFT マーカーが複数ある場合、drift_manager.add_drift がその数だけ呼ばれること。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        drifter.drift_from_text("[DRIFT:指針A][DRIFT:指針B]テキスト[DRIFT:指針C]")
        assert dm.add_drift.call_count == 3

    def test_drift_reset_calls_reset_drifts(self):
        """DRIFT_RESET マーカーがある場合、drift_manager.reset_drifts が呼ばれること。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        drifter.drift_from_text("全部リセットします。[DRIFT_RESET]")
        dm.reset_drifts.assert_called_once_with("sess-1", "char-1")
        dm.add_drift.assert_not_called()

    def test_drift_and_reset_coexist(self):
        """DRIFT と DRIFT_RESET が共存する場合、reset と add_drift がどちらも呼ばれること。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        drifter.drift_from_text("[DRIFT_RESET][DRIFT:新しい指針]リセットして新たに設定します。")
        dm.reset_drifts.assert_called_once()
        dm.add_drift.assert_called_once_with("sess-1", "char-1", "新しい指針")

    def test_drift_content_is_stripped(self):
        """DRIFT マーカー内のコンテンツが前後の空白をトリムされて add_drift に渡ること。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        drifter.drift_from_text("[DRIFT:  前後に空白がある指針  ]テキスト")
        dm.add_drift.assert_called_once_with("sess-1", "char-1", "前後に空白がある指針")

    def test_multiline_drift_content(self):
        """DRIFT マーカー内のコンテンツが複数行にわたる場合も正しく add_drift に渡ること。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        drifter.drift_from_text("[DRIFT:1行目\n2行目\n3行目]普通のテキスト")
        args = dm.add_drift.call_args[0]
        content = args[2]
        assert "1行目" in content
        assert "2行目" in content
        assert "3行目" in content

    def test_markers_removed_from_clean_text(self):
        """clean_text に [DRIFT:...] も [DRIFT_RESET] も含まれないこと。"""
        drifter = _make_drifter()
        clean = drifter.drift_from_text("前のテキスト。[DRIFT:指針X][DRIFT_RESET]後のテキスト。")
        assert "[DRIFT:" not in clean
        assert "[DRIFT_RESET]" not in clean
        assert "前のテキスト。" in clean
        assert "後のテキスト。" in clean

    def test_empty_string_input(self):
        """空文字列を入力した場合、clean_text は空でエラーにならないこと。"""
        drifter = _make_drifter()
        clean = drifter.drift_from_text("")
        assert clean == ""

    def test_japanese_content(self):
        """DRIFT マーカー内に日本語テキストが含まれる場合も正しく処理されること。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        drifter.drift_from_text("了解です。[DRIFT:敬語を使わずに話す。タメ口でいこう。]よろしく。")
        args = dm.add_drift.call_args[0]
        assert "敬語を使わずに話す" in args[2]
        assert "タメ口でいこう" in args[2]

    def test_no_session_id_skips_drift_manager(self):
        """session_id が None の場合、drift_manager を呼ばずにクリーンテキストを返すこと。"""
        dm = MagicMock()
        drifter = Drifter(session_id=None, character_id="char-1", drift_manager=dm)
        clean = drifter.drift_from_text("[DRIFT:指針A]テキスト")
        dm.add_drift.assert_not_called()
        assert "[DRIFT:" not in clean

    def test_no_drift_manager_skips_db(self):
        """drift_manager が None の場合、エラーにならずクリーンテキストを返すこと。"""
        drifter = Drifter(session_id="sess-1", character_id="char-1", drift_manager=None)
        clean = drifter.drift_from_text("[DRIFT:指針A]テキスト")
        assert "[DRIFT:" not in clean

    def test_does_not_touch_inscribe_memory_markers(self):
        """[INSCRIBE_MEMORY:...] マーカーは除去せず残し、[DRIFT:...] だけを処理すること。

        drifter.py は inscriber.py の処理後のテキストを受け取る想定であるが、
        仮に [INSCRIBE_MEMORY:...] が混在していても誤って除去・抽出しないことを検証する。
        """
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        clean = drifter.drift_from_text("テキスト[DRIFT:指針A][INSCRIBE_MEMORY:user|1.0|ユーザは猫が好き]追記")
        # [DRIFT:...] のみ処理されること
        dm.add_drift.assert_called_once()
        args = dm.add_drift.call_args[0]
        assert args[2] == "指針A"
        # [INSCRIBE_MEMORY:...] は clean_text に残ること
        assert "[INSCRIBE_MEMORY:user|1.0|ユーザは猫が好き]" in clean
        # [DRIFT:...] は clean_text から除去されること
        assert "[DRIFT:" not in clean


# ===== ツール呼び出し方式: drift / drift_reset のテスト =====

class TestDrifterToolCall:
    """Drifter.drift() / drift_reset() のツール呼び出し方式を検証する。"""

    def test_drift_calls_add_drift(self):
        """drift() が drift_manager.add_drift を正しく呼び出すこと。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        result = drifter.drift("クールに話す")
        dm.add_drift.assert_called_once_with("sess-1", "char-1", "クールに話す")
        assert result == "指針を設定した。"

    def test_drift_without_session_id_returns_unavailable(self):
        """session_id が None の場合、drift() は利用不可メッセージを返すこと。"""
        drifter = Drifter(session_id=None, character_id="char-1", drift_manager=MagicMock())
        result = drifter.drift("指針テキスト")
        assert result == "SELF_DRIFT は利用できない。"

    def test_drift_without_drift_manager_returns_unavailable(self):
        """drift_manager が None の場合、drift() は利用不可メッセージを返すこと。"""
        drifter = Drifter(session_id="sess-1", character_id="char-1", drift_manager=None)
        result = drifter.drift("指針テキスト")
        assert result == "SELF_DRIFT は利用できない。"

    def test_drift_handles_exception(self):
        """drift_manager.add_drift が例外を投げた場合、エラーメッセージを返すこと。"""
        dm = MagicMock()
        dm.add_drift.side_effect = RuntimeError("DB error")
        drifter = _make_drifter(drift_manager=dm)
        result = drifter.drift("指針テキスト")
        assert "[drift error:" in result

    def test_drift_reset_calls_reset_drifts(self):
        """drift_reset() が drift_manager.reset_drifts を正しく呼び出すこと。"""
        dm = MagicMock()
        drifter = _make_drifter(drift_manager=dm)
        result = drifter.drift_reset()
        dm.reset_drifts.assert_called_once_with("sess-1", "char-1")
        assert result == "指針をリセットした。"

    def test_drift_reset_without_session_id_returns_unavailable(self):
        """session_id が None の場合、drift_reset() は利用不可メッセージを返すこと。"""
        drifter = Drifter(session_id=None, character_id="char-1", drift_manager=MagicMock())
        result = drifter.drift_reset()
        assert result == "SELF_DRIFT は利用できない。"

    def test_drift_reset_handles_exception(self):
        """drift_manager.reset_drifts が例外を投げた場合、エラーメッセージを返すこと。"""
        dm = MagicMock()
        dm.reset_drifts.side_effect = RuntimeError("DB error")
        drifter = _make_drifter(drift_manager=dm)
        result = drifter.drift_reset()
        assert "[drift_reset error:" in result
