"""backend.character_actions.switcher モジュールのユニットテスト。

Switcher クラスの タグ方式（switch_from_text）・ツール呼び出し方式（switch_angle）
の両方を網羅的に検証する。

タグ抽出テスト（switch_from_text）:
    [SWITCH_ANGLE:preset_name|self_instruction] タグを正しく抽出・除去できるかを検証する。
    tag_parser.py の汎用パーサーを使っているため、エッジケースも含めて網羅的にテストする。

ツール呼び出しテスト（switch_angle）:
    switch_request への格納と返却メッセージを検証する。
"""

import pytest

from backend.character_actions.switcher import Switcher


class TestSwitcherFromTextBasic:
    """Switcher.switch_from_text() の基本的なタグ抽出とクリーンテキスト生成を検証する。"""

    def test_no_tag_returns_original_text_and_none(self):
        """タグが存在しない場合、テキストはそのままで switch_request は None を返す。"""
        switcher = Switcher()
        text = "こんにちは、元気ですか？"
        clean = switcher.switch_from_text(text)
        assert clean == text
        assert switcher.switch_request is None

    def test_basic_tag_extracted(self):
        """基本的な [SWITCH_ANGLE:name|instruction] タグを正しく抽出する。"""
        switcher = Switcher()
        text = "返事します。[SWITCH_ANGLE:gemini2FlashLite|軽くさっぱりと応答する]"
        switcher.switch_from_text(text)
        assert switcher.switch_request is not None
        assert switcher.switch_request[0] == "gemini2FlashLite"
        assert switcher.switch_request[1] == "軽くさっぱりと応答する"

    def test_tag_removed_from_clean_text(self):
        """抽出後のクリーンテキストからタグが除去されている。"""
        switcher = Switcher()
        clean = switcher.switch_from_text("本文テキスト[SWITCH_ANGLE:preset-A|指針テキスト]")
        assert "[SWITCH_ANGLE:" not in clean
        assert "本文テキスト" in clean

    def test_tag_at_end_of_text(self):
        """テキスト末尾にタグがある場合を正しく処理する。"""
        switcher = Switcher()
        clean = switcher.switch_from_text("以上です。\n[SWITCH_ANGLE:fastModel|短く答える]")
        assert switcher.switch_request == ("fastModel", "短く答える")
        assert "[SWITCH_ANGLE:" not in clean

    def test_tag_only(self):
        """タグのみのテキストでも正しく処理する。"""
        switcher = Switcher()
        clean = switcher.switch_from_text("[SWITCH_ANGLE:myPreset|my instruction]")
        assert switcher.switch_request == ("myPreset", "my instruction")
        assert clean == ""

    def test_whitespace_trimmed_in_parts(self):
        """preset_name と self_instruction の前後空白がトリムされる。"""
        switcher = Switcher()
        switcher.switch_from_text("[SWITCH_ANGLE:  presetX  |  指針  ]")
        assert switcher.switch_request is not None
        assert switcher.switch_request[0] == "presetX"
        assert switcher.switch_request[1] == "指針"


class TestSwitcherFromTextEdgeCases:
    """Switcher.switch_from_text() のエッジケースと境界条件を検証する。"""

    def test_self_instruction_empty_when_no_pipe(self):
        """パイプ区切りがない場合、self_instruction は空文字になる。"""
        switcher = Switcher()
        switcher.switch_from_text("[SWITCH_ANGLE:presetOnly]")
        assert switcher.switch_request is not None
        assert switcher.switch_request[0] == "presetOnly"
        assert switcher.switch_request[1] == ""

    def test_multiple_tags_uses_first_only(self):
        """複数タグがある場合は最初の1件のみ使用する。"""
        switcher = Switcher()
        switcher.switch_from_text("[SWITCH_ANGLE:first|first-inst][SWITCH_ANGLE:second|second-inst]")
        assert switcher.switch_request is not None
        assert switcher.switch_request[0] == "first"
        assert switcher.switch_request[1] == "first-inst"

    def test_tag_inside_code_fence_is_ignored(self):
        """コードフェンス内のタグはスキャンをスキップする。"""
        switcher = Switcher()
        switcher.switch_from_text("```\n[SWITCH_ANGLE:preset|instruction]\n```")
        assert switcher.switch_request is None

    def test_preset_name_with_hyphens_and_numbers(self):
        """ハイフンや数字を含むプリセット名を正しく処理する。"""
        switcher = Switcher()
        switcher.switch_from_text("[SWITCH_ANGLE:gemini-2.0-flash-lite|さっぱり応答]")
        assert switcher.switch_request is not None
        assert switcher.switch_request[0] == "gemini-2.0-flash-lite"

    def test_self_instruction_with_pipe_uses_first_split_only(self):
        """self_instruction にパイプが含まれる場合は最初の区切りのみで分割する。"""
        switcher = Switcher()
        switcher.switch_from_text("[SWITCH_ANGLE:preset|指針A|余分な内容]")
        assert switcher.switch_request is not None
        assert switcher.switch_request[0] == "preset"
        assert switcher.switch_request[1] == "指針A|余分な内容"

    def test_surrounding_text_preserved(self):
        """タグ前後のテキストが保持される。"""
        switcher = Switcher()
        clean = switcher.switch_from_text("前のテキスト[SWITCH_ANGLE:p|i]後のテキスト")
        assert "前のテキスト" in clean
        assert "後のテキスト" in clean
        assert switcher.switch_request == ("p", "i")

    def test_empty_string_input(self):
        """空文字入力でもエラーにならない。"""
        switcher = Switcher()
        clean = switcher.switch_from_text("")
        assert clean == ""
        assert switcher.switch_request is None


# ===== ツール呼び出し方式: switch_angle のテスト =====

class TestSwitcherToolCall:
    """Switcher.switch_angle() のツール呼び出し方式を検証する。"""

    def test_switch_angle_stores_request(self):
        """switch_angle() が switch_request に (preset_name, self_instruction) を格納すること。"""
        switcher = Switcher()
        switcher.switch_angle("gemini2FlashLite", "軽くさっぱりと応答する")
        assert switcher.switch_request == ("gemini2FlashLite", "軽くさっぱりと応答する")

    def test_switch_angle_returns_message(self):
        """switch_angle() が preset_name を含む確認メッセージを返すこと。"""
        switcher = Switcher()
        result = switcher.switch_angle("fastModel", "短く答える")
        assert "fastModel" in result

    def test_switch_angle_empty_instruction(self):
        """self_instruction が空文字の場合も switch_request に格納されること。"""
        switcher = Switcher()
        switcher.switch_angle("myPreset", "")
        assert switcher.switch_request == ("myPreset", "")

    def test_switch_request_initial_value_is_none(self):
        """初期化直後は switch_request が None であること。"""
        switcher = Switcher()
        assert switcher.switch_request is None

    def test_switch_angle_overwrites_previous_request(self):
        """switch_angle() を複数回呼んだ場合、最後の値で上書きされること。"""
        switcher = Switcher()
        switcher.switch_angle("first", "first-inst")
        switcher.switch_angle("second", "second-inst")
        assert switcher.switch_request == ("second", "second-inst")
