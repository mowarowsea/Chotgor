"""switcher.extract() のテスト。

[SWITCH_ANGLE:preset_name|self_instruction] タグを正しく抽出・除去できるかを検証する。
tag_parser.py の汎用パーサーを使っているため、エッジケースも含めて網羅的にテストする。
"""

import pytest

from backend.core.chat.switcher import extract


class TestSwitcherExtractBasic:
    """基本的なタグ抽出とクリーンテキスト生成を検証する。"""

    def test_no_tag_returns_original_text_and_none(self):
        """タグが存在しない場合、テキストはそのままで switch_request は None を返す。"""
        text = "こんにちは、元気ですか？"
        clean, switch_info = extract(text)
        assert clean == text
        assert switch_info is None

    def test_basic_tag_extracted(self):
        """基本的な [SWITCH_ANGLE:name|instruction] タグを正しく抽出する。"""
        text = "返事します。[SWITCH_ANGLE:gemini2FlashLite|軽くさっぱりと応答する]"
        clean, switch_info = extract(text)
        assert switch_info is not None
        assert switch_info[0] == "gemini2FlashLite"
        assert switch_info[1] == "軽くさっぱりと応答する"

    def test_tag_removed_from_clean_text(self):
        """抽出後のクリーンテキストからタグが除去されている。"""
        text = "本文テキスト[SWITCH_ANGLE:preset-A|指針テキスト]"
        clean, _ = extract(text)
        assert "[SWITCH_ANGLE:" not in clean
        assert "本文テキスト" in clean

    def test_tag_at_end_of_text(self):
        """テキスト末尾にタグがある場合を正しく処理する。"""
        text = "以上です。\n[SWITCH_ANGLE:fastModel|短く答える]"
        clean, switch_info = extract(text)
        assert switch_info == ("fastModel", "短く答える")
        assert "[SWITCH_ANGLE:" not in clean

    def test_tag_only(self):
        """タグのみのテキストでも正しく処理する。"""
        text = "[SWITCH_ANGLE:myPreset|my instruction]"
        clean, switch_info = extract(text)
        assert switch_info == ("myPreset", "my instruction")
        assert clean == ""

    def test_whitespace_trimmed_in_parts(self):
        """preset_name と self_instruction の前後空白がトリムされる。"""
        text = "[SWITCH_ANGLE:  presetX  |  指針  ]"
        clean, switch_info = extract(text)
        assert switch_info is not None
        assert switch_info[0] == "presetX"
        assert switch_info[1] == "指針"


class TestSwitcherExtractEdgeCases:
    """エッジケースと境界条件を検証する。"""

    def test_self_instruction_empty_when_no_pipe(self):
        """パイプ区切りがない場合、self_instruction は空文字になる。"""
        text = "[SWITCH_ANGLE:presetOnly]"
        clean, switch_info = extract(text)
        assert switch_info is not None
        assert switch_info[0] == "presetOnly"
        assert switch_info[1] == ""

    def test_multiple_tags_uses_first_only(self):
        """複数タグがある場合は最初の1件のみ使用する。"""
        text = "[SWITCH_ANGLE:first|first-inst][SWITCH_ANGLE:second|second-inst]"
        clean, switch_info = extract(text)
        assert switch_info is not None
        assert switch_info[0] == "first"
        assert switch_info[1] == "first-inst"

    def test_tag_inside_code_fence_is_ignored(self):
        """コードフェンス内のタグはスキャンをスキップする。"""
        text = "```\n[SWITCH_ANGLE:preset|instruction]\n```"
        clean, switch_info = extract(text)
        assert switch_info is None

    def test_preset_name_with_hyphens_and_numbers(self):
        """ハイフンや数字を含むプリセット名を正しく処理する。"""
        text = "[SWITCH_ANGLE:gemini-2.0-flash-lite|さっぱり応答]"
        clean, switch_info = extract(text)
        assert switch_info is not None
        assert switch_info[0] == "gemini-2.0-flash-lite"

    def test_self_instruction_with_pipe_uses_first_split_only(self):
        """self_instruction にパイプが含まれる場合は最初の区切りのみで分割する。"""
        text = "[SWITCH_ANGLE:preset|指針A|余分な内容]"
        clean, switch_info = extract(text)
        assert switch_info is not None
        assert switch_info[0] == "preset"
        assert switch_info[1] == "指針A|余分な内容"

    def test_surrounding_text_preserved(self):
        """タグ前後のテキストが保持される。"""
        text = "前のテキスト[SWITCH_ANGLE:p|i]後のテキスト"
        clean, switch_info = extract(text)
        assert "前のテキスト" in clean
        assert "後のテキスト" in clean
        assert switch_info == ("p", "i")

    def test_empty_string_input(self):
        """空文字入力でもエラーにならない。"""
        clean, switch_info = extract("")
        assert clean == ""
        assert switch_info is None
