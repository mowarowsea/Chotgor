"""backend.character_actions.switcher モジュールのユニットテスト。

extract_switch_angle_tags() のタグ抽出と Switcher.switch_angle() の switch_request 格納を
網羅的に検証する。タグ方式の実行口（タグ抽出 → switch_request 格納）は
ToolExecutor.apply_switch_angle_tags() に統一されているため、本ファイルでは抽出関数と
状態保持クラスを別々に検証する。
"""

import pytest

from backend.character_actions.switcher import Switcher, extract_switch_angle_tags


class TestExtractSwitchAngleTagsBasic:
    """extract_switch_angle_tags() の基本的なタグ抽出とクリーンテキスト生成を検証する。"""

    def test_no_tag_returns_original_text_and_none(self):
        """タグが存在しない場合、テキストはそのままで switch_request は None を返す。"""
        text = "こんにちは、元気ですか？"
        clean, switch_request = extract_switch_angle_tags(text)
        assert clean == text
        assert switch_request is None

    def test_basic_tag_extracted(self):
        """基本的な [SWITCH_ANGLE:name|instruction] タグを正しく抽出する。"""
        text = "返事します。[SWITCH_ANGLE:gemini2FlashLite|軽くさっぱりと応答する]"
        _, switch_request = extract_switch_angle_tags(text)
        assert switch_request == ("gemini2FlashLite", "軽くさっぱりと応答する")

    def test_tag_removed_from_clean_text(self):
        """抽出後のクリーンテキストからタグが除去されている。"""
        clean, _ = extract_switch_angle_tags("本文テキスト[SWITCH_ANGLE:preset-A|指針テキスト]")
        assert "[SWITCH_ANGLE:" not in clean
        assert "本文テキスト" in clean

    def test_tag_at_end_of_text(self):
        """テキスト末尾にタグがある場合を正しく処理する。"""
        clean, switch_request = extract_switch_angle_tags("以上です。\n[SWITCH_ANGLE:fastModel|短く答える]")
        assert switch_request == ("fastModel", "短く答える")
        assert "[SWITCH_ANGLE:" not in clean

    def test_tag_only(self):
        """タグのみのテキストでも正しく処理する。"""
        clean, switch_request = extract_switch_angle_tags("[SWITCH_ANGLE:myPreset|my instruction]")
        assert switch_request == ("myPreset", "my instruction")
        assert clean == ""

    def test_whitespace_trimmed_in_parts(self):
        """preset_name と self_instruction の前後空白がトリムされる。"""
        _, switch_request = extract_switch_angle_tags("[SWITCH_ANGLE:  presetX  |  指針  ]")
        assert switch_request == ("presetX", "指針")


class TestExtractSwitchAngleTagsEdgeCases:
    """extract_switch_angle_tags() のエッジケースと境界条件を検証する。"""

    def test_self_instruction_empty_when_no_pipe(self):
        """パイプ区切りがない場合、self_instruction は空文字になる。"""
        _, switch_request = extract_switch_angle_tags("[SWITCH_ANGLE:presetOnly]")
        assert switch_request == ("presetOnly", "")

    def test_multiple_tags_uses_first_only(self):
        """複数タグがある場合は最初の1件のみ使用する。"""
        _, switch_request = extract_switch_angle_tags(
            "[SWITCH_ANGLE:first|first-inst][SWITCH_ANGLE:second|second-inst]"
        )
        assert switch_request == ("first", "first-inst")

    def test_tag_inside_code_fence_is_ignored(self):
        """コードフェンス内のタグはスキャンをスキップする。"""
        _, switch_request = extract_switch_angle_tags("```\n[SWITCH_ANGLE:preset|instruction]\n```")
        assert switch_request is None

    def test_preset_name_with_hyphens_and_numbers(self):
        """ハイフンや数字を含むプリセット名を正しく処理する。"""
        _, switch_request = extract_switch_angle_tags("[SWITCH_ANGLE:gemini-2.0-flash-lite|さっぱり応答]")
        assert switch_request is not None
        assert switch_request[0] == "gemini-2.0-flash-lite"

    def test_self_instruction_with_pipe_uses_first_split_only(self):
        """self_instruction にパイプが含まれる場合は最初の区切りのみで分割する。"""
        _, switch_request = extract_switch_angle_tags("[SWITCH_ANGLE:preset|指針A|余分な内容]")
        assert switch_request == ("preset", "指針A|余分な内容")

    def test_surrounding_text_preserved(self):
        """タグ前後のテキストが保持される。"""
        clean, switch_request = extract_switch_angle_tags("前のテキスト[SWITCH_ANGLE:p|i]後のテキスト")
        assert "前のテキスト" in clean
        assert "後のテキスト" in clean
        assert switch_request == ("p", "i")

    def test_empty_string_input(self):
        """空文字入力でもエラーにならない。"""
        clean, switch_request = extract_switch_angle_tags("")
        assert clean == ""
        assert switch_request is None


# ===== ツール呼び出し方式: switch_angle のテスト =====

class TestSwitcherToolCall:
    """Switcher.switch_angle() のツール呼び出し方式（state 保持）を検証する。"""

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
