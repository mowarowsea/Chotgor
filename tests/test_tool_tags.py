"""backend.character_actions.tool_tags モジュールのユニットテスト。

ツール名→タグ名変換と引数→タグ本体文字列変換ロジックを網羅的に検証する。

対象関数:
    tool_call_to_tag_body() — ツール呼び出しを (tag_name, tag_body) に変換する

テスト方針:
    - 全 7 種の既知ツール（inscribe_memory / carve_narrative / drift / drift_reset /
      switch_angle / end_session / power_recall）を個別に検証する
    - 未知ツール名のフォールバック動作を確認する
    - 空引数辞書でも例外を送出しないことを確認する
    - tag_body フォーマットは _parse_tag_body() に渡せる形式であること
"""

import pytest

from backend.character_actions.tool_tags import TOOL_TO_TAG, tool_call_to_tag_body


class TestToolToTagMapping:
    """TOOL_TO_TAG 定数の構造を検証するテストクラス。"""

    def test_all_expected_tools_present(self):
        """全 7 種の既知ツールが TOOL_TO_TAG に登録されていること。"""
        expected = {
            "inscribe_memory",
            "carve_narrative",
            "drift",
            "drift_reset",
            "switch_angle",
            "end_session",
            "power_recall",
        }
        assert expected == set(TOOL_TO_TAG.keys())

    def test_tag_names_are_uppercase(self):
        """全タグ名が大文字であること。"""
        for tag_name in TOOL_TO_TAG.values():
            assert tag_name == tag_name.upper(), f"タグ名が大文字でない: {tag_name}"

    def test_tag_names_unique(self):
        """全タグ名が重複していないこと。"""
        tag_names = list(TOOL_TO_TAG.values())
        assert len(tag_names) == len(set(tag_names))


class TestToolCallToTagBodyInscribeMemory:
    """inscribe_memory ツールの変換を検証するテストクラス。"""

    def test_returns_correct_tag_name(self):
        """tag_name が INSCRIBE_MEMORY であること。"""
        tag_name, _ = tool_call_to_tag_body(
            "inscribe_memory",
            {"category": "contextual", "impact": 1.5, "content": "テスト内容"},
        )
        assert tag_name == "INSCRIBE_MEMORY"

    def test_body_pipe_format(self):
        """body が category|impact|content のパイプ区切りフォーマットであること。"""
        _, body = tool_call_to_tag_body(
            "inscribe_memory",
            {"category": "contextual", "impact": 1.5, "content": "ユーザはドン引きしていた"},
        )
        assert body == "contextual|1.5|ユーザはドン引きしていた"

    def test_impact_float_preserved(self):
        """impact の float 値が body に埋め込まれること。"""
        _, body = tool_call_to_tag_body(
            "inscribe_memory",
            {"category": "user_info", "impact": 0.8, "content": "猫好き"},
        )
        parts = body.split("|", 2)
        assert parts[1] == "0.8"

    def test_content_with_pipe_survives(self):
        """content にパイプ文字が含まれる場合も body に正しく埋め込まれること。"""
        _, body = tool_call_to_tag_body(
            "inscribe_memory",
            {"category": "semantic_knowledge", "impact": 1.0, "content": "A|B|C"},
        )
        # split(|, 2) で category|impact|content として分割できること
        parts = body.split("|", 2)
        assert parts[0] == "semantic_knowledge"
        assert parts[1] == "1.0"
        assert parts[2] == "A|B|C"

    def test_empty_args_returns_empty_body(self):
        """空引数辞書でも例外を送出せず空フィールドで body が返ること。"""
        tag_name, body = tool_call_to_tag_body("inscribe_memory", {})
        assert tag_name == "INSCRIBE_MEMORY"
        assert body == "||"


class TestToolCallToTagBodyCarveNarrative:
    """carve_narrative ツールの変換を検証するテストクラス。"""

    def test_returns_correct_tag_name(self):
        """tag_name が CARVE_NARRATIVE であること。"""
        tag_name, _ = tool_call_to_tag_body(
            "carve_narrative",
            {"mode": "append", "content": "新しい自己認識"},
        )
        assert tag_name == "CARVE_NARRATIVE"

    def test_body_format_mode_and_content(self):
        """body が mode|content のパイプ区切りであること。"""
        _, body = tool_call_to_tag_body(
            "carve_narrative",
            {"mode": "append", "content": "新しい方針が生まれた"},
        )
        assert body == "append|新しい方針が生まれた"

    def test_default_mode_is_append(self):
        """mode が省略された場合、デフォルト値 'append' が使われること。"""
        _, body = tool_call_to_tag_body(
            "carve_narrative",
            {"content": "内容のみ"},
        )
        assert body.startswith("append|")

    def test_overwrite_mode(self):
        """overwrite モードも正しく変換されること。"""
        _, body = tool_call_to_tag_body(
            "carve_narrative",
            {"mode": "overwrite", "content": "全書き換え"},
        )
        assert body == "overwrite|全書き換え"

    def test_empty_args(self):
        """空引数でも例外を送出せず空フィールドで返ること。"""
        tag_name, body = tool_call_to_tag_body("carve_narrative", {})
        assert tag_name == "CARVE_NARRATIVE"
        assert body == "append|"


class TestToolCallToTagBodyDrift:
    """drift ツールの変換を検証するテストクラス。"""

    def test_returns_correct_tag_name(self):
        """tag_name が DRIFT であること。"""
        tag_name, _ = tool_call_to_tag_body("drift", {"content": "ドリフト内容"})
        assert tag_name == "DRIFT"

    def test_body_is_content(self):
        """body が content の値そのものであること。"""
        _, body = tool_call_to_tag_body("drift", {"content": "新方針テキスト"})
        assert body == "新方針テキスト"

    def test_empty_args(self):
        """空引数でも例外を送出せず空文字 body が返ること。"""
        tag_name, body = tool_call_to_tag_body("drift", {})
        assert tag_name == "DRIFT"
        assert body == ""


class TestToolCallToTagBodyDriftReset:
    """drift_reset ツールの変換を検証するテストクラス。"""

    def test_returns_correct_tag_name(self):
        """tag_name が DRIFT_RESET であること。"""
        tag_name, _ = tool_call_to_tag_body("drift_reset", {})
        assert tag_name == "DRIFT_RESET"

    def test_body_is_empty_string(self):
        """引数なしツールのため body が空文字であること。"""
        _, body = tool_call_to_tag_body("drift_reset", {})
        assert body == ""

    def test_extra_args_ignored(self):
        """余分な引数があっても body が空文字のままであること。"""
        _, body = tool_call_to_tag_body("drift_reset", {"unexpected_key": "value"})
        assert body == ""


class TestToolCallToTagBodySwitchAngle:
    """switch_angle ツールの変換を検証するテストクラス。"""

    def test_returns_correct_tag_name(self):
        """tag_name が SWITCH_ANGLE であること。"""
        tag_name, _ = tool_call_to_tag_body(
            "switch_angle",
            {"preset_name": "Gemma4", "self_instruction": "大胆に"},
        )
        assert tag_name == "SWITCH_ANGLE"

    def test_body_format_preset_and_instruction(self):
        """body が preset_name|self_instruction のパイプ区切りであること。"""
        _, body = tool_call_to_tag_body(
            "switch_angle",
            {"preset_name": "ClaudeCode", "self_instruction": "静かに話せ"},
        )
        assert body == "ClaudeCode|静かに話せ"

    def test_empty_args(self):
        """空引数でも例外を送出せず空フィールドで返ること。"""
        tag_name, body = tool_call_to_tag_body("switch_angle", {})
        assert tag_name == "SWITCH_ANGLE"
        assert body == "|"


class TestToolCallToTagBodyEndSession:
    """end_session ツールの変換を検証するテストクラス。"""

    def test_returns_correct_tag_name(self):
        """tag_name が END_SESSION であること。"""
        tag_name, _ = tool_call_to_tag_body("end_session", {"reason": "会話完了"})
        assert tag_name == "END_SESSION"

    def test_body_is_reason(self):
        """body が reason の値であること。"""
        _, body = tool_call_to_tag_body("end_session", {"reason": "さようなら"})
        assert body == "さようなら"

    def test_empty_reason(self):
        """reason が省略されても空文字 body が返ること。"""
        _, body = tool_call_to_tag_body("end_session", {})
        assert body == ""


class TestToolCallToTagBodyPowerRecall:
    """power_recall ツールの変換を検証するテストクラス。"""

    def test_returns_correct_tag_name(self):
        """tag_name が POWER_RECALL であること。"""
        tag_name, _ = tool_call_to_tag_body(
            "power_recall",
            {"query": "初めて会った日", "top_k": 10},
        )
        assert tag_name == "POWER_RECALL"

    def test_body_format_query_and_top_k(self):
        """body が query|top_k のパイプ区切りであること。"""
        _, body = tool_call_to_tag_body(
            "power_recall",
            {"query": "感動した出来事", "top_k": 3},
        )
        assert body == "感動した出来事|3"

    def test_default_top_k_is_5(self):
        """top_k が省略された場合、デフォルト値 5 が使われること。"""
        _, body = tool_call_to_tag_body(
            "power_recall",
            {"query": "記憶を呼び起こせ"},
        )
        assert body == "記憶を呼び起こせ|5"

    def test_empty_args(self):
        """空引数でも例外を送出せず空クエリ・デフォルト top_k で返ること。"""
        tag_name, body = tool_call_to_tag_body("power_recall", {})
        assert tag_name == "POWER_RECALL"
        assert body == "|5"


class TestToolCallToTagBodyUnknownTool:
    """未知ツール名のフォールバック動作を検証するテストクラス。"""

    def test_tag_name_is_uppercased_tool_name(self):
        """未知ツール名はそのまま大文字化してタグ名として使われること。"""
        tag_name, _ = tool_call_to_tag_body("my_custom_tool", {"key": "value"})
        assert tag_name == "MY_CUSTOM_TOOL"

    def test_body_is_pipe_joined_key_value(self):
        """未知ツールの body は key=value 形式をパイプで結合した文字列であること。"""
        _, body = tool_call_to_tag_body(
            "unknown_tool",
            {"alpha": "aaa", "beta": "bbb"},
        )
        assert "alpha=aaa" in body
        assert "beta=bbb" in body
        assert "|" in body

    def test_empty_args_unknown_tool(self):
        """空引数の未知ツールでも例外を送出せず空文字 body が返ること。"""
        tag_name, body = tool_call_to_tag_body("no_args_tool", {})
        assert tag_name == "NO_ARGS_TOOL"
        assert body == ""
