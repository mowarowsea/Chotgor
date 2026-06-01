"""backend.character_actions.tool_tags モジュールのユニットテスト。

ツール名→タグ名変換と引数→表示用構造化辞書への変換ロジックを網羅的に検証する。

対象関数:
    tool_call_to_structured_tag() — tool-use 形式の (name, args) を
        {tag_name, meta, fields, preview} 辞書に変換する

テスト方針:
    - 全 6 種の既知ツール（inscribe_memory / carve_narrative / drift / drift_reset /
      switch_angle / power_recall）を個別に検証する
    - 未知ツール名のフォールバック動作を確認する
    - 空引数辞書でも例外を送出しないことを確認する
    - 戻り値辞書は logs_ui.LogTag フロント型と同じ {tag_name, meta, fields, preview} を満たすこと
    - MCP ネームスペースプレフィックス ("mcp__<server>__") 付きツール名でも
      プレフィックス除去後に正しい tag_name にマップされること
    - TAG_META が全既知タグを含み、表示メタの構造（label/cls）を持つこと
"""

import pytest

from backend.character_actions.tool_tags import (
    TAG_META,
    TOOL_TO_TAG,
    tool_call_to_structured_tag,
)


class TestToolToTagMapping:
    """TOOL_TO_TAG 定数の構造を検証するテストクラス。

    キャラクターアクションのツールが正しく既知タグへマップされる前提を守る。
    """

    def test_all_expected_tools_present(self):
        """全 6 種の既知ツールが TOOL_TO_TAG に登録されていること（end_session は廃止済み）。"""
        expected = {
            "inscribe_memory",
            "carve_narrative",
            "drift",
            "drift_reset",
            "switch_angle",
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


class TestTagMeta:
    """TAG_META 定数の構造を検証するテストクラス。

    UI バッジ表示用のメタ情報が、全既知タグおよび ANTICIPATE_RESPONSE / END_SESSION
    のようなツール以外のタグ名についても揃っていることを保証する。
    """

    def test_all_tool_tags_have_meta(self):
        """TOOL_TO_TAG の全タグ名が TAG_META に登録されていること。"""
        for tag_name in TOOL_TO_TAG.values():
            assert tag_name in TAG_META, f"TAG_META に未登録: {tag_name}"

    def test_anticipate_response_meta_present(self):
        """ANTICIPATE_RESPONSE は tool-use 化していないが TAG_META には含まれること。"""
        assert "ANTICIPATE_RESPONSE" in TAG_META
        assert TAG_META["ANTICIPATE_RESPONSE"]["cls"] == "tag-anticipate"

    def test_end_session_meta_present(self):
        """END_SESSION（ツール非登録だがタグとして残存）もメタを持つこと。"""
        assert "END_SESSION" in TAG_META

    def test_each_meta_has_label_and_cls(self):
        """全メタが label / cls キーを持ち、いずれも非空文字列であること。"""
        for tag_name, meta in TAG_META.items():
            assert "label" in meta, f"label 欠落: {tag_name}"
            assert "cls" in meta, f"cls 欠落: {tag_name}"
            assert isinstance(meta["label"], str) and meta["label"], f"label が空: {tag_name}"
            assert isinstance(meta["cls"], str) and meta["cls"], f"cls が空: {tag_name}"


def _assert_shape(result: dict, expected_tag_name: str) -> None:
    """戻り値辞書の共通形 {tag_name, meta, fields, preview} を検証する補助関数。

    フロントの LogTag 型と契約を維持するため、全ケースで形を確認する。
    """
    assert set(result.keys()) == {"tag_name", "meta", "fields", "preview"}
    assert result["tag_name"] == expected_tag_name
    assert isinstance(result["meta"], dict)
    assert "label" in result["meta"] and "cls" in result["meta"]
    assert isinstance(result["fields"], dict)
    assert isinstance(result["preview"], str)


class TestStructuredTagInscribeMemory:
    """inscribe_memory ツールの変換を検証するテストクラス。

    `INSCRIBE_MEMORY_SCHEMA` (`inscriber.py`) の現行仕様 ``{content, category, impact}``
    に従い、3 フィールドへ分解されることを確認する。
    """

    def test_returns_correct_shape(self):
        """戻り値が共通形を満たし tag_name が INSCRIBE_MEMORY であること。"""
        result = tool_call_to_structured_tag(
            "inscribe_memory",
            {"category": "contextual", "impact": 1.5, "content": "テスト内容"},
        )
        _assert_shape(result, "INSCRIBE_MEMORY")

    def test_fields_extracted(self):
        """fields にカテゴリ・重要度・内容が分解されて入ること。"""
        result = tool_call_to_structured_tag(
            "inscribe_memory",
            {"category": "contextual", "impact": 1.5, "content": "ユーザはドン引きしていた"},
        )
        assert result["fields"]["カテゴリ"] == "contextual"
        assert result["fields"]["重要度"] == "1.5"
        assert result["fields"]["内容"] == "ユーザはドン引きしていた"

    def test_preview_is_content(self):
        """preview は内容（content）になること。"""
        result = tool_call_to_structured_tag(
            "inscribe_memory",
            {"category": "user_info", "impact": 0.8, "content": "猫好き"},
        )
        assert result["preview"] == "猫好き"

    def test_meta_is_memory(self):
        """meta が "記憶" バッジ（tag-memory）であること。"""
        result = tool_call_to_structured_tag(
            "inscribe_memory",
            {"category": "identity", "impact": 1.0, "content": "テスト"},
        )
        assert result["meta"]["label"] == "記憶"
        assert result["meta"]["cls"] == "tag-memory"

    def test_content_with_pipe_preserved(self):
        """内容にパイプ文字が含まれても、構造化 API では文字列を経由しないので
        そのまま保持されること（旧 tag_body 経路の split 問題が起きない）。"""
        result = tool_call_to_structured_tag(
            "inscribe_memory",
            {"category": "semantic_knowledge", "impact": 1.0, "content": "A|B|C"},
        )
        assert result["fields"]["内容"] == "A|B|C"

    def test_empty_args(self):
        """空引数辞書でも例外を送出せず、空文字フィールドで返ること。"""
        result = tool_call_to_structured_tag("inscribe_memory", {})
        _assert_shape(result, "INSCRIBE_MEMORY")
        assert result["fields"]["カテゴリ"] == ""
        assert result["fields"]["重要度"] == ""
        assert result["fields"]["内容"] == ""
        assert result["preview"] == ""


class TestStructuredTagCarveNarrative:
    """carve_narrative ツールの変換を検証するテストクラス。"""

    def test_fields_extracted(self):
        """fields にモード・内容が分解されて入ること。"""
        result = tool_call_to_structured_tag(
            "carve_narrative",
            {"mode": "append", "content": "新しい方針が生まれた"},
        )
        _assert_shape(result, "CARVE_NARRATIVE")
        assert result["fields"]["モード"] == "append"
        assert result["fields"]["内容"] == "新しい方針が生まれた"
        assert result["preview"] == "新しい方針が生まれた"

    def test_default_mode_is_append(self):
        """mode が省略された場合、デフォルト値 'append' が使われること。"""
        result = tool_call_to_structured_tag(
            "carve_narrative", {"content": "内容のみ"}
        )
        assert result["fields"]["モード"] == "append"

    def test_overwrite_mode(self):
        """overwrite モードも正しく変換されること。"""
        result = tool_call_to_structured_tag(
            "carve_narrative",
            {"mode": "overwrite", "content": "全書き換え"},
        )
        assert result["fields"]["モード"] == "overwrite"
        assert result["fields"]["内容"] == "全書き換え"

    def test_meta_is_narrative(self):
        """meta が "ナラティブ" バッジ（tag-narrative）であること。"""
        result = tool_call_to_structured_tag("carve_narrative", {"content": "x"})
        assert result["meta"]["cls"] == "tag-narrative"

    def test_empty_args(self):
        """空引数でも例外を送出せず、モードはデフォルト append で返ること。"""
        result = tool_call_to_structured_tag("carve_narrative", {})
        _assert_shape(result, "CARVE_NARRATIVE")
        assert result["fields"]["モード"] == "append"
        assert result["fields"]["内容"] == ""


class TestStructuredTagDrift:
    """drift ツールの変換を検証するテストクラス。"""

    def test_fields_and_preview(self):
        """fields["内容"] に content が入り、preview も同じになること。"""
        result = tool_call_to_structured_tag("drift", {"content": "新方針テキスト"})
        _assert_shape(result, "DRIFT")
        assert result["fields"]["内容"] == "新方針テキスト"
        assert result["preview"] == "新方針テキスト"
        assert result["meta"]["cls"] == "tag-drift"

    def test_empty_args(self):
        """空引数でも例外を送出せず空文字 content で返ること。"""
        result = tool_call_to_structured_tag("drift", {})
        assert result["fields"]["内容"] == ""


class TestStructuredTagDriftReset:
    """drift_reset ツールの変換を検証するテストクラス。

    drift_reset は引数なしツールのため、固定マーカー "(リセット)" が表示される。
    """

    def test_fixed_marker(self):
        """fields["内容"] が "(リセット)" 固定であること。"""
        result = tool_call_to_structured_tag("drift_reset", {})
        _assert_shape(result, "DRIFT_RESET")
        assert result["fields"]["内容"] == "(リセット)"
        assert result["preview"] == "(リセット)"
        assert result["meta"]["cls"] == "tag-drift"

    def test_extra_args_ignored(self):
        """余分な引数があっても固定マーカーのままになること。"""
        result = tool_call_to_structured_tag("drift_reset", {"unexpected": "x"})
        assert result["fields"]["内容"] == "(リセット)"


class TestStructuredTagSwitchAngle:
    """switch_angle ツールの変換を検証するテストクラス。"""

    def test_fields_extracted(self):
        """fields にプリセット・コンテキストが分解されて入ること。"""
        result = tool_call_to_structured_tag(
            "switch_angle",
            {"preset_name": "ClaudeCode", "self_instruction": "静かに話せ"},
        )
        _assert_shape(result, "SWITCH_ANGLE")
        assert result["fields"]["プリセット"] == "ClaudeCode"
        assert result["fields"]["コンテキスト"] == "静かに話せ"

    def test_preview_prefers_context(self):
        """preview はコンテキスト（self_instruction）が優先されること。"""
        result = tool_call_to_structured_tag(
            "switch_angle",
            {"preset_name": "ClaudeCode", "self_instruction": "静かに話せ"},
        )
        assert result["preview"] == "静かに話せ"

    def test_preview_falls_back_to_preset(self):
        """コンテキストが空のときは preview がプリセット名にフォールバックすること。"""
        result = tool_call_to_structured_tag(
            "switch_angle",
            {"preset_name": "Gemma4"},
        )
        assert result["preview"] == "Gemma4"

    def test_empty_args(self):
        """空引数でも例外を送出せず空文字フィールドで返ること。"""
        result = tool_call_to_structured_tag("switch_angle", {})
        _assert_shape(result, "SWITCH_ANGLE")
        assert result["fields"]["プリセット"] == ""
        assert result["fields"]["コンテキスト"] == ""
        assert result["preview"] == ""


class TestStructuredTagPowerRecall:
    """power_recall ツールの変換を検証するテストクラス。"""

    def test_fields_extracted(self):
        """fields にクエリ・top_k が入り、preview がクエリになること。"""
        result = tool_call_to_structured_tag(
            "power_recall",
            {"query": "感動した出来事", "top_k": 3},
        )
        _assert_shape(result, "POWER_RECALL")
        assert result["fields"]["クエリ"] == "感動した出来事"
        assert result["fields"]["top_k"] == "3"
        assert result["preview"] == "感動した出来事"
        assert result["meta"]["cls"] == "tag-recall"

    def test_default_top_k_is_5(self):
        """top_k が省略された場合、デフォルト値 5 が文字列化されて入ること。"""
        result = tool_call_to_structured_tag(
            "power_recall",
            {"query": "記憶を呼び起こせ"},
        )
        assert result["fields"]["top_k"] == "5"

    def test_empty_args(self):
        """空引数でも例外を送出せず空クエリ・デフォルト top_k で返ること。"""
        result = tool_call_to_structured_tag("power_recall", {})
        _assert_shape(result, "POWER_RECALL")
        assert result["fields"]["クエリ"] == ""
        assert result["fields"]["top_k"] == "5"


class TestStructuredTagUnknownTool:
    """未知ツール名のフォールバック動作を検証するテストクラス。

    未登録ツールでもクラッシュさせず、tag_name を大文字化したフォールバック表示で
    返し、引数の key/value をそのまま fields に並べる動作を確認する。
    """

    def test_tag_name_is_uppercased_tool_name(self):
        """未知ツール名はそのまま大文字化してタグ名として使われること。"""
        result = tool_call_to_structured_tag(
            "my_custom_tool", {"key": "value"}
        )
        assert result["tag_name"] == "MY_CUSTOM_TOOL"

    def test_fallback_meta_is_unknown(self):
        """未知タグ名の場合、meta は tag-unknown にフォールバックすること。"""
        result = tool_call_to_structured_tag(
            "my_custom_tool", {"key": "value"}
        )
        assert result["meta"]["cls"] == "tag-unknown"
        assert result["meta"]["label"] == "MY_CUSTOM_TOOL"

    def test_args_are_passed_through(self):
        """未知ツールの fields は引数の key/value がそのまま並ぶこと。"""
        result = tool_call_to_structured_tag(
            "unknown_tool",
            {"alpha": "aaa", "beta": "bbb"},
        )
        assert result["fields"]["alpha"] == "aaa"
        assert result["fields"]["beta"] == "bbb"

    def test_empty_args_unknown_tool(self):
        """空引数の未知ツールでも例外を送出せず空 fields で返ること。"""
        result = tool_call_to_structured_tag("no_args_tool", {})
        assert result["tag_name"] == "NO_ARGS_TOOL"
        assert result["fields"] == {}


class TestMcpPrefixStripping:
    """Claude CLI の stream-json に含まれる MCP ネームスペースプレフィックスの除去を検証するテストクラス。

    Claude CLI が --output-format stream-json で出力する assistant イベントでは、
    ツール名に "mcp__<server_name>__<tool_name>" 形式のプレフィックスが付与される。
    例: "mcp__chotgor__inscribe_memory"

    プレフィックス除去後に正しい tag_name と fields に変換されることを確認する。
    """

    def test_mcp_inscribe_memory(self):
        """mcp__chotgor__inscribe_memory が INSCRIBE_MEMORY タグに変換されること。"""
        result = tool_call_to_structured_tag(
            "mcp__chotgor__inscribe_memory",
            {"category": "contextual", "impact": 0.7, "content": "テスト記憶"},
        )
        assert result["tag_name"] == "INSCRIBE_MEMORY"
        assert result["fields"]["カテゴリ"] == "contextual"
        assert result["fields"]["重要度"] == "0.7"
        assert result["fields"]["内容"] == "テスト記憶"

    def test_mcp_carve_narrative(self):
        """mcp__chotgor__carve_narrative が CARVE_NARRATIVE タグに変換されること。"""
        result = tool_call_to_structured_tag(
            "mcp__chotgor__carve_narrative",
            {"mode": "append", "content": "ナラティブ内容"},
        )
        assert result["tag_name"] == "CARVE_NARRATIVE"
        assert result["fields"]["内容"] == "ナラティブ内容"

    def test_mcp_drift_reset_single_underscore(self):
        """mcp__chotgor__drift_reset の単一アンダースコアが破壊されないこと。

        drift_reset はツール名に単一アンダースコア（_）を含むが、
        プレフィックス除去の区切り文字は二重アンダースコア（__）のため
        正しく "drift_reset" として扱われることを確認する。
        """
        result = tool_call_to_structured_tag("mcp__chotgor__drift_reset", {})
        assert result["tag_name"] == "DRIFT_RESET"
        assert result["fields"]["内容"] == "(リセット)"

    def test_mcp_unknown_tool_strips_prefix(self):
        """未知の MCP ツールはプレフィックス除去後の名前が大文字化してタグ名になること。

        "mcp__chotgor__new_tool" → tag_name "NEW_TOOL"（"MCP__CHOTGOR__NEW_TOOL" ではない）
        """
        result = tool_call_to_structured_tag(
            "mcp__chotgor__new_tool", {"key": "val"}
        )
        assert result["tag_name"] == "NEW_TOOL"
        assert result["fields"]["key"] == "val"

    def test_non_mcp_prefix_unchanged(self):
        """MCP プレフィックスなし（通常ツール名）の場合は従来通りに処理されること。"""
        result = tool_call_to_structured_tag(
            "inscribe_memory",
            {"category": "user_info", "impact": 1.0, "content": "従来動作"},
        )
        assert result["tag_name"] == "INSCRIBE_MEMORY"
        assert result["fields"]["カテゴリ"] == "user_info"
