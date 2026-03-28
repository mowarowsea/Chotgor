"""backend.character_actions.inscriber モジュールのユニットテスト。

[INSCRIBE_MEMORY:category|impact|content] マーカーの抽出・除去・書き込みを検証する。
Inscriber クラスのタグ方式（inscribe_memory_from_text）と
ツール呼び出し方式（inscribe_memory）の両方を網羅する。
"""

from unittest.mock import MagicMock, call

import pytest

from backend.character_actions.inscriber import (
    _extract,
    Inscriber,
    INSCRIBE_MEMORY_SCHEMA,
    INSCRIBE_MEMORY_TAG_GUIDE,
    INSCRIBE_MEMORY_TOOL_DESCRIPTION,
)


# ─── _extract: 基本動作 ────────────────────────────────────────────────────────


def test_extract_basic_three_part_format():
    """[INSCRIBE_MEMORY:category|impact|content] の3要素形式が正しく抽出されること。"""
    text = "こんにちは！ [INSCRIBE_MEMORY:user|0.8|ユーザは猫が好き]\nまた話しましょう。"
    clean, mems = _extract(text)

    assert len(mems) == 1
    category, impact_str, content = mems[0]
    assert category == "user"
    assert impact_str == "0.8"
    assert content == "ユーザは猫が好き"


def test_extract_removes_marker_from_clean_text():
    """抽出後のクリーンテキストにマーカーが残らないこと。"""
    text = "本文です。[INSCRIBE_MEMORY:contextual|1.0|覚えたこと]"
    clean, _ = _extract(text)

    assert "[INSCRIBE_MEMORY:" not in clean
    assert "本文です。" in clean


def test_extract_two_part_format_defaults_impact_to_1_0():
    """[INSCRIBE_MEMORY:category|content] の2要素形式は impact を 1.0 とみなすこと。"""
    text = "[INSCRIBE_MEMORY:semantic|Chotgorの設計を理解した]"
    clean, mems = _extract(text)

    assert len(mems) == 1
    category, impact_str, content = mems[0]
    assert category == "semantic"
    assert impact_str == "1.0"
    assert content == "Chotgorの設計を理解した"


def test_extract_multiple_markers_all_extracted():
    """複数の [INSCRIBE_MEMORY:...] マーカーがすべて抽出されること。"""
    text = "[INSCRIBE_MEMORY:user|0.8|ユーザA][INSCRIBE_MEMORY:contextual|1.0|出来事B]テキスト"
    clean, mems = _extract(text)

    assert len(mems) == 2
    assert mems[0][0] == "user"
    assert mems[1][0] == "contextual"
    assert "テキスト" in clean


# ─── _extract: Issue #49 由来のネストした角括弧処理 ─────────────────────────────


def test_extract_nested_bracket_in_content_is_parsed_correctly():
    """コンテンツ内に [INSCRIBE_MEMORY:] が含まれる場合も正しく抽出されること (Issue #49 相当)。

    旧実装では ([^]]+) がネストした ']' で止まり、残りのテキストが
    キャラクターの発言に漏れ出すバグがあった。
    """
    text = "[INSCRIBE_MEMORY:contextual|1.2|[INSCRIBE_MEMORY:]タグのパースバグで発言末尾に内容が漏れる事象。恥ずかしい。]"
    clean, mems = _extract(text)

    assert len(mems) == 1
    category, impact, content = mems[0]
    assert category == "contextual"
    assert impact == "1.2"
    assert "[INSCRIBE_MEMORY:]" in content
    assert "タグのパースバグ" in content
    # クリーンテキストに内容が漏れ出していないこと
    assert "タグのパースバグ" not in clean


def test_extract_nested_bracket_does_not_leak_to_clean_text():
    """ネストした角括弧を含むマーカーがクリーンテキストに漏れ出さないこと (Issue #49 相当)。"""
    text = "今日は元気です。[INSCRIBE_MEMORY:contextual|1.2|[INSCRIBE_MEMORY:]タグのバグ発生。恥ずかしい。]また話しましょう。"
    clean, _ = _extract(text)

    assert "[INSCRIBE_MEMORY:" not in clean
    assert "タグのバグ" not in clean
    assert "今日は元気です。" in clean
    assert "また話しましょう。" in clean


# ─── _extract: バッククォート内は無視 ──────────────────────────────────────────


def test_extract_skips_marker_inside_inline_code():
    """バッククォートで囲まれたインラインコード内のマーカーは抽出されないこと。"""
    text = "例: `[INSCRIBE_MEMORY:contextual|1.0|コード内]` はスキップされる。[INSCRIBE_MEMORY:user|0.5|実際の内容]"
    clean, mems = _extract(text)

    assert len(mems) == 1
    assert mems[0][0] == "user"
    assert mems[0][2] == "実際の内容"


# ─── Inscriber.inscribe_memory_from_text (タグ方式) ───────────────────────────


class TestInscriberInscribeMemoryFromText:
    """Inscriber.inscribe_memory_from_text() のタグ抽出・write_memory 呼び出しを検証する。"""

    def _make_inscriber(self, memory_manager=None):
        """テスト用の Inscriber インスタンスを生成するヘルパー。"""
        mm = memory_manager or MagicMock()
        return Inscriber(character_id="char-1", memory_manager=mm), mm

    def test_inscribe_memory_from_text_returns_clean_text(self):
        """マーカーを除去したクリーンテキストを返すこと。"""
        inscriber, _ = self._make_inscriber()
        text = "こんにちは！ [INSCRIBE_MEMORY:user|0.8|ユーザは猫が好き]\nまた話しましょう。"
        clean = inscriber.inscribe_memory_from_text(text)

        assert "[INSCRIBE_MEMORY:" not in clean
        assert "こんにちは！" in clean
        assert "また話しましょう。" in clean

    def test_inscribe_memory_from_text_calls_write_memory(self):
        """マーカーが存在するとき write_memory が呼ばれること。"""
        inscriber, mm = self._make_inscriber()
        text = "[INSCRIBE_MEMORY:user|0.8|ユーザは猫が好き]"
        inscriber.inscribe_memory_from_text(text)

        mm.write_memory.assert_called_once()
        kwargs = mm.write_memory.call_args.kwargs
        assert kwargs["character_id"] == "char-1"
        assert kwargs["content"] == "ユーザは猫が好き"
        assert kwargs["category"] == "user"

    def test_inscribe_memory_from_text_no_marker_does_not_call_write_memory(self):
        """マーカーが存在しないとき write_memory は呼ばれないこと。"""
        inscriber, mm = self._make_inscriber()
        text = "マーカーのない普通の発言です。"
        clean = inscriber.inscribe_memory_from_text(text)

        assert clean == text
        mm.write_memory.assert_not_called()

    def test_inscribe_memory_from_text_multiple_markers_calls_write_memory_multiple_times(self):
        """複数マーカーがあるとき write_memory が複数回呼ばれること。"""
        inscriber, mm = self._make_inscriber()
        text = "[INSCRIBE_MEMORY:user|1.0|ユーザA][INSCRIBE_MEMORY:contextual|0.5|出来事B]"
        inscriber.inscribe_memory_from_text(text)

        assert mm.write_memory.call_count == 2

    def test_inscribe_memory_from_text_with_source_preset_id(self):
        """source_preset_id が指定されたとき write_memory に渡されること。"""
        inscriber, mm = self._make_inscriber()
        text = "[INSCRIBE_MEMORY:semantic|1.0|Chotgorの設計]"
        inscriber.inscribe_memory_from_text(text, source_preset_id="preset-abc")

        kwargs = mm.write_memory.call_args.kwargs
        assert kwargs.get("source_preset_id") == "preset-abc"

    def test_inscribe_memory_from_text_empty_preset_id_passes_none(self):
        """source_preset_id が空文字列のとき None が渡されること。"""
        inscriber, mm = self._make_inscriber()
        text = "[INSCRIBE_MEMORY:contextual|1.0|内容]"
        inscriber.inscribe_memory_from_text(text, source_preset_id="")

        kwargs = mm.write_memory.call_args.kwargs
        assert kwargs.get("source_preset_id") is None

    def test_inscribe_memory_from_text_write_memory_exception_does_not_raise(self):
        """write_memory が例外を投げても inscribe_memory_from_text はクラッシュしないこと。"""
        mm = MagicMock()
        mm.write_memory.side_effect = RuntimeError("DB接続失敗")
        inscriber = Inscriber(character_id="char-1", memory_manager=mm)
        text = "[INSCRIBE_MEMORY:user|1.0|内容]残りのテキスト"
        clean = inscriber.inscribe_memory_from_text(text)

        # クラッシュせず、クリーンテキストを返すこと
        assert "残りのテキスト" in clean


# ─── Inscriber.inscribe_memory (ツール呼び出し方式) ───────────────────────────


class TestInscriberInscribeMemory:
    """Inscriber.inscribe_memory() の直接書き込みとインポータンス計算を検証する。"""

    def _run(self, category: str, impact: float) -> dict:
        """inscribe_memory を実行して write_memory に渡された kwargs を返すヘルパー。"""
        mm = MagicMock()
        inscriber = Inscriber(character_id="char-x", memory_manager=mm)
        inscriber.inscribe_memory(content="テスト内容", category=category, impact=impact)
        return mm.write_memory.call_args.kwargs

    def test_inscribe_memory_calls_write_memory_with_content(self):
        """content が write_memory に正しく渡されること。"""
        mm = MagicMock()
        inscriber = Inscriber(character_id="char-1", memory_manager=mm)
        inscriber.inscribe_memory(content="ユーザは音楽が好き", category="user", impact=1.2)

        kwargs = mm.write_memory.call_args.kwargs
        assert kwargs["character_id"] == "char-1"
        assert kwargs["content"] == "ユーザは音楽が好き"
        assert kwargs["category"] == "user"

    def test_identity_category_has_highest_identity_importance(self):
        """identity カテゴリでは identity_importance が最大になること。"""
        kwargs = self._run("identity", 1.0)
        assert kwargs["identity_importance"] > kwargs["contextual_importance"]
        assert kwargs["identity_importance"] > kwargs["user_importance"]

    def test_user_category_has_highest_user_importance(self):
        """user カテゴリでは user_importance が最大になること。"""
        kwargs = self._run("user", 1.0)
        assert kwargs["user_importance"] > kwargs["semantic_importance"]
        assert kwargs["user_importance"] > kwargs["contextual_importance"]

    def test_semantic_category_has_highest_semantic_importance(self):
        """semantic カテゴリでは semantic_importance が最大になること。"""
        kwargs = self._run("semantic", 1.0)
        assert kwargs["semantic_importance"] > kwargs["contextual_importance"]

    def test_contextual_category_has_highest_contextual_importance(self):
        """contextual カテゴリでは contextual_importance が最大になること。"""
        kwargs = self._run("contextual", 1.0)
        assert kwargs["contextual_importance"] > kwargs["semantic_importance"]
        assert kwargs["contextual_importance"] > kwargs["identity_importance"]

    def test_impact_multiplier_scales_all_importances(self):
        """impact 係数 2.0 は 1.0 の2倍の重要度になること。"""
        kwargs_1x = self._run("contextual", 1.0)
        kwargs_2x = self._run("contextual", 2.0)
        assert abs(kwargs_2x["contextual_importance"] - kwargs_1x["contextual_importance"] * 2) < 1e-9

    def test_unknown_category_uses_default_base_0_5(self):
        """未知のカテゴリでもエラーにならず、デフォルト基準値 0.5 を使うこと。"""
        kwargs = self._run("未知カテゴリ", 1.0)
        assert kwargs["contextual_importance"] == pytest.approx(0.5)
        assert kwargs["semantic_importance"] == pytest.approx(0.5)
        assert kwargs["identity_importance"] == pytest.approx(0.5)
        assert kwargs["user_importance"] == pytest.approx(0.5)

    def test_impact_default_is_1_0(self):
        """impact のデフォルト値は 1.0 であること。"""
        mm = MagicMock()
        inscriber = Inscriber(character_id="char-1", memory_manager=mm)
        inscriber.inscribe_memory(content="内容", category="user")

        kwargs = mm.write_memory.call_args.kwargs
        # impact=1.0 と impact=1.5 で呼んだ場合で値が変わることを確認（デフォルト1.0の検証）
        assert kwargs["user_importance"] == pytest.approx(0.9)  # user カテゴリの user 基準値 0.9 × 1.0


# ─── エクスポートされた定数のサニティチェック ─────────────────────────────────


class TestInscriberExportedConstants:
    """inscriber.py がエクスポートする定数の存在・内容を検証する。"""

    def test_inscribe_memory_schema_has_required_fields(self):
        """INSCRIBE_MEMORY_SCHEMA が content / category / impact を必須パラメータとして持つこと。"""
        required = INSCRIBE_MEMORY_SCHEMA["required"]
        assert "content" in required
        assert "category" in required
        assert "impact" in required

    def test_inscribe_memory_schema_category_enum(self):
        """INSCRIBE_MEMORY_SCHEMA の category が4カテゴリを列挙していること。"""
        enum = INSCRIBE_MEMORY_SCHEMA["properties"]["category"]["enum"]
        assert set(enum) == {"identity", "user", "semantic", "contextual"}

    def test_inscribe_memory_tag_guide_contains_correct_tag_name(self):
        """INSCRIBE_MEMORY_TAG_GUIDE に [INSCRIBE_MEMORY:...] タグ名が含まれること。"""
        assert "[INSCRIBE_MEMORY:" in INSCRIBE_MEMORY_TAG_GUIDE

    def test_inscribe_memory_tag_guide_does_not_contain_old_tag_name(self):
        """INSCRIBE_MEMORY_TAG_GUIDE に古い [MEMORY:...] タグ名が含まれないこと。"""
        # 使用例テキスト内ではなく実際のタグとして [MEMORY:] が残っていないこと
        # （説明文として "MEMORY" という単語が含まれることはあるが、タグ形式では不可）
        import re
        old_tag_pattern = re.compile(r'\[MEMORY:')
        assert not old_tag_pattern.search(INSCRIBE_MEMORY_TAG_GUIDE), (
            "INSCRIBE_MEMORY_TAG_GUIDE に旧タグ形式 [MEMORY:...] が残っています"
        )

    def test_inscribe_memory_tool_description_is_nonempty(self):
        """INSCRIBE_MEMORY_TOOL_DESCRIPTION が空でないこと。"""
        assert INSCRIBE_MEMORY_TOOL_DESCRIPTION.strip()
