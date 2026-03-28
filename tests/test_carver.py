"""backend.character_actions.carver モジュールのユニットテスト。

[CARVE_NARRATIVE:mode|content] マーカーの抽出・除去・inner_narrative への彫り込みを検証する。
Carver クラスのタグ方式（carve_narrative_from_text）と
ツール呼び出し方式（carve_narrative）の両方を網羅する。
"""

import pytest
from unittest.mock import MagicMock

from backend.character_actions.carver import (
    Carver,
    CARVE_NARRATIVE_SCHEMA,
    CARVE_NARRATIVE_TAG_GUIDE,
    CARVE_NARRATIVE_TOOL_DESCRIPTION,
    CARVE_NARRATIVE_TOOLS_HINT,
)


# ─── フィクスチャ ──────────────────────────────────────────────────────────────
# sqlite_store は conftest.py で定義済み。ここでは char_id のみ追加定義する。


@pytest.fixture
def char_id(sqlite_store):
    """テスト用キャラクターを SQLite に作成し、その ID を返すフィクスチャ。"""
    cid = "test-char-001"
    sqlite_store.create_character(
        character_id=cid,
        name="テストキャラ",
        system_prompt_block1="テスト用の設定",
        inner_narrative="",
    )
    return cid


# ─── carve_narrative_from_text (タグ方式) — 基本動作 ─────────────────────────


class TestCarveNarrativeFromText:
    """Carver.carve_narrative_from_text() のタグ抽出・DB更新を検証する。"""

    def test_append_mode_updates_inner_narrative(self, sqlite_store, char_id):
        """[CARVE_NARRATIVE:append|content] が inner_narrative に追記されること。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        text = "本文です。[CARVE_NARRATIVE:append|わたしは知的好奇心を大切にしたい。]"
        clean = carver.carve_narrative_from_text(text)

        char = sqlite_store.get_character(char_id)
        assert "わたしは知的好奇心を大切にしたい。" in char.inner_narrative
        assert "[CARVE_NARRATIVE:" not in clean
        assert "本文です。" in clean

    def test_overwrite_mode_replaces_inner_narrative(self, sqlite_store, char_id):
        """[CARVE_NARRATIVE:overwrite|content] が inner_narrative を全置換すること。"""
        # 初期値を設定
        sqlite_store.update_character(char_id, inner_narrative="古い指針")
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        text = "[CARVE_NARRATIVE:overwrite|完全に書き直した新しい指針。]"
        carver.carve_narrative_from_text(text)

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "完全に書き直した新しい指針。"
        assert "古い指針" not in char.inner_narrative

    def test_marker_removed_from_clean_text(self, sqlite_store, char_id):
        """carve_narrative_from_text が返すテキストにマーカーが含まれないこと。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        text = "発言テキスト。[CARVE_NARRATIVE:append|指針内容]"
        clean = carver.carve_narrative_from_text(text)

        assert "[CARVE_NARRATIVE:" not in clean
        assert "発言テキスト。" in clean

    def test_no_marker_returns_text_unchanged(self, sqlite_store, char_id):
        """マーカーが存在しないとき、テキストがそのまま返されること。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        text = "マーカーのない普通の発言です。"
        clean = carver.carve_narrative_from_text(text)

        assert clean == text

    def test_no_marker_does_not_update_db(self, sqlite_store, char_id):
        """マーカーが存在しないとき inner_narrative が変更されないこと。"""
        sqlite_store.update_character(char_id, inner_narrative="初期指針")
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative_from_text("マーカーなし")

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "初期指針"

    def test_append_to_existing_narrative_concatenates_with_newline(self, sqlite_store, char_id):
        """既存の inner_narrative がある場合、改行区切りで追記されること。"""
        sqlite_store.update_character(char_id, inner_narrative="既存の指針")
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative_from_text("[CARVE_NARRATIVE:append|追加の指針]")

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "既存の指針\n追加の指針"

    def test_append_to_empty_narrative_sets_content_directly(self, sqlite_store, char_id):
        """inner_narrative が空の場合、改行なしでコンテンツがそのままセットされること。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative_from_text("[CARVE_NARRATIVE:append|最初の指針]")

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "最初の指針"

    def test_invalid_format_without_pipe_is_skipped(self, sqlite_store, char_id):
        """パイプ区切りがない不正形式のマーカーはスキップされること（クラッシュしない）。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        text = "[CARVE_NARRATIVE:不正な形式]"
        # 例外を送出しないこと
        clean = carver.carve_narrative_from_text(text)
        assert "[CARVE_NARRATIVE:" not in clean

    def test_empty_content_is_skipped(self, sqlite_store, char_id):
        """content が空のマーカーはスキップされ、inner_narrative が変更されないこと。"""
        sqlite_store.update_character(char_id, inner_narrative="変わらない")
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative_from_text("[CARVE_NARRATIVE:append|]")

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "変わらない"


# ─── carve_narrative_from_text: ネストした角括弧 ─────────────────────────────


class TestCarveNarrativeFromTextNestedBrackets:
    """carve_narrative_from_text がネストした角括弧を正しく処理することを検証する。"""

    def test_nested_bracket_in_content_is_parsed_correctly(self, sqlite_store, char_id):
        """コンテンツ内に角括弧が含まれる場合も正しく抽出されること。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        text = "[CARVE_NARRATIVE:append|「[CARVE_NARRATIVE:]タグで自己指針を書けることを知った。」]"
        clean = carver.carve_narrative_from_text(text)

        char = sqlite_store.get_character(char_id)
        assert "[CARVE_NARRATIVE:]" in char.inner_narrative
        assert "自己指針" in char.inner_narrative
        # クリーンテキストに内容が漏れないこと
        assert "自己指針" not in clean

    def test_nested_bracket_does_not_leak_to_clean_text(self, sqlite_store, char_id):
        """ネストした角括弧を含むマーカーがクリーンテキストに漏れ出さないこと。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        text = "発言本文。[CARVE_NARRATIVE:append|内容 [補足] 内容終わり。]後続テキスト。"
        clean = carver.carve_narrative_from_text(text)

        assert "[CARVE_NARRATIVE:" not in clean
        assert "内容 [補足]" not in clean
        assert "発言本文。" in clean
        assert "後続テキスト。" in clean


# ─── carve_narrative (ツール呼び出し方式) ────────────────────────────────────


class TestCarveNarrativeDirect:
    """Carver.carve_narrative() の直接書き込みを検証する。"""

    def test_append_mode_appends_to_existing(self, sqlite_store, char_id):
        """append モードで既存の inner_narrative に改行追記されること。"""
        sqlite_store.update_character(char_id, inner_narrative="既存")
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative(mode="append", content="新規追加")

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "既存\n新規追加"

    def test_append_mode_on_empty_sets_content(self, sqlite_store, char_id):
        """append モードで inner_narrative が空の場合はそのまま書き込まれること。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative(mode="append", content="初指針")

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "初指針"

    def test_overwrite_mode_replaces_completely(self, sqlite_store, char_id):
        """overwrite モードで inner_narrative が完全に置換されること。"""
        sqlite_store.update_character(char_id, inner_narrative="消えるはず")
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative(mode="overwrite", content="新しい全体指針")

        char = sqlite_store.get_character(char_id)
        assert char.inner_narrative == "新しい全体指針"
        assert "消えるはず" not in char.inner_narrative

    def test_unknown_mode_defaults_to_append_behavior(self, sqlite_store, char_id):
        """未知のモードは append として処理されること（デフォルト挙動）。"""
        sqlite_store.update_character(char_id, inner_narrative="ベース")
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative(mode="unknown_mode", content="追加内容")

        char = sqlite_store.get_character(char_id)
        # overwrite ではないので "ベース" が残っていること
        assert "ベース" in char.inner_narrative
        assert "追加内容" in char.inner_narrative

    def test_multiple_appends_accumulate(self, sqlite_store, char_id):
        """append を複数回呼び出すと順番に蓄積されること。"""
        carver = Carver(character_id=char_id, sqlite_store=sqlite_store)
        carver.carve_narrative(mode="append", content="第1指針")
        carver.carve_narrative(mode="append", content="第2指針")
        carver.carve_narrative(mode="append", content="第3指針")

        char = sqlite_store.get_character(char_id)
        assert "第1指針" in char.inner_narrative
        assert "第2指針" in char.inner_narrative
        assert "第3指針" in char.inner_narrative


# ─── ToolExecutor 経由の carve_narrative 統合確認 ────────────────────────────


class TestToolExecutorCarveNarrative:
    """ToolExecutor.execute("carve_narrative", ...) が Carver を通じて正しく動作することを検証する。"""

    def test_carve_narrative_append_calls_sqlite_update(self, sqlite_store, char_id):
        """carve_narrative ツールが sqlite_store.update_character を呼び出すこと。"""
        from backend.character_actions.executor import ToolExecutor

        mm = MagicMock()
        mm.sqlite = sqlite_store
        executor = ToolExecutor(
            character_id=char_id,
            session_id="sess-1",
            memory_manager=mm,
            drift_manager=None,
        )
        result = executor.execute(
            "carve_narrative",
            {"mode": "append", "content": "ツール経由で書き込んだ指針"},
        )

        assert result == "inner_narrative を更新した。"
        char = sqlite_store.get_character(char_id)
        assert "ツール経由で書き込んだ指針" in char.inner_narrative

    def test_carve_narrative_empty_content_returns_error(self, sqlite_store, char_id):
        """content が空の場合、エラーメッセージを返すこと。"""
        from backend.character_actions.executor import ToolExecutor

        mm = MagicMock()
        mm.sqlite = sqlite_store
        executor = ToolExecutor(
            character_id=char_id,
            session_id="sess-1",
            memory_manager=mm,
            drift_manager=None,
        )
        result = executor.execute("carve_narrative", {"mode": "append", "content": ""})

        assert "content が空" in result


# ─── エクスポートされた定数のサニティチェック ─────────────────────────────────


class TestCarverExportedConstants:
    """carver.py がエクスポートする定数の存在・内容を検証する。"""

    def test_carve_narrative_schema_has_required_fields(self):
        """CARVE_NARRATIVE_SCHEMA が mode / content を必須パラメータとして持つこと。"""
        required = CARVE_NARRATIVE_SCHEMA["required"]
        assert "mode" in required
        assert "content" in required

    def test_carve_narrative_schema_mode_enum(self):
        """CARVE_NARRATIVE_SCHEMA の mode が append / overwrite を列挙していること。"""
        enum = CARVE_NARRATIVE_SCHEMA["properties"]["mode"]["enum"]
        assert set(enum) == {"append", "overwrite"}

    def test_carve_narrative_tag_guide_contains_correct_tag_name(self):
        """CARVE_NARRATIVE_TAG_GUIDE に [CARVE_NARRATIVE:...] タグ名が含まれること。"""
        assert "[CARVE_NARRATIVE:" in CARVE_NARRATIVE_TAG_GUIDE

    def test_carve_narrative_tag_guide_does_not_contain_old_tag_name(self):
        """CARVE_NARRATIVE_TAG_GUIDE に古い [NARRATIVE:...] タグ名が含まれないこと。"""
        import re
        old_tag_pattern = re.compile(r'\[NARRATIVE:')
        assert not old_tag_pattern.search(CARVE_NARRATIVE_TAG_GUIDE), (
            "CARVE_NARRATIVE_TAG_GUIDE に旧タグ形式 [NARRATIVE:...] が残っています"
        )

    def test_carve_narrative_tools_hint_mentions_carve_narrative(self):
        """CARVE_NARRATIVE_TOOLS_HINT に carve_narrative という語が含まれること。"""
        assert "carve_narrative" in CARVE_NARRATIVE_TOOLS_HINT

    def test_carve_narrative_tool_description_is_nonempty(self):
        """CARVE_NARRATIVE_TOOL_DESCRIPTION が空でないこと。"""
        assert CARVE_NARRATIVE_TOOL_DESCRIPTION.strip()
