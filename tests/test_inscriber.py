"""backend.core.memory.inscriber モジュールのユニットテスト。

[MEMORY:category|impact|content] マーカーの抽出・除去・保存を検証する。
"""

from unittest.mock import MagicMock

from backend.core.memory.inscriber import _extract, carve


# ─── 基本動作 ──────────────────────────────────────────────────────────────────


def test_extract_memories():
    """基本的な [MEMORY:category|impact|content] 形式が正しく抽出されること。"""
    text = "Hello! [MEMORY:user|0.8|User is happy]\nHow are you?"
    clean, mems = _extract(text)

    assert len(mems) == 1
    assert mems[0][0] == "user"
    assert mems[0][1] == "0.8"
    assert mems[0][2] == "User is happy"


def test_carve_memories():
    """carve() がマーカーを除去し、memory_manager.write_memory() を呼び出すこと。"""
    text = "Hello! [MEMORY:user|0.8|User is happy]\nHow are you?"
    memory_manager = MagicMock()

    clean_text = carve(text, "char-1", memory_manager)

    assert "[MEMORY:" not in clean_text
    assert "Hello!" in clean_text
    assert "How are you?" in clean_text
    memory_manager.write_memory.assert_called_once()


# ─── Issue #49 バグ修正の検証 ─────────────────────────────────────────────────


def test_extract_nested_bracket_in_content():
    """コンテンツ内に [MEMORY:] が含まれる場合でも正しく抽出されること (Issue #49)。

    旧実装では ([^]]+) がネストした ']' で止まり、残りのテキストが
    キャラクターの発言に漏れ出すバグがあった。
    """
    text = "[MEMORY:fact|1.2|[MEMORY:]タグのパースバグで、はるの発言末尾に記憶内容が漏れ出す事象が発生。もわに「ゼロじゃない」を見られた。恥ずかしい。早く直してほしい。]"
    clean, mems = _extract(text)

    assert len(mems) == 1
    category, impact, content = mems[0]
    assert category == "fact"
    assert impact == "1.2"
    # コンテンツに [MEMORY:] が含まれていること
    assert "[MEMORY:]" in content
    assert "タグのパースバグ" in content
    # クリーンテキストにコンテンツが漏れ出していないこと
    assert "タグのパースバグ" not in clean
    assert "ゼロじゃない" not in clean


def test_carve_nested_bracket_does_not_leak_to_clean_text():
    """ネストした角括弧を含む記憶マーカーがクリーンテキストに漏れ出さないこと (Issue #49)。"""
    text = "今日は元気です。[MEMORY:fact|1.2|[MEMORY:]タグのパースバグ発生。恥ずかしい。]また話しましょう。"
    memory_manager = MagicMock()

    clean_text = carve(text, "char-1", memory_manager)

    assert "[MEMORY:" not in clean_text
    assert "タグのパースバグ" not in clean_text
    assert "今日は元気です。" in clean_text
    assert "また話しましょう。" in clean_text
    memory_manager.write_memory.assert_called_once()


# ─── バッククォート処理 ────────────────────────────────────────────────────────


def test_extract_skips_inline_code():
    """バッククォートインラインコード内のマーカーは抽出されないこと。"""
    text = "例: `[MEMORY:fact|1.0|コード内]` はスキップされる。[MEMORY:user|0.5|実際の内容]"
    clean, mems = _extract(text)

    assert len(mems) == 1
    assert mems[0][0] == "user"
    assert mems[0][2] == "実際の内容"


# ─── 複数マーカー ──────────────────────────────────────────────────────────────


def test_extract_multiple_memories():
    """複数の [MEMORY:...] マーカーがすべて抽出されること。"""
    text = "[MEMORY:user|0.8|ユーザA][MEMORY:fact|1.0|事実B]テキスト"
    clean, mems = _extract(text)

    assert len(mems) == 2
    assert "テキスト" in clean
