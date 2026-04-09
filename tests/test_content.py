"""backend.services.chat.content モジュールのユニットテスト。

対象関数:
    apply_context_window() — chronicle済みメッセージ数を制限してコンテキストを圧縮する

テスト方針:
    - ChatMessage の代わりに SimpleNamespace で chronicled_at を持つ軽量オブジェクトを使う
    - DB・LLM へのアクセスは発生しないため外部 mock は不要
    - 境界値（0件・全件chronicle済み・全件未chronicle・混在）を網羅する
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from backend.services.chat.content import apply_context_window


# ─── ヘルパー ──────────────────────────────────────────────────────────────────

def _msg(chronicled: bool, label: str = "") -> SimpleNamespace:
    """テスト用メッセージオブジェクトを生成する。

    Args:
        chronicled: True なら chronicled_at に datetime をセット、False なら None。
        label: 識別用のラベル（content 相当）。

    Returns:
        chronicled_at と content を持つ SimpleNamespace。
    """
    return SimpleNamespace(
        chronicled_at=datetime(2026, 1, 1) if chronicled else None,
        content=label,
    )


# ─── apply_context_window ────────────────────────────────────────────────────


class TestApplyContextWindow:
    """apply_context_window() の動作を検証するテストクラス。"""

    def test_empty_history_returns_empty(self):
        """空リストを渡した場合、空リストを返すこと。"""
        assert apply_context_window([]) == []

    def test_all_unchronicled_returns_all(self):
        """chronicle未実行のメッセージはすべて返されること（圧縮なし）。"""
        msgs = [_msg(False, f"m{i}") for i in range(20)]
        result = apply_context_window(msgs, max_chronicled=10)
        assert result == msgs

    def test_all_chronicled_returns_last_n(self):
        """chronicle済みのみの場合、末尾 max_chronicled 件だけ返されること。"""
        msgs = [_msg(True, f"m{i}") for i in range(15)]
        result = apply_context_window(msgs, max_chronicled=5)
        assert result == msgs[-5:]

    def test_mixed_keeps_all_unchronicled_and_last_n_chronicled(self):
        """chronicle済みと未chronicle混在時、未chronicle全件 + chronicle済み末尾N件を返すこと。

        時系列順も保持されることを確認する。
        """
        # chronicle済み: 10件（インデックス 0〜9）
        # 未chronicle: 5件（インデックス 10〜14）
        chronicled = [_msg(True, f"c{i}") for i in range(10)]
        unchronicled = [_msg(False, f"u{i}") for i in range(5)]
        history = chronicled + unchronicled

        result = apply_context_window(history, max_chronicled=3)

        # chronicle済みの末尾3件 + 未chronicle全5件 = 8件
        assert len(result) == 8
        # 時系列順: chronicle済み末尾3件が先に来る
        assert result[0].content == "c7"
        assert result[1].content == "c8"
        assert result[2].content == "c9"
        assert result[3].content == "u0"
        assert result[-1].content == "u4"

    def test_max_chronicled_zero_removes_all_chronicled(self):
        """max_chronicled=0 の場合、chronicle済みメッセージをすべて除去すること。"""
        msgs = [_msg(True, f"c{i}") for i in range(5)] + [_msg(False, "u0")]
        result = apply_context_window(msgs, max_chronicled=0)
        assert len(result) == 1
        assert result[0].content == "u0"

    def test_max_chronicled_larger_than_total_returns_all(self):
        """max_chronicled が chronicle済み件数を超える場合、chronicle済みを全件保持すること。"""
        msgs = [_msg(True, f"c{i}") for i in range(3)] + [_msg(False, "u0")]
        result = apply_context_window(msgs, max_chronicled=100)
        assert result == msgs

    def test_preserves_original_order(self):
        """chronicle済みと未chronicleが交互に並ぶ場合、元の時系列順が保持されること。

        例: [c0, u1, c2, u3] で max_chronicled=1 の場合
        → c0 は除外、u1 は保持、c2 は保持（末尾1件）、u3 は保持
        → [u1, c2, u3] の順
        """
        c0 = _msg(True, "c0")
        u1 = _msg(False, "u1")
        c2 = _msg(True, "c2")
        u3 = _msg(False, "u3")
        history = [c0, u1, c2, u3]

        result = apply_context_window(history, max_chronicled=1)

        assert len(result) == 3
        assert result[0].content == "u1"
        assert result[1].content == "c2"
        assert result[2].content == "u3"

    def test_no_chronicled_at_attribute_treated_as_unchronicled(self):
        """chronicled_at 属性がないオブジェクトは未chronicle扱い（全件保持）になること。

        getattr(..., None) のフォールバック動作を確認する。
        """
        msgs = [SimpleNamespace(content=f"m{i}") for i in range(5)]
        result = apply_context_window(msgs, max_chronicled=2)
        # chronicled_at がなければ全件保持
        assert result == msgs

    def test_exact_boundary_max_chronicled_equals_count(self):
        """chronicle済みの件数と max_chronicled が等しい場合、全件保持されること。"""
        msgs = [_msg(True, f"c{i}") for i in range(5)]
        result = apply_context_window(msgs, max_chronicled=5)
        assert result == msgs
