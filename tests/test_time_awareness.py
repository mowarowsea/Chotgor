"""compute_time_awareness の有効/無効判定のテスト。"""
from datetime import datetime

import pytest

from backend.lib.time_awareness import compute_time_awareness


class TestEnableTimeAwareness:
    """enable_time_awareness 設定の型ゆれを吸収できること。

    設定ページは bool を保存し、get_all_settings() は JSON 復元後の bool を返す。
    以前は文字列 "true" との比較だったため、bool の True が「無効」と判定される
    食い違いがあった。旧データ・テストの文字列表現も引き続き受ける。
    """

    @pytest.mark.parametrize("value", [True, "true", "True", None])
    def test_有効になる値(self, value):
        settings = {} if value is None else {"enable_time_awareness": value}
        ta = compute_time_awareness(settings, "c1", None, datetime(2026, 1, 1, 12, 0, 0))
        assert ta.enabled is True
        assert ta.current_time_str != ""

    @pytest.mark.parametrize("value", [False, "false", "False"])
    def test_無効になる値(self, value):
        ta = compute_time_awareness(
            {"enable_time_awareness": value}, "c1", None, datetime(2026, 1, 1, 12, 0, 0)
        )
        assert ta.enabled is False
        assert ta.current_time_str == ""
        assert ta.time_since_last_interaction == ""
