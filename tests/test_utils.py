"""
core.utils モジュールのユニットテスト。

format_time_delta の全境界値・代表値を網羅する。
"""

from datetime import timedelta

import pytest

from backend.core.utils import format_time_delta


# --- 数分以内（60秒未満かつm=0になるケース） ---

def test_format_time_delta_zero_returns_suufu():
    """0秒は「数分以内」を返すこと。"""
    assert format_time_delta(timedelta(seconds=0)) == "数分以内"


def test_format_time_delta_30sec_returns_suufu():
    """30秒は「数分以内」を返すこと（m=0になるため）。"""
    assert format_time_delta(timedelta(seconds=30)) == "数分以内"


# --- 分表示（1分以上1時間未満） ---

def test_format_time_delta_1min():
    """1分は「約 1 分」を返すこと。"""
    result = format_time_delta(timedelta(minutes=1))
    assert result == "約 1 分"


def test_format_time_delta_45min():
    """45分は「約 45 分」を返すこと。"""
    result = format_time_delta(timedelta(minutes=45))
    assert result == "約 45 分"


def test_format_time_delta_59min():
    """59分は分表示になること（1時間未満）。"""
    result = format_time_delta(timedelta(minutes=59))
    assert "分" in result
    assert "時間" not in result


# --- 時間表示（1時間以上24時間未満） ---

def test_format_time_delta_1hour_boundary():
    """ちょうど1時間は時間表示になること。"""
    result = format_time_delta(timedelta(hours=1))
    assert "時間" in result


def test_format_time_delta_3hours():
    """3時間は「約 3.0 時間」を返すこと。"""
    result = format_time_delta(timedelta(hours=3))
    assert result == "約 3.0 時間"


def test_format_time_delta_23hours():
    """23時間は時間表示になること（24時間未満）。"""
    result = format_time_delta(timedelta(hours=23))
    assert "時間" in result
    assert "日" not in result


# --- 日表示（24時間以上） ---

def test_format_time_delta_24hours_boundary():
    """ちょうど24時間は日表示になること。"""
    result = format_time_delta(timedelta(hours=24))
    assert "日" in result


def test_format_time_delta_2days_3hours():
    """2日3時間は「約 2 日」を返すこと（端数切り捨て）。"""
    result = format_time_delta(timedelta(days=2, hours=3))
    assert result == "約 2 日"


def test_format_time_delta_7days():
    """7日は「約 7 日」を返すこと。"""
    result = format_time_delta(timedelta(days=7))
    assert result == "約 7 日"
