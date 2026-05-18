"""記憶・ワーキングメモリの時間減衰と類似度変換の共通ユーティリティ。

長期記憶（InscribedMemory）の4カテゴリ別 decay と、ワーキングメモリスレッドの
type 別 decay は、いずれも「指数減衰」という同一の数式を共有する。
本モジュールはその数式と、ベクトル検索の cosine 距離→類似度変換を
共通部品として切り出したもの（DRY）。
"""

import math
from datetime import datetime
from typing import Optional

# 自然対数 2。半減期から減衰定数 λ = ln(2) / half_life を導くために使う。
_LN2 = 0.69314718


def exp_decay(value: float, elapsed_days: float, half_life_days: float) -> float:
    """指数減衰した値を返す: ``value × e^(-ln2/half_life × elapsed_days)``。

    Args:
        value: 減衰前の値（importance など）。
        elapsed_days: 経過日数。負値は0として扱う。
        half_life_days: 半減期（日）。0以下なら減衰なし（value をそのまま返す）。

    Returns:
        減衰後の値。
    """
    if half_life_days <= 0:
        return value
    if elapsed_days < 0:
        elapsed_days = 0.0
    return value * math.exp(-(_LN2 / half_life_days) * elapsed_days)


def elapsed_days_since(base_time: datetime, now: Optional[datetime] = None) -> float:
    """``base_time`` から ``now`` までの経過日数を返す（負なら0）。"""
    if now is None:
        now = datetime.now()
    days = (now - base_time).total_seconds() / 86400.0
    return max(0.0, days)


def distance_to_similarity(distance: float) -> float:
    """cosine 距離（0=同一, 2=対極）を類似度（0.0-1.0）に変換する。"""
    return max(0.0, 1.0 - (distance / 2.0))
