"""意図のライフサイクル計算 — 意図圧の読み取り時導出と失効/不満化の候補挙げ。

意図に圧力カラムはない。意図圧は g(経過日数, source_kind の現在圧) の
読み取り時計算（docs/aliveness_plan.md §4.3）。増圧はイベントでも更新でもなく、
タイムラインには遷移だけが載る。

機械は候補を挙げ、本人が裁く:
    - 失効候補（expired）: 低圧のまま14日 → Chronicle 同乗で「これ、まだ心にある？」
    - 不満化候補（soured）: 高圧なのに7日遷移できない → 本人が不満を言語化したら
      soured ＋ その言葉を記憶へ刻む（不満化＝利害と合流）
"""

from datetime import datetime

# 失効候補: この日数以上、低圧のまま。
# 閾値 0.45 の根拠: 意図圧の式は源圧ゼロ・経過飽和でちょうど 0.4 に張り付くため、
# 「源の圧がほぼ死んでいる意図」を候補に含めるには 0.4 より僅かに上に置く必要がある。
_EXPIRE_AGE_DAYS = 14
_EXPIRE_PRESSURE_BELOW = 0.45
# 不満化候補: この日数以上、高圧なのに遷移できない
_SOUR_AGE_DAYS = 7
_SOUR_PRESSURE_ABOVE = 0.7

# source_kind="none" の意図に使う基準圧（圧に紐づかない意図もゆっくり増圧する）
_NONE_SOURCE_BASE = 0.5


def intent_pressure(intent, pressures: dict, now: datetime | None = None) -> float:
    """意図圧を読み取り時計算する — g(経過日数, source_kind の現在圧)。

    v1 の g: 経過日数の飽和項 × 源圧の混合。
        age_term = 0.3 + 0.7 × min(経過日数 / 14, 1)   … 時間とともに飽和増加
        src_term = 0.4 + 0.6 × 源圧                     … 源の圧が高いほど押される
        意図圧 = clamp(age_term × src_term, 0, 1)

    源圧は compute_pressures の現在値（source_kind="none" は固定基準）。
    保存しない — 封筒と圧力の純関数の合成なので、これもまた純関数。

    Args:
        intent: Intent ORM（created_at / source_kind を持つ）。
        pressures: compute_pressures の戻り値。
        now: 基準時刻。None なら現在時刻。

    Returns:
        意図圧（0.0〜1.0）。
    """
    now = now or datetime.now()
    age_days = max(0.0, (now - intent.created_at).total_seconds() / 86400.0)
    age_term = 0.3 + 0.7 * min(age_days / 14.0, 1.0)
    src = pressures.get(intent.source_kind, _NONE_SOURCE_BASE)
    if intent.source_kind == "none":
        src = _NONE_SOURCE_BASE
    src_term = 0.4 + 0.6 * float(src)
    return max(0.0, min(1.0, age_term * src_term))


def _age_days(intent, now: datetime) -> float:
    """意図の経過日数を返す。"""
    return max(0.0, (now - intent.created_at).total_seconds() / 86400.0)


def expired_candidates(
    intents: list, pressures: dict, now: datetime | None = None
) -> list:
    """失効候補 — 低圧のまま _EXPIRE_AGE_DAYS 日を超えた active 意図を返す。

    候補であって決定ではない。本人が「まだ心にある」と言えば active のまま。

    Args:
        intents: active な Intent のリスト。
        pressures: compute_pressures の戻り値。
        now: 基準時刻。

    Returns:
        失効候補の Intent リスト。
    """
    now = now or datetime.now()
    return [
        i for i in intents
        if _age_days(i, now) >= _EXPIRE_AGE_DAYS
        and intent_pressure(i, pressures, now) < _EXPIRE_PRESSURE_BELOW
    ]


def soured_candidates(
    intents: list, pressures: dict, now: datetime | None = None
) -> list:
    """不満化候補 — 高圧なのに _SOUR_AGE_DAYS 日遷移できない active 意図を返す。

    高圧の意図は本来その前に行動権（Phase 6）が拾う。不満化は「行動しても
    叶わなかった／出口がなかった」場合の受け皿であり、この受け皿の詰まりは
    将来枠の intent_no_exit 計器が見張る。

    Args:
        intents: active な Intent のリスト。
        pressures: compute_pressures の戻り値。
        now: 基準時刻。

    Returns:
        不満化候補の Intent リスト。
    """
    now = now or datetime.now()
    return [
        i for i in intents
        if _age_days(i, now) >= _SOUR_AGE_DAYS
        and intent_pressure(i, pressures, now) >= _SOUR_PRESSURE_ABOVE
    ]
