"""意図のライフサイクル計算 — 意図圧の読み取り時導出と終端遷移の候補挙げ。

意図に圧力カラムはない。意図圧は g(経過日数) の読み取り時計算
（docs/planned/aliveness_plan.md §4.3）。増圧はイベントでも更新でもなく、
タイムラインには遷移だけが載る。

**源圧は掛けない**（2026-08-27 改訂）。v1 は source_kind の現在圧を乗算していたが、
経過日数の項が 1.0 で飽和する以上、意図圧は源圧の項を超えられず、源圧が低い意図は
経過日数をいくら積んでも行動権にも不満化にも到達できなかった。社会圧は「誰でもいいから
話したい」、意図圧は「これを話したい」という別々の駆動であり、独立した発火源として扱う。

機械は候補を挙げ、本人が裁く:
    - active のまま _STALE_AGE_DAYS 日を超えたものを1リストで挙げ、
      手放す(expired) / 満ちた(fulfilled) / 不満(soured) / このまま継続 は本人が選ぶ
"""

from datetime import datetime

# 意図圧の飽和日数（この日数で経過項が 1.0 に達する）
_SATURATION_DAYS = 14

# 終端遷移の候補に挙げる日数（active のままこの日数を超えたもの）。
# 行動権（services/actions/runner.py）は意図圧 0.7 = 8.0 日で拾うため、ここに残るのは
# 「行動権で本人が見送った」「cap で順番が回ってこなかった」意図になる。
_STALE_AGE_DAYS = 14


def intent_pressure(intent, now: datetime | None = None) -> float:
    """意図圧を読み取り時計算する — g(経過日数)。

    g: 経過日数の飽和項のみ。
        意図圧 = 0.3 + 0.7 × min(経過日数 / 14, 1)

    保存しない — 封筒と時刻の純関数。行動権の閾値 0.7 には 8.0 日で到達する
    （旧実装で源圧が最大だったときと同じ発火日数）。

    Args:
        intent: Intent ORM（created_at を持つ）。
        now: 基準時刻。None なら現在時刻。

    Returns:
        意図圧（0.3〜1.0）。
    """
    now = now or datetime.now()
    age_days = max(0.0, (now - intent.created_at).total_seconds() / 86400.0)
    return 0.3 + 0.7 * min(age_days / _SATURATION_DAYS, 1.0)


def _age_days(intent, now: datetime) -> float:
    """意図の経過日数を返す。"""
    return max(0.0, (now - intent.created_at).total_seconds() / 86400.0)


def stale_candidates(intents: list, now: datetime | None = None) -> list:
    """終端遷移の候補 — active のまま _STALE_AGE_DAYS 日を超えた意図を返す。

    候補であって決定ではない。手放す／満ちた／不満／このまま継続のどれになるかは
    本人が裁く（Chronicle 同乗の問い）。機械が種別まで決めないのは、意図圧が
    経過日数のみになって低圧／高圧の区別が成立しなくなったためだが、種別の判定を
    本人へ返すほうが「機械は候補を挙げ、本人が裁く」の原則に対して純度が高い。

    Args:
        intents: active な Intent のリスト。
        now: 基準時刻。

    Returns:
        経過日数の降順に並んだ候補 Intent リスト。
    """
    now = now or datetime.now()
    stale = [i for i in intents if _age_days(i, now) >= _STALE_AGE_DAYS]
    stale.sort(key=lambda i: -_age_days(i, now))
    return stale
