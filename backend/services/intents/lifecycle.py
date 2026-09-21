"""意図のライフサイクル計算 — 意図圧の読み取り時導出と終端遷移の候補挙げ。

意図に圧力カラムはない。意図圧は g(経過日数) の読み取り時計算
（docs/planned/aliveness_plan.md §4.3）。増圧はイベントでも更新でもなく、
タイムラインには遷移だけが載る。

**源圧は掛けない**（2026-08-27 改訂）。v1 は source_kind の現在圧を乗算していたが、
経過日数の項が 1.0 で飽和する以上、意図圧は源圧の項を超えられず、源圧が低い意図は
経過日数をいくら積んでも行動権にも不満化にも到達できなかった。社会圧は「誰でもいいから
話したい」、意図圧は「これを話したい」という別々の駆動であり、独立した発火源として扱う。

**一区切り（settled）が唯一の減衰源**（2026-09-21 追加）。意図圧の起点は created_at ではなく
「created_at と最終 intent.settled の遅いほう」で、settled が打たれると圧は 0.3 へ戻り、
また日数ぶんだけ積み上がる。保存カラムは持たず封筒から導出する（「圧力は保存しない」）。

機械は候補を挙げ、本人が裁く:
    - active のまま _STALE_AGE_DAYS 日を超えたものを1リストで挙げ、
      手放す(expired) / 満ちた(fulfilled) / 不満(soured) / このまま継続 は本人が選ぶ
    - 経過日数は settled 起点で測る（一区切りついた意図が裁定候補に出ないように）
"""

from datetime import datetime, timedelta

# 意図圧の飽和日数（この日数で経過項が 1.0 に達する）
_SATURATION_DAYS = 14

# 終端遷移の候補に挙げる日数（active のままこの日数を超えたもの）。
# 行動権（services/actions/runner.py）は意図圧 0.7 = 8.0 日で拾うため、ここに残るのは
# 「行動権で本人が見送った」「cap で順番が回ってこなかった」意図になる。
_STALE_AGE_DAYS = 14


def pressure_origin(intent, settled_at: datetime | None = None) -> datetime:
    """意図圧・経過日数の起点を返す — created_at と最終 settled の遅いほう。

    Args:
        intent: Intent ORM。
        settled_at: この意図の最終 intent.settled 時刻（無ければ None）。

    Returns:
        起点時刻。
    """
    if settled_at is not None and settled_at > intent.created_at:
        return settled_at
    return intent.created_at


def intent_pressure(
    intent, now: datetime | None = None, settled_at: datetime | None = None
) -> float:
    """意図圧を読み取り時計算する — g(起点からの経過日数)。

    g: 経過日数の飽和項のみ。
        意図圧 = 0.3 + 0.7 × min(経過日数 / 14, 1)

    保存しない — 封筒と時刻の純関数。行動権の閾値 0.7 には 8.0 日で到達する
    （旧実装で源圧が最大だったときと同じ発火日数）。一区切り（settled）が打たれると
    起点がそこへ移るため、圧は 0.3 へ戻って 8 日かけて再燃する。

    Args:
        intent: Intent ORM（created_at を持つ）。
        now: 基準時刻。None なら現在時刻。
        settled_at: この意図の最終 intent.settled 時刻。None なら created_at 起点。

    Returns:
        意図圧（0.3〜1.0）。
    """
    now = now or datetime.now()
    age_days = _age_days(intent, now, settled_at)
    return 0.3 + 0.7 * min(age_days / _SATURATION_DAYS, 1.0)


def _age_days(intent, now: datetime, settled_at: datetime | None = None) -> float:
    """意図の経過日数を返す（settled 起点）。"""
    return max(0.0, (now - pressure_origin(intent, settled_at)).total_seconds() / 86400.0)


def stale_candidates(
    intents: list, now: datetime | None = None, settled_map: dict | None = None
) -> list:
    """終端遷移の候補 — active のまま _STALE_AGE_DAYS 日を超えた意図を返す。

    候補であって決定ではない。手放す／満ちた／不満／このまま継続のどれになるかは
    本人が裁く（Chronicle 同乗の問い）。機械が種別まで決めないのは、意図圧が
    経過日数のみになって低圧／高圧の区別が成立しなくなったためだが、種別の判定を
    本人へ返すほうが「機械は候補を挙げ、本人が裁く」の原則に対して純度が高い。

    Args:
        intents: active な Intent のリスト。
        now: 基準時刻。
        settled_map: {intent_id: 最終 settled 時刻}。一区切りついた意図は
            そこを起点に測り直すため、裁定候補から外れる。

    Returns:
        経過日数の降順に並んだ候補 Intent リスト。
    """
    now = now or datetime.now()
    settled_map = settled_map or {}
    stale = [
        i for i in intents
        if _age_days(i, now, settled_map.get(i.id)) >= _STALE_AGE_DAYS
    ]
    stale.sort(key=lambda i: -_age_days(i, now, settled_map.get(i.id)))
    return stale


def record_intent_meters(sqlite) -> int:
    """意図の在庫と decay の効き具合を計器メーターへ日次記録する。

    settled（§4.3）は乱発にも沈黙にも倒れうるため、ガイド文を調整する材料として
    発生数を観測する。在庫件数と最大意図圧は「出口が詰まっていないか」の指標で、
    在庫が単調増加して最大圧が 1.0 に張り付いていれば詰まりのサイン。

    Args:
        sqlite: SQLiteStore。

    Returns:
        記録したメーター行数。
    """
    import logging
    logger = logging.getLogger(__name__)
    now = datetime.now()
    since = now - timedelta(days=1)
    recorded = 0
    for char in sqlite.list_characters():
        try:
            active = sqlite.list_intents(char.id, status="active")
            settled_map = sqlite.latest_settled_map(char.id)
            settled_1d = sum(1 for at in settled_map.values() if at >= since)
            max_pressure = max(
                (intent_pressure(i, now=now, settled_at=settled_map.get(i.id))
                 for i in active),
                default=0.0,
            )
        except Exception:
            logger.exception("意図メーターの記録に失敗 char=%s", char.name)
            continue
        sqlite.record_meter("intent_active", float(len(active)), character_id=char.id)
        sqlite.record_meter("intent_settled_1d", float(settled_1d), character_id=char.id)
        sqlite.record_meter("intent_pressure_max", float(max_pressure), character_id=char.id)
        recorded += 3
    return recorded
