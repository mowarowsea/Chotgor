"""応答可能性の判定 — availability(character, now) の純関数実装。

判定材料（docs/planned/aliveness_plan.md §5.1）:
    1. 対面モード（既存）      : 対面中は常に available（目の前にいる）
    2. away 状態（動的）       : 疲労離席・take_leave が設定した不在期限
    3. うつつシーン進行中      : 無人シーンの最中は席にいない
    4. 生活時間割（週間スケジュール）: キャラクター設計者（ユーザ）が管理UIで設定

キャラ発の push（Phase 6）も同じゲートを通す — 仕事中のキャラから push は来ない。
旧・時間帯ゲート（7〜24時）の概念はこの生活時間割に吸収される。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

# 曜日キー（Python の weekday() 順: 月=0 〜 日=6）
_WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

# うつつシーン進行中マーカーの有効期限（分）。プロセスクラッシュで
# 掃除されなかったマーカーが永久に unavailable を返さないための保険。
_USUAL_RUNNING_TTL_MINUTES = 30


@dataclass
class Availability:
    """availability 判定の結果。

    Attributes:
        available: 応答可能なら True。
        reason: unavailable の理由（"away" / "usual_scene" / 時間割のラベル）。
            available=True のときは空文字列。
    """

    available: bool
    reason: str = ""


def _parse_hhmm(value: str) -> tuple[int, int] | None:
    """"HH:MM" 文字列を (時, 分) に変換する。不正なら None。"""
    try:
        h, m = map(int, str(value).strip().split(":"))
    except (ValueError, AttributeError):
        return None
    if not (0 <= h < 24 and 0 <= m < 60):
        return None
    return (h, m)


def _schedule_block(schedule: dict | None, now: datetime) -> str | None:
    """生活時間割から現在時刻が「応答不可時間帯」に入っているか判定する。

    Args:
        schedule: characters.availability_schedule の値
            （{"mon": [{"from": "09:00", "to": "18:00", "label": "仕事"}], ...}）。
        now: 基準時刻。

    Returns:
        該当する時間帯のラベル。該当なし（応答可能）なら None。
        from > to の指定は日跨ぎ（例 23:00〜06:00）として扱う。
    """
    if not schedule or not isinstance(schedule, dict):
        return None
    blocks = schedule.get(_WEEKDAY_KEYS[now.weekday()]) or []
    if not isinstance(blocks, list):
        return None
    minutes_now = now.hour * 60 + now.minute
    for block in blocks:
        if not isinstance(block, dict):
            continue
        start = _parse_hhmm(block.get("from", ""))
        end = _parse_hhmm(block.get("to", ""))
        if start is None or end is None:
            continue
        start_min = start[0] * 60 + start[1]
        end_min = end[0] * 60 + end[1]
        if start_min <= end_min:
            hit = start_min <= minutes_now < end_min
        else:
            # 日跨ぎ（例 23:00〜06:00）: 開始以降 or 終了前
            hit = minutes_now >= start_min or minutes_now < end_min
        if hit:
            return str(block.get("label") or "予定あり")
    return None


def check_availability(
    character,
    now: datetime | None = None,
    *,
    usual_scene_running: bool = False,
) -> Availability:
    """キャラクターが今この瞬間応答できるかを判定する純関数（LLM 不使用）。

    優先順位: 対面モード（available 確定）→ away → うつつシーン進行中 → 生活時間割。

    Args:
        character: Character ORM（face_to_face_mode / away_until /
            availability_schedule を持つ）。None なら available（ゲート対象外）。
        now: 基準時刻。None なら現在時刻。
        usual_scene_running: うつつシーンが進行中か（is_usual_scene_running の結果）。

    Returns:
        Availability（available / unavailable + 理由）。
    """
    if character is None:
        return Availability(available=True)
    now = now or datetime.now()

    # 対面中は目の前にいる — 他のすべての不在理由に優先して available
    if int(getattr(character, "face_to_face_mode", 0) or 0):
        return Availability(available=True)

    away_until = getattr(character, "away_until", None)
    if away_until is not None and away_until > now:
        return Availability(
            available=False,
            reason=str(getattr(character, "away_reason", None) or "away"),
        )

    if usual_scene_running:
        return Availability(available=False, reason="usual_scene")

    label = _schedule_block(getattr(character, "availability_schedule", None), now)
    if label is not None:
        return Availability(available=False, reason=label)

    return Availability(available=True)


def mark_usual_scene_running(sqlite, character_id: str, running: bool) -> None:
    """うつつシーン進行中マーカーを設定/解除する（run_usual_days_scene が呼ぶ）。

    値は ISO タイムスタンプ。クラッシュで解除されなくても
    _USUAL_RUNNING_TTL_MINUTES 分で自然失効する。

    Args:
        sqlite: SQLiteStore。
        character_id: うつつ世界の所有者キャラ。
        running: True で開始マーク、False で解除。
    """
    key = f"usual_scene_running_{character_id}"
    sqlite.set_setting(key, datetime.now().isoformat() if running else "")


def is_usual_scene_running(sqlite, character_id: str, now: datetime | None = None) -> bool:
    """うつつシーンが進行中かをマーカーから判定する。

    Args:
        sqlite: SQLiteStore。
        character_id: うつつ世界の所有者キャラ。
        now: 基準時刻。

    Returns:
        進行中なら True（TTL 超過の古いマーカーは無視）。
    """
    raw = sqlite.get_setting(f"usual_scene_running_{character_id}", "")
    # 文字列以外（設定未初期化・テストのモック等）は「進行中でない」に倒す
    if not raw or not isinstance(raw, str):
        return False
    try:
        started = datetime.fromisoformat(raw)
    except ValueError:
        return False
    now = now or datetime.now()
    return (now - started) <= timedelta(minutes=_USUAL_RUNNING_TTL_MINUTES)


def format_escrow_annotation(message) -> str:
    """預かりメッセージの配達時に付ける時間差注釈を組み立てる。

    LLM に渡すコピーにだけ付け、DB の本文は変更しない。

    Args:
        message: ChatMessage ORM（created_at / content を持つ）。

    Returns:
        注釈付きの本文テキスト。
    """
    sent_at = getattr(message, "created_at", None)
    stamp = f"{sent_at.month}/{sent_at.day} {sent_at:%H:%M}" if sent_at else "少し前"
    return (
        f"（{stamp}・あなたが席を外している間に届いていたメッセージ）\n"
        f"{message.content}"
    )
