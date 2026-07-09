"""応答可能性の判定 — availability(character, now) の純関数実装。

2系統の判定経路を持つ（生活カレンダーのキャラ単位トグルで切り替わる）:

**従来経路**（living_schedule_enabled = 0）— aliveness_plan.md §5.1 の二値ゲート:
    1. 対面モード（既存）      : 対面中は常に available（目の前にいる）
    2. away 状態（動的）       : 疲労離席・take_leave が設定した不在期限
    3. うつつシーン進行中      : 無人シーンの最中は席にいない
    4. 生活時間割（週間スケジュール）: キャラクター設計者（ユーザ）が管理UIで設定

**生活カレンダー経路**（living_schedule_enabled = 1）— schedule_plan.md §2/§5/§7.5:
    availability を二値から **(state, 占有圧, 返信率, チェック間隔)** の連続量へ一般化する。
    now を含む planned な schedule_entries のうち占有圧（occupancy）最大が勝つ。
    エントリ皆無の時間帯は OnTime（完全リアルタイム）。
    優先順位: 対面(OnTime強制) > away_until(offline相当) > schedule_entries(占有圧最大) >
              エントリなし(OnTime)。
    ※ この経路では「うつつシーン進行中 = unavailable」は削除された（§7.5）— シーンは
      ②④エントリの内側で走り、そのエントリの state が配達値を直接決めるため不要。

`available`（二値）は後方互換のため残す。生活カレンダー経路では
`available = (state != "offline")` で導出する（active/busy は「返事は来る、ただし遅れる」
ので available 側）。既存の呼び出し（delivery.py の配達ゲート等）は無傷で動く。

キャラ発の push も同じゲートを通す — 仕事中のキャラから push は来ない。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

# 曜日キー（Python の weekday() 順: 月=0 〜 日=6）
_WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

# うつつシーン進行中マーカーの有効期限（分）。プロセスクラッシュで
# 掃除されなかったマーカーが永久に unavailable を返さないための保険。
_USUAL_RUNNING_TTL_MINUTES = 30

# 配達値の状態プリセット（schedule_plan.md §5）。エントリ個別の
# reply_rate / check_interval が None のとき、この既定が使われる。
# check_interval の None は「∞（次の非 offline チェック点まで預かり）」を表す。
_STATE_PRESETS: dict[str, dict] = {
    "OnTime": {"reply_rate": 1.0, "check_interval": 0},       # 予定なし・完全リアルタイム
    "active": {"reply_rate": 0.9, "check_interval": 5},       # 仕事中・数分の非同期遅延
    "busy": {"reply_rate": 0.4, "check_interval": 60},        # 超繁忙・平均数時間後
    "offline": {"reply_rate": 0.0, "check_interval": None},   # 意識がない・∞
}
# 未知 state のフォールバック（生成側のタイポ等で壊れないように active へ寄せる）
_DEFAULT_STATE = "active"


@dataclass
class Availability:
    """availability 判定の結果。

    従来の二値（available / reason）に加え、生活カレンダー（schedule_plan.md §5）の
    連続量 (state, occupancy, reply_rate, check_interval) を保持する。従来経路では
    state は available から導出され（True→OnTime / False→offline）、配達値は
    プリセット既定が入るため、二値しか見ない既存呼び出しはそのまま動く。

    Attributes:
        available: 応答可能なら True（生活カレンダー経路では state != "offline"）。
        reason: unavailable の理由（"away" / "usual_scene" / 時間割・エントリのラベル）。
            available=True のときは空文字列。
        state: 配達プリセット状態（OnTime / active / busy / offline）。
        occupancy: 占有圧 0.0–1.0（従来経路では available→0.0 / unavailable→1.0）。
        reply_rate: 返信率 0.0–1.0（見た時に実際に返信を生成する確率）。
        check_interval: チェック間隔 [分]。None は ∞（次の非 offline チェック点まで預かり）。
    """

    available: bool
    reason: str = ""
    state: str = "OnTime"
    occupancy: float = 0.0
    reply_rate: float = 1.0
    check_interval: int | None = 0


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


def _resolve_delivery_values(state: str, entry=None) -> tuple[float, int | None]:
    """state プリセット既定に、エントリ個別の上書きを重ねて (返信率, チェック間隔) を返す。

    Args:
        state: 配達プリセット状態（OnTime / active / busy / offline）。
        entry: ScheduleEntry ORM（reply_rate / check_interval を持ちうる）。None なら
            プリセット既定のみ。個別値が None のフィールドはプリセット既定を使う。

    Returns:
        (返信率 0.0–1.0, チェック間隔 [分]／None=∞)。
    """
    preset = _STATE_PRESETS.get(state, _STATE_PRESETS[_DEFAULT_STATE])
    reply_rate = preset["reply_rate"]
    check_interval = preset["check_interval"]
    if entry is not None:
        override_reply = getattr(entry, "reply_rate", None)
        if override_reply is not None:
            reply_rate = float(override_reply)
        override_check = getattr(entry, "check_interval", None)
        if override_check is not None:
            check_interval = int(override_check)
    return reply_rate, check_interval


def _availability_from_state(
    state: str, *, reason: str = "", occupancy: float = 0.0, entry=None
) -> Availability:
    """state（＋任意のエントリ個別値）から Availability を組み立てる。

    `available` は二値互換のため state != "offline" で導出する。

    Args:
        state: 配達プリセット状態。
        reason: unavailable 理由・ラベル（available 側でも表示ラベルとして使える）。
        occupancy: 占有圧 0.0–1.0。
        entry: 配達値の個別上書き元 ScheduleEntry（任意）。

    Returns:
        連続量を埋めた Availability。
    """
    reply_rate, check_interval = _resolve_delivery_values(state, entry)
    return Availability(
        available=(state != "offline"),
        reason=reason if state != "OnTime" else "",
        state=state,
        occupancy=occupancy,
        reply_rate=reply_rate,
        check_interval=check_interval,
    )


def check_availability(
    character,
    now: datetime | None = None,
    *,
    usual_scene_running: bool = False,
    sqlite=None,
) -> Availability:
    """キャラクターが今この瞬間応答できるかを判定する純関数（LLM 不使用）。

    生活カレンダー有効キャラ（living_schedule_enabled=1）で sqlite が渡された場合は
    実現層エントリを引く一般化経路（§7.5）、それ以外は従来の二値経路を通す。

    Args:
        character: Character ORM（face_to_face_mode / away_until /
            availability_schedule / living_schedule_enabled を持つ）。
            None なら available（ゲート対象外）。
        now: 基準時刻。None なら現在時刻。
        usual_scene_running: うつつシーンが進行中か（従来経路のみ参照。生活カレンダー
            経路では §7.5 により削除済みで無視される）。
        sqlite: SQLiteStore。生活カレンダー経路で schedule_entries を引くのに使う。
            None のときは有効キャラでも従来経路にフォールバックする（エントリを読めない
            ため。実運用の呼び出し側は sqlite を渡す）。

    Returns:
        Availability（二値 + 連続量 (state, 占有圧, 返信率, チェック間隔)）。
    """
    if character is None:
        return Availability(available=True)
    now = now or datetime.now()

    if int(getattr(character, "living_schedule_enabled", 0) or 0) and sqlite is not None:
        return _check_availability_scheduled(character, now, sqlite)
    return _check_availability_legacy(character, now, usual_scene_running)


def _check_availability_legacy(
    character, now: datetime, usual_scene_running: bool
) -> Availability:
    """従来の二値ゲート（living_schedule_enabled=0）。

    優先順位: 対面モード（available 確定）→ away → うつつシーン進行中 → 生活時間割。
    連続量は available から導出する（True→OnTime / False→offline）ため、二値しか見ない
    既存呼び出しはそのまま動く。
    """
    # 対面中は目の前にいる — 他のすべての不在理由に優先して available
    if int(getattr(character, "face_to_face_mode", 0) or 0):
        return _availability_from_state("OnTime")

    away_until = getattr(character, "away_until", None)
    if away_until is not None and away_until > now:
        return _availability_from_state(
            "offline",
            reason=str(getattr(character, "away_reason", None) or "away"),
            occupancy=1.0,
        )

    if usual_scene_running:
        return _availability_from_state("offline", reason="usual_scene", occupancy=1.0)

    label = _schedule_block(getattr(character, "availability_schedule", None), now)
    if label is not None:
        return _availability_from_state("offline", reason=label, occupancy=1.0)

    return _availability_from_state("OnTime")


def _check_availability_scheduled(character, now: datetime, sqlite) -> Availability:
    """生活カレンダー経路（living_schedule_enabled=1）— schedule_plan.md §2/§7.5。

    優先順位: 対面(OnTime強制) > away_until(offline相当) > schedule_entries(占有圧最大) >
              エントリなし(OnTime)。うつつシーン進行中は判定材料にしない（§7.5で削除）。

    Args:
        character: Character ORM。
        now: 基準時刻。
        sqlite: SQLiteStore（get_active_schedule_entries を持つ）。

    Returns:
        占有圧最大エントリの (state, 占有圧, 返信率, チェック間隔)。エントリ皆無なら OnTime。
    """
    # 対面中は割り込まれない — OnTime を強制（§7 (a)）
    if int(getattr(character, "face_to_face_mode", 0) or 0):
        return _availability_from_state("OnTime")

    # away_until（疲労離席・take_leave 由来）は配達値とは別系統の動的オーバーライド（§7.5）
    away_until = getattr(character, "away_until", None)
    if away_until is not None and away_until > now:
        return _availability_from_state(
            "offline",
            reason=str(getattr(character, "away_reason", None) or "away"),
            occupancy=1.0,
        )

    # now を含む planned エントリのうち占有圧最大が勝つ（重なりは読み取り時に解決）
    entries = sqlite.get_active_schedule_entries(character.id, now)
    if entries:
        top = max(entries, key=lambda e: float(getattr(e, "occupancy", 0.0) or 0.0))
        state = str(getattr(top, "state", None) or _DEFAULT_STATE)
        return _availability_from_state(
            state,
            reason=str(getattr(top, "label", None) or ""),
            occupancy=float(getattr(top, "occupancy", 0.0) or 0.0),
            entry=top,
        )

    # 何も予定がなければ完全リアルタイム（§2）
    return _availability_from_state("OnTime")


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
