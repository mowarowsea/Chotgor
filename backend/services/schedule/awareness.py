"""予定コンテキスト（要件③） — キャラ本人へ渡す「いま・この後の予定」の淡白な行を組む。

「③はるへの予定の通知」（2026-07-11 要件）:
    - 現状、本人は自分の予定（就寝など）を知らされておらず、会話の途中で突然
      offline に落ちる（返信が翌朝まで預かりになる）。「この後の予定が分かっている」
      「それを意志で塗り替えられる（override_schedule・④）」の両輪で、寝落ちしか
      なかった夜の反応を本人のその場の選択に開く。
    - 注入先はターン注釈（request_builder の {block_schedule}）。1on1 とうつつ PC の
      両方が chat_flow/flow.py 経由でこのモジュールを通る。GM には渡さない
      （GM の予定入力は別経路・weekly_batch の framing）。

ネタバレ防止（要件の絶対条件）:
    - ランダムイベントは通知してはならない。③伏せ枠（world/adhoc・status="pending"）は
      status フィルタで、発火済みの world/adhoc 実イベントは「次の予定」の origin/source
      フィルタで構造的に除外する（未来の world イベントを先に知らせない）。
    - 「いまの予定」は例外的に origin を問わない — 現在進行中の出来事は本人が
      体験している現実であり、隠す方が不整合になる。

出力は淡白な事実の行リスト（圧力ブロックと同じ思想 — どう受け取るかは本人次第）。

対面中の「二つの時計」（docs/planned/schedule_plan.md §7.6）:
    対面中は予定を push しない（発話予約の段取り行も含めてブロックごと出さない）。
    本人は peek_at_schedule で覗きたいときだけ引く。対面が終わった直後の最初のターンにだけ
    「外は◯◯。対面中に過ぎた予定：◯◯」を事実だけで先頭へ置く（つなぎ直し）。
    対面の開始・終了は chat_messages.face_to_face から割り出し、状態は持たない。

発話予約（speak_later）の相乗り（docs/planned/speak_later_plan.md）:
    pending の予約がある間、「{speak_at} になったら『{note}』をしようと思っている」を
    本人の段取りとして同じ枠に注入する。予約はセッションに紐づくが、注入はセッションを
    問わない（自分の段取りはどの場でも本人の頭の中にあるもの）。生活カレンダー無効の
    キャラにも載る（予約行だけのブロックになる）。
"""

import logging
from datetime import datetime, timedelta

from backend.lib.time_awareness import japanese_weekday

# 本人上書きエントリの payload.kind（rescheduler.py と共有の種別マーカー）
from backend.character_actions.rescheduler import OVERRIDE_PAYLOAD_KIND

logger = logging.getLogger(__name__)

# 「次の予定」を探す先読み幅（時間）。丸1日先まで見れば就寝・翌日の主要予定を拾える。
_LOOKAHEAD_HOURS = 24


def _fmt_span(minutes: int) -> str:
    """分数を「X分」「X時間Y分」の読みやすい表記にする。"""
    minutes = max(0, int(minutes))
    if minutes < 60:
        return f"{minutes}分"
    hours, rest = divmod(minutes, 60)
    return f"{hours}時間{rest}分" if rest else f"{hours}時間"


def _is_self_override(entry) -> bool:
    """エントリが本人の意志上書き（override_schedule・④）かを返す。"""
    payload = getattr(entry, "payload", None) or {}
    return (
        getattr(entry, "source", "") == "haru"
        and getattr(entry, "origin", "") == "adhoc"
        and isinstance(payload, dict)
        and payload.get("kind") == OVERRIDE_PAYLOAD_KIND
    )


def face_to_face_started_at(sqlite, session_id: str | None, now: datetime) -> datetime:
    """いまの対面が始まった時刻を返す（セッション末尾から続く対面メッセージ列の先頭）。

    対面メッセージがまだ無い（切り替えた直後の最初のターン等）なら now。
    """
    if sqlite is None or not session_id:
        return now
    started = sqlite.get_face_to_face_run_start(session_id)
    return started if started is not None and started <= now else now


def build_reconnect_lines(
    sqlite, character_id: str, session_id: str | None, now: datetime
) -> list[str]:
    """対面が終わった直後の最初のターンだけ、外の時刻と対面中に過ぎた予定を返す。

    判定: いまが非対面で、セッション内の最新のキャラ発話が対面のもの。キャラが
    テキストで一度返せば条件が外れるため、状態を持たずに一度きりになる。
    """
    if sqlite is None or not session_id:
        return []
    last_char = sqlite.get_last_character_message(session_id)
    if last_char is None or not int(getattr(last_char, "face_to_face", 0) or 0):
        return []
    started = sqlite.get_face_to_face_run_start(session_id, before=last_char.created_at)
    if started is None:
        return []

    passed: list[str] = []
    for entry in sqlite.list_schedule_entries(
        character_id, since=started, until=now, statuses=["planned", "done"],
    ):
        # 外の時刻でもう終わっている予定だけ（進行中のものは「いまの予定」に載る）。
        # 本人の上書きは過ぎた「予定」ではないので除く。
        if entry.end_at > now or _is_self_override(entry):
            continue
        label = (getattr(entry, "label", None) or "予定").strip()
        if label not in passed:
            passed.append(label)

    lines = [f"外は {now.month}/{now.day}({japanese_weekday(now)}) {now:%H:%M}。"]
    if passed:
        lines.append(f"対面中に過ぎた予定：{'、'.join(passed)}")
    return lines


def build_schedule_lines(
    sqlite,
    character_id: str,
    now: datetime | None = None,
    *,
    face_to_face: bool = False,
    session_id: str | None = None,
) -> list[str]:
    """キャラ本人へ渡す予定コンテキストの行リストを組む（LLM 不使用の純関数）。

    構成（素材がある行だけ並ぶ）:
        1. いまの予定       — now を含む planned エントリの占有圧最大。開始からの経過分。
        2. 上書きの超過     — 現在が本人の意志上書き（④）中で、本来の予定を轢いている
                              場合、「本来の予定をX分超過中」を明示（夜更かしの自覚材料）。
        3. 次の予定         — 固定予定（template）のうち直近の未来。あと何分かを添える。
                              world/adhoc（伏せ枠・突発）は含めない（ネタバレ防止）。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター ID。
        now: 基準時刻（テスト注入用）。
        face_to_face: 対面中なら True（ブロックごと出さない）。
        session_id: 1on1 セッション ID（対面終了直後のつなぎ直し判定に使う）。

    Returns:
        淡白な事実の行リスト。素材なしなら空リスト。
        生活カレンダー無効キャラは 1〜3 を出さない（発話予約の行だけ載りうる）。
    """
    if face_to_face:
        return []
    char = sqlite.get_character(character_id)
    if char is None:
        return []
    now = now or datetime.now()
    if not int(getattr(char, "living_schedule_enabled", 0) or 0):
        return _build_reservation_lines(sqlite, character_id, now)
    lines: list[str] = build_reconnect_lines(sqlite, character_id, session_id, now)

    # --- 1. いまの予定（占有圧最大が現実） ---
    active = sqlite.get_active_schedule_entries(character_id, now)
    current = active[0] if active else None  # get_active_schedule_entries は占有圧降順
    if current is not None:
        elapsed = int((now - current.start_at).total_seconds() // 60)
        label = (getattr(current, "label", None) or "予定").strip()
        if _is_self_override(current):
            lines.append(
                f"いまは、あなたが自分の意志で {current.end_at:%H:%M} まで"
                f"予定を上書きしている時間（理由: {label}）"
            )
            # --- 2. 上書きの超過（本来の予定との差分を自覚できるように） ---
            override_occ = float(getattr(current, "occupancy", 0.0) or 0.0)
            for beaten in active[1:]:
                if _is_self_override(beaten):
                    continue
                if float(getattr(beaten, "occupancy", 0.0) or 0.0) >= override_occ:
                    continue
                beaten_label = (getattr(beaten, "label", None) or "予定").strip()
                overrun = int((now - beaten.start_at).total_seconds() // 60)
                if overrun > 0:
                    lines.append(
                        f"本来の予定「{beaten_label}」（{beaten.start_at:%H:%M}〜）を"
                        f"{_fmt_span(overrun)}超過中"
                    )
        else:
            state = getattr(current, "state", "") or ""
            lines.append(
                f"いまの予定: {label}"
                f"（{state}・{current.start_at:%H:%M}に始まって{_fmt_span(elapsed)}経過・"
                f"{current.end_at:%H:%M}まで）"
            )

    # --- 3. 次の予定（固定予定のみ。world/adhoc はネタバレ防止で見せない） ---
    upcoming = sqlite.list_schedule_entries(
        character_id,
        since=now,
        until=now + timedelta(hours=_LOOKAHEAD_HOURS),
        statuses=["planned"],
        origins=["template"],
    )
    next_entry = next((e for e in upcoming if e.start_at > now), None)
    if next_entry is not None:
        remaining = int((next_entry.start_at - now).total_seconds() // 60)
        label = (getattr(next_entry, "label", None) or "予定").strip()
        day_note = "翌日" if next_entry.start_at.date() != now.date() else ""
        lines.append(
            f"次の予定: {label}"
            f"（{day_note}{next_entry.start_at:%H:%M}から・あと{_fmt_span(remaining)}）"
        )

    # --- 4. 発話予約（speak_later）— 本人の段取りの想起 ---
    lines.extend(_build_reservation_lines(sqlite, character_id, now))

    return lines


def _build_reservation_lines(sqlite, character_id: str, now: datetime) -> list[str]:
    """pending の発話予約を「本人の段取り」の行にする。

    fired / expired / cancelled / superseded になった予約は
    list_pending_speech_reservations に載らないため、自然に消える。
    """
    lines: list[str] = []
    for r in sqlite.list_pending_speech_reservations(character_id=character_id):
        if r.speak_at.date() == now.date():
            stamp = f"{r.speak_at:%H:%M}"
        else:
            stamp = f"{r.speak_at.month}/{r.speak_at.day} {r.speak_at:%H:%M}"
        lines.append(f"{stamp} になったら「{r.note}」をしようと思っている")
    return lines
