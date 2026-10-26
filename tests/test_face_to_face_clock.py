"""対面中の「二つの時計」（docs/planned/schedule_plan.md §7.6）のテスト。

背景:
    チャットは入力に時間がかかるため、対面が長引くと場の時計（会話の流れの時刻）と
    外の時計（実時間）がずれる。従来は実時刻と予定を毎ターン push していたため、
    本人が会話の途中で慌てて帰る・思考の中で二つの時計を選ばされる、が起きていた。

検証対象:
    1. 対面開始時刻の割り出し（chat_store.get_face_to_face_run_start）
       - 最後の非対面メッセージより後の最初の対面メッセージの時刻
       - システムメッセージ（退席通知等）はモードの区切りに数えない
       - 末尾が非対面なら None / before 指定で終わった直後の列も引ける
       - 対面メッセージがまだ無ければ now（awareness.face_to_face_started_at）
    2. 対面中は予定ブロックを出さない（発話予約の段取り行も含めて）
    3. 対面が終わった直後の最初のターンだけ「外は◯◯。対面中に過ぎた予定：◯◯」が先頭に載る
       - キャラがテキストで一度返したら出なくなる（状態を持たない一度きり）
       - 過ぎた予定は外の時刻で終わったものだけ。進行中・本人の上書きは含めない
    4. 時刻ブロック: 対面中は実時刻・前回の交流からの経過を出さず、対面開始時刻だけ
    5. peek_at_schedule
       - at の緩いパース（"/"・"T"・秒付き・時刻なし・時だけ）、読めなければエラー文字列
       - at の時刻の予定＋前後2件、at 省略時は外の時刻を添える
       - 外の時刻より未来の突発（world/adhoc）は出さない・発火前の伏せ枠（pending）も出さない
       - 生活カレンダー無効キャラはエラー文字列
"""

import uuid
from datetime import datetime, timedelta

from sqlalchemy import text

from backend.character_actions.schedule_peeker import SchedulePeeker, parse_peek_at
from backend.services.chat.request_builder import _build_time_block
from backend.services.schedule.awareness import (
    build_schedule_lines,
    face_to_face_started_at,
)

# 外の時計: 月曜の朝。対面は土曜の夜に始まった想定。
_NOW = datetime(2026, 10, 26, 9, 0)
_SAT_NIGHT = datetime(2026, 10, 24, 21, 0)


def _make_character(sqlite_store, living=1, **kwargs):
    """テスト用キャラクターを作成して ID を返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name="はるテスト")
    sqlite_store.update_character(char_id, living_schedule_enabled=living, **kwargs)
    return char_id


def _msg(sqlite_store, session_id, role, at, face_to_face, is_system=False):
    """created_at を指定してメッセージを作るヘルパ（既定は作成時刻の now になるため上書きする）。"""
    mid = str(uuid.uuid4())
    sqlite_store.create_chat_message(
        message_id=mid, session_id=session_id, role=role, content="…",
        face_to_face=face_to_face, is_system_message=is_system,
    )
    with sqlite_store.engine.begin() as conn:
        conn.execute(
            text("UPDATE chat_messages SET created_at = :at WHERE id = :id"),
            {"at": at, "id": mid},
        )
    return mid


def _session(sqlite_store):
    sid = str(uuid.uuid4())
    sqlite_store.create_chat_session(session_id=sid, model_id="はるテスト@default")
    return sid


def _entry(sqlite_store, char_id, start, end, label, *, origin="template",
           source="haru", status="planned", state="active", occupancy=0.5, payload=None):
    sqlite_store.create_schedule_entry(
        character_id=char_id, start_at=start, end_at=end, state=state,
        source=source, origin=origin, occupancy=occupancy, status=status,
        label=label, payload=payload,
    )


# --- 1. 対面開始時刻の割り出し ---

def test_run_start_is_first_face_message_after_last_text(sqlite_store):
    """テキスト→対面→対面 の並びで、対面列の先頭の時刻が返ること。"""
    sid = _session(sqlite_store)
    _msg(sqlite_store, sid, "user", _SAT_NIGHT - timedelta(hours=1), 0)
    _msg(sqlite_store, sid, "user", _SAT_NIGHT, 1)
    _msg(sqlite_store, sid, "character", _SAT_NIGHT + timedelta(minutes=5), 1)
    assert sqlite_store.get_face_to_face_run_start(sid) == _SAT_NIGHT


def test_run_start_ignores_system_messages(sqlite_store):
    """対面中に挟まった非対面のシステムメッセージで列が途切れないこと。"""
    sid = _session(sqlite_store)
    _msg(sqlite_store, sid, "user", _SAT_NIGHT, 1)
    _msg(sqlite_store, sid, "character", _SAT_NIGHT + timedelta(hours=1), 0, is_system=True)
    _msg(sqlite_store, sid, "character", _SAT_NIGHT + timedelta(hours=2), 1)
    assert sqlite_store.get_face_to_face_run_start(sid) == _SAT_NIGHT


def test_run_start_none_when_tail_is_text(sqlite_store):
    """末尾が非対面なら None。before に対面発話の時刻を渡せば終わった列を引けること。"""
    sid = _session(sqlite_store)
    _msg(sqlite_store, sid, "user", _SAT_NIGHT, 1)
    last_face = _SAT_NIGHT + timedelta(hours=3)
    _msg(sqlite_store, sid, "character", last_face, 1)
    _msg(sqlite_store, sid, "user", _NOW, 0)
    assert sqlite_store.get_face_to_face_run_start(sid) is None
    assert sqlite_store.get_face_to_face_run_start(sid, before=last_face) == _SAT_NIGHT


def test_started_at_falls_back_to_now(sqlite_store):
    """対面メッセージがまだ無い・セッション無しなら now が返ること。"""
    sid = _session(sqlite_store)
    assert face_to_face_started_at(sqlite_store, sid, _NOW) == _NOW
    assert face_to_face_started_at(sqlite_store, None, _NOW) == _NOW


# --- 2. 対面中は予定ブロックを出さない ---

def test_face_to_face_hides_schedule_block(sqlite_store):
    """対面中は「いまの予定」「次の予定」が出ないこと（予定を push しない）。"""
    char_id = _make_character(sqlite_store)
    _entry(sqlite_store, char_id, _NOW - timedelta(hours=1), _NOW + timedelta(hours=3), "仕事")
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW)  # 通常時は出る
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW, face_to_face=True) == []


def test_face_to_face_hides_reservation_lines(sqlite_store):
    """対面中は発話予約の段取り行も出ないこと（生活カレンダー無効キャラでも）。"""
    char_id = _make_character(sqlite_store, living=0)
    sqlite_store.create_speech_reservation(
        character_id=char_id, session_id="s1",
        speak_at=_NOW + timedelta(hours=2), note="様子を聞く",
    )
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW)
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW, face_to_face=True) == []


# --- 3. 対面終了直後のつなぎ直し ---

def _ended_face_session(sqlite_store):
    """土曜21時〜日曜深夜まで対面し、月曜朝にテキストへ戻ったセッションを作る。"""
    sid = _session(sqlite_store)
    _msg(sqlite_store, sid, "user", _SAT_NIGHT, 1)
    _msg(sqlite_store, sid, "character", _SAT_NIGHT + timedelta(hours=27), 1)
    _msg(sqlite_store, sid, "user", _NOW, 0)
    return sid


def test_reconnect_lists_passed_entries(sqlite_store):
    """対面終了直後は、外の時刻と対面中に過ぎた予定が先頭に事実だけで載ること。"""
    char_id = _make_character(sqlite_store)
    sid = _ended_face_session(sqlite_store)
    _entry(sqlite_store, char_id, datetime(2026, 10, 24, 23, 0), datetime(2026, 10, 25, 7, 0),
           "就寝", state="offline")
    _entry(sqlite_store, char_id, datetime(2026, 10, 26, 6, 0), datetime(2026, 10, 26, 8, 30),
           "朝の支度")
    # 進行中の予定は「過ぎた予定」ではなく「いまの予定」に載る
    _entry(sqlite_store, char_id, datetime(2026, 10, 26, 8, 30), datetime(2026, 10, 26, 12, 0),
           "仕事（午前）")
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW, session_id=sid)
    assert lines[0] == "外は 10/26(月) 09:00。"
    assert lines[1] == "対面中に過ぎた予定：就寝、朝の支度"
    assert any("いまの予定: 仕事（午前）" in ln for ln in lines[2:])
    # 気遣いの文言は付けない（本人の希望: 事実だけ）
    assert not any("ごめん" in ln or "大丈夫" in ln for ln in lines)


def test_reconnect_without_passed_entries(sqlite_store):
    """過ぎた予定が無ければ外の時刻の行だけが載ること。"""
    char_id = _make_character(sqlite_store)
    sid = _ended_face_session(sqlite_store)
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW, session_id=sid) == [
        "外は 10/26(月) 09:00。"
    ]


def test_reconnect_excludes_self_override(sqlite_store):
    """本人の意志上書き（override_schedule）は過ぎた「予定」に数えないこと。"""
    from backend.character_actions.rescheduler import OVERRIDE_PAYLOAD_KIND
    char_id = _make_character(sqlite_store)
    sid = _ended_face_session(sqlite_store)
    _entry(sqlite_store, char_id, _SAT_NIGHT, _SAT_NIGHT + timedelta(hours=3), "夜更かし",
           origin="adhoc", state="OnTime", occupancy=0.85,
           payload={"kind": OVERRIDE_PAYLOAD_KIND, "reason": "夜更かし"})
    lines = build_schedule_lines(sqlite_store, char_id, now=_NOW, session_id=sid)
    assert lines == ["外は 10/26(月) 09:00。"]


def test_reconnect_only_once(sqlite_store):
    """キャラがテキストで一度返したら、つなぎ直しは出なくなること（状態を持たない一度きり）。"""
    char_id = _make_character(sqlite_store)
    sid = _ended_face_session(sqlite_store)
    _msg(sqlite_store, sid, "character", _NOW + timedelta(minutes=1), 0)
    lines = build_schedule_lines(
        sqlite_store, char_id, now=_NOW + timedelta(minutes=5), session_id=sid,
    )
    assert not any(ln.startswith("外は") for ln in lines)


def test_no_reconnect_without_face_history(sqlite_store):
    """ずっとテキストだったセッションではつなぎ直しが出ないこと。"""
    char_id = _make_character(sqlite_store)
    sid = _session(sqlite_store)
    _msg(sqlite_store, sid, "user", _NOW - timedelta(minutes=10), 0)
    _msg(sqlite_store, sid, "character", _NOW - timedelta(minutes=9), 0)
    assert build_schedule_lines(sqlite_store, char_id, now=_NOW, session_id=sid) == []


# --- 4. 時刻ブロック ---

def test_time_block_in_face_to_face_shows_start_only():
    """対面中は実時刻・経過時間を出さず、対面開始時刻だけを渡すこと。"""
    block = _build_time_block(True, "2026-10-26(月) 09:00:00", "2日", "2026-10-24(土) 21:00")
    assert "【対面開始：2026-10-24(土) 21:00】" in block
    assert "2026-10-26" not in block
    assert "前回の交流から" not in block


def test_time_block_outside_face_to_face_unchanged():
    """対面外は従来どおり現在時刻と経過が出ること。"""
    block = _build_time_block(True, "2026-10-26(月) 09:00:00", "2日")
    assert "【現在時刻：2026-10-26(月) 09:00:00】" in block
    assert "【前回の交流から：2日】" in block


def test_time_block_respects_time_awareness_off():
    """時刻認識 OFF なら対面中でもブロック自体を出さないこと。"""
    assert _build_time_block(False, "", "", "2026-10-24(土) 21:00") == ""


# --- 5. peek_at_schedule ---

def test_parse_peek_at_is_lenient():
    """「大体でいい」の約束どおり、表記揺れを受けること。"""
    want = datetime(2026, 10, 24, 23, 0)
    assert parse_peek_at("2026-10-24 23:00") == want
    assert parse_peek_at("2026/10/24 23:00") == want
    assert parse_peek_at("2026-10-24T23:00") == want
    assert parse_peek_at("2026-10-24 23:00:00") == want
    assert parse_peek_at("2026-10-24 23") == want
    assert parse_peek_at("2026-10-24") == datetime(2026, 10, 24, 0, 0)
    assert parse_peek_at("土曜の夜") is None
    assert parse_peek_at("") is None


def _seed_weekend(sqlite_store, char_id):
    """土曜夜〜月曜の固定予定を並べる。"""
    for start, end, label in [
        (datetime(2026, 10, 24, 12, 0), datetime(2026, 10, 24, 13, 0), "昼ごはん"),
        (datetime(2026, 10, 24, 19, 0), datetime(2026, 10, 24, 20, 0), "夕ごはん"),
        (datetime(2026, 10, 24, 23, 0), datetime(2026, 10, 25, 7, 0), "就寝"),
        (datetime(2026, 10, 25, 10, 0), datetime(2026, 10, 25, 12, 0), "ゲーム"),
        (datetime(2026, 10, 25, 23, 0), datetime(2026, 10, 26, 6, 0), "就寝"),
        (datetime(2026, 10, 26, 9, 0), datetime(2026, 10, 26, 18, 0), "仕事"),
    ]:
        _entry(sqlite_store, char_id, start, end, label)


def test_peek_at_scene_time_returns_neighbors(sqlite_store):
    """at の時刻の予定と、前後2件ずつが返ること（場の時計で予定を読む）。"""
    char_id = _make_character(sqlite_store)
    _seed_weekend(sqlite_store, char_id)
    out = SchedulePeeker(char_id, sqlite_store).peek_at_schedule(
        at="2026-10-24 23:30", now=_NOW,
    )
    lines = out.splitlines()
    assert lines[0] == "2026-10-24(土) 23:30 ごろの予定:"
    assert [ln for ln in lines if ln.startswith("- 前:")] == [
        "- 前: 10/24(土) 12:00〜13:00 昼ごはん（active）",
        "- 前: 10/24(土) 19:00〜20:00 夕ごはん（active）",
    ]
    assert "- その時刻: 10/24(土) 23:00〜10/25(日) 07:00 就寝（active）" in lines
    assert [ln for ln in lines if ln.startswith("- 後:")] == [
        "- 後: 10/25(日) 10:00〜12:00 ゲーム（active）",
        "- 後: 10/25(日) 23:00〜10/26(月) 06:00 就寝（active）",
    ]
    assert "外の世界" not in out  # 場の時刻で覗いたときは外の時刻を押しつけない


def test_peek_without_at_shows_outside_clock(sqlite_store):
    """at 省略時は外の時刻と、その周りの予定が返ること。"""
    char_id = _make_character(sqlite_store)
    _seed_weekend(sqlite_store, char_id)
    out = SchedulePeeker(char_id, sqlite_store).peek_at_schedule(now=_NOW)
    assert out.startswith("外の世界はいま 2026-10-26(月) 09:00。")
    assert "- その時刻: 10/26(月) 09:00〜18:00 仕事（active）" in out


def test_peek_hides_future_world_events(sqlite_store):
    """外の時刻より未来の突発（world/adhoc）と伏せ枠（pending）は出ないこと（ネタバレ防止）。"""
    char_id = _make_character(sqlite_store)
    _entry(sqlite_store, char_id, _NOW + timedelta(hours=2), _NOW + timedelta(hours=3),
           "急な呼び出し", origin="adhoc", source="world")
    _entry(sqlite_store, char_id, _NOW + timedelta(hours=4), _NOW + timedelta(hours=5),
           "伏せ枠", origin="adhoc", source="world", status="pending")
    _entry(sqlite_store, char_id, _NOW + timedelta(hours=6), _NOW + timedelta(hours=7), "散歩")
    out = SchedulePeeker(char_id, sqlite_store).peek_at_schedule(now=_NOW)
    assert "急な呼び出し" not in out
    assert "伏せ枠" not in out
    assert "散歩" in out


def test_peek_shows_past_world_events(sqlite_store):
    """外の時刻で既に始まった突発は、本人が体験した現実なので見えること。"""
    char_id = _make_character(sqlite_store)
    _entry(sqlite_store, char_id, _NOW - timedelta(hours=3), _NOW - timedelta(hours=2),
           "停電", origin="adhoc", source="world")
    out = SchedulePeeker(char_id, sqlite_store).peek_at_schedule(now=_NOW)
    assert "停電" in out


def test_peek_errors(sqlite_store):
    """形式不正・生活カレンダー無効・sqlite 無しはエラー文字列で返ること（例外を投げない）。"""
    char_id = _make_character(sqlite_store)
    assert "形式が読めません" in SchedulePeeker(char_id, sqlite_store).peek_at_schedule(at="あとで")
    off = _make_character(sqlite_store, living=0)
    assert "有効になっていません" in SchedulePeeker(off, sqlite_store).peek_at_schedule()
    assert "error" in SchedulePeeker(char_id, None).peek_at_schedule()
