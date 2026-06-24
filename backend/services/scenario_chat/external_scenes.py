"""うつつ PC へ「同じ世界軸上の最近のシーン全て」をまとめて渡すための整形ヘルパ。

設計哲学:
    キャラクターにとって、うつつ・ユーザとの1on1・TRPG・グループは、すべて同じ一本の
    世界軸上で続いている時間である。仕事をする→帰宅して話す→翌朝また仕事、というのと
    同じ連続性。本モジュールはそれをモデルへ素直に渡すための部品で、「ハレ／ケ」「幕間」
    のようなフレーミングは加えない。シーン境界は XML 風タグで包んで示すだけにする。

出力フォーマット（単一メッセージのテキスト本文）:

    <history>
    <{キャラ名}の日常>
    <Narrator>...</Narrator>
    <同僚>...</同僚>
    <{キャラ名}>...</{キャラ名}>
    <Narrator>... [SCENE_CLOSE]</Narrator>
    </{キャラ名}の日常>
    <{ユーザ名}とのテキストのやり取り>
    <{ユーザ名}>...</{ユーザ名}>
    <{キャラ名}>...</{キャラ名}>
    </{ユーザ名}とのテキストのやり取り>
    <{ユーザ名}との対面>
    ...
    </{ユーザ名}との対面>
    <TRPG「{シナリオ名}」プレイログ>
    <Narrator>...</Narrator>
    <{役名}@{キャラ名}>...</{役名}@{キャラ名}>
    </TRPG「{シナリオ名}」プレイログ>
    <{キャラ名}の日常>
    <Narrator>...</Narrator>
    <同僚>...</同僚>
    </{キャラ名}の日常>
    </history>

役割:
    - 過去のシーンも進行中シーンも、すべてシーンタグで閉じる（XML 整形）
    - キャラ本人の発話は他の登場人物と同じ `<{名前}>...</{名前}>` 形式（role 区別を持たない）
    - TRPG 内の自分の PC 発話だけ `<{役名}@{キャラ名}>` の二段表記で「中の人」を明示
    - ユーザ発話も `<{user_label}>...</{user_label}>` でタグ化
    - 単一メッセージ（role="user"）として送ることで、claude_cli の `<human>` ラッパー
      を回避する（claude_cli は messages 長 1 のとき内容を素通しでダンプする）

時間軸（since_dt の決め方）:
    - 起点: 最新の SCENE_CLOSE 直後（前回うつつシーンが閉じた時刻）
    - 該当無し: 最古のうつつターン時刻、それも無ければ 24h 前
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from backend.services.scenario_chat.format_speech import (
    _escape_xml_content,
    _sanitize_xml_tag_name,
)

logger = logging.getLogger(__name__)

# SCENE_CLOSE 起点が決まらないときの遡及ウィンドウ。
_DEFAULT_LOOKBACK_HOURS = 24


# ---------------------------------------------------------------------------
# 中間表現
# ---------------------------------------------------------------------------


@dataclass
class SpeechTurn:
    """1 発話を表す中間表現。XML タグ整形済み speaker と content を持つ。"""

    created_at: datetime
    speaker_tag: str  # `<speaker_tag>content</speaker_tag>` のタグ部分
    content: str       # 本文（XML エスケープ済み・タグなし）


@dataclass
class Scene:
    """1 シーンを表す中間表現。

    シーンタグ名（"はるの日常" / "太郎との対面" / "TRPG「ダンジョン編」プレイログ" など）と、
    そのシーンに属する SpeechTurn のリストを持つ。time_key はマージ用ソートに使う。
    """

    scene_tag: str
    scene_key: tuple
    started_at: datetime
    turns: list[SpeechTurn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 起点時刻の解決（最新 SCENE_CLOSE）
# ---------------------------------------------------------------------------


def _latest_scene_close_time(sqlite, character_id: str) -> datetime | None:
    """指定キャラの usual_days セッションで、SCENE_CLOSE を含む最新ターン時刻を返す。"""
    from backend.repositories.sqlite.store import Scenario, ScenarioSession, ScenarioTurn

    with sqlite.get_session() as session:
        rows = (
            session.query(ScenarioTurn.created_at, ScenarioTurn.raw_response, ScenarioTurn.content)
            .join(ScenarioSession, ScenarioTurn.session_id == ScenarioSession.id)
            .join(Scenario, ScenarioSession.scenario_id == Scenario.id)
            .filter(
                Scenario.owner_character_id == character_id,
                ScenarioSession.engine_type == "usual_days",
            )
            .order_by(ScenarioTurn.created_at.desc())
            .all()
        )
    for created_at, raw, content in rows:
        haystack = f"{raw or ''}\n{content or ''}".lower()
        if "[scene_close]" in haystack:
            return created_at
    return None


def _earliest_usual_turn_time(sqlite, character_id: str) -> datetime | None:
    """うつつターン最古時刻。SCENE_CLOSE がまだ無い場合のフォールバック起点。"""
    from backend.repositories.sqlite.store import Scenario, ScenarioSession, ScenarioTurn

    with sqlite.get_session() as session:
        row = (
            session.query(ScenarioTurn.created_at)
            .join(ScenarioSession, ScenarioTurn.session_id == ScenarioSession.id)
            .join(Scenario, ScenarioSession.scenario_id == Scenario.id)
            .filter(
                Scenario.owner_character_id == character_id,
                ScenarioSession.engine_type == "usual_days",
            )
            .order_by(ScenarioTurn.created_at.asc())
            .first()
        )
        return row[0] if row else None


def resolve_since_dt(sqlite, character_id: str, now: datetime | None = None) -> datetime:
    """external シーン収集の起点時刻を決める。

    優先順: 最新 SCENE_CLOSE → 最古うつつターン → now-24h。
    PC runner 起動時には「今シーンの GM ターン」が保存済みなので、最大 created_at を
    使ってはいけない（それを使うと当の今シーンが起点になり、external が全て対象外になる）。
    """
    last_close = _latest_scene_close_time(sqlite, character_id)
    if last_close is not None:
        return last_close
    earliest = _earliest_usual_turn_time(sqlite, character_id)
    if earliest is not None:
        return earliest
    base = now or datetime.now()
    return base - timedelta(hours=_DEFAULT_LOOKBACK_HOURS)


# ---------------------------------------------------------------------------
# シーン構築（うつつ scenario_turns）
# ---------------------------------------------------------------------------


def _scene_close_in(raw: str, content: str) -> bool:
    """raw_response または content に [SCENE_CLOSE] を含むか（大小・空白ゆれ無視）。"""
    haystack = f"{raw or ''}\n{content or ''}".lower()
    return "[scene_close]" in haystack


def _resolve_trpg_role_name(sqlite, session_id: str, character_id: str) -> str:
    """TRPG セッションの pc_assignments から、当該キャラの役名（slot name）を解決する。

    pc_assignments: [{"slot_id": "...", "player_type": "character", "character_id": "..."}]
    対応する slot を scenario.pc_slots から引いて name を返す。見つからなければ空文字。
    """
    sess = sqlite.get_scenario_session(session_id)
    if not sess:
        return ""
    pc_assignments = getattr(sess, "pc_assignments", None) or []
    if isinstance(pc_assignments, str):
        try:
            pc_assignments = json.loads(pc_assignments)
        except (json.JSONDecodeError, TypeError):
            pc_assignments = []
    target_slot_id = None
    for a in pc_assignments:
        if not isinstance(a, dict):
            continue
        if a.get("player_type") == "character" and a.get("character_id") == character_id:
            target_slot_id = a.get("slot_id")
            break
    if not target_slot_id:
        return ""
    scenario = sqlite.get_scenario(sess.scenario_id)
    if not scenario:
        return ""
    slots = getattr(scenario, "pc_slots", None) or []
    if isinstance(slots, str):
        try:
            slots = json.loads(slots)
        except (json.JSONDecodeError, TypeError):
            slots = []
    for s in slots:
        if isinstance(s, dict) and s.get("slot_id") == target_slot_id:
            return str(s.get("name") or "")
    return ""


def _build_usual_scenes(
    history: list[Any],
    self_character_id: str,
    self_speaker_tag: str,
    user_label: str,
    narrator_name: str,
) -> list[Scene]:
    """うつつ scenario_turns（同一セッションの全ターン）をシーン単位に分割して Scene 列を返す。

    SCENE_CLOSE 観測後、別 raw_response の GM ターンに切り替わった地点で次シーンに分割する。
    PC ターン（raw="" のことが多い）が間に挟まっても境界とは見なさない（既存 _format_scenario_history_for_pc と同じ判定）。
    """
    scenes: list[Scene] = []
    user_tag = _sanitize_xml_tag_name(user_label or "user")
    narrator_tag = _sanitize_xml_tag_name(narrator_name)
    self_tag = _sanitize_xml_tag_name(self_speaker_tag)

    pending_close_raw: str | None = None
    current: Scene | None = None
    scene_idx = 0

    def _new_scene(at: datetime) -> Scene:
        # シーンタグ名は本人の主観: 「{キャラ名}の日常」
        return Scene(
            scene_tag=f"{self_speaker_tag}の日常",
            scene_key=("usual", scene_idx),
            started_at=at,
        )

    for turn in history:
        content = (getattr(turn, "content", "") or "").strip()
        if not content:
            continue
        raw = getattr(turn, "raw_response", "") or ""
        # シーン境界判定: SCENE_CLOSE 既観測 & 別 raw_response の非空 raw 到来 → 次シーンへ
        if pending_close_raw is not None and raw and raw != pending_close_raw:
            scene_idx += 1
            current = None
            pending_close_raw = None
        if current is None:
            current = _new_scene(turn.created_at)
            scenes.append(current)
        stype = getattr(turn, "speaker_type", "")
        speaker_id = getattr(turn, "speaker_id", None)
        speaker_name = getattr(turn, "speaker_name", "") or ""
        if stype == "pc" and speaker_id == self_character_id:
            current.turns.append(SpeechTurn(turn.created_at, self_tag, content))
        elif stype == "user":
            current.turns.append(SpeechTurn(turn.created_at, user_tag, content))
        elif stype == "narrator":
            current.turns.append(SpeechTurn(turn.created_at, narrator_tag, content))
        else:
            tag = _sanitize_xml_tag_name(speaker_name)
            current.turns.append(SpeechTurn(turn.created_at, tag, content))
        if _scene_close_in(raw, content):
            pending_close_raw = raw
    return scenes


# ---------------------------------------------------------------------------
# シーン構築（external: 1on1 / Group / TRPG）
# ---------------------------------------------------------------------------


def _build_1on1_scenes(
    sqlite,
    character_name: str,
    user_label: str,
    since_dt: datetime,
    until_dt: datetime,
) -> list[Scene]:
    """1on1 セッションのメッセージから Scene 列を作る。

    同じセッション内で face_to_face モードが切り替わったら別シーン扱い。
    Scene 単位: (session_id, face_to_face) ごとに 1 つ。
    シーンタグ名: 対面なら「{user_label}との対面」、テキストなら「{user_label}とのテキストのやり取り」。
    """
    from backend.repositories.sqlite.store import ChatMessage, ChatSession

    scenes: list[Scene] = []
    user_tag = _sanitize_xml_tag_name(user_label or "user")
    char_tag = _sanitize_xml_tag_name(character_name)
    with sqlite.get_session() as session:
        rows = (
            session.query(ChatMessage, ChatSession.session_type)
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .filter(
                ChatSession.model_id.like(f"{character_name}@%"),
                ChatMessage.created_at >= since_dt,
                ChatMessage.created_at < until_dt,
                (ChatMessage.is_system_message == None) | (ChatMessage.is_system_message == 0),  # noqa: E711
            )
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
    current: Scene | None = None
    current_key: tuple | None = None
    for msg, session_type in rows:
        if (session_type or "1on1") == "group":
            continue
        is_face = bool(getattr(msg, "face_to_face", 0))
        key = ("1on1", msg.session_id, 1 if is_face else 0)
        if key != current_key:
            tag_label = (
                f"{user_label or 'ユーザ'}との対面"
                if is_face else
                f"{user_label or 'ユーザ'}とのテキストのやり取り"
            )
            current = Scene(scene_tag=tag_label, scene_key=key, started_at=msg.created_at)
            scenes.append(current)
            current_key = key
        content = (msg.content or "").strip()
        if not content:
            continue
        if msg.role == "user":
            current.turns.append(SpeechTurn(msg.created_at, user_tag, content))
        elif msg.role == "character":
            current.turns.append(SpeechTurn(msg.created_at, char_tag, content))
    return [s for s in scenes if s.turns]


def _build_trpg_scenes(
    sqlite,
    character_id: str,
    character_name: str,
    user_label: str,
    since_dt: datetime,
    until_dt: datetime,
    narrator_name: str = "Narrator",
) -> list[Scene]:
    """TRPG（engine_type="ensemble_pc"）で本人が PC 参加するセッションの発話から Scene 列を作る。

    Scene 単位: (session_id, scene_index)。scene_index は SCENE_CLOSE 観測後、別 raw_response の
    GM ターンに切り替わった地点で +1。
    自分の PC 発話タグ = `{役名}@{キャラ名}`（中の人を明示）。役名は pc_assignments から解決。
    """
    try:
        turns = sqlite.get_trpg_turns_for_character_on_date(character_id, since_dt, until_dt)
    except Exception:
        logger.exception("external_scenes: TRPG ターン取得失敗 character=%s", character_id)
        return []

    user_tag = _sanitize_xml_tag_name(user_label or "user")
    narrator_tag = _sanitize_xml_tag_name(narrator_name)
    char_tag = _sanitize_xml_tag_name(character_name)
    role_name_cache: dict[str, str] = {}
    scenes: list[Scene] = []

    pending_close_by_session: dict[str, str] = {}
    scene_idx_by_session: dict[str, int] = {}
    current_by_session: dict[str, Scene] = {}

    def _self_tag_for(session_id: str) -> str:
        if session_id in role_name_cache:
            role = role_name_cache[session_id]
        else:
            role = _resolve_trpg_role_name(sqlite, session_id, character_id)
            role_name_cache[session_id] = role
        if role and role != character_name:
            # `<役名@キャラ名>` 形式。`@` を残すため、各パートを sanitize して `@` で連結する
            # （_sanitize_xml_tag_name そのままだと `@` が `_` に置換されてしまう）。
            safe_role = _sanitize_xml_tag_name(role)
            safe_char = _sanitize_xml_tag_name(character_name)
            return f"{safe_role}@{safe_char}"
        return char_tag

    for t in turns:
        content = (getattr(t, "content", "") or "").strip()
        if not content:
            continue
        sid = getattr(t, "session_id", "") or ""
        raw = getattr(t, "raw_response", "") or ""
        title = getattr(t, "scenario_title", "TRPG") or "TRPG"
        # シーン境界判定（usual と同じ）
        if sid in pending_close_by_session and raw and raw != pending_close_by_session[sid]:
            scene_idx_by_session[sid] = scene_idx_by_session.get(sid, 0) + 1
            pending_close_by_session.pop(sid, None)
            current_by_session.pop(sid, None)
        if sid not in current_by_session:
            scene_idx = scene_idx_by_session.get(sid, 0)
            scene = Scene(
                scene_tag=f"TRPG「{title}」プレイログ",
                scene_key=("trpg", sid, scene_idx),
                started_at=t.created_at,
            )
            current_by_session[sid] = scene
            scenes.append(scene)
        scene = current_by_session[sid]
        stype = getattr(t, "speaker_type", "")
        speaker_id = getattr(t, "speaker_id", None)
        speaker_name = getattr(t, "speaker_name", "") or ""
        if stype == "pc" and speaker_id == character_id:
            scene.turns.append(SpeechTurn(t.created_at, _self_tag_for(sid), content))
        elif stype == "user":
            scene.turns.append(SpeechTurn(t.created_at, user_tag, content))
        elif stype == "narrator":
            scene.turns.append(SpeechTurn(t.created_at, narrator_tag, content))
        else:
            scene.turns.append(SpeechTurn(t.created_at, _sanitize_xml_tag_name(speaker_name), content))
        if _scene_close_in(raw, content):
            pending_close_by_session[sid] = raw
    return [s for s in scenes if s.turns]


# ---------------------------------------------------------------------------
# 統合 → テキスト化 → メッセージ化
# ---------------------------------------------------------------------------


def collect_all_scenes(
    sqlite,
    history: list[Any],
    self_character_id: str,
    character_name: str,
    user_label: str,
    narrator_name: str = "Narrator",
    until_dt: datetime | None = None,
) -> list[Scene]:
    """全シーン（うつつ過去+今、external 1on1/Group/TRPG）を時系列でマージして返す。

    シーンの順序は started_at 昇順。各シーン内の turns は created_at 昇順。
    """
    if until_dt is None:
        until_dt = datetime.now()
    since_dt = resolve_since_dt(sqlite, self_character_id)
    usual_scenes = _build_usual_scenes(
        history, self_character_id, character_name, user_label, narrator_name
    )
    ext_scenes: list[Scene] = []
    ext_scenes.extend(_build_1on1_scenes(sqlite, character_name, user_label, since_dt, until_dt))
    ext_scenes.extend(_build_trpg_scenes(
        sqlite, self_character_id, character_name, user_label, since_dt, until_dt, narrator_name,
    ))
    all_scenes = sorted(usual_scenes + ext_scenes, key=lambda s: s.started_at)
    return all_scenes


# scene_tag は「TRPG「ダンジョン編」プレイログ」のように、`「」` などの記号を含む
# 人間可読な見出し文字列。speaker_tag 用の厳しい _sanitize_xml_tag_name を通すと「」が
# `_` に潰れてしまうため、scene_tag 専用に「XML として真に危険な文字だけ落とす」
# 軽い置換を別途用意する。許可するもの: 漢字・かな・カナ・記号類（「」『』 等）。
# 落とすもの: `<` `>` `&` `"` `'` 空白系。
_SCENE_TAG_SANITIZE = str.maketrans({
    "<": "_",
    ">": "_",
    "&": "_",
    "\"": "_",
    "'": "_",
    " ": "_",
    "\t": "_",
    "\n": "_",
    "\r": "_",
})


def _sanitize_scene_tag(tag: str) -> str:
    """scene_tag を XML 要素名として安全に整形する（speaker より許容範囲が広い）。"""
    if not tag:
        return "scene"
    safe = tag.translate(_SCENE_TAG_SANITIZE).strip()
    return safe or "scene"


def _render_speech_line(speaker_tag: str, content: str) -> str:
    """speaker_tag は事前検証済みの前提で、content だけ XML 特殊文字をエスケープして
    `<speaker_tag>content</speaker_tag>` の 1 行に組む。

    speaker_tag は build 段で `_sanitize_xml_tag_name` を通している（または役名@キャラ名
    のように意図的に `@` を残した形）ため、render では再整形しない。
    """
    safe = _escape_xml_content((content or "").strip())
    return f"<{speaker_tag}>{safe}</{speaker_tag}>"


def render_scenes_to_text(scenes: list[Scene]) -> str:
    """Scene 列を `<history>...</history>` 形式のテキストにレンダリングする。

    各シーンを `<{scene_tag}>...</{scene_tag}>` で包み、内部の各 SpeechTurn は
    `<{speaker_tag}>{content}</{speaker_tag}>` で行を組み立てる。本文は XML 特殊
    文字をエスケープする。シーン無しの場合は空の `<history>\n</history>` を返す。

    scene_tag は build 段で `_sanitize_xml_tag_name` を通過してい**ない**生の人間可読名
    （「はるの日常」「太郎との対面」など）を保持する。XML タグとして危険な文字は
    キャラ名・ユーザ呼称・シナリオタイトルの構成上ほとんど混入しないが、念のため
    タグ書き出し時に `_sanitize_xml_tag_name` を通す。
    """
    lines: list[str] = ["<history>"]
    for scene in scenes:
        scene_open_tag = _sanitize_scene_tag(scene.scene_tag)
        lines.append(f"<{scene_open_tag}>")
        for t in scene.turns:
            lines.append(_render_speech_line(t.speaker_tag, t.content))
        lines.append(f"</{scene_open_tag}>")
    lines.append("</history>")
    return "\n".join(lines)


def build_unified_pc_messages(
    sqlite,
    history: list[Any],
    self_character_id: str,
    character_name: str,
    self_role_name: str,  # 互換のため引数は残すが本実装では未使用（TRPG役名は session ごとに解決）
    user_label: str,
    narrator_name: str = "Narrator",
    until_dt: datetime | None = None,
) -> list[dict]:
    """うつつ scenario_turns と external シーンを統合し、scene-wrap 形式の単一メッセージを返す。

    返却: `[{"role": "user", "content": "<history>...</history>"}]`。
    claude_cli は messages 長 1 のとき content を素通しでダンプするため、`<human>`
    ラッパー無しでこのテキストがそのまま LLM へ渡る。Anthropic/OpenAI API でも単一
    user メッセージとして自然に解釈される。

    シーンが 1 つも無い場合（履歴も external も空）は空リストを返す。
    """
    scenes = collect_all_scenes(
        sqlite,
        history=history,
        self_character_id=self_character_id,
        character_name=character_name,
        user_label=user_label,
        narrator_name=narrator_name,
        until_dt=until_dt,
    )
    if not scenes:
        return []
    text = render_scenes_to_text(scenes)
    return [{"role": "user", "content": text}]
