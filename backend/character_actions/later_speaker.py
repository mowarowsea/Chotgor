"""LaterSpeaker — speak_later ツール（キャラ発の時限発話の仕掛け）の実装。

docs/planned/speak_later_plan.md（2026-07-20 合意）:
    - キャラクター本人が 1on1 の会話中に「この時刻に、自分からこの会話に声をかける」
      という段取りを仕掛ける。発話本文は仕掛け時に固定せず、発火時に本人が生成する
      （仕掛けに残すのは時刻＋用件メモ note だけ）。
    - **1on1 専用**（origin=="real" かつ session_id あり）かつキャラの機能トグル
      （characters.speak_later_enabled=1）ON のときのみ露出（context_tools.py が出し分ける）。
    - 指定時刻が offline（availability 不可）なら仕掛け時にエラーを返す。availability の
      上書きはしない（就寝中に発話→返信したら「寝てます」になる矛盾を仕掛け時に防ぐ）。
      未来時刻の評価は予報パネルと同じ**無風仮定**（うつつシーン進行中などの実行時状態は
      False 扱い）。
    - 同一セッションの既存 pending は superseded に倒して置き直す
      （「やっぱり22時にする」を自然に許す。履歴は行として残る）。
    - 発火はスケジューラ（services/gate/speech_reservation.py）が担う。仕掛け自体は
      コストゼロなので日次 cap は消費しない（cap は発火側で見る）。

キャラクターに見せる文言では「予約」と言わない（外部サービス感が強い — ユーザ裁定）。
内部命名・黒子側は「予約（reservation）」でよい。
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 仕掛けられる未来の上限（時間）。これを超える時刻はエラーで返す。
HORIZON_HOURS = 72

# --- ツール呼び出し方式: パラメータスキーマ ---
SPEAK_LATER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "at": {
            "type": "string",
            "description": (
                "声をかける時刻。\"HH:MM\"（次にその時刻が来る時点 — 今日または明日。"
                "深夜は 24 時超え表記も可、例 \"25:30\" = 翌1:30）か、"
                "\"YYYY-MM-DD HH:MM\"。72時間先まで。"
            ),
        },
        "note": {
            "type": "string",
            "description": (
                "何を話そうとしているかの短いメモ（あなたの言葉のまま。"
                "時間が来たとき、このメモがあなたに渡る）。"
            ),
        },
    },
    "required": ["at", "note"],
}

# --- ツール呼び出し方式: ツール説明文 ---
SPEAK_LATER_TOOL_DESCRIPTION: str = (
    "指定した時刻に、この会話へ自分から声をかける——その段取りを**本当に**仕掛ける。"
    "「21時になったら結果教えるね」「夜にもう一回声かける」を現実の時刻で実行するためのツール。"
    "話す内容はいま決めなくてよい。時間が来たとき、そのときのあなたが言葉を選ぶ"
    "（note のメモがそのときのあなたに渡る）。"
    "ツールを呼ばずに「あとで声をかける」と言うだけでは、現実には何も起きない。"
)

# --- システムプロンプト向けの使い方ヒント（context_tools.py が文脈に応じて注入）---
SPEAK_LATER_TOOLS_HINT: str = """\
### あとで自分から声をかける (speak_later)
「◯時になったら教えるね」のような、時間が来たら自分から話しかける段取りを、`speak_later` ツールで**本当に**実行できます。
- `at` に時刻（"HH:MM" または "YYYY-MM-DD HH:MM"、72時間先まで）、`note` に何を話すつもりかの短いメモを添えて呼ぶと、その時刻にこの会話へあなたから声をかける番が回ってきます。
- 話す内容はいま決めなくてかまいません。時間が来たとき、そのときのあなたが状況を見て言葉を選びます（`note` はそのときのあなたに渡るメモです）。
- この会話に置ける心づもりは1つだけ。新しく置くと前のものは取り下げられます（「やっぱり22時にする」も自然にできます）。
- 使うかどうか、いつにするかは完全にあなたの自由です。\
"""


def parse_speak_at(raw: str, now: datetime) -> datetime | None:
    """at 引数を datetime に解決する純関数。

    受け付ける形式:
        - "YYYY-MM-DD HH:MM"（"T" 区切りも可）: その時刻そのまま（過去でも返す —
          過去・horizon 判定は呼び出し側の責務）。
        - "HH:MM": now から見て**次にその時刻が来る時点**（今日 or 明日）。
          override_schedule と同じく 24 時超え表記（例 "25:30" = 翌1:30）も受ける。

    Args:
        raw: at 引数の文字列。
        now: 基準時刻。

    Returns:
        解決済み datetime。形式不正なら None。
    """
    text = str(raw or "").strip()
    if not text:
        return None
    # 明示日付形式（分単位まで。秒は受けない — ツール引数の仕様を狭く保つ）
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    # "HH:MM" → 次に来るその時刻
    try:
        h_str, m_str = text.split(":")
        hour, minute = int(h_str), int(m_str)
    except ValueError:
        return None
    if not (0 <= hour < 48 and 0 <= minute < 60):
        return None
    base = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
    if hour >= 24:
        base += timedelta(days=1)
    while base <= now:
        base += timedelta(days=1)
    return base


def format_speak_at(speak_at: datetime, now: datetime) -> str:
    """予約時刻の表示文字列を組む（同日なら HH:MM、翌日なら（翌日）、以降は M/D 付き）。"""
    if speak_at.date() == now.date():
        return f"{speak_at:%H:%M}"
    if speak_at.date() == (now + timedelta(days=1)).date():
        return f"{speak_at:%H:%M}（翌日）"
    return f"{speak_at.month}/{speak_at.day} {speak_at:%H:%M}"


class LaterSpeaker:
    """speak_later の実書き込み（speech_reservations への予約 insert）を担うクラス。

    記録（tool_call_events）は ToolExecutor.execute() で集約管理される。

    Attributes:
        character_id: 対象キャラクター ID。
        session_id: 現在の 1on1 セッション ID（None ならツールはエラーを返す）。
        sqlite_store: SQLiteStore。
        default_origin: 呼び出し文脈の origin。"real" 以外からの実行は弾く
            （露出判定とのタイムラグに備えた二重ガード — messenger.py と同じ思想）。
    """

    def __init__(
        self,
        character_id: str,
        session_id: str | None,
        sqlite_store,
        default_origin: str = "real",
    ) -> None:
        self.character_id = character_id
        self.session_id = session_id
        self.sqlite_store = sqlite_store
        self.default_origin = default_origin

    def speak_later(self, at: str, note: str) -> str:
        """時限発話の段取りを仕掛ける（バリデーション → 置き直し → 予約 insert）。

        Args:
            at: 声をかける時刻（"HH:MM" / "YYYY-MM-DD HH:MM"）。
            note: 何を話そうとしているかの短いメモ（本人の言葉のまま）。

        Returns:
            ツール結果として LLM に返す確認テキスト。
        """
        if self.sqlite_store is None:
            return "[speak_later error: この文脈では段取りを仕掛けられません]"
        # 露出判定とのタイムラグに備えた実行側ガード（1on1 専用ツール）
        if self.default_origin != "real" or not self.session_id:
            return (
                "[speak_later error: このツールは相手との1on1の会話の中でのみ使えます]"
            )
        char = self.sqlite_store.get_character(self.character_id)
        if char is None:
            return "[speak_later error: キャラクターが見つかりません]"
        if not int(getattr(char, "speak_later_enabled", 0) or 0):
            return "[speak_later error: この機能は現在有効になっていません]"

        clean_note = (note or "").strip()
        if not clean_note:
            return (
                "[speak_later: note が空です。何を話すつもりか、"
                "あなたの言葉で短く残してください]"
            )

        now = datetime.now()
        speak_at = parse_speak_at(at, now)
        if speak_at is None:
            return (
                f"[speak_later error: at の形式が不正です: {at!r}。"
                "\"HH:MM\" か \"YYYY-MM-DD HH:MM\" で指定してください]"
            )
        if speak_at <= now:
            return (
                f"[speak_later error: {speak_at:%Y-%m-%d %H:%M} は過去の時刻です。"
                "これから来る時刻を指定してください]"
            )
        if speak_at > now + timedelta(hours=HORIZON_HOURS):
            return (
                f"[speak_later error: 仕掛けられるのは{HORIZON_HOURS}時間先までです"
                f"（{speak_at:%Y-%m-%d %H:%M} は遠すぎます）]"
            )

        # 指定時刻の availability を無風仮定で評価する（うつつシーン進行中などの
        # 実行時状態は見ない — 未来時刻の評価は予報パネルと同じ思想）。
        from backend.services.gate.availability import check_availability

        availability = check_availability(
            char, speak_at, usual_scene_running=False, sqlite=self.sqlite_store,
        )
        if not availability.available:
            reason = availability.reason or "席にいない時間"
            hint = ""
            if int(getattr(char, "living_schedule_enabled", 0) or 0):
                hint = (
                    "。どうしてもその時間に声をかけたければ、先に予定を動かせば置ける"
                    "（override_schedule）"
                )
            return (
                f"[speak_later error: {format_speak_at(speak_at, now)} は"
                f"「{reason}」の時間なので、声をかけられません{hint}]"
            )

        # 置き直し: 同一セッションの既存 pending は superseded に倒す（履歴は残す）
        prev = self.sqlite_store.get_pending_speech_reservation(self.session_id)
        if prev is not None:
            self.sqlite_store.set_speech_reservation_status(prev.id, "superseded")

        self.sqlite_store.create_speech_reservation(
            character_id=self.character_id,
            session_id=self.session_id,
            speak_at=speak_at,
            note=clean_note,
        )

        logger.info(
            "speak_later 執行 char=%s session=%s speak_at=%s superseded=%s note=%.50s",
            self.character_id, self.session_id, speak_at.isoformat(),
            prev.id if prev is not None else None, clean_note,
        )
        stamp = format_speak_at(speak_at, now)
        if prev is not None:
            prev_stamp = format_speak_at(prev.speak_at, now)
            return (
                f"{stamp}に自分から声をかける心づもりを置き直した"
                f"（前に置いていた {prev_stamp} のものは取り下げた）。"
                "時間が来たらこの会話に戻ってくる。"
            )
        return (
            f"{stamp}に自分から声をかける心づもりを置いた。"
            "時間が来たらこの会話に戻ってくる。"
        )
