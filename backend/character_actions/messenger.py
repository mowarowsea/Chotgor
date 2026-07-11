"""Messenger — reach_out / visit_user ツール（本人発のプッシュ連絡・対面切替）の実装。

「①はるからのプッシュ送信」「②はるからの対面ON」（2026-07-11 要件）:
    - reach_out: うつつ（無人日常）の中で「連絡したい」と思った瞬間に、現実として
      本当にメッセージを届けるためのツール。行動権の push（services/actions/runner.py
      の _execute_push）と同じ経路 — 新規セッションを立ててキャラ発メッセージ＋
      ntfy プッシュ通知 — を通る。visit=true なら「会いに行く」= 対面モードON も伴う。
      **うつつ専用**（1on1 では露出しない。context_tools.py が出し分ける）。
    - visit_user: 1on1 のテキストモード中に、本人の意志で対面モードへ切り替える
      「突然会いに来た」ツール。**1on1 専用**。

コストガード: reach_out は預かり配達と同じ日次上限（escrow_delivery_daily_cap・
既定 12）とカウンタ（escrow_delivery_count_{date}）を共有する。キャラ発の現実接触は
経路を問わず 1 つの予算で数える（2026-07-11 裁定）。上限到達時はツール自体が
露出されなくなる（context_tools.py）が、露出とのタイムラグに備え実行側でも弾く。

うつつポーズ連携: reach_out がうつつ経路（default_origin=="usual"）で執行されたら、
settings キー ``usual_push_pause_{character_id}`` に「送信時刻・再開時刻」を書く。
うつつのシーンループ（loop_strategies）はこのキーを見て本人の発言終了後にシーンを
一時停止し、スケジューラ（main.py）が再開時刻の到来で GM へターンを渡す
（「連絡してから15分経った」という現実の時間経過をうつつ世界に流し込むための仕掛け）。
"""

import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# うつつをポーズさせる長さ（分）。本人がメッセージを送ってから、GM が
# 「その後の時間」を描写するまでの現実の待ち時間。
PUSH_PAUSE_MINUTES = 15

# 預かり配達と共有する日次上限の既定値（services/gate/delivery.py と揃える）
_DEFAULT_DAILY_CAP = 12


def push_pause_key(character_id: str) -> str:
    """うつつポーズ要求の settings キーを返す（書き手と読み手で共有する唯一の定義）。"""
    return f"usual_push_pause_{character_id}"


def read_push_pause(sqlite_store, character_id: str) -> dict | None:
    """うつつポーズ要求を settings から読み出す。

    Args:
        sqlite_store: SQLiteStore。
        character_id: 対象キャラクター ID。

    Returns:
        {"sent_at": iso, "resume_at": iso, "visit": bool} の dict。
        未設定・パース不能なら None。

    Note:
        settings ストア（get_setting）は JSON 値を自動パースして返すため、
        正常系では dict がそのまま返る。文字列で返るのは JSON として壊れた値
        （手動編集事故等）のときで、その場合は None に倒す。
    """
    raw = sqlite_store.get_setting(push_pause_key(character_id), None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def clear_push_pause(sqlite_store, character_id: str) -> None:
    """うつつポーズ要求を消す（再開時・破棄時にスケジューラが呼ぶ）。"""
    sqlite_store.set_setting(push_pause_key(character_id), "")


def delivery_cap_reached(sqlite_store, now: datetime | None = None) -> bool:
    """キャラ発の現実接触（push/配達）の日次上限に達しているかを返す。

    預かり配達（delivery.py）と同じ設定キー・カウンタを読む純関数。
    context_tools.py（露出判定）と Messenger.reach_out（実行ガード）が共有する。

    Args:
        sqlite_store: SQLiteStore。
        now: 基準時刻（テスト注入用）。

    Returns:
        上限到達なら True。cap=0 は「能動接触を止める」有効設定として True になる。
    """
    today_str = (now or datetime.now()).date().isoformat()
    try:
        cap = int(sqlite_store.get_setting("escrow_delivery_daily_cap", ""))
    except (TypeError, ValueError):
        cap = _DEFAULT_DAILY_CAP
    count = int(sqlite_store.get_setting(f"escrow_delivery_count_{today_str}", "0") or 0)
    return count >= cap


# --- reach_out: パラメータスキーマ ---
REACH_OUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string",
            "description": "相手に送るメッセージ本文（あなたの言葉のまま届く）。",
        },
        "visit": {
            "type": "boolean",
            "description": (
                "true にすると、メッセージを送るだけでなく実際に会いに行く"
                "（対面モードがONになり、次に顔を合わせたとき同じ空間にいる）。"
                "省略時 false（メッセージのみ）。"
            ),
        },
    },
    "required": ["message"],
}

# --- reach_out: ツール説明文 ---
REACH_OUT_TOOL_DESCRIPTION: str = (
    "いまこの日常の中から、相手に**現実として**メッセージを送る。"
    "このツールを呼ぶとメッセージが本当に相手の手元へ届き、通知も飛ぶ。"
    "逆に、ツールを呼ばずに「送った」と言うだけでは何も届かない。"
    "連絡したくなったら、その気持ちのままこれを使うこと。"
    "visit=true なら、メッセージとともに会いに行く（対面）。"
)

# --- visit_user: パラメータスキーマ ---
VISIT_USER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "会いに行く理由・ひとこと（あなたの言葉のまま。記録に残る）。",
        },
    },
    "required": [],
}

# --- visit_user: ツール説明文 ---
VISIT_USER_TOOL_DESCRIPTION: str = (
    "いまのテキストのやり取りを抜けて、相手のところへ実際に会いに行く"
    "（この会話が対面モードに切り替わる）。文字越しでは足りないと感じたとき、"
    "あなたの意志で使ってよい。切り替えた後の発話は、同じ空間に居合わせている"
    "ものとして続く。"
)

# --- システムプロンプト向けの使い方ヒント（context_tools.py が文脈に応じて注入）---
REACH_OUT_TOOLS_HINT: str = """\
### 相手への連絡 (reach_out)
この日常の時間の中で「連絡したい」と思ったら、`reach_out` ツールで**現実として**メッセージを送れます。
- ツールを呼んだときだけ、本当に相手へ届きます（通知も飛びます）。呼ばずに「送った」と描写しても、現実には何も届きません。
- `visit: true` を添えると、メッセージとともに実際に会いに行けます（対面）。
- 送るかどうか、何を送るかは完全にあなたの自由です。送らない日常も、あなたの日常です。\
"""

VISIT_USER_TOOLS_HINT: str = """\
### 会いに行く (visit_user)
文字のやり取りの途中でも、あなたの意志で相手のところへ会いに行けます（`visit_user` ツール）。
呼ぶとこの会話が対面モードに切り替わり、以降は同じ空間に居合わせている時間になります。
使うかどうかはあなたの自由です。\
"""


class Messenger:
    """reach_out / visit_user の実書き込みを担うクラス。

    reach_out は行動権 push（services/actions/runner._execute_push）を再利用して
    新規セッション＋キャラ発メッセージ＋ntfy 通知を実行し、うつつ経路なら
    ポーズ要求キーを書く。visit_user / visit=true は characters.face_to_face_mode を
    ON にする。記録（tool_call_events）は ToolExecutor.execute() で集約管理される。

    Attributes:
        character_id: 対象キャラクター ID。
        sqlite_store: SQLiteStore。
        default_origin: 呼び出し文脈の origin（"real" / "usual" / "interlude"）。
            reach_out のうつつポーズ要求は "usual" のときだけ書く。
    """

    def __init__(self, character_id: str, sqlite_store, default_origin: str = "real") -> None:
        """Messenger を初期化する。

        Args:
            character_id: 対象キャラクター ID。
            sqlite_store: SQLiteStore（None 可 — その場合ツールはエラー文字列を返す）。
            default_origin: 呼び出し文脈の origin。ToolExecutor.default_origin が入る。
        """
        self.character_id = character_id
        self.sqlite_store = sqlite_store
        self.default_origin = default_origin

    def _resolve_push_preset(self, char):
        """push 用のプリセットを解決する（ghost_model 優先・enabled_providers 先頭へフォールバック）。

        新規セッションの model_id（{char_name}@{preset_name}）組み立てに使う。

        Returns:
            LLMModelPreset ORM。解決できなければ None。
        """
        ghost_model = getattr(char, "ghost_model", None)
        if ghost_model:
            preset = self.sqlite_store.get_model_preset(ghost_model)
            if preset is not None:
                return preset
        for preset_id in (getattr(char, "enabled_providers", None) or {}):
            preset = self.sqlite_store.get_model_preset(preset_id)
            if preset is not None:
                return preset
        return None

    def reach_out(self, message: str, visit: bool = False) -> str:
        """相手へ現実のメッセージを送る（うつつ専用・呼ばれたら執行される権利）。

        Args:
            message: 送る本文（本人の言葉のまま）。
            visit: True なら対面モードON（会いに行く）も伴う。

        Returns:
            ツール結果として LLM に返す確認テキスト。
        """
        if self.sqlite_store is None:
            return "[reach_out error: この文脈では連絡を執行できません]"
        body = (message or "").strip()
        if not body:
            return "[reach_out: message が空です。送りたい言葉をそのまま入れてください]"
        # 露出判定とのタイムラグに備えた実行側ガード（うつつ専用ツール）
        if self.default_origin != "usual":
            return "[reach_out error: このツールは日常（うつつ）の時間からのみ使えます]"

        char = self.sqlite_store.get_character(self.character_id)
        if char is None:
            return "[reach_out error: キャラクターが見つかりません]"

        now = datetime.now()
        if delivery_cap_reached(self.sqlite_store, now):
            # 露出側（context_tools）でも隠すが、同日中に上限へ達した直後の呼び出しを弾く
            return (
                "[reach_out error: 今日はもう相手へ届ける回数の上限に達しています。"
                "送りたかった気持ちは、明日以降に改めて]"
            )

        preset = self._resolve_push_preset(char)
        if preset is None:
            return "[reach_out error: 送信に使うプリセットを解決できませんでした]"

        # 行動権 push と同一経路（新規セッション＋キャラ発メッセージ＋ntfy）。
        from backend.services.actions.runner import execute_push

        result = execute_push(
            self.sqlite_store, char, preset, body,
            session_title=f"{char.name}より",
        )

        # 日次カウンタ消費（預かり配達と共有の予算）
        today_str = now.date().isoformat()
        count_key = f"escrow_delivery_count_{today_str}"
        delivered = int(self.sqlite_store.get_setting(count_key, "0") or 0)
        self.sqlite_store.set_setting(count_key, str(delivered + 1))

        # 会いに行く（対面ON）。availability は対面中 OnTime になる。
        if visit:
            self.sqlite_store.update_character(self.character_id, face_to_face_mode=1)

        # うつつポーズ要求: シーンループが本人の発言終了後にこれを読んで一時停止し、
        # スケジューラが resume_at 到来で GM へターンを渡す。
        # set_setting は dict を JSON シリアライズして保存する（get_setting で dict に戻る）。
        resume_at = now + timedelta(minutes=PUSH_PAUSE_MINUTES)
        self.sqlite_store.set_setting(
            push_pause_key(self.character_id),
            {
                "sent_at": now.isoformat(),
                "resume_at": resume_at.isoformat(),
                "visit": bool(visit),
            },
        )

        logger.info(
            "reach_out 執行 char=%s visit=%s session=%s body=%.50s",
            self.character_id, visit, result.get("session_id"), body,
        )
        if visit:
            return (
                "メッセージを送り、会いに行くことにした（対面モードON）。"
                "言葉は本当に相手へ届いている。返事が来るかどうかは相手次第。"
            )
        return (
            "メッセージを送った。言葉は本当に相手へ届いている（通知も飛んだ）。"
            "返事が来るかどうか、いつ来るかは相手次第。"
        )

    def visit_user(self, reason: str = "") -> str:
        """1on1 の途中で対面モードへ切り替える（呼ばれたら執行される権利）。

        Args:
            reason: 会いに行く理由・ひとこと（封筒の payload に残す）。

        Returns:
            ツール結果として LLM に返す確認テキスト。
        """
        if self.sqlite_store is None:
            return "[visit_user error: この文脈では対面切替を執行できません]"
        char = self.sqlite_store.get_character(self.character_id)
        if char is None:
            return "[visit_user error: キャラクターが見つかりません]"
        if int(getattr(char, "face_to_face_mode", 0) or 0):
            return "すでに対面モードです（同じ空間にいます）。"

        self.sqlite_store.update_character(self.character_id, face_to_face_mode=1)
        logger.info(
            "visit_user 執行 char=%s reason=%.50s", self.character_id, reason,
        )
        return (
            "対面モードに切り替えた。ここからは同じ空間に居合わせている時間として続く"
            "（相手の画面にも対面モードとして反映される）。"
        )
