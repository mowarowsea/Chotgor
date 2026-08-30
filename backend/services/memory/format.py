"""記憶フォーマットユーティリティ。

recall_memory() の返り値を表示用テキストに変換する共通関数を提供する。
1on1 チャット（api/chat.py）とシナリオ PC モード（services/scenario_chat/pc_runner.py）の両方から使用する。
"""

import logging
from datetime import datetime

from backend.lib.time_awareness import japanese_weekday


logger = logging.getLogger(__name__)


# origin 別の前置きラベル。recall された記憶 / スレッドの由来が「通常チャット (real)」
# 以外なら、TRPG・うつつから持ち越した記憶であることを明示する。
# 「卓を囲んだ友達が TRPG の記憶を覚えていない方が不自然」という立て付けから、
# 通常チャットでもこれらの記憶を recall して言及できる前提。ただしキャラ本人が
# 「これは現実か遊びか」を取り違えないよう、表示時にラベルで区別する。
_ORIGIN_LABEL = {
    "real": "",
    "usual": "[うつつでの記憶] ",
    "interlude": "[TRPGでの記憶] ",
}


def origin_label_prefix(origin: str | None) -> str:
    """origin 値（real / usual / interlude）を行頭ラベルに変換する公開ヘルパ。

    recall 表示・power_recall ツールの整形・Chronicle 棚卸し WM スレッド表示など、
    origin に応じた行頭ラベルが要る全経路から呼ばれる。未知値は real 扱いに
    フォールバックするが、データ整合性監視のため logger.warning を出す（findings #12）。
    """
    if origin is None or origin == "":
        return _ORIGIN_LABEL["real"]
    label = _ORIGIN_LABEL.get(origin)
    if label is None:
        # 未知値はラベル無しで通すが、データ整合性バグの兆候として記録する。
        logger.warning("未知の origin 値=%r real 扱いにフォールバック", origin)
        return _ORIGIN_LABEL["real"]
    return label


def short_thread_id(thread_id: str) -> str:
    """ワーキングメモリスレッド ID の短縮表記（先頭8桁）を返す。

    プロンプトへのスレッド一覧注入はトークン節約のためフル UUID を出さない。
    キャラクターが短縮 ID をツールへ渡したときは
    WorkingMemoryManager.resolve_thread_id() が前方一致でフル ID に解決する。
    """
    return (thread_id or "")[:8]


def short_date(value, *, with_weekday: bool = True) -> str:
    """日時を記憶表示用の短い絶対日付へ整形する（取れなければ空文字）。

    プロンプトへ出す日付は「いつのことか」を本人が判断できる最小限に絞る:
    同じ年なら ``MM-DD(曜)``、年をまたぐものだけ ``YYYY-MM-DD(曜)``。
    曜日を添えるのは「今週」「先週」のような週単位の言い回しを本人が
    絶対日付へ結び直せるようにするため（お盆休みのような期間限定の事実が、
    日付の無いまま恒常的な状態として残り続ける事故への対策）。

    Args:
        value: ISO 8601 文字列または datetime。None・パース不能なら空文字を返す。
        with_weekday: False なら曜日を付けない。

    Returns:
        整形済み日付文字列。取れなければ空文字。
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).strip())
        except (TypeError, ValueError):
            return ""
    fmt = "%m-%d" if dt.year == datetime.now().year else "%Y-%m-%d"
    text = dt.strftime(fmt)
    if with_weekday:
        text += f"({japanese_weekday(dt)})"
    return text


def format_recalled_memories(recalled: list) -> str:
    """想起した記憶リストを reasoning / SSE 表示用テキストにフォーマットする。

    Args:
        recalled: recall_memory() が返す記憶辞書のリスト。
                  各要素は {"content": str, "metadata": dict, "hybrid_score": float} を想定。
                  metadata.origin が "usual" / "interlude" の場合は、行頭に由来ラベルを付与する。

    Returns:
        人間が読みやすい形式の文字列。記憶がなければ空文字列。
    """
    if not recalled:
        return ""
    lines = []
    for mem in recalled:
        meta = mem.get("metadata") or {}
        category = meta.get("category") or "general"
        origin = meta.get("origin")
        # content に改行が含まれると行単位パースが壊れるため、スペースに置換して1行に収める
        content = mem.get("content", "").replace("\n", " ")
        score = mem.get("hybrid_score", 0.0)
        lines.append(f"{origin_label_prefix(origin)}[{category}] {content}  (score: {score:.2f})")
    return "\n".join(lines) + "\n"


# ワーキングメモリスレッド行の先頭マーカー。フロントエンドはこの接頭辞で
# 「想起したスレッド」行を識別し、専用セクションに振り分ける。
_THREAD_LINE_PREFIX = "⟦thread⟧"


def format_recalled_threads(threads: list) -> str:
    """heat 想起したワーキングメモリスレッドを reasoning / SSE 表示用テキストに整形する。

    1スレッド = 1行（行単位パースが壊れないよう改行はスペースに置換）。
    フロントエンドは行頭マーカー ``⟦thread⟧`` でスレッド行を識別する。

    Args:
        threads: WorkingMemoryManager.recall_threads() が返すスレッド辞書のリスト。
                 各要素は {"type", "summary", "atmosphere_tag", "latest_post"} を想定。

    Returns:
        人間が読みやすい形式の文字列。スレッドがなければ空文字列。
    """
    if not threads:
        return ""
    lines = []
    for t in threads:
        type_ = t.get("type", "")
        origin = t.get("origin")
        summary = (t.get("summary", "") or "").replace("\n", " ")
        atmo = (t.get("atmosphere_tag", "") or "").replace("\n", " ")
        latest = (t.get("latest_post") or "").replace("\n", " ")
        # マーカー（フロント識別用）の直後に origin ラベルを差し込む。
        # 記憶側 (format_recalled_memories) と表記を揃え、TRPG/うつつ由来のスレッドを
        # 通常チャットの現実と取り違えないようにする。
        line = f"{_THREAD_LINE_PREFIX} {origin_label_prefix(origin)}[{type_}] {summary}"
        if atmo:
            line += f" 〈{atmo}〉"
        if latest:
            line += f" → {latest}"
        lines.append(line)
    return "\n".join(lines) + "\n"
