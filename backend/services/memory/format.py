"""記憶フォーマットユーティリティ。

recall_memory() の返り値を表示用テキストに変換する共通関数を提供する。
1on1チャット（api/chat.py）とグループチャット（core/group_chat/service.py）の両方から使用する。
"""

import logging


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
