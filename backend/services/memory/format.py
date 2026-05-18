"""記憶フォーマットユーティリティ。

recall_memory() の返り値を表示用テキストに変換する共通関数を提供する。
1on1チャット（api/chat.py）とグループチャット（core/group_chat/service.py）の両方から使用する。
"""


def format_recalled_memories(recalled: list) -> str:
    """想起した記憶リストを reasoning / SSE 表示用テキストにフォーマットする。

    Args:
        recalled: recall_memory() が返す記憶辞書のリスト。
                  各要素は {"content": str, "metadata": dict, "hybrid_score": float} を想定。

    Returns:
        人間が読みやすい形式の文字列。記憶がなければ空文字列。
    """
    if not recalled:
        return ""
    lines = []
    for mem in recalled:
        category = mem.get("metadata", {}).get("category") or "general"
        # content に改行が含まれると行単位パースが壊れるため、スペースに置換して1行に収める
        content = mem.get("content", "").replace("\n", " ")
        score = mem.get("hybrid_score", 0.0)
        lines.append(f"[{category}] {content}  (score: {score:.2f})")
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
                 各要素は {"type", "summary", "atmosphere", "latest_post"} を想定。

    Returns:
        人間が読みやすい形式の文字列。スレッドがなければ空文字列。
    """
    if not threads:
        return ""
    lines = []
    for t in threads:
        type_ = t.get("type", "")
        summary = (t.get("summary", "") or "").replace("\n", " ")
        atmo = (t.get("atmosphere", "") or "").replace("\n", " ")
        latest = (t.get("latest_post") or "").replace("\n", " ")
        line = f"{_THREAD_LINE_PREFIX} [{type_}] {summary}"
        if atmo:
            line += f" 〈{atmo}〉"
        if latest:
            line += f" → {latest}"
        lines.append(line)
    return "\n".join(lines) + "\n"
