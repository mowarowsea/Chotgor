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
