"""Memory Inscriber — [MEMORY:...] マーカーの抽出と刻み込み。

LLMの応答テキストから記憶マーカーを読み取り、ChromaDB + SQLite に永続化する。
"""

import re

from .manager import MemoryManager

# [MEMORY:category|impact|content] 正規形式  impact は float (例: 1.5)
MEMORY_PATTERN = re.compile(r"\[MEMORY:(\w+)\|([\d.]+)\|([^\]]+)\]", re.DOTALL)
# フォールバック: [MEMORY:category] content (パイプ/impact なし)
MEMORY_PATTERN_FALLBACK = re.compile(
    r"\[MEMORY:(\w+)\]\s*(.+?)(?=\[MEMORY:|\Z)", re.DOTALL
)

# カテゴリごとの重要度ベースマトリクス
_BASE_IMPORTANCE = {
    "contextual": {"contextual": 0.8, "semantic": 0.2, "identity": 0.1, "user": 0.1},
    "semantic":   {"contextual": 0.1, "semantic": 0.9, "identity": 0.3, "user": 0.1},
    "identity":   {"contextual": 0.2, "semantic": 0.4, "identity": 0.9, "user": 0.3},
    "user":       {"contextual": 0.3, "semantic": 0.2, "identity": 0.3, "user": 0.9},
}


def _extract(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """テキストからマーカーを取り出し (clean_text, [(category, impact_str, content)]) を返す。"""
    memories = MEMORY_PATTERN.findall(text)
    clean = MEMORY_PATTERN.sub("", text).strip()

    if not memories and "[MEMORY:" in clean:
        fb = MEMORY_PATTERN_FALLBACK.findall(clean)
        if fb:
            memories = [(cat, "1.0", content.strip()) for cat, content in fb]
            clean = MEMORY_PATTERN_FALLBACK.sub("", clean).strip()

    return clean, memories


def carve(text: str, character_id: str, memory_manager: MemoryManager) -> str:
    """LLM応答から [MEMORY:...] マーカーを読み取り、記憶として刻み込む。

    Args:
        text: LLMの生応答テキスト
        character_id: 保存先キャラクターID
        memory_manager: 記憶の書き込みを担うマネージャー

    Returns:
        マーカーを除去したクリーンなテキスト
    """
    clean, memories = _extract(text)

    for category, impact_str, content in memories:
        impact = float(impact_str) if impact_str else 1.0
        default_base = {k: 0.5 for k in ["contextual", "semantic", "identity", "user"]}
        base = _BASE_IMPORTANCE.get(category, default_base)
        # impact係数をそのまま掛ける (1.0を超える値も許容)
        scores = {f"{k}_importance": (v * impact) for k, v in base.items()}
        try:
            memory_manager.write_memory(
                character_id=character_id,
                content=content.strip(),
                category=category.strip(),
                **scores,
            )
        except Exception:
            pass

    return clean
