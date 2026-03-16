"""Memory Inscriber — [MEMORY:...] マーカーの抽出と刻み込み。

LLMの応答テキストから記憶マーカーを読み取り、ChromaDB + SQLite に永続化する。
タグ抽出は tag_parser.parse_tags() に委譲し、ネストした角括弧・バッククォートを正しく処理する。
"""

import logging

from ..tag_parser import parse_tags
from .manager import MemoryManager

logger = logging.getLogger(__name__)

# カテゴリごとの重要度ベースマトリクス
_BASE_IMPORTANCE = {
    "contextual": {"contextual": 0.8, "semantic": 0.2, "identity": 0.1, "user": 0.1},
    "semantic":   {"contextual": 0.1, "semantic": 0.9, "identity": 0.3, "user": 0.1},
    "identity":   {"contextual": 0.2, "semantic": 0.4, "identity": 0.9, "user": 0.3},
    "user":       {"contextual": 0.3, "semantic": 0.2, "identity": 0.3, "user": 0.9},
}


def _extract(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """テキストから [MEMORY:category|impact|content] マーカーを取り出す。

    tag_parser.parse_tags() を使用して文字単位でスキャンし、
    ネストした角括弧・バッククォートを正しく処理する。

    Args:
        text: LLMの生応答テキスト。

    Returns:
        tuple:
            clean_text (str): マーカーを除去したテキスト。
            memories (list[tuple[str, str, str]]): [(category, impact_str, content)] のリスト。
    """
    clean, matches = parse_tags(text, ["MEMORY"])
    memories: list[tuple[str, str, str]] = []
    for m in matches["MEMORY"]:
        parts = m.body.split("|", 2)
        if len(parts) == 3:
            category, impact_str, content = parts
            memories.append((category.strip(), impact_str.strip(), content.strip()))
        elif len(parts) == 2:
            # impact なし形式: "category|content"
            category, content = parts
            memories.append((category.strip(), "1.0", content.strip()))
        elif len(parts) == 1 and parts[0]:
            # カテゴリのみ: フォールバック
            memories.append((parts[0].strip(), "1.0", ""))
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
            logger.exception("記憶の書き込みに失敗: category=%s content=%.50s...", category, content)

    return clean
