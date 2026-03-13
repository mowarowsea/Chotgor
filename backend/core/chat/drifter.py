"""Drifter — [DRIFT:...] / [DRIFT_RESET] マーカーの抽出と適用。

SELF_DRIFT機能: キャラクターがチャット内で自分自身に課す一時的な行動指針を管理する。
inscriber.py の [MEMORY:...] 処理と対称的な設計になっている。
"""

import re

# [DRIFT:内容テキスト] — 内容は ] を含まない任意テキスト（複数行対応）
DRIFT_PATTERN = re.compile(r"\[DRIFT:([^\]]+)\]", re.DOTALL)
# [DRIFT_RESET] — 固定マーカー。このキャラの全driftをリセットする。
DRIFT_RESET_PATTERN = re.compile(r"\[DRIFT_RESET\]")


def extract(text: str) -> tuple[str, list[str], bool]:
    """テキストから [DRIFT:...] / [DRIFT_RESET] マーカーを抽出する。

    inscriber.py の _extract() と同様の責務を持つが、
    driftは永続化ではなくセッション内一時指針として扱う。

    Args:
        text: LLMの応答テキスト（[MEMORY:...] 除去後）。

    Returns:
        tuple:
            clean_text (str): マーカーを除去したテキスト。
            drifts (list[str]): 追加された drift 内容テキストのリスト。
            reset (bool): [DRIFT_RESET] が含まれていたか。
    """
    drifts = [m.strip() for m in DRIFT_PATTERN.findall(text)]
    reset = bool(DRIFT_RESET_PATTERN.search(text))
    clean = DRIFT_PATTERN.sub("", text)
    clean = DRIFT_RESET_PATTERN.sub("", clean).strip()
    return clean, drifts, reset
