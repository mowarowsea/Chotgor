"""[SCENE_CLOSE] マーカー — GM のシーン幕引き宣言の検出と本文除去。

うつつ（Usual Days）無人ループの主たる停止条件だが、GM 出力処理としては
シナリオ一般の機構なので service / usual_days / loop_strategies / pc_runner が共有する。
"""

import re

# GM がシーンの幕引きを宣言するマーカー。うつつ無人ループの主たる停止条件。
# 検出は raw_response に対して行い、表示用 content からは extract_scene_close で除去する
# （anticipator と同じく「タグは機能、本文には残さない」方針）。
_SCENE_CLOSE_MARKER = "[SCENE_CLOSE]"

# [SCENE_CLOSE] を大小・全角半角の揺れも込みで拾う正規表現（検出と本文除去の共通源）。
_SCENE_CLOSE_RE = re.compile(r"\[\s*scene[_\s]?close\s*\]", re.IGNORECASE)


def _has_scene_close(text: str | None) -> bool:
    """GM の生出力にシーン幕引きマーカー（[SCENE_CLOSE]）が含まれるか判定する。

    うつつ無人ループの主たる停止判断（判断主体は GM）。大文字小文字・区切り揺れを許容する。
    """
    if not text:
        return False
    return bool(_SCENE_CLOSE_RE.search(text))


def extract_scene_close(text: str | None) -> tuple[str, bool]:
    """テキストから [SCENE_CLOSE] マーカーを除去し、(クリーン本文, 検出フラグ) を返す。

    anticipator.extract_anticipation と同じ思想で、マーカーは「機能」として扱い、
    表示・要約に残る本文からは取り除く。検出は揺れ（大小・全半角・空白）に寛容。

    Args:
        text: GM の発話テキスト。

    Returns:
        (marker を除いた本文, 1 つ以上検出されたか) のタプル。
    """
    if not text:
        return "", False
    found = bool(_SCENE_CLOSE_RE.search(text))
    if not found:
        return text, False
    cleaned = _SCENE_CLOSE_RE.sub("", text)
    # 除去で生じた行末の余白・空行を軽く整える。
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()
    return cleaned, True
