"""Drifter — [DRIFT:...] / [DRIFT_RESET] マーカーの抽出と適用。

SELF_DRIFT機能: キャラクターがチャット内で自分自身に課す一時的な行動指針を管理する。
タグ抽出は tag_parser.parse_tags() に委譲し、inscriber.py と対称的な設計になっている。
"""

from ..tag_parser import parse_tags


def extract(text: str) -> tuple[str, list[str], bool]:
    """テキストから [DRIFT:...] / [DRIFT_RESET] マーカーを抽出する。

    tag_parser.parse_tags() を使用して文字単位でスキャンし、
    ネストした角括弧・バッククォートを正しく処理する。
    inscriber.py の _extract() と同様の責務を持つが、
    driftは永続化ではなくセッション内一時指針として扱う。

    Args:
        text: LLMの応答テキスト（[INSCRIBE_MEMORY:...] 除去後）。

    Returns:
        tuple:
            clean_text (str): マーカーを除去したテキスト。
            drifts (list[str]): 追加された drift 内容テキストのリスト。
            reset (bool): [DRIFT_RESET] が含まれていたか。
    """
    # multiline=True: DRIFT内容は複数行にわたる場合がある
    # 内部で長さ降順ソートされるため、列挙順は不問
    clean, matches = parse_tags(text, ["DRIFT", "DRIFT_RESET"], multiline=True)
    drifts = [m.body.strip() for m in matches["DRIFT"]]
    reset = len(matches["DRIFT_RESET"]) > 0
    return clean, drifts, reset
