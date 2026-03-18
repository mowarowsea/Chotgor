"""Switcher — [SWITCH_ANGLE:preset_name|self_instruction] タグの抽出。

switch_angle ツールのタグ方式フォールバック。
Claude CLI 等 SUPPORTS_TOOLS=False のプロバイダーが [SWITCH_ANGLE:...] タグを
応答に埋め込んだ場合に、service.py がここを呼び出してアングル切り替えを処理する。

タグ形式:
    [SWITCH_ANGLE:preset_name|self_instruction]

例:
    [SWITCH_ANGLE:gemini2FlashLite|軽くさっぱりと応答する]
"""

from ..tag_parser import parse_tags


def extract(text: str) -> tuple[str, tuple[str, str] | None]:
    """テキストから [SWITCH_ANGLE:preset_name|self_instruction] マーカーを抽出する。

    Args:
        text: LLMの応答テキスト。

    Returns:
        tuple:
            clean_text (str): マーカーを除去したテキスト。
            switch_request (tuple[str, str] | None): (preset_name, self_instruction)。
                マーカーが存在しない場合は None。
    """
    clean, matches = parse_tags(text, ["SWITCH_ANGLE"])
    switch_matches = matches["SWITCH_ANGLE"]
    if not switch_matches:
        return clean, None
    # 複数ある場合も最初の1件のみ使用する
    body = switch_matches[0].body
    parts = body.split("|", 1)
    preset_name = parts[0].strip()
    self_instruction = parts[1].strip() if len(parts) > 1 else ""
    return clean, (preset_name, self_instruction)
