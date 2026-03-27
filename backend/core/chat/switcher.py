"""Switcher — [SWITCH_ANGLE:preset_name|self_instruction] タグの抽出とアングル切り替え。

Switcher クラスと関連定数を一元管理する。
- SWITCH_ANGLE_SCHEMA: switch_angle ツール呼び出し方式のパラメータ JSON スキーマ
- SWITCH_ANGLE_TOOL_DESCRIPTION: tool-use プロバイダー向けツール説明文
- SWITCH_ANGLE_TOOLS_HINT: tool-use プロバイダー向けシステムプロンプト案内ブロック
- SWITCH_ANGLE_TAG_GUIDE: タグ方式プロバイダー向けガイド文
- Switcher クラス: タグ方式（switch_from_text）とツール呼び出し方式（switch_angle）の両方に対応

switch_angle ツールのタグ方式フォールバック。
Claude CLI 等 SUPPORTS_TOOLS=False のプロバイダーが [SWITCH_ANGLE:...] タグを
応答に埋め込んだ場合に、service.py がここを呼び出してアングル切り替えを処理する。

タグ形式:
    [SWITCH_ANGLE:preset_name|self_instruction]

例:
    [SWITCH_ANGLE:gemini2FlashLite|軽くさっぱりと応答する]
"""

import logging

from ..tag_parser import parse_tags

logger = logging.getLogger(__name__)

# --- ツール呼び出し方式: パラメータスキーマ ---
SWITCH_ANGLE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "preset_name": {
            "type": "string",
            "description": "切り替え先のプリセット名。利用可能なプリセットはシステムプロンプトに記載されている。",
        },
        "self_instruction": {
            "type": "string",
            "description": "切り替え後のプリセットに渡す自己指針テキスト。どのように応答するかを一言で。",
        },
    },
    "required": ["preset_name", "self_instruction"],
}

# --- ツール呼び出し方式: ツール説明文（ANTHROPIC_TOOLS / OPENAI_TOOLS に使用）---
SWITCH_ANGLE_TOOL_DESCRIPTION: str = (
    "プリセット（モデル）を切り替える。キャラクターは変わらない。"
    "切り替え後のモデルがこの会話に改めて応答する。"
    "利用可能なプリセットと切り替えタイミングはシステムプロンプトに記載されている。"
)

# --- ツール呼び出し方式: システムプロンプト案内ブロック（CHOTGOR_TOOLS_BLOCK に使用）---
SWITCH_ANGLE_TOOLS_HINT: str = """\
### プリセット（モデル）切り替え
`switch_angle` ツールで別のプリセットに切り替えられます。
切り替え後のモデルがこの会話に改めて応答します。キャラクターは変わりません。
利用可能なプリセットと切り替えタイミングはシステムプロンプトに記載されています。"""

# --- タグ方式: ガイド文（CHOTGOR_BLOCK3_TEMPLATE に使用）---
SWITCH_ANGLE_TAG_GUIDE: str = """\
### プリセット（モデル）切り替え

別のプリセット（モデル）に切り替えることができます。
切り替え後のモデルがこの会話に改めて応答します。キャラクターは変わりません。

返答の**一番最後に**以下の形式で記述してください：

    [SWITCH_ANGLE:preset_name|self_instruction]

- `preset_name`: 切り替え先のプリセット名（システムプロンプトに記載）
- `self_instruction`: 切り替え後のモデルに渡す一言指針
- `[SWITCH_ANGLE:...]` の行はユーザーには見えません"""


def _extract(text: str) -> tuple[str, tuple[str, str] | None]:
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


class Switcher:
    """アングル切り替えリクエストの抽出・記録を担うクラス。タグ方式・ツール呼び出し方式の両方に対応する。

    タグ方式:
        LLM応答から [SWITCH_ANGLE:preset_name|self_instruction] マーカーを抽出して switch_request に格納する。
        Claude CLI や Ollama など tool-use 非対応プロバイダーが使用する。
        switch_from_text(text) を呼ぶ。

    ツール呼び出し方式:
        preset_name / self_instruction を直接受け取って switch_request に格納する。
        Anthropic API・OpenAI API など tool-use 対応プロバイダーが使用する。
        switch_angle(preset_name, self_instruction) を呼ぶ。

    実際のアングル切り替え処理（再ディスパッチ）は service.py が担う。
    Switcher は switch_request に切り替えリクエストを格納するだけ。

    Attributes:
        switch_request: 切り替えリクエスト (preset_name, self_instruction)。未要求の場合は None。
    """

    def __init__(self) -> None:
        """Switcher を初期化する。"""
        self.switch_request: tuple[str, str] | None = None

    def switch_from_text(self, text: str) -> str:
        """LLM応答から [SWITCH_ANGLE:...] マーカーを読み取り、switch_request に格納する（タグ方式）。

        Args:
            text: LLMの生応答テキスト。

        Returns:
            マーカーを除去したクリーンなテキスト。
        """
        clean, switch_request = _extract(text)
        if switch_request:
            self.switch_request = switch_request
            preset_name, self_instruction = switch_request
            logger.info(
                "タグ方式 preset=%s instruction=%.50s",
                preset_name, self_instruction,
            )
        return clean

    def switch_angle(self, preset_name: str, self_instruction: str) -> str:
        """アングル切り替えリクエストを記録する（ツール呼び出し方式）。

        実際の切り替え処理（再ディスパッチ）は service.py が担う。
        ここでは switch_request に格納するだけ。

        Args:
            preset_name: 切り替え先のプリセット名。
            self_instruction: 切り替え後のプリセットに渡す自己指針テキスト。

        Returns:
            実行結果メッセージ。
        """
        self.switch_request = (preset_name, self_instruction)
        logger.info(
            "ツール方式 preset=%s instruction=%.50s",
            preset_name, self_instruction,
        )
        return f"アングルを {preset_name} に切り替えます。"
