"""Switcher — [SWITCH_ANGLE:preset_name|self_instruction] タグの抽出とアングル切り替え。

Switcher クラスと関連定数を一元管理する。
- SWITCH_ANGLE_SCHEMA: switch_angle ツール呼び出し方式のパラメータ JSON スキーマ
- SWITCH_ANGLE_TOOL_DESCRIPTION: tool-use プロバイダー向けツール説明文
- SWITCH_ANGLE_TOOLS_HINT: tool-use プロバイダー向けシステムプロンプト案内ブロック
- SWITCH_ANGLE_TAG_GUIDE: タグ方式プロバイダー向けガイド文
- Switcher クラス: switch_request の保持（実書き込み相当）
- extract_switch_angle_tags(text): タグ抽出ヘルパ。タグ方式の execute() 経路で使う

タグ方式の実行口は ToolExecutor.apply_switch_angle_tags(text) に統一されている。
実際のアングル切り替え処理（再ディスパッチ）は service.py が ToolExecutor.switch_request を
読んで行う。本モジュールは record_tool_event を直接呼ばない（ToolExecutor.execute() が記録する）。

タグ形式:
    [SWITCH_ANGLE:preset_name|self_instruction]

例:
    [SWITCH_ANGLE:gemini2FlashLite|軽くさっぱりと応答する]
"""

import logging

from backend.lib.tag_parser import parse_tags

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


def extract_switch_angle_tags(text: str) -> tuple[str, tuple[str, str] | None]:
    """テキストから [SWITCH_ANGLE:preset_name|self_instruction] マーカーを抽出する。

    Args:
        text: LLMの応答テキスト。

    Returns:
        tuple:
            clean_text (str): マーカーを除去したテキスト。
            switch_request (tuple[str, str] | None): (preset_name, self_instruction)。
                マーカーが存在しない場合は None。複数あっても最初の1件のみ採用する。
    """
    clean, matches = parse_tags(text, ["SWITCH_ANGLE"])
    switch_matches = matches["SWITCH_ANGLE"]
    if not switch_matches:
        return clean, None
    body = switch_matches[0].body
    parts = body.split("|", 1)
    preset_name = parts[0].strip()
    self_instruction = parts[1].strip() if len(parts) > 1 else ""
    return clean, (preset_name, self_instruction)


class Switcher:
    """アングル切り替えリクエストの保持を担うクラス（実書き込み相当の状態保持）。

    タグ方式・JSON 方式・tool-use 方式 のいずれの入口から来ても、最終的にこのクラスの
    switch_angle() を経由して switch_request に格納する。実際のアングル切り替え処理
    （再ディスパッチ）は service.py が ToolExecutor.switch_request を読んで担う。
    記録（tool_call_events）は ToolExecutor.execute() で集約管理される。

    Attributes:
        switch_request: 切り替えリクエスト (preset_name, self_instruction)。未要求の場合は None。
    """

    def __init__(self) -> None:
        """Switcher を初期化する。"""
        self.switch_request: tuple[str, str] | None = None

    def switch_angle(self, preset_name: str, self_instruction: str) -> str:
        """アングル切り替えリクエストを記録する。

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
            "切替リクエスト preset=%s instruction=%.50s",
            preset_name, self_instruction,
        )
        return f"アングルを {preset_name} に切り替えます。"
