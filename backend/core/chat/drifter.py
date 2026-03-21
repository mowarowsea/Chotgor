"""Drifter — [DRIFT:...] / [DRIFT_RESET] マーカーの抽出と適用。

Drifter クラスと関連定数を一元管理する。
- DRIFT_SCHEMA: drift ツール呼び出し方式のパラメータ JSON スキーマ
- DRIFT_RESET_SCHEMA: drift_reset ツール呼び出し方式のパラメータ JSON スキーマ
- DRIFT_TOOL_DESCRIPTION: tool-use プロバイダー向け drift ツール説明文
- DRIFT_RESET_TOOL_DESCRIPTION: tool-use プロバイダー向け drift_reset ツール説明文
- DRIFT_TAG_GUIDE: タグ方式プロバイダー向けガイド文
- Drifter クラス: タグ方式（drift_from_text）とツール呼び出し方式（drift / drift_reset）の両方に対応

SELF_DRIFT機能: キャラクターがチャット内で自分自身に課す一時的な行動指針を管理する。
タグ抽出は tag_parser.parse_tags() に委譲し、inscriber.py と対称的な設計になっている。
"""

import logging

from ..tag_parser import parse_tags

logger = logging.getLogger(__name__)

# --- ツール呼び出し方式: drift パラメータスキーマ ---
DRIFT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "このチャット内でのみ有効な一時的な行動指針のテキスト。",
        },
    },
    "required": ["content"],
}

# --- ツール呼び出し方式: drift_reset パラメータスキーマ ---
DRIFT_RESET_SCHEMA: dict = {
    "type": "object",
    "properties": {},
    "required": [],
}

# --- ツール呼び出し方式: ツール説明文 ---
DRIFT_TOOL_DESCRIPTION: str = (
    "このチャット内でのみ有効な一時的な行動指針を設定する。最大3件まで保持。"
)

DRIFT_RESET_TOOL_DESCRIPTION: str = "現在有効な全SELF_DRIFT指針をリセットする。"

# --- タグ方式: ガイド文（CHOTGOR_BLOCK3_TEMPLATE に使用）---
DRIFT_TAG_GUIDE: str = """\
### 一時的な行動指針（SELF_DRIFT）

このチャット内でのみ有効な一時的な行動指針を設定できます。
次のターンからその指針に従って応答します。最大3件まで保持されます。

返答の**一番最後に**以下の形式で記述してください：

    [DRIFT:指針テキスト]
    [DRIFT_RESET]

- `[DRIFT:...]`: 一時的な行動指針を追加する（複数行OK）
- `[DRIFT_RESET]`: 現在有効な全指針をリセットする
- これらの行はユーザーには見えません"""


def _extract(text: str) -> tuple[str, list[str], bool]:
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


class Drifter:
    """SELF_DRIFT指針の抽出・適用を担うクラス。タグ方式・ツール呼び出し方式の両方に対応する。

    タグ方式:
        LLM応答から [DRIFT:...] / [DRIFT_RESET] マーカーを抽出して DriftManager に反映する。
        Claude CLI や Ollama など tool-use 非対応プロバイダーが使用する。
        drift_from_text(text) を呼ぶ。

    ツール呼び出し方式:
        content を直接受け取って DriftManager に追加する。
        Anthropic API・OpenAI API など tool-use 対応プロバイダーが使用する。
        drift(content) / drift_reset() を呼ぶ。

    Attributes:
        session_id: 現在のセッションID。
        character_id: 操作対象のキャラクターID。
        drift_manager: SELF_DRIFT指針の読み書きを担うマネージャー。
    """

    def __init__(self, session_id: str | None, character_id: str, drift_manager) -> None:
        """Drifter を初期化する。

        Args:
            session_id: 現在のセッションID（None の場合は SELF_DRIFT 処理をスキップする）。
            character_id: 操作対象のキャラクターID。
            drift_manager: SELF_DRIFT指針の読み書きを担うマネージャー（None の場合はスキップ）。
        """
        self.session_id = session_id
        self.character_id = character_id
        self.drift_manager = drift_manager

    def drift_from_text(self, text: str) -> str:
        """LLM応答から [DRIFT:...] / [DRIFT_RESET] マーカーを読み取り、DriftManager に反映する（タグ方式）。

        Args:
            text: LLMの生応答テキスト（[INSCRIBE_MEMORY:...] 除去後）。

        Returns:
            マーカーを除去したクリーンなテキスト。
        """
        clean, drifts, reset = _extract(text)
        if not self.drift_manager or not self.session_id:
            return clean
        if reset:
            try:
                self.drift_manager.reset_drifts(self.session_id, self.character_id)
            except Exception:
                logger.exception("SELF_DRIFT リセットに失敗: session_id=%s", self.session_id)
        for content in drifts:
            try:
                self.drift_manager.add_drift(self.session_id, self.character_id, content)
            except Exception:
                logger.exception("SELF_DRIFT 追加に失敗: session_id=%s content=%.50s...", self.session_id, content)
        return clean

    def drift(self, content: str) -> str:
        """SELF_DRIFT指針を追加する（ツール呼び出し方式）。

        Args:
            content: 追加する一時指針テキスト。

        Returns:
            実行結果メッセージ。
        """
        if not self.drift_manager or not self.session_id:
            return "SELF_DRIFT は利用できない。"
        try:
            self.drift_manager.add_drift(self.session_id, self.character_id, content)
            return "指針を設定した。"
        except Exception as e:
            return f"[drift error: {e}]"

    def drift_reset(self) -> str:
        """有効な全SELF_DRIFT指針をリセットする（ツール呼び出し方式）。

        Returns:
            実行結果メッセージ。
        """
        if not self.drift_manager or not self.session_id:
            return "SELF_DRIFT は利用できない。"
        try:
            self.drift_manager.reset_drifts(self.session_id, self.character_id)
            return "指針をリセットした。"
        except Exception as e:
            return f"[drift_reset error: {e}]"
