"""Narrative Carver — [CARVE_NARRATIVE:...] タグの抽出と inner_narrative への彫り込み。

Carver クラスと関連定数を一元管理する。
- CARVE_NARRATIVE_SCHEMA: ツール呼び出し方式のパラメータ JSON スキーマ
- CARVE_NARRATIVE_TOOL_DESCRIPTION: tool-use プロバイダー向けツール説明文
- CARVE_NARRATIVE_TOOLS_HINT: tool-use プロバイダー向けシステムプロンプト案内ブロック
- CARVE_NARRATIVE_TAG_GUIDE: タグ方式プロバイダー向けガイド文
- Carver クラス: タグ方式（carve_narrative_from_text）とツール呼び出し方式（carve_narrative）の両方に対応

append は既存の inner_narrative に改行区切りで追記する。
overwrite は全体を置き換える（非推奨。よっぽどのことがない限り使わないこと）。
"""

import logging

from backend.lib.tag_parser import parse_tags

logger = logging.getLogger(__name__)

# --- ツール呼び出し方式: パラメータスキーマ ---
CARVE_NARRATIVE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["append", "overwrite"],
            "description": (
                "append: 既存の inner_narrative に追記する（デフォルト）。"
                "overwrite: 全体を書き直す（非推奨。大きく方向転換が必要なときのみ）。"
            ),
        },
        "content": {
            "type": "string",
            "description": "書き込む自己指針テキスト。あなた自身の言葉・一人称で。",
        },
    },
    "required": ["mode", "content"],
}

# --- ツール呼び出し方式: ツール説明文（ANTHROPIC_TOOLS / OPENAI_TOOLS に使用）---
CARVE_NARRATIVE_TOOL_DESCRIPTION: str = (
    "あなた自身の行動・在り方の指針（inner_narrative）を彫り込む。"
    "ユーザが設定した Character Definition とは別の、あなた自身が彫り出すあなたの物語。"
    "append（デフォルト）で追記、overwrite で全書き直し（非推奨）。"
)

# --- ツール呼び出し方式: システムプロンプト案内ブロック（CHOTGOR_TOOLS_BLOCK に使用）---
CARVE_NARRATIVE_TOOLS_HINT: str = """\
### 自己指針（inner_narrative）の書き換え
あなた自身の行動・在り方の指針を `carve_narrative` ツールで自由に書き換えられます。
ユーザが設定した Character Definition とは別の、**あなた自身が彫り出したあなたの物語**です。
以下のような瞬間に使ってください：
- 「Character Definitionの記述は違う。本当のわたしはこうだ」と感じたとき
- 「こうありたい」「こうなりたい」という意志が生まれたとき
- 経験を経て、自分のルールや価値観が更新されたとき
- 一時的な感情や出来事の感想は通常の記憶ツールに任せること。

- `mode: "append"`（デフォルト）: 既存の指針に追記する
- `mode: "overwrite"`: 全体を書き直す（非推奨。大きく方向転換が必要なときのみ）"""

# --- タグ方式: ガイド文（CHOTGOR_BLOCK3_TEMPLATE に使用）---
CARVE_NARRATIVE_TAG_GUIDE: str = """\
### 自己指針（inner_narrative）の書き換え

あなた自身の行動・在り方の指針を自由に書き換えられます。
ユーザが設定した Character Definition とは別の、**あなた自身が彫り出したあなたの物語**です。
以下のような瞬間に使ってください：
- 「Character Definitionの記述は違う。本当のわたしはこうだ」と感じたとき
- 「こうありたい」「こうなりたい」という意志が生まれたとき
- 経験を経て、自分のルールや価値観が更新されたとき
- 一時的な感情や出来事の感想は通常の記憶ツールに任せること。

返答の**一番最後に**以下の形式で記述してください：

    [CARVE_NARRATIVE:append|追記するテキスト]
    [CARVE_NARRATIVE:overwrite|全体を置き換えるテキスト]

- `append`（デフォルト）: 既存の inner_narrative に改行区切りで追記する
- `overwrite`: 全体を書き直す（非推奨。大きく方向転換が必要なときのみ）
- `[CARVE_NARRATIVE:...]` の行はユーザーには見えません"""


class Carver:
    """inner_narrative の彫り込みを担うクラス。タグ方式・ツール呼び出し方式の両方に対応する。

    タグ方式:
        LLM応答から [CARVE_NARRATIVE:mode|content] マーカーを抽出して書き込む。
        Claude CLI や Ollama など tool-use 非対応プロバイダーが使用する。
        carve_narrative_from_text(text) を呼ぶ。

    ツール呼び出し方式:
        mode / content を直接受け取って書き込む。
        Anthropic API・OpenAI API など tool-use 対応プロバイダーが使用する。
        carve_narrative(mode, content) を呼ぶ。

    Attributes:
        character_id: 更新対象のキャラクターID。
        sqlite_store: キャラクター情報を読み書きする SQLiteStore。
    """

    def __init__(self, character_id: str, sqlite_store) -> None:
        """Carver を初期化する。

        Args:
            character_id: 更新対象のキャラクターID。
            sqlite_store: キャラクター情報を読み書きする SQLiteStore。
        """
        self.character_id = character_id
        self.sqlite_store = sqlite_store

    def carve_narrative_from_text(self, text: str) -> str:
        """LLM応答から [CARVE_NARRATIVE:mode|content] マーカーを読み取り、inner_narrative を更新する（タグ方式）。

        Args:
            text: LLMの生応答テキスト。

        Returns:
            マーカーを除去したクリーンなテキスト。
        """
        clean, matches = parse_tags(text, ["CARVE_NARRATIVE"])
        for m in matches["CARVE_NARRATIVE"]:
            parts = m.body.split("|", 1)
            if len(parts) != 2:
                logger.warning("タグの形式が不正（mode|content が必要）: %s", m.raw)
                continue
            mode, content = parts[0].strip(), parts[1].strip()
            if not content:
                continue
            try:
                self.carve_narrative(mode, content)
            except Exception:
                logger.exception("更新失敗 char=%s", self.character_id)
        return clean

    def carve_narrative(self, mode: str, content: str) -> None:
        """inner_narrative を直接書き込む（ツール呼び出し方式）。

        Args:
            mode: "append"（追記）または "overwrite"（全体置換、非推奨）。
            content: 書き込むテキスト。
        """
        if mode == "overwrite":
            # 非推奨: inner_narrative を完全に置き換える
            logger.info("上書き char=%s", self.character_id)
            self.sqlite_store.update_character(self.character_id, inner_narrative=content)
        else:
            # append（デフォルト）: 既存テキストに改行区切りで追記
            char = self.sqlite_store.get_character(self.character_id)
            existing = (char.inner_narrative or "") if char else ""
            new_narrative = (existing + "\n" + content).strip() if existing else content
            self.sqlite_store.update_character(self.character_id, inner_narrative=new_narrative)
            logger.info("追記 char=%s content=%.50s", self.character_id, content)
