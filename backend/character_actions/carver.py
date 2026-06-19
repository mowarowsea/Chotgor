"""Narrative Carver — [CARVE_NARRATIVE:...] タグの抽出と inner_narrative への彫り込み。

Carver クラスと関連定数を一元管理する。
- CARVE_NARRATIVE_SCHEMA: ツール呼び出し方式のパラメータ JSON スキーマ
- CARVE_NARRATIVE_TOOL_DESCRIPTION: tool-use プロバイダー向けツール説明文
- CARVE_NARRATIVE_TOOLS_HINT: tool-use プロバイダー向けシステムプロンプト案内ブロック
- CARVE_NARRATIVE_TAG_GUIDE: タグ方式プロバイダー向けガイド文
- Carver クラス: 実書き込みの実装（タグ・JSON・tool-use 全方式の終点）
- extract_carve_narrative_tags(text): タグ抽出ヘルパ。タグ方式の execute() 経路で使う

タグ方式の実行口は ToolExecutor.apply_carve_narrative_tags(text) に統一されている。
記録（tool_call_events）も ToolExecutor.execute() で集約管理されるため、本モジュールは
record_tool_event を直接呼ばない。

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
            "description": "書き込む内的叙述テキスト。あなた自身の言葉・一人称で。",
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
### 内的叙述（inner_narrative）の書き換え
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
### 内的叙述（inner_narrative）の書き換え

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


_RECOMMENDED_NARRATIVE_LEN = 2000


def build_carve_narrative_tools_hint(inner_narrative_len: int) -> str:
    """文字数情報を含む carve_narrative ツール案内文を生成する。

    推奨文字数以内ならそのまま返し、超過時のみ冒頭に警告行を追加する。

    Args:
        inner_narrative_len: 現在の inner_narrative の文字数。

    Returns:
        システムプロンプトに挿入するツール案内テキスト。
    """
    if inner_narrative_len <= _RECOMMENDED_NARRATIVE_LEN:
        return CARVE_NARRATIVE_TOOLS_HINT
    warning = (
        f"⚠️ inner_narrative が {inner_narrative_len}文字あります"
        f"（推奨: {_RECOMMENDED_NARRATIVE_LEN}文字以内）。overwrite での圧縮を検討してください。"
    )
    return f"{warning}\n\n{CARVE_NARRATIVE_TOOLS_HINT}"


def build_carve_narrative_tag_guide(inner_narrative_len: int) -> str:
    """文字数情報を含む carve_narrative タグ方式ガイドを生成する。

    推奨文字数以内ならそのまま返し、超過時のみ冒頭に警告行を追加する。

    Args:
        inner_narrative_len: 現在の inner_narrative の文字数。

    Returns:
        システムプロンプトに挿入するタグ方式ガイドテキスト。
    """
    if inner_narrative_len <= _RECOMMENDED_NARRATIVE_LEN:
        return CARVE_NARRATIVE_TAG_GUIDE
    warning = (
        f"⚠️ inner_narrative が {inner_narrative_len}文字あります"
        f"（推奨: {_RECOMMENDED_NARRATIVE_LEN}文字以内）。overwrite での圧縮を検討してください。"
    )
    return f"{warning}\n\n{CARVE_NARRATIVE_TAG_GUIDE}"


def extract_carve_narrative_tags(text: str) -> tuple[str, list[tuple[str, str]]]:
    """テキストから [CARVE_NARRATIVE:mode|content] マーカーを抽出する。

    Args:
        text: LLMの生応答テキスト。

    Returns:
        tuple:
            clean_text (str): マーカーを除去したテキスト。
            narratives (list[tuple[str, str]]): [(mode, content)] のリスト。
                mode が空のものは "append" にフォールバックし、content が空のものは除外する。
    """
    clean, matches = parse_tags(text, ["CARVE_NARRATIVE"])
    narratives: list[tuple[str, str]] = []
    for m in matches["CARVE_NARRATIVE"]:
        parts = m.body.split("|", 1)
        if len(parts) != 2:
            logger.warning("タグの形式が不正（mode|content が必要）: %s", m.raw)
            continue
        mode = parts[0].strip() or "append"
        content = parts[1].strip()
        if not content:
            continue
        narratives.append((mode, content))
    return clean, narratives


class Carver:
    """inner_narrative の彫り込みを担うクラス（実書き込みの実装）。

    タグ方式・JSON 方式・tool-use 方式 のいずれの入口から来ても、最終的にこのクラスの
    carve_narrative() を経由して inner_narrative を更新する。記録（tool_call_events）は
    ToolExecutor.execute() で集約管理されるため、本クラスは記録を行わない。

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
