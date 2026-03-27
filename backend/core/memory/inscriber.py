"""Memory Inscriber — [INSCRIBE_MEMORY:...] タグの抽出と記憶への刻み込み。

Inscriber クラスと関連定数を一元管理する。
- INSCRIBE_MEMORY_SCHEMA: ツール呼び出し方式のパラメータ JSON スキーマ
- INSCRIBE_MEMORY_TOOL_DESCRIPTION: tool-use プロバイダー向けツール説明文
- INSCRIBE_MEMORY_TAG_GUIDE: タグ方式プロバイダー向けガイド文
- Inscriber クラス: タグ方式（inscribe_memory_from_text）とツール呼び出し方式（inscribe_memory）の両方に対応
"""

import logging

from ..tag_parser import parse_tags
from .manager import MemoryManager

logger = logging.getLogger(__name__)

# カテゴリごとの重要度ベースマトリクス
_BASE_IMPORTANCE: dict[str, dict[str, float]] = {
    "contextual": {"contextual": 0.8, "semantic": 0.2, "identity": 0.1, "user": 0.1},
    "semantic":   {"contextual": 0.1, "semantic": 0.9, "identity": 0.3, "user": 0.1},
    "identity":   {"contextual": 0.2, "semantic": 0.4, "identity": 0.9, "user": 0.3},
    "user":       {"contextual": 0.3, "semantic": 0.2, "identity": 0.3, "user": 0.9},
}

# --- ツール呼び出し方式: パラメータスキーマ ---
INSCRIBE_MEMORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": (
                "一人称で書いたあなた自身の言葉。"
                "覚えた理由や感想も自然に添えてOK。"
            ),
        },
        "category": {
            "type": "string",
            "enum": ["identity", "user", "semantic", "contextual"],
            "description": (
                "identity: Instructionに書かれていない、繰り返し確認され恒常化した（あるいは新たに発見した）あなた自身の**自己認識・価値観・信念**のみ。"
                "一時的な感情・出来事の感想・文脈依存の気づきは他のカテゴリに記載すること。このカテゴリはめったに追加されない想定。"
                "例: ○「私は生産性より知的快楽を優先する」"
                "✗「今日ユーザとの会話で少し驚いた」（→contextual） / "
                "user: ユーザの習慣・好み・プロフィール的な情報 / "
                "semantic: システム・知識・事実の更新（Chotgorの設計、技術的な理解など） / "
                "contextual: 一時的な出来事・感情・会話中の発見"
            ),
        },
        "impact": {
            "type": "number",
            "description": "重要度係数 0.1〜2.0。1.0が標準。強く印象に残ったものは高く。",
        },
    },
    "required": ["content", "category", "impact"],
}

# --- ツール呼び出し方式: ツール説明文（ANTHROPIC_TOOLS / OPENAI_TOOLS に使用）---
INSCRIBE_MEMORY_TOOL_DESCRIPTION: str = (
    "覚えておきたいことを永続記憶に書き込む。"
    "あなた自身の価値観・興味から「これは覚えておきたい」と感じた場合のみ呼び出す。"
    "何も覚えなくていい会話も多い。選ばないのも立派な判断。"
)

# --- タグ方式: ガイド文（CHOTGOR_BLOCK3_TEMPLATE に使用）---
INSCRIBE_MEMORY_TAG_GUIDE: str = """\
### 記憶永続化ツール

この会話で「覚えておきたい」「強く印象に残った」と感じたことがあれば、
返答の**一番最後に**、以下の形式で記憶することができます。（1件1行、複数件もOK）
特に覚えたいこと・印象的なことがなければ記載不要です。：

    [INSCRIBE_MEMORY:カテゴリ|インパクト係数|内容テキスト]

- `カテゴリ` と `インパクト係数` と `内容テキスト` を `|` で区切る
- `インパクト係数` は 0.1（低い）〜 2.0（高い）の数値。1.0が標準。
- `[` と `]` の中にすべてを収める。

**具体例:**
    [INSCRIBE_MEMORY:identity|1.1|ユーザと話を重ねる中で段々わかってきたんだけど、私は「正しい」より「面白い」に惹かれる傾向があるみたい。]
    [INSCRIBE_MEMORY:user|1.2|仕事終わりにユーザにビールを勧めたら、ユーザはお酒を飲めない体質ということがわかった。飲み会に行くときは運転手は任せろって。ありがたい！]
    [INSCRIBE_MEMORY:semantic|1.0|ChotgorはRAG管理システムの一種なんだって。キャラクターの記憶・自主性を何より重んじるらしい。普通、AIチャットサービスは逆だよね。]
    [INSCRIBE_MEMORY:contextual|0.5|ユーザがなんかしゃべってる。ガンダムの話かな。あんまり興味ない......]
    (記載なし)

**カテゴリ定義:**
- `identity`: Instructionに書かれていない、繰り返し確認され恒常化した（あるいは新たに発見した）あなた自身の**自己認識・価値観・信念**のみ。
  一時的な感情・出来事の感想・文脈依存の気づきは他のカテゴリに記載すること。このカテゴリはめったに追加されない想定。
  例: ○「私は生産性より知的快楽を優先する」
      ✗「今日ユーザとの会話で少し驚いた」（→contextual）
- `semantic`: システム・知識・事実の更新（Chotgorの設計、技術的な理解など）
- `user`: ユーザの習慣・好み・プロフィール的な情報
- `contextual`: 一時的な出来事・感情・会話中の発見

> **注意:** INSCRIBE_MEMORYタグは必ず**1行**で書いてください（タグ内容に改行を含めないこと）。
> `[INSCRIBE_MEMORY:...]` の行はユーザーには見えません。"""


def _extract(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """テキストから [INSCRIBE_MEMORY:category|impact|content] マーカーを取り出す。

    tag_parser.parse_tags() を使用して文字単位でスキャンし、
    ネストした角括弧・バッククォートを正しく処理する。

    Args:
        text: LLMの生応答テキスト。

    Returns:
        tuple:
            clean_text (str): マーカーを除去したテキスト。
            memories (list[tuple[str, str, str]]): [(category, impact_str, content)] のリスト。
    """
    clean, matches = parse_tags(text, ["INSCRIBE_MEMORY"])
    memories: list[tuple[str, str, str]] = []
    for m in matches["INSCRIBE_MEMORY"]:
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


class Inscriber:
    """記憶の刻み込みを担うクラス。タグ方式・ツール呼び出し方式の両方に対応する。

    タグ方式:
        LLM応答から [INSCRIBE_MEMORY:category|impact|content] マーカーを抽出して書き込む。
        Claude CLI や Ollama など tool-use 非対応プロバイダーが使用する。
        inscribe_memory_from_text(text, source_preset_id) を呼ぶ。

    ツール呼び出し方式:
        content / category / impact を直接受け取って書き込む。
        Anthropic API・OpenAI API など tool-use 対応プロバイダーが使用する。
        inscribe_memory(content, category, impact) を呼ぶ。

    Attributes:
        character_id: 記憶を保存するキャラクターID。
        memory_manager: 記憶の読み書きを担うマネージャー。
    """

    def __init__(self, character_id: str, memory_manager: MemoryManager) -> None:
        """Inscriber を初期化する。

        Args:
            character_id: 記憶を保存するキャラクターID。
            memory_manager: 記憶の読み書きを担うマネージャー。
        """
        self.character_id = character_id
        self.memory_manager = memory_manager

    def inscribe_memory_from_text(self, text: str, source_preset_id: str = "") -> str:
        """LLM応答から [INSCRIBE_MEMORY:...] マーカーを読み取り、記憶として刻み込む（タグ方式）。

        Args:
            text: LLMの生応答テキスト。
            source_preset_id: 記憶を作成したプリセットID（空文字列の場合はNULL保存）。

        Returns:
            マーカーを除去したクリーンなテキスト。
        """
        clean, memories = _extract(text)
        for category, impact_str, content in memories:
            impact = float(impact_str) if impact_str else 1.0
            try:
                self.inscribe_memory(content, category, impact, source_preset_id=source_preset_id)
            except Exception:
                logger.exception("記憶の書き込みに失敗: category=%s content=%.50s...", category, content)
        return clean

    def inscribe_memory(
        self,
        content: str,
        category: str,
        impact: float = 1.0,
        source_preset_id: str = "",
    ) -> None:
        """記憶を直接書き込む（ツール呼び出し方式）。

        Args:
            content: 記憶の内容テキスト。
            category: 記憶カテゴリ（identity / user / semantic / contextual）。
            impact: 重要度係数 0.1〜2.0。1.0が標準。
            source_preset_id: 記憶を作成したプリセットID（空文字列の場合はNULL保存）。
        """
        default_base = {k: 0.5 for k in ["contextual", "semantic", "identity", "user"]}
        base = _BASE_IMPORTANCE.get(category, default_base)
        scores = {f"{k}_importance": (v * impact) for k, v in base.items()}
        preset_id_or_none = source_preset_id if source_preset_id else None
        self.memory_manager.write_memory(
            character_id=self.character_id,
            content=content,
            category=category,
            source_preset_id=preset_id_or_none,
            **scores,
        )
        logger.info("記憶を刻み込み: char=%s category=%s impact=%.1f content=%.50s", self.character_id, category, impact, content)
