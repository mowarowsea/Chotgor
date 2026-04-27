"""Recaller — キャラクターの能動的記憶検索（PowerRecall）タグを処理するモジュール。

キャラクターが [POWER_RECALL:クエリ|top_k] タグを出力したとき、
Chotgorバックエンドが記憶・チャット履歴を検索し、結果を加えて再呼び出しする仕組みを担う。

タグ仕様:
  [POWER_RECALL:クエリ文字列|top_k]
    - クエリ文字列: 検索クエリ（日本語可）
    - top_k: 各コレクションから取得する最大件数（省略時は5）
  例: [POWER_RECALL:ユーザが前に言っていた好きな食べ物|5]

処理フロー（execute_stream から呼ばれる）:
  1. ストリーム終了後、full_text から [POWER_RECALL:...] を検出する
  2. マーカー前テキストはすでにUIへ流れているため、それを assistant ターンとして history に追加
  3. 記憶コレクション（char_*）とチャット履歴コレクション（chat_*）を横断検索
  4. 検索結果を power_recalled に乗せた新しい ChatRequest で execute_stream を再帰呼び出し
"""

import logging

from backend.lib.tag_parser import parse_tags

logger = logging.getLogger(__name__)

# --- tool-use プロバイダー向けシステムプロンプト案内 ---
POWER_RECALL_TOOLS_HINT: str = """\
### 能動的な記憶検索（power_recall）
自動想起で見つからなかった情報のみ使用可。**1ターン1回**、結果受取後は再検索禁止。
返答テキストなしでツール呼び出しだけ終わらせないこと。\
"""

# --- タグ方式プロバイダー向けガイド ---
POWER_RECALL_TAG_GUIDE: str = """\
### 能動的な記憶検索（power_recall）

**精神的コストが高い操作です。会話冒頭に自動想起された記憶で見つからなかった場合のみ使用してください。**

どうしても思い出せない・自動想起に出てこなかった情報が必要なときに限り、
返答の途中に以下の形式で能動的に記憶を検索できます。

    [POWER_RECALL:検索クエリ|件数]

- `検索クエリ` は自然文で（例: ユーザが前に言ってた好きな食べ物）
- `件数` は省略可能（省略時は5件）
- 記憶コレクション（あなたが刻んだ記憶）とチャット履歴コレクション（過去の全会話）を横断して検索します
- 検索結果はシステムプロンプトに追記されて再度あなたへ届きます。それを踏まえて続きを応答してください
- **1ターンに1回だけ使用できます。検索結果を受け取った後は再検索せず、そのまま応答してください**
- タグの前に「ちょっと思い出してみる」などひとこと添えるのがおすすめです
- `[POWER_RECALL:...]` の行はユーザーには見えません

**⚠️ 禁止事項:**
- タグだけで応答を終わらせないこと。必ず何らかの返答テキストを出した後にタグを書くこと
- 検索結果が届いた後（システムプロンプトに「Power Recall」ブロックがある状態）は、必ず自分の言葉で答えること。再度 `[POWER_RECALL:...]` を使うことは禁止

**具体例:**
    ちょっと頑張って思い出してみる
    [POWER_RECALL:ユーザが前に言ってた好きな音楽|5]\
"""

# --- ツールのスキーマ定義（Anthropic input_schema / OpenAI parameters 形式）---
POWER_RECALL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "検索クエリ。思い出したい内容を自然文で記述する。",
        },
        "top_k": {
            "type": "integer",
            "description": "各コレクション（記憶・会話履歴）から取得する最大件数。デフォルト10。",
            "default": 10,
        },
    },
    "required": ["query"],
}

# power_recall ツールの説明文
POWER_RECALL_TOOL_DESCRIPTION: str = (
    "記憶コレクション（过去に刻んだ記憶）とチャット履歴コレクション（過去の全会話）を横断して検索し、"
    "クエリに意味的に近い記録を取得する。"
    "通常のRAG想起では見つからなかった情報を能動的に掘り起こしたいときに使う。"
    "時間減衰スコアは適用されず、意味的類似度のみで検索される。"
)


def format_power_recall_turn(results: dict, query: str) -> str:
    """PowerRecall 検索結果を Chotgor ユーザーターンのテキストに整形する。

    LLM の会話履歴に `role="user"` として注入することで、
    キャラクターターン終了 → Chotgor からの新しいターン、という会話構造を作る。

    Args:
        results: MemoryManager.power_recall() の戻り値
                 {"memories": [...], "chat_turns": [...]}。
        query: キャラクターが指定した検索クエリ。

    Returns:
        Chotgor ユーザーターンとして注入するテキスト。
    """
    memories = results.get("memories", [])
    chat_turns = results.get("chat_turns", [])

    lines = [
        "【Chotgor】POWER_RECALL COMPLETE — 再検索禁止",
        f"クエリ「{query}」の検索結果です。",
        "これはあなたが思い出した記録です。内容を要約・整理するのではなく、思い出した内容を踏まえてそのまま応答してください。",
        "**`[POWER_RECALL:...]` タグをこれ以上出力することは禁止です。**",
    ]

    if not memories and not chat_turns:
        lines.append("\n（該当する記憶・会話は見つかりませんでした）")
    else:
        if memories:
            lines.append(f"\n### 記憶 ({len(memories)}件)")
            for i, mem in enumerate(memories, 1):
                category = mem.get("metadata", {}).get("category", "general")
                lines.append(f"{i}. [{category}] {mem['content']}")
        if chat_turns:
            lines.append(f"\n### 過去の会話 ({len(chat_turns)}件)")
            for i, turn in enumerate(chat_turns, 1):
                context = turn.get("context", [])
                if context:
                    lines.append(f"{i}. 前後の会話:")
                    for msg in context:
                        marker = "（ヒット）" if msg.get("is_hit") else ""
                        lines.append(f"   {msg['speaker_name']}: {msg['content']}{marker}")
                else:
                    lines.append(f"{i}. {turn['content']}")

    return "\n".join(lines)


class Recaller:
    """[POWER_RECALL:query|top_k] タグを抽出し、再呼び出し情報を保持するクラス。

    Attributes:
        recall_request: タグが見つかった場合の (query, top_k) タプル。
                        タグがなければ None。
    """

    def __init__(self) -> None:
        """Recaller を初期化する。"""
        self.recall_request: tuple[str, int] | None = None

    def power_recall_from_text(self, text: str) -> str:
        """テキストから [POWER_RECALL:...] タグを抽出し、マーカー前のテキストを返す。

        マーカーが見つかった場合:
          - recall_request に (query, top_k) をセットする
          - マーカーより前のテキストのみを返す（マーカー以降は破棄）
            ※ マーカー前テキストはすでにストリーミングでUIへ送信済み

        マーカーがない場合:
          - recall_request は None のまま
          - テキストをそのまま返す

        Args:
            text: LLM の応答テキスト全体（full_text）。

        Returns:
            マーカー前のテキスト（マーカーなしの場合は元テキストのstrip結果）。
        """
        _, matches = parse_tags(text, ["POWER_RECALL"])
        if not matches["POWER_RECALL"]:
            return text.strip()

        # 最初の [POWER_RECALL:...] タグのみ処理する
        m = matches["POWER_RECALL"][0]
        pre_marker = text[: m.start].strip()

        # タグ本体を "クエリ|top_k" としてパースする
        parts = m.body.split("|", 1)
        query = parts[0].strip()
        if len(parts) > 1 and parts[1].strip().isdigit():
            top_k = int(parts[1].strip())
        else:
            top_k = 5

        self.recall_request = (query, top_k)
        logger.info("検出（タグ方式）query=%.80s top_k=%d", query, top_k)
        return pre_marker
