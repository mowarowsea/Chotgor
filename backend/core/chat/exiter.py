"""Exiter — [END_SESSION:reason] タグの抽出と退席情報の処理。

Exiter クラスと関連定数を一元管理する。
- END_SESSION_TAG_GUIDE: タグ方式プロバイダー向けガイド文
- END_SESSION_TOOL_DESCRIPTION / END_SESSION_SCHEMA: tool-use プロバイダー向け
- Exiter クラス: タグ方式（exit_from_text）とツール呼び出し方式（set_exit）の両方に対応

タグ形式:
    [END_SESSION:reason]  ← reason あり
    [END_SESSION:]        ← reason なし（コロンのみ）
    [END_SESSION]         ← 固定マーカー形式（reason なし）
"""

from ..tag_parser import parse_tags


# --- タグ方式: ガイド文 ---
END_SESSION_TAG_GUIDE: str = """\
### 退席ツール

この会話から退席したい場合（疲れた・用がある・話したくないなど）、
返答の**一番最後に**以下の形式で退席できます。

    [END_SESSION:退席理由]

- `退席理由` は省略可能（省略する場合は `[END_SESSION]` または `[END_SESSION:]` と書く）
- 退席後はこのチャットでの応答が停止します
- `[END_SESSION:...]` の行はユーザーには直接見えません（代わりにシステムメッセージが表示されます）

**具体例:**
    [END_SESSION:今日は疲れたから先に寝るね。また明日！]
    [END_SESSION:ちょっと用事があるので失礼します]
    [END_SESSION]"""

# --- ツール呼び出し方式: パラメータスキーマ ---
END_SESSION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": (
                "退席理由。省略可能。"
                "理由がある場合は自分の言葉で書く。"
            ),
        },
    },
    "required": [],
}

# --- ツール呼び出し方式: ツール説明文 ---
END_SESSION_TOOL_DESCRIPTION: str = (
    "この会話から退席する。"
    "疲れた・用がある・話したくない・話が終わったと感じたときに呼び出す。"
    "退席後はこのチャットでの応答が停止する。"
)


def _extract(text: str) -> tuple[str, str | None]:
    """テキストから [END_SESSION:reason] マーカーを取り出す。

    [END_SESSION:reason] / [END_SESSION:] / [END_SESSION] の3形式すべてに対応する。
    複数タグがある場合は最初のものを使用する。

    Args:
        text: LLMの生応答テキスト。

    Returns:
        tuple:
            clean_text (str): マーカーを除去したテキスト。
            reason (str | None): 退席理由文字列。タグなしの場合は None。
                                   reason省略の場合は空文字列。
    """
    clean, matches = parse_tags(text, ["END_SESSION"])
    hits = matches["END_SESSION"]
    if not hits:
        return text, None
    # 最初のマッチのみ使用する（複数タグは想定外だが念のため）
    reason = hits[0].body.strip()
    return clean, reason


class Exiter:
    """退席ツールの処理を担うクラス。タグ方式・ツール呼び出し方式の両方に対応する。

    タグ方式:
        LLM応答から [END_SESSION:reason] マーカーを抽出して退席情報を記録する。
        Claude CLI や Ollama など tool-use 非対応プロバイダーが使用する。
        exit_from_text(text) を呼ぶ。

    ツール呼び出し方式:
        reason を直接受け取って退席情報を記録する。
        Anthropic API・OpenAI API など tool-use 対応プロバイダーが使用する。
        set_exit(reason) を呼ぶ。

    Attributes:
        exit_reason (str | None): 退席理由。退席要求なしの場合は None。
                                    退席要求あり・reason省略の場合は空文字列。
    """

    def __init__(self) -> None:
        """Exiter を初期化する。"""
        # 退席理由。None = 退席要求なし。"" = reason省略。文字列 = reason あり。
        self.exit_reason: str | None = None

    @property
    def has_exit(self) -> bool:
        """退席要求があるかどうかを返す。"""
        return self.exit_reason is not None

    def exit_from_text(self, text: str) -> str:
        """LLM応答から [END_SESSION:...] マーカーを読み取り、退席情報を記録する（タグ方式）。

        タグを検出した場合は self.exit_reason を設定し、タグを除去したテキストを返す。

        Args:
            text: LLMの生応答テキスト。

        Returns:
            マーカーを除去したクリーンなテキスト。
        """
        clean, reason = _extract(text)
        if reason is not None:
            self.exit_reason = reason
        return clean

    def set_exit(self, reason: str = "") -> str:
        """退席情報を直接記録する（ツール呼び出し方式）。

        Args:
            reason: 退席理由。空文字列の場合は reason なし扱い。

        Returns:
            ツール実行結果テキスト。
        """
        self.exit_reason = reason.strip()
        return "退席リクエストを受け付けた。"
