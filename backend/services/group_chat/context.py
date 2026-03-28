"""グループチャット用コンテキスト整形モジュール。

各キャラクターへ渡すメッセージ履歴を OpenAI messages 形式に変換する。
自分の発言は role="assistant"（API 仕様上の呼称）、それ以外はすべて role="user" でタグ付き表現にまとめる。
連続する同ロールのメッセージは1つにマージしてAPIの仕様制約を回避する。
画像が添付されているユーザーメッセージは OpenAI vision 形式のコンテンツリストとして構築する。
"""

from typing import Any

from backend.services.chat.content import build_message_content


def format_group_history_for_character(
    history: list,
    self_char_name: str,
    sqlite=None,
    uploads_dir: str = "",
) -> list[dict[str, Any]]:
    """グループチャット履歴を指定キャラクター視点の OpenAI messages 形式に変換する。

    - 自分の発言 → role="assistant", content=本文（タグなし）
    - ユーザーの発言 → role="user", content=<user>本文</user>（画像ありの場合はvision形式）
    - 他キャラクターの発言 → role="user", content=<キャラ名>本文</キャラ名>
    - 連続する同ロールのメッセージは1つにマージする（OpenAI API仕様対応）

    Args:
        history: ChatMessageオブジェクトのリスト（時系列順）。
        self_char_name: 現在応答生成するキャラクターの名前。
        sqlite: SQLiteStoreインスタンス（画像メタデータ取得用。省略時は画像を無視）。
        uploads_dir: 画像ファイルの保存ディレクトリパス（省略時は画像を無視）。

    Returns:
        OpenAI messages 形式の辞書リスト。
    """
    result: list[dict[str, Any]] = []

    for msg in history:
        if msg.role == "user":
            oai_role = "user"
            image_ids = list(getattr(msg, "images", None) or [])
            tagged_text = f"<user>{msg.content}</user>"
            tagged = build_message_content(tagged_text, image_ids, sqlite, uploads_dir)
        elif msg.role == "character":
            char_name = getattr(msg, "character_name", None) or self_char_name
            if char_name == self_char_name:
                # 自分自身の発言はそのまま渡す（API 仕様上 role="assistant"）
                tagged = msg.content
                oai_role = "assistant"  # API 仕様上の呼称。内部ではキャラクターターンとして扱う
            else:
                # 他キャラクターの発言はタグ付きで user として渡す
                tagged = f"<{char_name}>{msg.content}</{char_name}>"
                oai_role = "user"
        else:
            continue

        # 連続する同ロールのメッセージをマージする（テキスト同士のみ）
        if result and result[-1]["role"] == oai_role and isinstance(tagged, str) and isinstance(result[-1]["content"], str):
            result[-1]["content"] += "\n" + tagged
        else:
            result.append({"role": oai_role, "content": tagged})

    return result
