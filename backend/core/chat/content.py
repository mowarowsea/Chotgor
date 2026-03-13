"""メッセージコンテンツ構築ユーティリティ。

画像付きメッセージの OpenAI vision 形式変換と、
1on1チャット履歴の Message リスト変換を提供する。
"""

import base64
import os
from typing import Any, Union

from .models import Message


def build_message_content(
    text: str,
    image_ids: list[str],
    sqlite,
    uploads_dir: str,
) -> Any:
    """1件分のメッセージ content を構築する。

    画像が添付されている場合は OpenAI vision 形式のコンテンツリストを返す。
    画像がない場合、またはsqlite/uploads_dirが未指定の場合はテキスト文字列をそのまま返す。
    1枚も読み込めなかった場合もテキスト文字列を返す。

    Args:
        text: メッセージ本文テキスト。
        image_ids: 添付画像IDのリスト。
        sqlite: SQLiteStoreインスタンス（画像メタデータ取得用）。
        uploads_dir: 画像ファイルの保存ディレクトリパス。

    Returns:
        str または list（vision形式）のコンテンツ。
    """
    if not image_ids or not sqlite or not uploads_dir:
        return text

    parts: list[dict] = [{"type": "text", "text": text}]
    for img_id in image_ids:
        img_meta = sqlite.get_chat_image(img_id)
        if not img_meta:
            continue
        img_path = os.path.join(uploads_dir, img_id)
        if not os.path.exists(img_path):
            continue
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img_meta.mime_type};base64,{b64}"},
        })

    return parts if len(parts) > 1 else text


def build_1on1_history(
    history: list,
    sqlite,
    uploads_dir: str,
) -> list[Message]:
    """1on1チャットの履歴を ChatRequest 用 Message リストに変換する。

    - character ロール → role="assistant"
    - user ロール → role="user"（画像ありの場合は vision 形式に変換）

    Args:
        history: ChatMessageオブジェクトのリスト（時系列順）。
        sqlite: SQLiteStoreインスタンス（画像メタデータ取得用）。
        uploads_dir: 画像ファイルの保存ディレクトリパス。

    Returns:
        Message オブジェクトのリスト。
    """
    messages: list[Message] = []
    for msg in history:
        if msg.role == "character":
            messages.append(Message(role="assistant", content=msg.content))
        else:
            image_ids = list(getattr(msg, "images", None) or [])
            content: Union[str, list] = build_message_content(
                msg.content, image_ids, sqlite, uploads_dir
            )
            messages.append(Message(role="user", content=content))
    return messages
