"""メッセージコンテンツ構築ユーティリティ。

添付（画像・音声）付きメッセージの OpenAI 準拠形式への変換と、
1on1チャット履歴の Message リスト変換を提供する。
"""

import base64
import os
from typing import Any

from backend.services.chat.models import Message


def build_message_content(
    text: str,
    attachment_ids: list[str],
    sqlite,
    uploads_dir: str,
) -> Any:
    """1件分のメッセージ content を構築する。

    添付がある場合は OpenAI 準拠のコンテンツリストを返す。
    添付がない場合、またはsqlite/uploads_dirが未指定の場合はテキスト文字列をそのまま返す。
    1件も読み込めなかった場合もテキスト文字列を返す。

    Args:
        text: メッセージ本文テキスト。
        attachment_ids: 添付IDのリスト。
        sqlite: SQLiteStoreインスタンス（添付メタデータ取得用）。
        uploads_dir: 添付ファイルの保存ディレクトリパス。

    Returns:
        str または list（コンテンツパート形式）のコンテンツ。
    """
    if not attachment_ids or not sqlite or not uploads_dir:
        return text

    parts: list[dict] = [{"type": "text", "text": text}]
    for att_id in attachment_ids:
        att_meta = sqlite.get_chat_attachment(att_id)
        if not att_meta:
            continue
        att_path = os.path.join(uploads_dir, att_id)
        if not os.path.exists(att_path):
            continue
        with open(att_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{att_meta.mime_type};base64,{b64}"},
        })

    return parts if len(parts) > 1 else text


def apply_context_window(history: list, max_chronicled: int = 10) -> list:
    """chronicle済みメッセージ数を制限し、コンテキストウィンドウを圧縮する。

    chronicle済み（chronicled_at が非NULL）のメッセージは末尾 max_chronicled 件のみ残す。
    未chronicle（chronicled_at が NULL）のメッセージはセッション中の生きた文脈として全件保持する。
    これにより、記憶に昇華済みの古い会話がトークンを圧迫するのを防ぐ。

    Args:
        history: ChatMessageオブジェクトのリスト（時系列順）。
        max_chronicled: chronicle済みメッセージの保持上限件数（デフォルト: 10）。

    Returns:
        フィルタリング後のメッセージリスト（時系列順を保持）。
    """
    chronicled = [m for m in history if getattr(m, "chronicled_at", None) is not None]
    unchronicled = [m for m in history if getattr(m, "chronicled_at", None) is None]
    # max_chronicled=0 のとき -0 は 0 と等しく全件になるため > 0 で分岐する
    trimmed_chronicled = chronicled[-max_chronicled:] if max_chronicled > 0 else []
    # 時系列順を復元するため、元のリストから順に選別する
    kept_ids = {id(m) for m in trimmed_chronicled} | {id(m) for m in unchronicled}
    return [m for m in history if id(m) in kept_ids]


def build_1on1_history(
    history: list,
    sqlite,
    uploads_dir: str,
) -> list[Message]:
    """1on1チャットの履歴を ChatRequest 用 Message リストに変換する。

    - character ロール → role="assistant"（※ API 仕様上の呼称。内部ではキャラクターターンとして扱う）
    - user ロール → role="user"（添付ありの場合はコンテンツパート形式に変換）

    Args:
        history: ChatMessageオブジェクトのリスト（時系列順）。
        sqlite: SQLiteStoreインスタンス（添付メタデータ取得用）。
        uploads_dir: 添付ファイルの保存ディレクトリパス。

    Returns:
        Message オブジェクトのリスト。
    """
    messages: list[Message] = []
    for msg in history:
        # システムメッセージ（退席通知・預かり通知などの掲示）はキャラクターの
        # 発話ではないため LLM 履歴に載せない（キャラが自分の台詞と誤認するのを防ぐ）。
        if getattr(msg, "is_system_message", None):
            continue
        if msg.role == "character":
            messages.append(Message(role="assistant", content=msg.content))
        else:
            attachment_ids = list(getattr(msg, "attachments", None) or [])
            content: str | list = build_message_content(
                msg.content, attachment_ids, sqlite, uploads_dir
            )
            messages.append(Message(role="user", content=content))
    return messages
