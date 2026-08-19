"""メッセージコンテンツ構築ユーティリティ。

添付（画像・音声）付きメッセージの OpenAI 準拠形式への変換と、
1on1チャット履歴の Message リスト変換を提供する。
"""

import base64
import os
from typing import Any

from backend.lib.attachments import attachment_kind, audio_format
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

    パートの形式は mime から導出した種別で出し分ける:
        image → `{"type": "image_url", "image_url": {"url": "data:...;base64,..."}}`
        audio → `{"type": "input_audio", "input_audio": {"data": ..., "format": "mp3"}}`
    どちらも OpenAI 準拠。独自形式を作らずに済み、将来 openai_provider が音声へ
    対応したときそのまま乗る（Anthropic は音声非対応なので寄せる理由がない）。

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
        mime = getattr(att_meta, "mime_type", None)
        part = _build_attachment_part(att_path, mime)
        if part:
            parts.append(part)

    return parts if len(parts) > 1 else text


def _build_attachment_part(att_path: str, mime_type) -> dict | None:
    """添付ファイル1件をコンテンツパートへ変換する。種別を導出できなければ None。"""
    kind = attachment_kind(mime_type)
    if kind is None:
        return None
    with open(att_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    if kind == "audio":
        fmt = audio_format(mime_type)
        if not fmt:
            return None
        return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


#: 過去ターンの添付を置き換える痕跡テキスト。種別ごとに1行（同種は1行にまとめる）。
_ATTACHMENT_TRACES = {
    "image": "[画像を見せた]",
    "audio": "[音声を聴かせた]",
}
#: 種別を導出できなかった添付（DB に残る旧レコード等）の痕跡。
_ATTACHMENT_TRACE_FALLBACK = "[ファイルを渡した]"


def attachment_trace(text: str, attachment_ids: list[str], sqlite) -> str:
    """過去ターンの添付を、実体の代わりに置く痕跡テキストへ変換する。

    添付は最新ターンにしか載せない（§添付の寿命）。過去ターンから実体を落とすと
    「何かを渡した」事実まで消えてしまうため、種別だけを残した1行を本文に足す。

    種別が同じ添付は1行にまとめる（枚数は残さない）。
    """
    if not attachment_ids or not sqlite:
        return text
    traces: list[str] = []
    for att_id in attachment_ids:
        att_meta = sqlite.get_chat_attachment(att_id)
        kind = attachment_kind(getattr(att_meta, "mime_type", None)) if att_meta else None
        trace = _ATTACHMENT_TRACES.get(kind, _ATTACHMENT_TRACE_FALLBACK)
        if trace not in traces:
            traces.append(trace)
    if not traces:
        return text
    joined = "\n".join(traces)
    return f"{text}\n{joined}" if text else joined


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
    - user ロール → role="user"（添付は痕跡テキストへ置換する）

    添付の寿命 — 実体を載せるのは最新ターンだけで、履歴の添付は痕跡テキストへ落とす。
    呼び出し側は「最新ターンを除いた履歴」を渡す契約になっており（api/chat.py・
    services/gate/delivery.py の双方が最新ターンを user_content として別に組む）、
    ここに来る添付はすべて過去のもの。曲は「聴かせた瞬間」にだけ存在し、印象を残すか
    どうかはキャラクター自身が inscribe_memory で決める。

    これはプロバイダー共通の層なので google / claude_cli 等すべてに効く
    （claude_cli 側の _extract_latest_images は二重の安全網として残す）。

    Args:
        history: ChatMessageオブジェクトのリスト（時系列順・最新ターンを含まない）。
        sqlite: SQLiteStoreインスタンス（添付メタデータ取得用）。
        uploads_dir: 添付ファイルの保存ディレクトリパス（互換のため受けるが未使用）。

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
            content = attachment_trace(msg.content, attachment_ids, sqlite)
            messages.append(Message(role="user", content=content))
    return messages
