"""チャット履歴インデクサー — メッセージを LanceDB の ``chat_turns`` テーブルへ upsert する。

asyncio.to_thread 経由で呼び出すことを想定した同期関数を提供する。
LanceDB 書き込みは SQLite コミット後のベストエフォート。失敗時は 30 秒ごとに最大 5 回リトライする。

文書・メタデータの構築仕様は ``build_chat_doc_and_metadata`` に集約しており、
embedding model 変更時の再インデックス処理（``services.memory.reindex_service``）からも
同関数を再利用する。indexer と reindex の構築ロジックがズレないように同関数経由で行うこと。
"""

import json
import logging
import time
from datetime import datetime
logger = logging.getLogger(__name__)

# チャット履歴インデックス書き込みのリトライ設定
_CHAT_INDEX_MAX_RETRIES = 5
_CHAT_INDEX_RETRY_INTERVAL_SEC = 30.0


def build_chat_doc_and_metadata(
    message,
    user_name: str = "ユーザ",
) -> tuple[str, dict] | None:
    """メッセージから LanceDB upsert 用の (document, metadata) を構築する。

    indexer（通常書き込み）と reindex_service（embedding 再構築）の双方から
    呼び出される。両者で構築ロジックが乖離しないよう、ここに一本化する。

    is_system_message が立っているメッセージはインデックス対象外として None を返す。

    Args:
        message: ChatMessage ORM オブジェクト。``role`` / ``content`` / ``session_id`` /
            ``character_name`` / ``created_at`` / ``is_system_message`` を読み取る。
        user_name: role="user" のときのスピーカー名表示。

    Returns:
        ``(doc, metadata)`` のタプル。system message の場合は ``None``。
    """
    if getattr(message, "is_system_message", None):
        return None

    role = message.role
    if role == "user":
        speaker_name = user_name
    else:
        speaker_name = getattr(message, "character_name", None) or "キャラクター"

    doc = f"{speaker_name}: {message.content}"
    created_at = message.created_at
    metadata = {
        "session_id": message.session_id,
        "role": role,
        "speaker_name": speaker_name,
        "created_at": (
            created_at.isoformat(timespec="seconds")
            if created_at
            else datetime.now().isoformat(timespec="seconds")
        ),
    }
    return doc, metadata


def get_participant_char_ids(session, sqlite) -> list[str]:
    """セッションに参加しているキャラクターのIDリストを返す。

    1on1セッション: model_id から char_name を取得してIDを解決する。
    グループセッション: group_config の participants から解決する。
    （司会はキャラクターではなくシステムモデルのため対象外）
    重複キャラ名は除外する。

    Args:
        session: ChatSession ORM オブジェクト。
        sqlite: SQLiteStore インスタンス。

    Returns:
        キャラクターIDのリスト（解決できなかったキャラクターは除外）。
    """
    session_type = getattr(session, "session_type", "1on1")
    if session_type == "group":
        raw = getattr(session, "group_config", None)
        if not raw:
            return []
        try:
            cfg = json.loads(raw)
        except Exception:
            return []

        # participants を重複除去しながら解決する
        seen_names: set[str] = set()
        char_ids: list[str] = []
        names = [p["char_name"] for p in cfg.get("participants", [])]

        for name in names:
            if name in seen_names:
                continue
            seen_names.add(name)
            char = sqlite.get_character_by_name(name)
            if char:
                char_ids.append(char.id)
        return char_ids
    else:
        # 1on1: model_id = "{char_name}@{preset_name}"
        model_id = getattr(session, "model_id", "")
        char_name = model_id.split("@")[0] if "@" in model_id else model_id
        char = sqlite.get_character_by_name(char_name)
        return [char.id] if char else []


def index_message_sync(
    message,
    character_ids: list[str],
    vector_store,
    user_name: str = "ユーザ",
) -> None:
    """メッセージを参加キャラ全員の chat_turns テーブルへ upsert する（同期関数）。

    is_system_message が設定されたメッセージはインデックス対象外。
    失敗してもチャット本体への影響がないよう、例外を握り潰す（embedding API 一時失敗対策）。

    Args:
        message: ChatMessage ORM オブジェクト。
        character_ids: upsert 先のキャラクター ID リスト。
        vector_store: LanceStore インスタンス。
        user_name: role="user" のときに使用するスピーカー名。
    """
    built = build_chat_doc_and_metadata(message, user_name=user_name)
    if built is None:
        return
    doc, metadata = built

    for char_id in character_ids:
        for attempt in range(_CHAT_INDEX_MAX_RETRIES):
            try:
                vector_store.add_chat_turn(
                    message_id=message.id,
                    content=doc,
                    character_id=char_id,
                    metadata=metadata,
                )
                break
            except Exception as e:
                if attempt < _CHAT_INDEX_MAX_RETRIES - 1:
                    logger.warning(
                        "チャット履歴インデックス失敗（%d回目） msg=%s char=%s error=%s → %gs後にリトライ",
                        attempt + 1, message.id, char_id, e, _CHAT_INDEX_RETRY_INTERVAL_SEC,
                    )
                    time.sleep(_CHAT_INDEX_RETRY_INTERVAL_SEC)
                else:
                    logger.error(
                        "チャット履歴インデックス最終失敗（%d回試行） msg=%s char=%s error=%s",
                        _CHAT_INDEX_MAX_RETRIES, message.id, char_id, e,
                    )
