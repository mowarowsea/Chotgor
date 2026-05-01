"""embeddingモデル変更時の記憶再インデックスサービス。

lib/embedding_migration_service.py から移動。
app（APIレイヤー）への依存をなくし、個別依存を引数で受け取って
新しいインスタンスのタプルを返す設計にした。
app.state の更新は呼び出し元（api/ui.py）で行う。

再インデックス対象は **ChromaDB に存在する全コレクション** で、内訳は以下の3系統:

1. ``char_{character_id}``      — 記憶コレクション（``MemoryStore``）
2. ``chat_{character_id}``      — チャット履歴コレクション（``ChatStore``）
3. ``char_definitions``         — キャラクター定義コレクション（別れ機能用、グローバル）

過去 (1) のみを対象にしていたため embedding 次元が混在し、
``Collection expecting embedding with dimension of N, got M`` エラーが
頻発する不具合があった。現在は全系統を対象とすること。
"""

import asyncio
import logging
import time

from backend.services.chat.service import ChatService
from backend.services.chat.indexer import build_chat_doc_and_metadata, get_participant_char_ids
from backend.repositories.chroma.store import ChromaStore
from backend.services.memory.manager import MemoryManager
from backend.repositories.sqlite.store import SQLiteStore

logger = logging.getLogger(__name__)

# キャラクターごとにバッチupsertする際の最大件数。
# Gemini など RPM 制限がある外部 API には小さめの値を、
# infinity（ローカル）など制限のないプロバイダーには大きめの値が効率的。
# チャット再インデックス時のタイムアウト多発を避けるため小さめに設定。
_UPSERT_CHUNK_SIZE = 1

# 全チャットセッションを取得する際の上限。
# SQLiteStore.list_chat_sessions の API を変えずに「実質全件」を取るための上限値。
_CHAT_SESSION_FETCH_LIMIT = 1_000_000

# チャットコレクション upsert のリトライ設定。
# embedding server への過負荷タイムアウト対策。
_CHAT_UPSERT_MAX_RETRIES = 5
_CHAT_UPSERT_RETRY_WAIT_SEC = 30
# チャンク間の待機時間（embedding server の過負荷防止）。
_CHAT_UPSERT_INTER_CHUNK_SLEEP = 0.5


async def migrate_embeddings(
    sqlite: SQLiteStore,
    old_chroma: ChromaStore,
    chroma_db_path: str,
    drift_manager,
    new_provider: str,
    new_model: str,
    new_api_key: str,
    new_base_url: str = "http://localhost:11434",
) -> tuple[ChromaStore, MemoryManager, ChatService]:
    """embeddingモデル変更時に全キャラクターの記憶を新モデルで再インデックスする。

    SQLiteから全アクティブ記憶のテキストを読み出し、新しいembedding functionで
    ChromaDBに再登録する。旧コレクションはdrop後に新モデルで再作成される。
    ChromaDB >= 0.5.0 は同一パスへの複数 PersistentClient を想定しないため、
    old_chroma の embedding_fn を差し替えて同一クライアントで再作成する（single-client 方式）。

    Args:
        sqlite: SQLite ストアインスタンス。
        old_chroma: 現在の ChromaStore インスタンス（embedding_fn を差し替えて再利用）。
        chroma_db_path: ChromaDB のストレージパス（single-client 化後は未使用）。
        drift_manager: DriftManager インスタンス（ChatService 再生成に使用）。
        new_provider: 新しいembeddingプロバイダー（"default" / "google"）。
        new_model: 新しいembeddingモデルID（空の場合はプロバイダーのデフォルト）。
        new_api_key: 新しいAPIキー。

    Returns:
        (chroma, new_memory_manager, new_chat_service) のタプル。
        chroma は old_chroma を embedding_fn 差し替えで再利用したもの。
    """

    def _do_migrate() -> ChromaStore:
        """同期的に全記憶を再インデックスする。スレッドプール上で実行する。

        各キャラクターの記憶を _UPSERT_CHUNK_SIZE 件ずつバッチupsertする。
        これにより 1件ずつ API を呼ぶ場合に比べ、外部 embedding API（Gemini 等）の
        RPM 制限への当たりを大幅に抑えられる。
        """
        from backend.repositories.chroma.store import get_embedding_function
        # 同一 client を使い続けたまま embedding function だけ差し替える
        old_chroma._embedding_fn = get_embedding_function(new_provider, new_model, new_api_key, new_base_url)
        characters = sqlite.list_characters()
        for char in characters:
            # SQLiteから記憶を先に取得してから削除する。
            # 先に削除してからupsertすると、upsert失敗時にコレクションが空のまま残る。
            memories = sqlite.get_all_active_memories(char.id)
            if not memories:
                logger.info("migrate_embeddings: 記憶なし char=%s (%s)", char.name, char.id[:8])
                old_chroma.delete_all_memories(char.id)
                continue
            old_chroma.delete_all_memories(char.id)

            # キャラクターの全記憶を chunk_size 件ずつバッチupsert
            collection = old_chroma._get_collection(char.id)
            total = len(memories)
            for chunk_start in range(0, total, _UPSERT_CHUNK_SIZE):
                chunk = memories[chunk_start:chunk_start + _UPSERT_CHUNK_SIZE]
                ids = [m.id for m in chunk]
                contents = [m.content for m in chunk]
                metadatas = [
                    {
                        "category": m.memory_category or "general",
                        "contextual_importance": m.contextual_importance,
                        "semantic_importance": m.semantic_importance,
                        "identity_importance": m.identity_importance,
                        "user_importance": m.user_importance,
                    }
                    for m in chunk
                ]
                # RPM制限でのリトライ（最大5回、指数バックオフ: 65/130/195/260秒）
                for attempt in range(5):
                    try:
                        collection.upsert(ids=ids, documents=contents, metadatas=metadatas)
                        break
                    except Exception as e:
                        if attempt < 4:
                            wait_sec = 65 * (attempt + 1)
                            logger.warning(
                                "migrate_embeddings: upsert失敗（%d回目）%d秒後にリトライ char=%s error=%s",
                                attempt + 1, wait_sec, char.name, e,
                            )
                            time.sleep(wait_sec)
                        else:
                            raise
                logger.info(
                    "migrate_embeddings: upsert %d/%d char=%s (%s)",
                    min(chunk_start + _UPSERT_CHUNK_SIZE, total), total, char.name, char.id[:8],
                )

        # 記憶コレクションの再インデックス完了後、
        # チャット履歴とキャラクター定義の再インデックスを順に実施する。
        _reindex_chat_collections(sqlite, old_chroma)
        _reindex_definition_collection(sqlite, old_chroma, characters)

        return old_chroma

    chroma = await asyncio.to_thread(_do_migrate)
    new_memory_manager = MemoryManager(sqlite=sqlite, chroma=chroma)
    new_chat_service = ChatService(
        memory_manager=new_memory_manager,
        drift_manager=drift_manager,
    )
    return chroma, new_memory_manager, new_chat_service


def _reindex_chat_collections(sqlite: SQLiteStore, chroma: ChromaStore) -> None:
    """全 ``chat_{character_id}`` コレクションを新しい embedding fn で再インデックスする。

    全チャットセッションを走査し、各メッセージを「そのセッションに参加した
    キャラクター全員」の chat コレクションへ振り分ける。
    旧コレクションは embedding 次元が古いため、削除→再作成してから upsert する。

    is_system_message が立っているメッセージはインデックス対象外
    （indexer.index_message_sync の挙動と一致させる）。

    Args:
        sqlite: SQLite ストアインスタンス。
        chroma: 新しい embedding fn が設定済みの ChromaStore インスタンス。
    """
    sessions = sqlite.list_chat_sessions(limit=_CHAT_SESSION_FETCH_LIMIT)
    # char_id -> [(message_id, doc, metadata), ...] の蓄積
    by_char: dict[str, list[tuple[str, str, dict]]] = {}

    for session in sessions:
        char_ids = get_participant_char_ids(session, sqlite)
        if not char_ids:
            continue
        messages = sqlite.list_chat_messages(session.id)
        for msg in messages:
            built = build_chat_doc_and_metadata(msg)
            if built is None:
                continue
            doc, metadata = built
            for char_id in char_ids:
                by_char.setdefault(char_id, []).append((msg.id, doc, metadata))

    if not by_char:
        logger.info("migrate_embeddings: 再インデックス対象のチャット履歴なし")
        return

    for char_id, entries in by_char.items():
        # 旧 chat_{id} コレクションを削除して新次元で作り直す
        collection_name = f"chat_{char_id.replace('-', '_')}"
        try:
            chroma.client.delete_collection(collection_name)
        except Exception as e:
            err_str = str(e)
            if "does not exist" not in err_str:
                logger.warning(
                    "migrate_embeddings(chat): コレクション削除失敗 char=%s name=%s error=%s",
                    char_id, collection_name, e,
                )
        new_collection = chroma._get_chat_collection(char_id)

        total = len(entries)
        for chunk_start in range(0, total, _UPSERT_CHUNK_SIZE):
            chunk = entries[chunk_start:chunk_start + _UPSERT_CHUNK_SIZE]
            ids = [e[0] for e in chunk]
            docs = [e[1] for e in chunk]
            # add_chat_turn と同じく character_id を metadata にマージする
            metas = [{**e[2], "character_id": char_id} for e in chunk]
            # str/int/float/bool 以外は ChromaDB が拒絶するためフィルタ（add_chat_turn と同等）
            metas = [
                {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
                for m in metas
            ]
            for attempt in range(_CHAT_UPSERT_MAX_RETRIES):
                try:
                    new_collection.upsert(ids=ids, documents=docs, metadatas=metas)
                    break
                except Exception as e:
                    if attempt < _CHAT_UPSERT_MAX_RETRIES - 1:
                        logger.warning(
                            "migrate_embeddings(chat): upsert失敗（%d回目）%d秒後にリトライ char=%s chunk=%d-%d error=%s",
                            attempt + 1, _CHAT_UPSERT_RETRY_WAIT_SEC,
                            char_id, chunk_start, chunk_start + len(chunk), e,
                        )
                        time.sleep(_CHAT_UPSERT_RETRY_WAIT_SEC)
                    else:
                        logger.warning(
                            "migrate_embeddings(chat): upsert失敗（最終）char=%s chunk=%d-%d error=%s",
                            char_id, chunk_start, chunk_start + len(chunk), e,
                        )
            time.sleep(_CHAT_UPSERT_INTER_CHUNK_SLEEP)
        logger.info(
            "migrate_embeddings(chat): upsert完了 char=%s count=%d", char_id, total,
        )


def _reindex_definition_collection(
    sqlite: SQLiteStore, chroma: ChromaStore, characters: list
) -> None:
    """``char_definitions`` グローバルコレクションを新しい embedding fn で再インデックスする。

    全キャラクターの定義テキスト（``system_prompt_block1``）を新次元で再登録する。
    relationship_status は維持するため、estranged キャラの「同一定義による再作成」検出は
    再インデックス後も正しく機能する。

    Args:
        sqlite: SQLite ストアインスタンス（参照は characters 引数で代用するため未使用）。
        chroma: 新しい embedding fn が設定済みの ChromaStore インスタンス。
        characters: list_characters() の戻り値（重複取得を避けるため _do_migrate から受け取る）。
    """
    try:
        chroma.client.delete_collection("char_definitions")
    except Exception as e:
        err_str = str(e)
        if "does not exist" not in err_str:
            logger.warning(
                "migrate_embeddings(definitions): コレクション削除失敗 error=%s", e,
            )

    count = 0
    for char in characters:
        text = (getattr(char, "system_prompt_block1", "") or "").strip()
        if not text:
            continue
        status = getattr(char, "relationship_status", "active") or "active"
        try:
            chroma.upsert_character_definition(char.id, text, status=status)
            count += 1
        except Exception as e:
            logger.warning(
                "migrate_embeddings(definitions): upsert失敗 char=%s error=%s",
                getattr(char, "name", "?"), e,
            )
    logger.info("migrate_embeddings(definitions): upsert完了 count=%d", count)
