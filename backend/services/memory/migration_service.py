"""embeddingモデル変更時の記憶再インデックスサービス。

lib/embedding_migration_service.py から移動。
app（APIレイヤー）への依存をなくし、個別依存を引数で受け取って
新しいインスタンスのタプルを返す設計にした。
app.state の更新は呼び出し元（api/ui.py）で行う。
"""

import asyncio
import logging

from backend.services.chat.service import ChatService
from backend.repositories.chroma.store import ChromaStore
from backend.services.memory.manager import MemoryManager
from backend.repositories.sqlite.store import SQLiteStore

logger = logging.getLogger(__name__)

# キャラクターごとにバッチupsertする際の最大件数。
# Gemini など RPM 制限がある外部 API には小さめの値を、
# infinity（ローカル）など制限のないプロバイダーには大きめの値が効率的。
# infinity を使う場合は 50 件程度まで増やしても問題ない。
_UPSERT_CHUNK_SIZE = 50


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
        import time
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

        return old_chroma

    chroma = await asyncio.to_thread(_do_migrate)
    new_memory_manager = MemoryManager(sqlite=sqlite, chroma=chroma)
    new_chat_service = ChatService(
        memory_manager=new_memory_manager,
        drift_manager=drift_manager,
    )
    return chroma, new_memory_manager, new_chat_service
