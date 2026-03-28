"""embeddingモデル変更時の記憶再インデックスサービス。

lib/embedding_migration_service.py から移動。
app（APIレイヤー）への依存をなくし、個別依存を引数で受け取って
新しいインスタンスのタプルを返す設計にした。
app.state の更新は呼び出し元（api/ui.py）で行う。
"""

import asyncio

from backend.services.chat.service import ChatService
from backend.repositories.chroma.store import ChromaStore
from backend.services.memory.manager import MemoryManager
from backend.repositories.sqlite.store import SQLiteStore


async def migrate_embeddings(
    sqlite: SQLiteStore,
    old_chroma: ChromaStore,
    chroma_db_path: str,
    drift_manager,
    new_provider: str,
    new_model: str,
    new_api_key: str,
) -> tuple[ChromaStore, MemoryManager, ChatService]:
    """embeddingモデル変更時に全キャラクターの記憶を新モデルで再インデックスする。

    SQLiteから全アクティブ記憶のテキストを読み出し、新しいembedding functionで
    ChromaDBに再登録する。旧コレクションはdrop後に新モデルで再作成される。
    新しいインスタンス（chroma / memory_manager / chat_service）を返すので、
    呼び出し元が app.state を更新すること。

    Args:
        sqlite: SQLite ストアインスタンス。
        old_chroma: 現在の ChromaStore インスタンス（旧コレクション削除に使用）。
        chroma_db_path: ChromaDB のストレージパス。
        drift_manager: DriftManager インスタンス（ChatService 再生成に使用）。
        new_provider: 新しいembeddingプロバイダー（"default" / "google"）。
        new_model: 新しいembeddingモデルID（空の場合はプロバイダーのデフォルト）。
        new_api_key: 新しいAPIキー。

    Returns:
        (new_chroma, new_memory_manager, new_chat_service) のタプル。
    """

    def _do_migrate() -> ChromaStore:
        """同期的に全記憶を再インデックスする。スレッドプール上で実行する。"""
        new_chroma = ChromaStore(
            db_path=chroma_db_path,
            embedding_provider=new_provider,
            embedding_model=new_model,
            api_key=new_api_key,
        )
        characters = sqlite.list_characters()
        for char in characters:
            old_chroma.delete_all_memories(char.id)
            memories = sqlite.get_all_active_memories(char.id)
            for mem in memories:
                new_chroma.add_memory(
                    memory_id=mem.id,
                    content=mem.content,
                    character_id=char.id,
                    metadata={
                        "category": mem.memory_category,
                        "contextual_importance": mem.contextual_importance,
                        "semantic_importance": mem.semantic_importance,
                        "identity_importance": mem.identity_importance,
                        "user_importance": mem.user_importance,
                    },
                )
        return new_chroma

    new_chroma = await asyncio.to_thread(_do_migrate)
    new_memory_manager = MemoryManager(sqlite=sqlite, chroma=new_chroma)
    new_chat_service = ChatService(
        memory_manager=new_memory_manager,
        drift_manager=drift_manager,
    )
    return new_chroma, new_memory_manager, new_chat_service
