"""embeddingモデル変更時の記憶再インデックスサービス。

設定UIからの呼び出しに対応する。APIレイヤーからビジネスロジックを分離し、
将来的な呼び出し元変更（CLI・バックグラウンドタスク等）に対応できるようにする。
"""

import asyncio

from .chat.service import ChatService
from .memory.chroma_store import ChromaStore
from .memory.manager import MemoryManager


async def migrate_embeddings(
    app,
    new_provider: str,
    new_model: str,
    new_api_key: str,
) -> None:
    """embeddingモデル変更時に全キャラクターの記憶を新モデルで再インデックスする。

    SQLiteから全アクティブ記憶のテキストを読み出し、新しいembedding functionで
    ChromaDBに再登録する。旧コレクションはdrop後に新モデルで再作成される。
    app.state.chroma / memory_manager / chat_service も新しいインスタンスに差し替える。

    Args:
        app: FastAPIアプリケーション（app.stateへのアクセスに使用）。
        new_provider: 新しいembeddingプロバイダー（"default" / "google"）。
        new_model: 新しいembeddingモデルID（空の場合はプロバイダーのデフォルト）。
        new_api_key: 新しいAPIキー。
    """
    sqlite = app.state.sqlite
    old_chroma = app.state.chroma
    chroma_db_path = app.state.chroma_db_path

    def _do_migrate():
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

    app.state.chroma = new_chroma
    new_memory_manager = MemoryManager(sqlite=sqlite, chroma=new_chroma)
    app.state.memory_manager = new_memory_manager
    app.state.chat_service = ChatService(
        memory_manager=new_memory_manager,
        drift_manager=app.state.drift_manager,
    )
