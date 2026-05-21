"""Embedding model 変更時の再インデックスサービス。

ユーザが Settings UI から embedding プロバイダー / モデルを変更した際、
全キャラクターの保存記憶・チャット履歴・キャラクター定義・ワーキングメモリスレッドを
新しい model で再 embed する。再インデックス本体は ``LanceStore.reindex_all`` に
集約されており、本サービスはそれを async ラップして UI から呼び出すだけの薄い層。
"""

import asyncio
import logging

from backend.repositories.embeddings import get_embedding_function
from backend.repositories.lance.store import LanceStore
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.chat.service import ChatService
from backend.services.memory.manager import InscribedMemoryManager

logger = logging.getLogger(__name__)


async def reindex_with_new_embeddings(
    sqlite: SQLiteStore,
    vector_store: LanceStore,
    working_memory_manager,
    new_provider: str,
    new_model: str,
    new_api_key: str,
    new_base_url: str = "http://localhost:11434",
) -> tuple[LanceStore, InscribedMemoryManager, ChatService]:
    """embedding モデル変更時に全キャラクターの保存記憶・履歴・定義を再インデックスする。

    SQLite を source of truth として全データを読み出し、新 embedding fn で再 embed して
    LanceStore に書き戻す。再インデックス自体は ``LanceStore.reindex_all`` に
    集約されているため、本関数は同期処理を ``asyncio.to_thread`` で別スレッドへ逃すのと、
    完了後の ``InscribedMemoryManager`` / ``ChatService`` 再生成を担当するだけの薄い層。

    Args:
        sqlite: SQLite ストアインスタンス。
        vector_store: 現在の LanceStore インスタンス（embedding fn が差し替えられて再利用される）。
        working_memory_manager: WorkingMemoryManager インスタンス（ChatService 再生成に使用）。
        new_provider: 新しい embedding プロバイダー（``"infinity"`` / ``"google"``）。
        new_model: 新しい embedding モデル ID（空の場合はプロバイダーのデフォルト）。
        new_api_key: 新しい API キー。
        new_base_url: infinity サーバーの BaseURL（infinity の場合のみ使用）。

    Returns:
        ``(vector_store, new_memory_manager, new_chat_service)`` のタプル。
        vector_store は引数で受け取ったものを embedding fn 差し替えで再利用したもの。
    """
    new_embedding_fn = get_embedding_function(new_provider, new_model, new_api_key, new_base_url)
    if new_embedding_fn is None:
        raise ValueError(
            f"未対応の embedding provider: {new_provider!r}。"
            "infinity または google を指定すること。"
        )

    def _do_reindex() -> dict:
        """同期的に全件再インデックスする（スレッドプール上で実行）。"""
        return vector_store.reindex_all(new_embedding_fn=new_embedding_fn, sqlite=sqlite)

    counts = await asyncio.to_thread(_do_reindex)
    logger.info("reindex_with_new_embeddings 完了: %s", counts)

    new_memory_manager = InscribedMemoryManager(sqlite=sqlite, vector_store=vector_store)
    new_chat_service = ChatService(
        memory_manager=new_memory_manager,
        working_memory_manager=working_memory_manager,
    )
    return vector_store, new_memory_manager, new_chat_service
