"""Embedding model 変更時の再インデックスサービス。

# 役割

ユーザが Settings UI から embedding プロバイダー / モデルを変更した際、
全キャラクターの記憶・チャット履歴・キャラクター定義を新しい model で再 embed する。

# 旧 ChromaStore 時代との違い

旧版は ChromaStore 内部の ``_get_collection`` / ``_write_lock`` を直接触りながら
コレクション単位で削除→再作成→チャンク upsert を行っていたが、
LanceStore に移行したことで ``LanceStore.reindex_all`` という
メソッド一発で全テーブルを再構築できるようになった。
本サービスはそれを async ラップして UI から呼び出すだけの薄い層になっている。
"""

import asyncio
import logging

from backend.repositories.embeddings import get_embedding_function
from backend.repositories.lance.store import LanceStore
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.chat.service import ChatService
from backend.services.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


async def migrate_embeddings(
    sqlite: SQLiteStore,
    old_vector_store: LanceStore,
    drift_manager,
    new_provider: str,
    new_model: str,
    new_api_key: str,
    new_base_url: str = "http://localhost:11434",
) -> tuple[LanceStore, MemoryManager, ChatService]:
    """embedding モデル変更時に全キャラクターの記憶・履歴・定義を再インデックスする。

    SQLite を source of truth として全データを読み出し、新 embedding fn で再 embed して
    LanceStore に書き戻す。LanceStore の単一テーブル構造のため、コレクションが
    キャラクター数だけ増減する旧 ChromaStore の懸念は解消されている。

    再インデックス自体は ``LanceStore.reindex_all`` に集約されているため、
    本関数は同期処理を ``asyncio.to_thread`` で別スレッドへ逃すのと、
    完了後の ``MemoryManager`` / ``ChatService`` 再生成を担当するだけの薄い層。

    Args:
        sqlite: SQLite ストアインスタンス。
        old_vector_store: 現在の LanceStore インスタンス（embedding fn が差し替えられて再利用される）。
        drift_manager: DriftManager インスタンス（ChatService 再生成に使用）。
        new_provider: 新しい embedding プロバイダー（``"infinity"`` / ``"google"``）。
        new_model: 新しい embedding モデル ID（空の場合はプロバイダーのデフォルト）。
        new_api_key: 新しい API キー。
        new_base_url: infinity サーバーの BaseURL（infinity の場合のみ使用）。

    Returns:
        ``(vector_store, new_memory_manager, new_chat_service)`` のタプル。
        vector_store は old_vector_store を embedding fn 差し替えで再利用したもの。
    """
    new_embedding_fn = get_embedding_function(new_provider, new_model, new_api_key, new_base_url)
    if new_embedding_fn is None:
        raise ValueError(
            f"未対応の embedding provider: {new_provider!r}。"
            "infinity または google を指定すること。"
        )

    def _do_migrate() -> dict:
        """同期的に全件再インデックスする（スレッドプール上で実行）。"""
        return old_vector_store.reindex_all(new_embedding_fn=new_embedding_fn, sqlite=sqlite)

    counts = await asyncio.to_thread(_do_migrate)
    logger.info("migrate_embeddings 完了: %s", counts)

    new_memory_manager = MemoryManager(sqlite=sqlite, vector_store=old_vector_store)
    new_chat_service = ChatService(
        memory_manager=new_memory_manager,
        drift_manager=drift_manager,
    )
    return old_vector_store, new_memory_manager, new_chat_service
