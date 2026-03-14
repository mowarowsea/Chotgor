"""ChromaDB integration for semantic memory retrieval."""

import os
from typing import Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Google Gemini Embeddingを使用するEmbeddingFunction実装。

    google-genaiパッケージを使用してテキストをベクトル化する。
    ChromaDBのEmbeddingFunctionインターフェースに準拠する。
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-2-preview"):
        """初期化。クライアントをインスタンス変数として保持し、呼び出しごとの生成コストを省く。

        Args:
            api_key: Google APIキー。
            model: 使用するembeddingモデルID。
        """
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:
        """テキストリストをembeddingベクトルに変換する。

        バッチリクエストで一括変換する。

        Args:
            input: ChromaDBから渡されるテキストのリスト。

        Returns:
            各テキストに対応するembeddingベクトルのリスト。
        """
        response = self._client.models.embed_content(model=self._model, contents=list(input))
        return [e.values for e in response.embeddings]


def get_embedding_function(
    provider: str,
    model: str,
    api_key: str,
) -> Optional[EmbeddingFunction]:
    """プロバイダーに応じたEmbeddingFunctionを返す。

    Args:
        provider: "default"（ChromaDB組み込み）または "google"。
        model: embeddingモデルID。空の場合はプロバイダーのデフォルトを使用。
        api_key: プロバイダーのAPIキー。

    Returns:
        EmbeddingFunctionの実装。"default"の場合はNone（ChromaDB組み込みを使用）。
    """
    if provider == "google":
        return GeminiEmbeddingFunction(
            api_key=api_key,
            model=model or "gemini-embedding-2-preview",
        )
    return None


class ChromaStore:
    """ChromaDB永続化ストア — キャラクターごとのベクトルコレクションを管理する。"""

    def __init__(
        self,
        db_path: str,
        embedding_provider: str = "default",
        embedding_model: str = "",
        api_key: str = "",
    ):
        """ChromaDBクライアントを初期化する。

        Args:
            db_path: ChromaDBデータディレクトリのパス。
            embedding_provider: "default"（ChromaDB組み込み）または "google"。
            embedding_model: embeddingモデルID。空の場合はプロバイダーのデフォルトを使用。
            api_key: プロバイダーのAPIキー。
        """
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedding_fn = get_embedding_function(embedding_provider, embedding_model, api_key)

    def _get_collection(self, character_id: str):
        """キャラクターごとのコレクションを取得または作成する。

        embeddingモデル変更による次元不一致エラーが発生した場合は、
        コレクションを削除して再作成する（記憶はSQLiteから再インデックス可能）。
        """
        collection_name = f"char_{character_id.replace('-', '_')}"
        kwargs: dict = {
            "name": collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        if self._embedding_fn is not None:
            kwargs["embedding_function"] = self._embedding_fn
        try:
            return self.client.get_or_create_collection(**kwargs)
        except Exception:
            # embeddingモデル変更による次元不一致エラーの場合、コレクションを再作成する
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
            return self.client.get_or_create_collection(**kwargs)

    def add_memory(
        self,
        memory_id: str,
        content: str,
        character_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """記憶のembeddingをChromaDBにupsertする。

        Args:
            memory_id: 記憶の一意ID（SQLiteのMemory.idと同一）。
            content: embeddingするテキスト本文。
            character_id: キャラクターID。
            metadata: 追加メタデータ（str/int/float/bool型の値のみ有効）。
        """
        collection = self._get_collection(character_id)
        meta = {"character_id": character_id}
        if metadata:
            # ChromaDBのメタデータ値はstr/int/float/bool型のみ許可
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
        collection.upsert(
            ids=[memory_id],
            documents=[content],
            metadatas=[meta],
        )

    def recall_memory(
        self,
        query: str,
        character_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """類似度検索で記憶を取得する。

        Args:
            query: 検索クエリテキスト。
            character_id: キャラクターID。
            top_k: 取得する最大件数。

        Returns:
            id / content / distance / metadata キーを持つdictのリスト。
        """
        collection = self._get_collection(character_id)
        count = collection.count()
        if count == 0:
            return []

        n = min(top_k, count)
        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                memories.append(
                    {
                        "id": mem_id,
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i],
                    }
                )
        return memories

    def delete_memory(self, memory_id: str, character_id: str) -> None:
        """指定IDの記憶をChromaDBから物理削除する。

        Args:
            memory_id: 削除する記憶のID。
            character_id: キャラクターID。
        """
        collection = self._get_collection(character_id)
        collection.delete(ids=[memory_id])

    def delete_all_memories(self, character_id: str) -> None:
        """キャラクターのコレクション全体を削除する。

        Args:
            character_id: コレクションを削除するキャラクターID。
        """
        collection_name = f"char_{character_id.replace('-', '_')}"
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
