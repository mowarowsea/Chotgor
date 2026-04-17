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

    def _safe_get_or_create_collection(self, collection_name: str):
        """コレクションを安全に取得または作成する共通ヘルパー。

        get_or_create_collection が失敗した場合、データが存在しない空コレクションのみ
        削除して再作成する。データが残っているコレクションは削除しない（データ保護）。
        embeddingモデル変更による次元不一致エラーへの対応を想定している。

        また、取得できたコレクションでも HNSW ファイルが存在しない（破損）場合は
        強制削除して再作成する。これは migration 中断などで起きる稀な状態を自動修復する。
        """
        import logging
        logger = logging.getLogger(__name__)
        kwargs: dict = {
            "name": collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        if self._embedding_fn is not None:
            kwargs["embedding_function"] = self._embedding_fn
        try:
            collection = self.client.get_or_create_collection(**kwargs)
            # HNSW ファイルの整合性を確認（"Nothing found on disk" 等の破損を早期検出）
            try:
                collection.count()
            except Exception as e:
                logger.warning(
                    "コレクション破損を検出（HNSWファイルなし）name=%s error=%s → 強制再作成",
                    collection_name, e,
                )
                try:
                    self.client.delete_collection(collection_name)
                except Exception:
                    pass
                return self.client.get_or_create_collection(**kwargs)
            return collection
        except Exception:
            # データが存在するコレクションは削除しない
            try:
                existing = self.client.get_collection(collection_name)
                if existing.count() > 0:
                    return existing
            except Exception:
                pass
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
            return self.client.get_or_create_collection(**kwargs)

    def _get_collection(self, character_id: str):
        """キャラクターの記憶コレクションを取得または作成する。"""
        return self._safe_get_or_create_collection(f"char_{character_id.replace('-', '_')}")

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
        where: Optional[dict] = None,
    ) -> list[dict]:
        """類似度検索で記憶を取得する。

        Args:
            query: 検索クエリテキスト。
            character_id: キャラクターID。
            top_k: 取得する最大件数。
            where: ChromaDB の where フィルタ（カテゴリ絞り込みなどに使用）。
                   例: {"category": "identity"} / {"category": {"$ne": "identity"}}

        Returns:
            id / content / distance / metadata キーを持つdictのリスト。
        """
        collection = self._get_collection(character_id)
        count = collection.count()
        if count == 0:
            return []

        n = min(top_k, count)
        query_kwargs: dict = {
            "query_texts": [query],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
            # where フィルタ適用時は全件数ではなくフィルタ後の件数で n_results を調整する。
            # コレクション全件数 < n_results でなくても、フィルタ後の件数 < n_results の場合に
            # ChromaDB が "Number of requested results > elements in index" を投げるため。
            try:
                filtered_ids = collection.get(where=where, limit=n, include=[])["ids"]
                filtered_count = len(filtered_ids)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "ChromaDB pre-filter (get) 失敗 where=%s error=%s", where, e
                )
                filtered_count = 0
            if filtered_count == 0:
                return []
            query_kwargs["n_results"] = min(n, filtered_count)
        try:
            results = collection.query(**query_kwargs)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "ChromaDB query 失敗 where=%s n_results=%s error=%s",
                where, query_kwargs.get("n_results"), e,
            )
            return []

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

    def find_similar_in_category(
        self,
        content: str,
        character_id: str,
        category: str,
        threshold: float = 0.15,
    ) -> Optional[str]:
        """同一キャラクター・カテゴリ内で類似する記憶IDを返す。

        コサイン距離の目安:
          ~0.05 : ほぼ同じ文（「コーヒーが好き」vs「コーヒーが大好き」）
                  ← identity カテゴリはここまで近い場合のみ上書き（threshold=0.05）
          ~0.15 : 同系統の内容のボーダーライン（「コーヒーが好き」vs「カフェラテが好き」）
                  ← identity 以外はここまで更新対象（threshold=0.15）
          ~0.2  : 近いが別内容（「昨日映画を見た」vs「先週映画を観た」）← 更新しない
          ~0.6+ : 別トピック（「コーヒーが好き」vs「犬を飼っている」）← 新規作成

        Args:
            content: 検索クエリとなる新しい記憶テキスト。
            character_id: キャラクターID。
            category: 検索対象のカテゴリ（このカテゴリ内のみ検索する）。
            threshold: 更新判定のコサイン距離しきい値。この値未満の場合に類似とみなす。

        Returns:
            類似記憶が見つかった場合はそのmemory_id、見つからない場合はNone。
        """
        collection = self._get_collection(character_id)
        count = collection.count()
        if count == 0:
            return None

        try:
            results = collection.query(
                query_texts=[content],
                n_results=1,
                include=["distances"],
                where={"category": category},
            )
        except Exception:
            return None

        if not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        if distance < threshold:
            return results["ids"][0][0]

        return None

    def delete_memory(self, memory_id: str, character_id: str) -> None:
        """指定IDの記憶をChromaDBから物理削除する。

        Args:
            memory_id: 削除する記憶のID。
            character_id: キャラクターID。
        """
        collection = self._get_collection(character_id)
        collection.delete(ids=[memory_id])

    def _get_chat_collection(self, character_id: str):
        """キャラクターのチャット履歴コレクションを取得または作成する。"""
        return self._safe_get_or_create_collection(f"chat_{character_id.replace('-', '_')}")

    def add_chat_turn(
        self,
        message_id: str,
        content: str,
        character_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """チャット履歴ターンを chat_{character_id} コレクションへupsertする。

        Args:
            message_id: SQLite の chat_messages.id と同一。
            content: "{speaker_name}: {content}" 形式のテキスト。
            character_id: upsert先のキャラクターID。
            metadata: 追加メタデータ（str/int/float/bool型の値のみ有効）。
        """
        collection = self._get_chat_collection(character_id)
        meta = {"character_id": character_id}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
        collection.upsert(
            ids=[message_id],
            documents=[content],
            metadatas=[meta],
        )

    def recall_chat_turns(
        self,
        query: str,
        character_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """類似度検索でチャット履歴ターンを取得する。

        chat_{character_id} コレクションを対象に、クエリと意味的に近い
        過去の会話ターンを返す。時間減衰は適用しない（PowerRecall用）。

        Args:
            query: 検索クエリテキスト。
            character_id: キャラクターID。
            top_k: 取得する最大件数。

        Returns:
            id / content / distance / metadata キーを持つdictのリスト。
        """
        collection = self._get_chat_collection(character_id)
        count = collection.count()
        if count == 0:
            return []

        n = min(top_k, count)
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        turns = []
        if results["ids"] and results["ids"][0]:
            for i, turn_id in enumerate(results["ids"][0]):
                turns.append(
                    {
                        "id": turn_id,
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i],
                    }
                )
        return turns

    # ─── キャラクター定義 Embedding（別れ機能） ─────────────────────────────────

    def _get_definition_collection(self):
        """キャラクター定義コレクション（char_definitions）を取得または作成する。

        全キャラクターの定義テキストを1コレクションに格納する。
        estrangedキャラとの類似度確認に使用する。
        """
        return self._safe_get_or_create_collection("char_definitions")

    def upsert_character_definition(
        self,
        character_id: str,
        definition_text: str,
        status: str = "active",
    ) -> str:
        """キャラクター定義をembeddingしてchar_definitionsコレクションへupsertする。

        Args:
            character_id: キャラクターID（doc IDとして使用）。
            definition_text: キャラクター定義テキスト（system_prompt_block1）。
            status: "active" または "estranged"。

        Returns:
            upsertしたdoc ID（character_idと同一）。
        """
        collection = self._get_definition_collection()
        collection.upsert(
            ids=[character_id],
            documents=[definition_text],
            metadatas=[{"character_id": character_id, "status": status}],
        )
        return character_id

    def find_similar_definition(
        self,
        definition_text: str,
        exclude_character_id: str = "",
        threshold: float = 0.1,
    ) -> list[dict]:
        """estrangedキャラクターの定義と類似するものを返す。

        コサイン距離がthreshold未満（ほぼ同一の定義）のestrangedキャラクターを検出する。
        キャラクター再作成による「なかったことにする」防止に使用する。

        Args:
            definition_text: 比較対象の定義テキスト（新規作成キャラクターのもの）。
            exclude_character_id: 除外するキャラクターID（更新時に自分自身を除くため）。
            threshold: コサイン距離のしきい値。この値未満を類似とみなす。

        Returns:
            類似するestrangedキャラクターの情報リスト（character_id, distance）。
        """
        import logging
        logger = logging.getLogger(__name__)
        collection = self._get_definition_collection()
        if collection.count() == 0:
            return []
        try:
            # estrangedのみを対象にフィルタ
            estranged_ids = collection.get(
                where={"status": "estranged"},
                include=[],
            )["ids"]
        except Exception as e:
            logger.warning("find_similar_definition: estrangedフィルタ失敗 error=%s", e)
            return []

        if not estranged_ids:
            return []

        try:
            results = collection.query(
                query_texts=[definition_text],
                n_results=min(5, len(estranged_ids)),
                include=["metadatas", "distances"],
                where={"status": "estranged"},
            )
        except Exception as e:
            logger.warning("find_similar_definition: query失敗 error=%s", e)
            return []

        similar = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                if doc_id == exclude_character_id:
                    continue
                if results["distances"][0][i] < threshold:
                    similar.append({
                        "character_id": doc_id,
                        "distance": results["distances"][0][i],
                    })
        return similar

    def mark_definition_estranged(self, character_id: str) -> None:
        """キャラクター定義のstatusをestrangedに更新する。

        Args:
            character_id: 疎遠になったキャラクターのID。
        """
        import logging
        logger = logging.getLogger(__name__)
        collection = self._get_definition_collection()
        try:
            collection.update(
                ids=[character_id],
                metadatas=[{"character_id": character_id, "status": "estranged"}],
            )
        except Exception as e:
            logger.warning("mark_definition_estranged: 更新失敗 char=%s error=%s", character_id, e)

    def delete_all_memories(self, character_id: str) -> None:
        """キャラクターのコレクション全体を削除する。

        Args:
            character_id: コレクションを削除するキャラクターID。
        """
        import logging
        collection_name = f"char_{character_id.replace('-', '_')}"
        try:
            self.client.delete_collection(collection_name)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "delete_all_memories: コレクション削除失敗 char=%s name=%s error=%s",
                character_id, collection_name, e,
            )
