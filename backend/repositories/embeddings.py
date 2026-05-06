"""Embedding 関数群 — ベクトルストア非依存の embedding プロバイダー実装。

LanceStore（および以前の ChromaStore）の双方から利用される embedding 実装をここに集約する。
ベクトル DB の差し替えで毎回 embedding 実装を書き換えなくて済むよう、専用モジュールに分離した。

# 提供するプロバイダー

  * ``InfinityEmbeddingFunction`` — infinity サーバーの OpenAI 互換 ``/embeddings``
    （ローカル実行、高速、API課金なし）
  * ``GeminiEmbeddingFunction``   — Google Gemini Embedding API
  * ``get_embedding_function``    — プロバイダー名からインスタンスを解決するファクトリ

# プロトコル

各 embedding fn は以下の最小プロトコルを満たす:

  * ``__call__(texts: list[str]) -> list[list[float]]``
    文書としての embedding（infinity は ``文章: `` プレフィックス）

  * ``InfinityEmbeddingFunction`` のみ追加で ``embed_query(texts: list[str]) -> list[list[float]]`` を持つ
    （ruri-v3 系はクエリ用にプレフィックスを変える必要があるため）

# 旧 ChromaDB 依存の整理

歴史的経緯で ``chromadb.EmbeddingFunction`` を継承していたが、現在は LanceStore 移行に伴い
ChromaDB への依存を切った。``chromadb`` 由来の型ヒントは削除したが、シグネチャ互換性は維持して
いるため、外部から呼ぶ側のコードは変更不要。
"""

from __future__ import annotations

from typing import Optional, Protocol


class EmbeddingFunction(Protocol):
    """embedding fn の最小プロトコル。

    LanceStore は ``__call__(list[str]) -> list[list[float]]`` のみを必須としている。
    ``embed_query`` を持つ実装（infinity）は LanceStore 内で isinstance 判定して使い分ける。
    """

    def __call__(self, texts: list[str]) -> list[list[float]]:
        ...


class InfinityEmbeddingFunction:
    """infinity サーバーの OpenAI 互換 ``/embeddings`` を使う embedding fn 実装。

    ruri-v3-310m は文書には「文章: 」、クエリには「クエリ: 」プレフィックスが必要。
    ``__call__`` は文書プレフィックスを付加する。
    クエリ埋め込みは ``embed_query()`` を使い、LanceStore 側で query vector に渡す。
    """

    def __init__(
        self,
        model: str = "cl-nagoya/ruri-v3-310m",
        base_url: str = "http://localhost:7997",
        doc_prefix: str = "文章: ",
        query_prefix: str = "クエリ: ",
    ):
        """初期化。

        Args:
            model: infinity サーバーで提供するモデル ID。
            base_url: infinity サーバーの BaseURL（例: ``http://localhost:7997``）。
            doc_prefix: ドキュメント埋め込み時に先頭に付加するプレフィックス。
            query_prefix: クエリ埋め込み時に先頭に付加するプレフィックス。
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._doc_prefix = doc_prefix
        self._query_prefix = query_prefix

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """テキストリストを ``/embeddings`` でベクトル化する（プレフィックス付加前提）。

        Args:
            texts: プレフィックス付加済みのテキストリスト。

        Returns:
            各テキストに対応する embedding ベクトルのリスト。
        """
        import json
        import urllib.request

        payload = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        # タイムアウトを設定（デフォルト60秒）。大きなバッチでも余裕を持たせる。
        # infinity v2 は /v1/ プレフィックスなし（/embeddings が正しいエンドポイント）
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return [item["embedding"] for item in result["data"]]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """文書として embedding する（``文章: `` プレフィックス付加）。

        Args:
            texts: 文書テキストのリスト。

        Returns:
            各テキストに対応する embedding ベクトルのリスト。
        """
        prefixed = [f"{self._doc_prefix}{t}" for t in texts]
        return self._embed(prefixed)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """クエリとして embedding する（``クエリ: `` プレフィックス付加）。

        Args:
            texts: クエリテキストのリスト。

        Returns:
            各クエリに対応する embedding ベクトルのリスト。
        """
        prefixed = [f"{self._query_prefix}{t}" for t in texts]
        return self._embed(prefixed)


class GeminiEmbeddingFunction:
    """Google Gemini Embedding を使用する embedding fn 実装。

    google-genai パッケージを使用してテキストをベクトル化する。
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-2-preview"):
        """初期化。クライアントをインスタンス変数として保持し、呼び出しごとの生成コストを省く。

        Args:
            api_key: Google API キー。
            model: 使用する embedding モデル ID。
        """
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """テキストリストを embedding ベクトルに変換する（バッチリクエスト）。

        Args:
            texts: 文書テキストのリスト。

        Returns:
            各テキストに対応する embedding ベクトルのリスト。
        """
        response = self._client.models.embed_content(model=self._model, contents=list(texts))
        return [e.values for e in response.embeddings]


def get_embedding_function(
    provider: str,
    model: str,
    api_key: str,
    base_url: str = "http://localhost:7997",
) -> Optional[EmbeddingFunction]:
    """プロバイダー名から EmbeddingFunction を解決する。

    LanceStore は ``"default"`` を非サポート（None を返すと初期化エラーになる）。
    実用上は ``"infinity"`` か ``"google"`` を指定する。

    Args:
        provider: ``"google"`` / ``"infinity"`` / その他は None。
        model: embedding モデル ID。空の場合はプロバイダーのデフォルトを使用。
        api_key: プロバイダーの API キー（infinity の場合は未使用）。
        base_url: infinity サーバーの BaseURL（infinity の場合のみ使用）。

    Returns:
        EmbeddingFunction の実装。未対応プロバイダーの場合は None。
    """
    if provider == "google":
        return GeminiEmbeddingFunction(
            api_key=api_key,
            model=model or "gemini-embedding-2-preview",
        )
    if provider == "infinity":
        return InfinityEmbeddingFunction(
            model=model or "cl-nagoya/ruri-v3-310m",
            base_url=base_url,
        )
    return None
