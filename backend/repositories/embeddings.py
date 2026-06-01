"""Embedding 関数群 — ベクトルストア非依存の embedding プロバイダー実装。

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
"""

from __future__ import annotations

from typing import Protocol


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


class OllamaEmbeddingFunction:
    """Ollama サーバの ``POST /api/embed`` を使う embedding fn 実装。

    Ollama の embed API は文書／クエリの区別を持たない（prefix も不要）。
    bge-m3 等のモデルを ``ollama pull`` 済み前提で、Chotgor 側からは LLM と同じ
    ``ollama_base_url`` を流用する。
    """

    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        """初期化。

        Args:
            model: Ollama 上で提供する embedding モデル名（例: ``bge-m3``）。
            base_url: Ollama サーバの BaseURL（例: ``http://localhost:11434``）。
        """
        self._model = model
        self._base_url = base_url.rstrip("/")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """テキストリストを ``/api/embed`` でベクトル化する（バッチ可）。

        Ollama の ``/api/embed`` は ``input`` に list/str 両方を受け付ける。
        list を渡すと ``embeddings`` も同順で list 返却される。

        Args:
            texts: 文書テキストのリスト。

        Returns:
            各テキストに対応する embedding ベクトルのリスト。
        """
        import json
        import urllib.request

        payload = json.dumps({"model": self._model, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return list(result["embeddings"])


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
    ollama_base_url: str = "http://localhost:11434",
) -> EmbeddingFunction | None:
    """プロバイダー名から EmbeddingFunction を解決する。

    LanceStore は ``"default"`` を非サポート（None を返すと初期化エラーになる）。
    実用上は ``"infinity"`` / ``"google"`` / ``"ollama"`` のいずれかを指定する。

    Args:
        provider: ``"google"`` / ``"infinity"`` / ``"ollama"`` / その他は None。
        model: embedding モデル ID。空の場合はプロバイダーのデフォルトを使用。
        api_key: プロバイダーの API キー（infinity/ollama の場合は未使用）。
        base_url: infinity サーバーの BaseURL（infinity の場合のみ使用）。
        ollama_base_url: Ollama サーバーの BaseURL（ollama の場合のみ使用）。
            LLM 用と同じ Ollama インスタンスを流用する想定。

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
    if provider == "ollama":
        return OllamaEmbeddingFunction(
            model=model or "bge-m3",
            base_url=ollama_base_url,
        )
    return None
