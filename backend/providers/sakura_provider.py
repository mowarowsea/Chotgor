"""さくらの AI Engine provider — OpenAI互換APIを使う薄いサブクラス。

SakuraProvider は OpenAIProvider のサブクラスとして、
ベースURL・デフォルトモデル・APIキー設定キーを上書きする。

さくらの AI Engine は OpenAI互換の /v1/chat/completions / /v1/embeddings を提供する。
ただし /v1/models は公開されていないため、モデル一覧は静的に保持する
（コントロールパネルから利用可能なモデルを参照する仕様）。

tool-use サポートはモデルごとに分かれる（vLLM の --enable-auto-tool-choice が
サーバー側で有効化されているモデルだけが OpenAI互換 tool_choice="auto" を受け付ける）。
非対応モデルでは Chotgor 側で SUPPORTS_TOOLS=False として扱い、タグ方式へフォールバックする。
"""

from __future__ import annotations

from backend.providers.openai_provider import OpenAIProvider

_SAKURA_BASE_URL = "https://api.ai.sakura.ad.jp/v1"

# コントロールパネルで利用可能なチャット生成モデルの静的リスト。
# /v1/models が公開されていないため、こちらで管理する。
# supports_tools は実機検証 or 妥当な推測値。実機で挙動が変わったら更新する。
_SAKURA_CHAT_MODELS: list[dict] = [
    {"id": "gpt-oss-120b", "label": "gpt-oss-120b", "supports_tools": True},
    {"id": "Qwen3-Coder-30B-A3B-Instruct", "label": "Qwen3-Coder-30B-A3B-Instruct (コーディング)", "supports_tools": True},
    {"id": "Qwen3-Coder-480B-A35B-Instruct-FP8", "label": "Qwen3-Coder-480B-A35B-Instruct-FP8 (コーディング)", "supports_tools": True},
    # llm-jp-3.1: 実機確認で tool-use 非対応（vLLM 側で auto-tool-choice 未設定）
    {"id": "llm-jp-3.1-8x13b-instruct4", "label": "llm-jp-3.1-8x13b-instruct4 (国産)", "supports_tools": False},
    # Kimi-K2.6: 実機確認で tool-use 対応
    {"id": "preview/Kimi-K2.6", "label": "preview/Kimi-K2.6 (プレビュー・マルチモーダル)", "supports_tools": True},
    # CPU 軽量モデル群は安全側に倒す
    {"id": "preview/Phi-4-mini-instruct-cpu", "label": "preview/Phi-4-mini-instruct-cpu (プレビュー・CPU)", "supports_tools": False},
    {"id": "preview/Phi-4-multimodal-instruct", "label": "preview/Phi-4-multimodal-instruct (プレビュー・マルチモーダル)", "supports_tools": False},
    {"id": "preview/Qwen3-0.6B-cpu", "label": "preview/Qwen3-0.6B-cpu (プレビュー・CPU)", "supports_tools": False},
    {"id": "preview/Qwen3-VL-30B-A3B-Instruct", "label": "preview/Qwen3-VL-30B-A3B-Instruct (プレビュー・マルチモーダル)", "supports_tools": True},
    {"id": "preview/Qwen3.6-35B-A3B", "label": "preview/Qwen3.6-35B-A3B (プレビュー・マルチモーダル)", "supports_tools": True},
]

# モデルID → tool-use 対応フラグの辞書（クラスメソッドからの参照を高速化するためのキャッシュ）
_SAKURA_TOOL_SUPPORT: dict[str, bool] = {m["id"]: m["supports_tools"] for m in _SAKURA_CHAT_MODELS}


class SakuraProvider(OpenAIProvider):
    """さくらの AI Engine — OpenAI互換APIを異なるベースURLで呼び出すプロバイダー。"""

    PROVIDER_ID = "sakura"
    DEFAULT_MODEL = "gpt-oss-120b"
    BASE_URL = _SAKURA_BASE_URL
    # _api_guard デコレータがエラーメッセージで参照するキー名
    _API_SETTINGS_KEY = "sakura_api_key"

    # さくら AI Engine は reasoning_effort 系をサポートしないため空dictで無効化
    _REASONING_MAP: dict = {}

    # 出力トークン上限を撤廃する（max_tokens を送らず、サーバー側の上限＝残りコンテキスト全部に委ねる）。
    # 理由: さくら AI Engine はトークン数ではなく「リクエスト数」課金のため、トークン節約の動機がない。
    #       かつ推論モデル（preview/Kimi-K2.6 など）は本文を出す前に大量に思考する。既定の 4096 では
    #       思考だけで上限へ到達し（finish_reason=length）、本文が一文字も返らない事象が起きる。
    #       enable_thinking=false / reasoning_effort=low はさくら側で無視され思考を止められないため、
    #       上限自体を撤廃して思考＋本文を最後まで生成させる方針とする（実機検証済み）。
    DEFAULT_MAX_TOKENS = None

    # クラス属性の SUPPORTS_TOOLS は「楽観的デフォルト」として True。
    # 実際のtool-use対応有無は __init__ でインスタンス属性に上書きされる
    # （モデルIDが静的リストにあればそのフラグ、未登録モデルは True 扱い）。
    SUPPORTS_TOOLS = True

    def __init__(self, api_key: str, model: str = "", thinking_level: str = "default"):
        """APIキーとモデルIDを保持し、モデル別 tool-use 判定をインスタンス属性にセットする。"""
        super().__init__(
            api_key=api_key,
            model=model or self.DEFAULT_MODEL,
            base_url=self.BASE_URL,
            thinking_level=thinking_level,
        )
        # モデル別 tool-use 判定をインスタンス属性で上書きする。
        # 静的リスト未登録のモデルは True（楽観的）にフォールバックする。
        self.SUPPORTS_TOOLS = _SAKURA_TOOL_SUPPORT.get(self.model, True)

    @classmethod
    def from_config(cls, model: str, settings: dict, thinking_level: str = "default", **kwargs) -> "SakuraProvider":
        """設定 dict から SakuraProvider インスタンスを生成する。"""
        return cls(api_key=settings.get("sakura_api_key", ""), model=model, thinking_level=thinking_level)

    @classmethod
    def supports_tools_for_preset(cls, model_id: str) -> bool:
        """指定モデルIDで tool-use を使えるかをインスタンス生成なしで判定する。

        プリセット情報からのみ tool 方式/タグ方式を判定したいケースで使う。
        静的リスト未登録モデルは True（楽観的）にフォールバックする。
        """
        return _SAKURA_TOOL_SUPPORT.get(model_id, True)

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """さくら AI Engine の利用可能チャットモデル一覧を返す。

        /v1/models が公開されていないため、静的リストをそのまま返す。
        APIキー未設定でも一覧を返す（モデル選択UIで候補を出すため）。
        tool-use 非対応モデルは name に「(no tools)」を付加し UI で区別可能にする。
        """
        result: list[dict] = []
        for m in _SAKURA_CHAT_MODELS:
            label = m["label"]
            if not m["supports_tools"]:
                label = f"{label} (no tools)"
            result.append({"id": m["id"], "name": label})
        return result
