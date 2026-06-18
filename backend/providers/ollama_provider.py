"""Ollama プロバイダー — ローカルOllamaサーバーへのHTTPクライアント。

主に司会AI（ディレクターモデル）用途で使用する。
ストリーミングは非対応（司会AIには不要）。
"""

import asyncio
import json
import re
import urllib.error
import urllib.request

from backend.providers.base import BaseLLMProvider

DEFAULT_MODEL = "qwen2.5:latest"

# Qwen3系の思考ブロック <think>...</think> をマッチする正規表現。
# DOTALLで改行を含むテキストにマッチさせ、複数ブロックにも対応する。
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _split_think(text: str) -> tuple[str, str]:
    """Qwen3系の <think>...</think> ブロックを分離する。

    複数の思考ブロックが含まれる場合はすべて結合して thinking 側に寄せる。
    閉じタグのない未完了 <think> は本文側にそのまま残す（安全側に倒す）。

    Args:
        text: モデル出力テキスト。

    Returns:
        (thinking, clean_text) のタプル。両方とも前後空白除去済み。
    """
    thinking_parts: list[str] = []

    def _collect(m: re.Match) -> str:
        thinking_parts.append(m.group(1))
        return ""

    clean = _THINK_RE.sub(_collect, text)
    return "\n".join(p.strip() for p in thinking_parts).strip(), clean.strip()


class OllamaProvider(BaseLLMProvider):
    """Ollama経由でローカルLLMを呼び出すプロバイダー。

    Attributes:
        base_url: OllamaサーバーのベースURL（例: http://localhost:11434）。
        model: 使用するモデル名（例: qwen2.5:latest）。
        timeout: リクエストタイムアウト秒数。
    """

    PROVIDER_ID = "ollama"
    DEFAULT_MODEL = DEFAULT_MODEL
    REQUIRES_API_KEY = False

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "", timeout: int = 300, no_think: bool = False):
        """OllamaProviderを初期化する。

        Args:
            timeout: リクエストタイムアウト秒数。デフォルトは5分（300秒）。
                     プリセット単位の timeout_seconds が from_config 経由で渡される。
            no_think: Trueのとき、Qwen3系のthinkingモードを/no_thinkトークンで無効化する。
                      Ollama 0.7+のthink: Falseパラメータが効かない場合の代替手段。
        """
        self.base_url = base_url.rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.no_think = no_think

    @classmethod
    def from_config(cls, model: str, settings: dict, **kwargs) -> "OllamaProvider":
        """設定からOllamaProviderを生成するファクトリメソッド。

        timeout_seconds が kwargs で渡されればそれを優先する（プリセット単位の設定）。
        未指定時は 300秒（5分）をデフォルトとする。
        """
        base_url = settings.get("ollama_base_url", "http://localhost:11434")
        timeout_kw = kwargs.get("timeout_seconds")
        try:
            timeout = int(timeout_kw) if timeout_kw is not None else 300
        except (TypeError, ValueError):
            timeout = 300
        if timeout <= 0:
            timeout = 300
        no_think_raw = settings.get("ollama_no_think", "false")
        no_think = str(no_think_raw).lower() in ("true", "1", "yes")
        return cls(base_url=base_url, model=model, timeout=timeout, no_think=no_think)

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """ローカルOllamaサーバーからモデル一覧を取得して返す。"""
        import httpx
        base_url = settings.get("ollama_base_url", "http://localhost:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
            return sorted(
                [{"id": m["name"], "name": m["name"]} for m in data.get("models", [])],
                key=lambda m: m["id"],
            )
        except Exception:
            return []

    @classmethod
    async def list_embedding_models(cls, settings: dict) -> list[dict]:
        """ローカル Ollama サーバから embedding 候補モデル一覧を取得して返す。

        Ollama 側には「embedding 専用」フラグが無いので、``/api/tags`` で取れる
        全モデルをそのまま列挙する（``list_models`` と同じ実装に委譲）。
        ユーザが UI 上で bge-m3 等の embedding モデルを選ぶ前提。
        """
        return await cls.list_models(settings)

    def _build_messages(self, messages: list[dict]) -> list[dict]:
        """messagesリストをOllama chat形式（user/assistantのみ）に変換する内部ヘルパー。

        マルチモーダル形式（list）のcontentはテキストパートのみを結合してプレーンテキスト化する。
        role が user/assistant 以外（systemなど）は除外する（systemはsystem_promptで渡す）。
        """
        result = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if role in ("user", "assistant"):
                result.append({"role": role, "content": content})
        return result

    def _call_api(self, system_prompt: str, messages: list[dict]) -> dict:
        """/api/generate エンドポイントを同期呼び出しして生のレスポンスdictを返す。

        asyncio.to_thread() 経由でスレッドから呼ぶ想定。
        ログ記録は asyncioコンテキスト（generate()）で行うため、ここでは行わない。

        /api/chat（messages形式）ではなく /api/generate（prompt形式）を使う理由:
            Ollamaのカスタムモデルでは、Modelfileのテンプレートが旧来の
            .Prompt/.System 変数を使った /api/generate 形式のことが多い。
            /api/chat で messages を送っても .Messages が展開されずに
            システムプロンプトのみが渡されるケースがある。
            シナリオチャットでは会話履歴が system_prompt 内に含まれるため、
            最後の user メッセージだけを prompt に渡せば十分。

        Args:
            system_prompt: システムプロンプト文字列。
            messages: _build_messages() で変換済みのメッセージリスト。最後のuserメッセージのみ使用。

        Returns:
            Ollama APIの生レスポンスdict。
        """
        # 最後の user メッセージを取り出す
        prompt = ""
        for m in reversed(messages):
            if m["role"] == "user":
                prompt = m["content"]
                break

        # no_think=True のとき、/no_think トークンをプロンプト先頭に付ける。
        # Qwen3の学習レベルで認識されるため、テンプレートやAPIフラグに依存しない。
        if self.no_think:
            prompt = "/no_think\n" + prompt

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                # Modelfileの num_predict 1024 を上書きして打ち切りを防ぐ。
                "num_predict": 2048,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def _call_and_log_raw(self, system_prompt: str, messages: list[dict]) -> str:
        """_call_api を実行してログ出力し、生レスポンス文字列を返す内部ヘルパー。

        generate / generate_stream_typed の共通API呼び出しロジックを集約する。
        エラー時はエラーメッセージ文字列を返す（例外は送出しない）。

        Args:
            system_prompt: システムプロンプト文字列。
            messages: _build_messages() で変換済みのメッセージリスト。

        Returns:
            /api/generate の "response" フィールド文字列。エラー時はエラー文字列。
        """
        clean_messages = self._build_messages(messages)
        self._log_request({"model": self.model, "system_prompt": system_prompt, "messages": clean_messages})
        try:
            result = await asyncio.to_thread(self._call_api, system_prompt, clean_messages)
            self._log_response(result)
            return result.get("response", "")
        except urllib.error.URLError as e:
            err = f"[Ollama接続エラー: {e.reason}. Ollamaが起動しているか確認してください]"
            self._log_error(err)
            return err
        except Exception as e:
            err = f"[Ollama エラー: {e}]"
            self._log_error(err)
            return err

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Ollamaから応答テキストを非同期で生成する。

        Qwen3系の <think>...</think> ブロックは本文から除去して返す。
        司会AIや非ストリームパスでは思考は使わないため破棄でよい。
        """
        raw = await self._call_and_log_raw(system_prompt, messages)
        _, clean = _split_think(raw)
        return clean

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """generate() の結果を単一チャンクとしてyieldするストリーム互換メソッド。"""
        text = await self.generate(system_prompt, messages)
        yield text

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """非ストリームで取得した結果を ("thinking", ...) と ("text", ...) に分けてyieldする。

        Ollama本体は stream: false で一括取得するため、Claudeのような逐次表示にはならない。
        ただし思考ブロックと本文を別チャンクとして流すことで、UI側はClaudeと同じ
        ThinkingBlock扱いで描画できる（chat service の typed-stream 契約に準拠）。

        Qwen3系の <think>...</think> がない通常モデル（qwen2.5など）では
        thinking 側が空文字となり、("text", ...) のみが流れる。

        _call_and_log_raw が "[Ollama接続エラー:..." / "[Ollama エラー:..." を返した
        場合は ("error", raw) として yield する。呼び出し側は ("error", ...) を出力に
        積まず、UI 表示・蒸留スキップなどの分岐を行う。
        """
        raw = await self._call_and_log_raw(system_prompt, messages)
        # エラー文字列は _call_and_log_raw の except 分岐のみで生成される（必ず "[Ollama" 始まり）。
        # 通常応答が偶然これに当たることはない想定。
        if raw.startswith("[Ollama接続エラー:") or raw.startswith("[Ollama エラー:"):
            yield ("error", raw)
            return
        thinking, clean = _split_think(raw)
        if thinking:
            yield ("thinking", thinking)
        if clean:
            yield ("text", clean)
