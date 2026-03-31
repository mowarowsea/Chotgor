"""自己参照ループ — 返答生成後にキャラクターが自己観察・内省を行う機構。

Ollama（ローカルLLM）を契機判断フィルターとして使い、
必要なターンのみパブリックLLMへ2次コールして自己参照を実行する。

動作モード:
    disabled       : 自己参照なし（デフォルト）
    local_trigger  : Ollamaで契機判断 → YESのときのみパブリックLLMで自己参照
    always         : 毎ターンパブリックLLMで自己参照（Ollama不要・コスト大）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.providers.registry import create_provider

if TYPE_CHECKING:
    from backend.providers.base import BaseLLMProvider
    from backend.services.memory.drift_manager import DriftManager
    from backend.services.memory.manager import MemoryManager

_log = logging.getLogger(__name__)

# Ollama向け契機判断プロンプト。
# 誤検知はパブリックLLMが最終判断するため感度は高めに設定する。
_TRIGGER_PROMPT_TEMPLATE = """\
以下の会話を読んで判断してください。

キャラクターが「自己参照」すべき重要な瞬間があるかを判定します。
「自己参照」とは：自分自身の気づき・感情の変化・価値観の変容・成長などを記録する行為です。

少しでもそのような兆しがあればYESとしてください（誤検知歓迎）。

「YES」または「NO」のみ答えてください。

[会話]
{conversation}
"""

# パブリックLLM向け自己参照プロンプト。
# キャラクタープロンプトは注入しない（内容の判断はキャラクター自身に委ねる）。
_REFLECTION_SYSTEM = """\
以下の会話を読み、自己参照が必要な気づきや変化があればタグのみで出力してください。

使えるタグ:
[CARVE_NARRATIVE:append|内容] — 自己の認識・成長・変容を inner_narrative に刻む
[DRIFT:内容]                   — 一時的な感情・状態の変化を記録する

内省が不要な場合は何も出力しないこと。説明文は一切不要。タグのみ出力してください。
"""


def _format_conversation(messages: list[dict]) -> str:
    """メッセージリストを人が読める会話テキストに整形する。

    Args:
        messages: {"role": str, "content": str} の辞書リスト。

    Returns:
        "[ユーザー]: ...\n[キャラクター]: ..." 形式の文字列。
    """
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "user":
            lines.append(f"[ユーザー]: {content}")
        elif role in ("assistant", "character"):
            lines.append(f"[キャラクター]: {content}")
    return "\n".join(lines)


class SelfReflector:
    """自己参照ループを実行するクラス。

    Attributes:
        memory_manager: 記憶の読み書きを担うマネージャー。
        drift_manager: SELF_DRIFT指針の読み書きを担うマネージャー。
    """

    def __init__(
        self,
        memory_manager: "MemoryManager",
        drift_manager: "DriftManager | None",
    ) -> None:
        """SelfReflector を初期化する。"""
        self.memory_manager = memory_manager
        self.drift_manager = drift_manager

    async def _detect_trigger(
        self,
        trigger_preset_id: str,
        settings: dict,
        conversation_window: list[dict],
    ) -> bool:
        """Ollamaに契機判断を問い合わせ、自己参照すべきか否かを返す。

        Args:
            trigger_preset_id: 使用するOllamaモデルプリセットID。
            settings: グローバル設定 dict（ollama_base_url 等を含む）。
            conversation_window: 直近Nターンのメッセージリスト。

        Returns:
            自己参照すべきであれば True。Ollama接続エラー等の場合も False。
        """
        # プリセットIDからモデル名を解決する
        preset = self.memory_manager.sqlite.get_model_preset(trigger_preset_id)
        if preset is None:
            _log.warning("自己参照: 契機判断プリセット未発見 preset_id=%s", trigger_preset_id)
            return False

        try:
            ollama_provider = create_provider("ollama", preset.model_id or "", settings)
        except Exception as e:
            _log.warning("自己参照: Ollamaプロバイダー生成失敗 preset=%s error=%s", trigger_preset_id, e)
            return False

        conversation_text = _format_conversation(conversation_window)
        prompt = _TRIGGER_PROMPT_TEMPLATE.format(conversation=conversation_text)

        try:
            response = await ollama_provider.generate("", [{"role": "user", "content": prompt}])
        except Exception as e:
            _log.warning("自己参照: Ollama契機判断失敗 error=%s", e)
            return False

        result = response.strip().upper()
        triggered = result.startswith("YES")
        _log.info("自己参照: 契機判断結果=%s raw=%.30s", "YES" if triggered else "NO", response)
        return triggered

    async def _run_reflection(
        self,
        public_provider: "BaseLLMProvider",
        conversation_window: list[dict],
        character_id: str,
        session_id: str,
        current_preset_id: str,
    ) -> None:
        """パブリックLLMに自己参照コールし、タグを解析してDBに反映する。

        生成されたテキストから [CARVE_NARRATIVE:...] / [DRIFT:...] タグを抽出し、
        Carver / Drifter を通じてDBに書き込む。

        Args:
            public_provider: 自己参照コールに使うプロバイダー（メイン会話と同じもの）。
            conversation_window: 直近Nターンのメッセージリスト。
            character_id: 自己参照を行うキャラクターID。
            session_id: 現在のセッションID（DRIFT操作に必要）。
            current_preset_id: 記憶作成時の出所プリセットID。
        """
        from backend.character_actions.carver import Carver
        from backend.character_actions.drifter import Drifter

        messages = [{"role": "user", "content": _format_conversation(conversation_window)}]

        try:
            reflection_text = await public_provider.generate(_REFLECTION_SYSTEM, messages)
        except Exception as e:
            _log.warning("自己参照: パブリックLLMコール失敗 char=%s error=%s", character_id, e)
            return

        if not reflection_text or not reflection_text.strip():
            _log.info("自己参照: 内省なし char=%s", character_id)
            return

        _log.info("自己参照: 内省テキスト取得 char=%s text=%.100s", character_id, reflection_text)

        # CARVE_NARRATIVE タグを処理して inner_narrative を更新する
        carver = Carver(character_id, self.memory_manager.sqlite)
        after_carve = carver.carve_narrative_from_text(reflection_text)

        # DRIFT タグを処理してセッション指針を更新する
        drifter = Drifter(session_id, character_id, self.drift_manager)
        drifter.drift_from_text(after_carve)

    async def run(
        self,
        request_mode: str,
        trigger_preset_id: str,
        n_turns: int,
        public_provider: "BaseLLMProvider",
        settings: dict,
        messages: list[dict],
        character_id: str,
        session_id: str,
        current_preset_id: str,
    ) -> None:
        """自己参照ループを実行する。

        メインの返答生成後に呼び出す。ユーザーへのストリームには一切影響しない。

        Args:
            request_mode: 動作モード（disabled / local_trigger / always）。
            trigger_preset_id: Ollama契機判断プリセットID（local_trigger 時に使用）。
            n_turns: 自己参照に使う直近ターン数。
            public_provider: 自己参照コールに使うプロバイダー（メイン会話と同じもの）。
            settings: グローバル設定 dict。
            messages: 会話履歴（最新のキャラクター応答を含む）。
            character_id: 自己参照を行うキャラクターID。
            session_id: 現在のセッションID。
            current_preset_id: 記憶作成時の出所プリセットID。
        """
        if request_mode == "disabled":
            return

        # 直近Nターンの会話ウィンドウを切り出す
        window = messages[-n_turns:] if len(messages) > n_turns else messages

        if not window:
            return

        if request_mode == "local_trigger":
            if not trigger_preset_id:
                _log.warning("自己参照: local_trigger だが self_reflection_preset_id 未設定 char=%s", character_id)
                return
            triggered = await self._detect_trigger(trigger_preset_id, settings, window)
            if not triggered:
                return

        # always もしくは local_trigger で契機あり → 自己参照実行
        await self._run_reflection(
            public_provider=public_provider,
            conversation_window=window,
            character_id=character_id,
            session_id=session_id,
            current_preset_id=current_preset_id,
        )
