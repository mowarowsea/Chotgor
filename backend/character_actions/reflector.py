"""自己参照ループ — 返答生成後にキャラクターが自己観察・内省を行う機構。

登録済みプリセット（任意のプロバイダー）を契機判断フィルターとして使い、
必要なターンのみパブリックLLMへ2次コールして自己参照を実行する。

動作モード:
    disabled       : 自己参照なし（デフォルト）
    local_trigger  : プリセットで契機判断 → YESのときのみパブリックLLMで自己参照
    always         : 毎ターンパブリックLLMで自己参照（コスト大）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.providers.registry import create_provider
from backend.lib.log_context import current_log_feature
from backend.lib.character_context import build_character_context

if TYPE_CHECKING:
    from backend.providers.base import BaseLLMProvider
    from backend.services.memory.drift_manager import DriftManager
    from backend.services.memory.manager import MemoryManager

_log = logging.getLogger(__name__)

# 契機判断のユーザーメッセージ。
# system_prompt にキャラクター設定を渡し、「あなたにとって」という問いかけにする。
_TRIGGER_USER_TEMPLATE = """\
以下の会話を読んで判断してください。

あなたが「自己参照」すべき重要な瞬間があるかを判定します。
「自己参照」とは：自分自身の気づき・感情の変化・価値観の変容・成長などを記録する行為です。

少しでもそのような兆しがあればYESとしてください（誤検知歓迎）。

「YES」または「NO」のみ答えてください。

[会話]
{conversation}
"""

# パブリックLLM向け自己参照のユーザーメッセージ。
# system_prompt にキャラクター設定を渡し、「あなたにとって」という問いかけにする。
_REFLECTION_USER_TEMPLATE = """\
以下の会話を読み、あなたとして自己参照が必要な気づきや変化があればタグのみで出力してください。

使えるタグ:
[CARVE_NARRATIVE:append|内容] — 自己の認識・成長・変容を inner_narrative に刻む
[DRIFT:内容]                   — 一時的な感情・状態の変化を記録する

内省が不要な場合は何も出力しないこと。説明文は一切不要。タグのみ出力してください。

[会話]
{conversation}
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
        character_system_prompt: str = "",
        inner_narrative: str = "",
        self_history: str = "",
        relationship_state: str = "",
    ) -> bool:
        """登録済みプリセットに契機判断を問い合わせ、自己参照すべきか否かを返す。

        キャラクター設定を先頭に添えることで、
        「このキャラクターにとって自己参照が必要か」という文脈で判断させる。

        Args:
            trigger_preset_id: 使用するモデルプリセットID（任意のプロバイダー）。
            settings: グローバル設定 dict。
            conversation_window: 直近Nターンのメッセージリスト。
            character_system_prompt: キャラクター基本設定テキスト。
            inner_narrative: キャラクターが自己記述した inner_narrative。
            self_history: キャラクターの歴史・経緯。
            relationship_state: ユーザ・他キャラとの現在の関係。

        Returns:
            自己参照すべきであれば True。接続エラー等の場合も False。
        """
        # プリセットIDからプロバイダー・モデル名を解決する
        preset = self.memory_manager.sqlite.get_model_preset(trigger_preset_id)
        if preset is None:
            _log.warning("自己参照: 契機判断プリセット未発見 preset_id=%s", trigger_preset_id)
            return False

        try:
            current_log_feature.set("trigger")
            trigger_provider = create_provider(
                preset.provider, preset.model_id or "", settings,
                preset_name=preset.name,
            )
        except Exception as e:
            _log.warning(
                "自己参照: 契機判断プロバイダー生成失敗 preset=%s provider=%s error=%s",
                trigger_preset_id, preset.provider, e,
            )
            return False

        # キャラクター設定をシステムプロンプトに、問いかけをユーザーメッセージに分離する
        system_prompt = build_character_context(
            character_system_prompt, inner_narrative, self_history, relationship_state,
        )
        conversation_text = _format_conversation(conversation_window)
        user_message = _TRIGGER_USER_TEMPLATE.format(conversation=conversation_text)

        try:
            response = await trigger_provider.generate(system_prompt, [{"role": "user", "content": user_message}])
        except Exception as e:
            _log.warning("自己参照: 契機判断失敗 provider=%s error=%s", preset.provider, e)
            return False

        result = response.strip().upper()
        triggered = result.startswith("YES")
        _log.info(
            "自己参照: 契機判断結果=%s provider=%s raw=%.30s",
            "YES" if triggered else "NO", preset.provider, response,
        )
        return triggered

    async def _run_reflection(
        self,
        public_provider: "BaseLLMProvider",
        conversation_window: list[dict],
        character_id: str,
        session_id: str,
        current_preset_id: str,
        character_system_prompt: str = "",
        inner_narrative: str = "",
        self_history: str = "",
        relationship_state: str = "",
    ) -> None:
        """パブリックLLMに自己参照コールし、タグを解析してDBに反映する。

        キャラクター設定（character_system_prompt / inner_narrative / self_history /
        relationship_state）をシステムプロンプトに組み込んで「キャラクターに聞いた」
        体で呼び出す。

        生成されたテキストから [CARVE_NARRATIVE:...] / [DRIFT:...] タグを抽出し、
        Carver / Drifter を通じてDBに書き込む。

        Args:
            public_provider: 自己参照コールに使うプロバイダー（メイン会話と同じもの）。
            conversation_window: 直近Nターンのメッセージリスト。
            character_id: 自己参照を行うキャラクターID。
            session_id: 現在のセッションID（DRIFT操作に必要）。
            current_preset_id: 記憶作成時の出所プリセットID。
            character_system_prompt: キャラクター基本設定テキスト。
            inner_narrative: キャラクターが自己記述した inner_narrative。
            self_history: キャラクターの歴史・経緯。
            relationship_state: ユーザ・他キャラとの現在の関係。
        """
        from backend.character_actions.carver import Carver
        from backend.character_actions.drifter import Drifter

        # キャラクター設定をシステムプロンプトに、問いかけをユーザーメッセージに分離する
        system_prompt = build_character_context(
            character_system_prompt, inner_narrative, self_history, relationship_state,
        )
        conversation_text = _format_conversation(conversation_window)
        user_message = _REFLECTION_USER_TEMPLATE.format(conversation=conversation_text)
        messages = [{"role": "user", "content": user_message}]

        current_log_feature.set("reflection")
        try:
            reflection_text = await public_provider.generate(system_prompt, messages)
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
        character_system_prompt: str = "",
        inner_narrative: str = "",
        self_history: str = "",
        relationship_state: str = "",
    ) -> None:
        """自己参照ループを実行する。

        メインの返答生成後に呼び出す。ユーザーへのストリームには一切影響しない。

        Args:
            request_mode: 動作モード（disabled / local_trigger / always）。
            trigger_preset_id: 契機判断プリセットID（local_trigger 時に使用、任意プロバイダー可）。
            n_turns: 自己参照に使う直近ターン数。
            public_provider: 自己参照コールに使うプロバイダー（メイン会話と同じもの）。
            settings: グローバル設定 dict。
            messages: 会話履歴（最新のキャラクター応答を含む）。
            character_id: 自己参照を行うキャラクターID。
            session_id: 現在のセッションID。
            current_preset_id: 記憶作成時の出所プリセットID。
            character_system_prompt: キャラクター基本設定テキスト。
            inner_narrative: キャラクターが自己記述した inner_narrative。
            self_history: キャラクターの歴史・経緯。
            relationship_state: ユーザ・他キャラとの現在の関係。
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
            triggered = await self._detect_trigger(
                trigger_preset_id, settings, window,
                character_system_prompt=character_system_prompt,
                inner_narrative=inner_narrative,
                self_history=self_history,
                relationship_state=relationship_state,
            )
            if not triggered:
                return

        # always もしくは local_trigger で契機あり → 自己参照実行
        await self._run_reflection(
            public_provider=public_provider,
            conversation_window=window,
            character_id=character_id,
            session_id=session_id,
            current_preset_id=current_preset_id,
            character_system_prompt=character_system_prompt,
            inner_narrative=inner_narrative,
            self_history=self_history,
            relationship_state=relationship_state,
        )
