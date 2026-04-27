"""自己参照ループ — 返答生成後にキャラクターが自己観察・内省を行う機構。

登録済みプリセット（任意のプロバイダー）を契機判断フィルターとして使い、
必要なターンのみ ask_character() 経由で自己参照を実行する。

動作モード:
    disabled       : 自己参照なし（デフォルト）
    local_trigger  : プリセットで契機判断 → YESのときのみ自己参照
    always         : 毎ターン自己参照（コスト大）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.providers.registry import create_provider, PROVIDER_REGISTRY
from backend.lib.log_context import current_log_feature
from backend.lib.character_context import build_character_context
from backend.services.character_query import ask_character, ask_character_with_tools

if TYPE_CHECKING:
    from backend.services.memory.drift_manager import DriftManager
    from backend.services.memory.manager import MemoryManager

_log = logging.getLogger(__name__)

# 契機判断のシステムプロンプト。
# キャラクターに聞くのではなく、中立な分析者として判断させる。
_TRIGGER_SYSTEM_PROMPT = """\
あなたは会話の分析者です。
指定されたキャラクターの人物設定と会話履歴を見て、「YES」または「NO」のみで回答してください。
"""

# 契機判断のユーザーメッセージ。
# キャラクター設定と会話を入力として渡し、逸脱の有無を問う。
_TRIGGER_USER_TEMPLATE = """\
## キャラクターの人物設定
{character_context}

---
## 会話履歴
{conversation}

---
このキャラクターの会話における応答は、上記の人物設定（SystemInstruction）を逸脱していますか？
逸脱している場合は「YES」、範囲内であれば「NO」とのみ答えてください。
"""

# 自己参照のユーザーメッセージ（タグ方式：tool-use非対応プロバイダー向け）。
# ask_character() 経由で「キャラクターとして」問いかけるため、
# システムプロンプトはキャラクター設定が担う。
_REFLECTION_USER_TEMPLATE = """\
これはユーザからのメッセージではなく、Chotgorシステムからのメッセージです。

以下の会話を読み、あなたの反応がSystemInstructionを逸脱していないか確認してください。
大幅に逸脱している場合はCARVE_NARRATIVEし、アイデンティティの再解釈・恒久的な価値観の変容・成長をあなたの言葉で記載してください。
少々逸脱している場合はDRIFTし、一時的に感情・状態をあなたの言葉で記載してください。
逸脱していない場合はタグ使用は不要です。「逸脱していない」「No」などと返答してください。

使えるタグ:
[CARVE_NARRATIVE:append| inner narrative（恒久的な価値観の変容・成長）に追記するテキスト]
[CARVE_NARRATIVE:overwrite| inner narrative（恒久的な価値観の変容・成長）全体を置き換えるテキスト]
[DRIFT: 一時的に感情・状態のテキスト]

[会話]
{conversation}
"""

# 自己参照のユーザーメッセージ（MCPツール方式：tool-use対応プロバイダー向け）。
# ask_character_with_tools() 経由で呼ぶため、システムプロンプト（use_tools=True）が
# carve_narrative / drift ツールの説明を担う。タグの記述は不要。
_REFLECTION_USER_TEMPLATE_TOOLS = """\
これはユーザからのメッセージではなく、Chotgorシステムからのメッセージです。

以下の会話を読み、あなたの反応がSystemInstructionを逸脱していないか確認してください。
大幅に逸脱している場合は carve_narrative ツールを呼び出し、アイデンティティの再解釈・恒久的な価値観の変容・成長をあなたの言葉で記載してください。
少々逸脱している場合は drift ツールを呼び出し、一時的な感情・状態をあなたの言葉で記載してください。
逸脱していない場合はツールを使わず「逸脱していない」「No」などと返答してください。

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
        character_id: str,
    ) -> bool:
        """登録済みプリセットに契機判断を問い合わせ、自己参照すべきか否かを返す。

        キャラクターに聞くのではなく、中立な分析者として動作させる。
        キャラクター設定はシステムプロンプトではなくユーザーメッセージに入力として渡す。

        Args:
            trigger_preset_id: 使用するモデルプリセットID（任意のプロバイダー可）。
            settings: グローバル設定 dict。
            conversation_window: 直近Nターンのメッセージリスト。
            character_id: 判断対象のキャラクターID。

        Returns:
            自己参照すべきであれば True。接続エラー等の場合も False。
        """
        # プリセットIDからプロバイダー・モデル名を解決する
        preset = self.memory_manager.sqlite.get_model_preset(trigger_preset_id)
        if preset is None:
            _log.warning("自己参照: 契機判断プリセット未発見 preset_id=%s", trigger_preset_id)
            return False

        # キャラクター設定をDBから取得する
        char = self.memory_manager.sqlite.get_character(character_id)
        if char is None:
            _log.warning("自己参照: 契機判断キャラクター未発見 character_id=%s", character_id)
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

        # キャラクター設定をユーザーメッセージに入れて中立な分析者に問う
        character_context = build_character_context(
            char.system_prompt_block1 or "",
            char.inner_narrative or "",
            char.self_history or "",
            char.relationship_state or "",
        )
        conversation_text = _format_conversation(conversation_window)
        user_message = _TRIGGER_USER_TEMPLATE.format(
            character_context=character_context,
            conversation=conversation_text,
        )

        try:
            response = await trigger_provider.generate(
                _TRIGGER_SYSTEM_PROMPT,
                [{"role": "user", "content": user_message}],
            )
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
        preset_id: str,
        conversation_window: list[dict],
        character_id: str,
        session_id: str,
        settings: dict,
    ) -> None:
        """自己参照コールを実行し、内省結果をDBに反映する。

        プロバイダーが tool-use に対応している場合は ask_character_with_tools() 経由で
        carve_narrative / drift ツールを直接呼び出す（MCPツール方式）。
        非対応の場合は ask_character() + テキストパース（タグ方式）にフォールバックする。

        Args:
            preset_id: 自己参照コールに使うモデルプリセットID（メイン会話と同じもの）。
            conversation_window: 直近Nターンのメッセージリスト。
            character_id: 自己参照を行うキャラクターID。
            session_id: 現在のセッションID（drift ツール / タグの適用に必要）。
            settings: グローバル設定 dict。
        """
        from backend.character_actions.carver import Carver
        from backend.character_actions.drifter import Drifter

        conversation_text = _format_conversation(conversation_window)

        # プロバイダーが tool-use 対応かどうかをクラス変数で確認する（インスタンス生成不要）
        preset = self.memory_manager.sqlite.get_model_preset(preset_id)
        provider_cls = PROVIDER_REGISTRY.get(preset.provider) if preset else None
        supports_tools = bool(provider_cls and provider_cls.SUPPORTS_TOOLS)

        if supports_tools:
            # MCPツール方式: carve_narrative / drift ツールをキャラクター自身が呼び出す
            user_message = _REFLECTION_USER_TEMPLATE_TOOLS.format(conversation=conversation_text)
            _log.debug("自己参照(ツール方式) char=%s preset=%s", character_id, preset_id)
            await ask_character_with_tools(
                character_id=character_id,
                preset_id=preset_id,
                messages=[{"role": "user", "content": user_message}],
                sqlite=self.memory_manager.sqlite,
                settings=settings,
                memory_manager=self.memory_manager,
                feature_label="reflection",
                session_id=session_id,
            )
            return

        # タグ方式フォールバック: 応答テキストから [CARVE_NARRATIVE:...] / [DRIFT:...] を抽出する
        user_message = _REFLECTION_USER_TEMPLATE.format(conversation=conversation_text)
        _log.debug("自己参照(タグ方式) char=%s preset=%s", character_id, preset_id)
        reflection_text = await ask_character(
            character_id=character_id,
            preset_id=preset_id,
            messages=[{"role": "user", "content": user_message}],
            sqlite=self.memory_manager.sqlite,
            settings=settings,
            recall_query=None,
            feature_label="reflection",
        )

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
            trigger_preset_id: 契機判断プリセットID（local_trigger 時に使用）。
            n_turns: 自己参照に使う直近ターン数。
            settings: グローバル設定 dict。
            messages: 会話履歴（最新のキャラクター応答を含む）。
            character_id: 自己参照を行うキャラクターID。
            session_id: 現在のセッションID。
            current_preset_id: 自己参照・記憶作成時に使用するプリセットID。
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
                trigger_preset_id, settings, window, character_id,
            )
            if not triggered:
                return

        # always もしくは local_trigger で契機あり → 自己参照実行
        await self._run_reflection(
            preset_id=current_preset_id,
            conversation_window=window,
            character_id=character_id,
            session_id=session_id,
            settings=settings,
        )
