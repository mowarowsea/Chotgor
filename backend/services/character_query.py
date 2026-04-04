"""「キャラクターに聞く」共通サービス。

通常チャット以外（自己参照・バッチ処理など）でキャラクターへ問い合わせる際の
共通エントリポイント。プリセット解決・システムプロンプト構築・LLMコールを一本化する。

使い方
------
recall_query を渡すと記憶想起を実行してシステムプロンプトに注入する（chat 相当）。
None の場合は想起をスキップする（バッチ処理向け）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.lib.log_context import current_log_feature
from backend.providers.registry import create_provider
from backend.services.chat.request_builder import build_system_prompt

if TYPE_CHECKING:
    from backend.services.memory.manager import MemoryManager
    from backend.repositories.sqlite.store import SQLiteStore

_log = logging.getLogger(__name__)


async def ask_character(
    character_id: str,
    preset_id: str,
    messages: list[dict],
    sqlite: "SQLiteStore",
    settings: dict,
    memory_manager: "MemoryManager | None" = None,
    recall_query: str | None = None,
    feature_label: str = "",
) -> str | None:
    """キャラクターに問いかけ、テキスト応答を返す。

    プリセット解決・システムプロンプト構築・LLMコールを統一的に処理する。
    recall_query を渡すと記憶想起を実行してシステムプロンプトに記憶を注入する。
    None の場合は想起をスキップし、キャラクター基本設定のみでシステムプロンプトを構築する。

    Args:
        character_id: 問い合わせ対象のキャラクターID。
        preset_id: 使用するモデルプリセットID。
        messages: LLMに渡す会話メッセージリスト（user/assistant ロール）。
        sqlite: SQLiteStore インスタンス。
        settings: グローバル設定 dict（get_all_settings() の結果）。
        memory_manager: 記憶想起に使用するマネージャー。recall_query が None の場合は不要。
        recall_query: 記憶想起クエリ。指定するとChromaDB検索を実行してシステムプロンプトに注入する。
                      None の場合は想起をスキップする。
        feature_label: ログ識別用のフィーチャーラベル（例: "chronicle", "forget", "reflection"）。

    Returns:
        LLM の応答テキスト。エラー時は None。
    """
    char = sqlite.get_character(character_id)
    if not char:
        _log.warning("ask_character: キャラクター未発見 character_id=%s", character_id)
        return None

    preset = sqlite.get_model_preset(preset_id)
    if preset is None:
        _log.warning(
            "ask_character: プリセット未発見 character_id=%s preset_id=%s",
            character_id, preset_id,
        )
        return None

    try:
        if feature_label:
            current_log_feature.set(feature_label)
        provider = create_provider(
            preset.provider,
            preset.model_id or "",
            settings,
            thinking_level=preset.thinking_level or "default",
            preset_name=preset.name,
        )
    except Exception as e:
        _log.warning(
            "ask_character: プロバイダー生成失敗 character_id=%s preset=%s error=%s",
            character_id, preset_id, e,
        )
        return None

    recalled_identity: list[dict] = []
    recalled: list[dict] = []
    if recall_query and memory_manager:
        try:
            recalled_identity, recalled = memory_manager.recall_with_identity(
                character_id, recall_query
            )
        except Exception as e:
            _log.warning(
                "ask_character: 記憶想起失敗 character_id=%s error=%s",
                character_id, e,
            )

    system_prompt = build_system_prompt(
        character_system_prompt=char.system_prompt_block1 or "",
        recalled_memories=recalled,
        recalled_identity_memories=recalled_identity,
        self_history=char.self_history or "",
        relationship_state=char.relationship_state or "",
        inner_narrative=char.inner_narrative or "",
    )

    try:
        result = await provider.generate(system_prompt, messages)
    except Exception as e:
        _log.warning(
            "ask_character: LLMコール失敗 character_id=%s preset=%s error=%s",
            character_id, preset_id, e,
        )
        return None

    char_label = f"{char.name}@{preset.name or preset.provider}"
    _log.info(
        "ask_character: 完了 char=%s feature=%s response_chars=%d",
        char_label, feature_label or "(none)", len(result or ""),
    )
    return result
