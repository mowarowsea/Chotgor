"""「キャラクターに聞く」共通サービス。

通常チャット以外（自己参照・バッチ処理など）でキャラクターへ問い合わせる際の
共通エントリポイント。プリセット解決・システムプロンプト構築・LLMコールを一本化する。

システムプロンプトは 1on1 チャット基準に統一する。
working_memory_manager を渡せば、1on1 チャットと同じ形でワーキングメモリの
全スレッド一覧（Block 6）・emotion/body/relation 固定注入（Block 7）が
システムプロンプトへ入る。recall_query も渡せば heat 想起（Block 8）も入る。
バッチ処理側でユーザメッセージにスレッドを別途埋め込んでいても、システム
プロンプトは 1on1 基準で統一する方針（重複は許容する）。

使い方
------
recall_query を渡すと記憶想起を実行してシステムプロンプトに注入する（chat 相当）。
None の場合は想起をスキップする（バッチ処理向け）。
ask_character_with_tools() は tool-use MCPループが必要なバッチ処理（forget蒸留等）向け。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.lib.log_context import current_log_feature
from backend.providers.registry import create_provider
from backend.services.chat.request_builder import build_system_prompt

if TYPE_CHECKING:
    from backend.character_actions.executor import ToolExecutor
    from backend.services.memory.manager import MemoryManager
    from backend.services.memory.working_memory_manager import WorkingMemoryManager
    from backend.repositories.sqlite.store import SQLiteStore


def _collect_wm_blocks(
    working_memory_manager: "WorkingMemoryManager | None",
    character_id: str,
    recall_query: str | None,
) -> tuple[list[dict] | None, list[dict] | None, list[dict] | None]:
    """システムプロンプト用のワーキングメモリ3系統を取得する。

    1on1 チャット（ChatService._build_context）と同じ3系統:
      - 全スレッド一覧（Open/Close 問わず・Block 6）
      - emotion/body/relation の固定注入（Block 7）
      - recall_query 指定時のみ heat 上位の task/topic 想起（Block 8）

    Args:
        working_memory_manager: WM マネージャー。None なら全て None を返す。
        character_id: 対象キャラクターID。
        recall_query: heat 想起のクエリ。None なら Block 8 をスキップ。

    Returns:
        (wm_all_threads, wm_fixed_threads, wm_recalled_threads) のタプル。
    """
    if working_memory_manager is None:
        return None, None, None
    wm_all = wm_fixed = wm_recalled = None
    try:
        wm_all = working_memory_manager.list_all_threads(character_id) or None
        wm_fixed = working_memory_manager.get_fixed_threads(character_id) or None
        if recall_query:
            wm_recalled = (
                working_memory_manager.recall_threads(character_id, recall_query) or None
            )
    except Exception as e:
        _log.warning("ワーキングメモリ取得失敗 character_id=%s error=%s", character_id, e)
    return wm_all, wm_fixed, wm_recalled

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
    working_memory_manager: "WorkingMemoryManager | None" = None,
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
        working_memory_manager: ワーキングメモリのマネージャー。指定すると 1on1 チャットと
                          同じ形で全スレッド一覧（Block 6）・emotion/body/relation 固定注入
                          （Block 7）をシステムプロンプトへ入れる。recall_query も併せて
                          渡せば heat 想起（Block 8）も入る。None なら WM ブロックは入らない。

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

    wm_all_threads, wm_fixed_threads, wm_recalled_threads = _collect_wm_blocks(
        working_memory_manager, character_id, recall_query
    )

    system_prompt = build_system_prompt(
        character_system_prompt=char.system_prompt_block1 or "",
        recalled_memories=recalled,
        recalled_identity_memories=recalled_identity,
        inner_narrative=char.inner_narrative or "",
        wm_all_threads=wm_all_threads,
        wm_fixed_threads=wm_fixed_threads,
        wm_recalled_threads=wm_recalled_threads,
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


async def ask_character_with_tools(
    character_id: str,
    preset_id: str,
    messages: list[dict],
    sqlite: "SQLiteStore",
    settings: dict,
    memory_manager: "MemoryManager",
    feature_label: str = "",
    session_id: str = "",
    working_memory_manager: "WorkingMemoryManager | None" = None,
    recall_query: str | None = None,
    batch_context: dict | None = None,
) -> bool:
    """tool-use MCPループを使ってキャラクターに問いかける。

    inscribe_memory 等のツールを呼び出しながら応答を生成する。
    テキスト応答は捨てる（バッチ処理向け）。

    Args:
        character_id: 問い合わせ対象のキャラクターID。
        preset_id: 使用するモデルプリセットID。
        messages: LLMに渡す会話メッセージリスト。
        sqlite: SQLiteStore インスタンス。
        settings: グローバル設定 dict。
        memory_manager: 記憶の読み書きに使用するマネージャー。
        feature_label: ログ識別用のフィーチャーラベル。
        session_id: セッションID。ツール実行のコンテキストとして使用する。
        working_memory_manager: ワーキングメモリのマネージャー。指定すると 1on1 チャットと
                          同じ形で全スレッド一覧（Block 6）・固定注入（Block 7）を
                          システムプロンプトへ入れ、ツール実行にもこのインスタンスを使う。
                          None の場合は内部で生成する。
        recall_query: heat 想起（Block 8）のクエリ。None ならスキップ。
        batch_context: バッチ処理由来のツール挙動切り替えフラグ。例えば forget 蒸留バッチは
            ``{"force_insert_memory": True}`` を渡すことで、inscribe_memory ツールが
            類似既存記憶を上書きせず必ず新規 ID で挿入するようになる。通常チャットでは None。

    Returns:
        True: tool-use ループを正常に実行した。
        False: プロバイダーが tool-use に非対応、またはエラー発生（呼び出し元でフォールバック）。
    """
    from backend.character_actions.executor import ToolExecutor
    from backend.services.memory.working_memory_manager import WorkingMemoryManager

    char = sqlite.get_character(character_id)
    if not char:
        _log.warning("ask_character_with_tools: キャラクター未発見 character_id=%s", character_id)
        return False

    preset = sqlite.get_model_preset(preset_id)
    if preset is None:
        _log.warning(
            "ask_character_with_tools: プリセット未発見 character_id=%s preset_id=%s",
            character_id, preset_id,
        )
        return False

    try:
        if feature_label:
            current_log_feature.set(feature_label)
        provider = create_provider(
            preset.provider,
            preset.model_id or "",
            settings,
            thinking_level=preset.thinking_level or "default",
            preset_name=preset.name,
            character_id=character_id,
            session_id=session_id,
        )
    except Exception as e:
        _log.warning(
            "ask_character_with_tools: プロバイダー生成失敗 character_id=%s preset=%s error=%s",
            character_id, preset_id, e,
        )
        return False

    if not provider.SUPPORTS_TOOLS:
        _log.info(
            "ask_character_with_tools: tool-use非対応プロバイダー character_id=%s provider=%s",
            character_id, preset.provider,
        )
        return False

    # WM マネージャーは未指定なら内部生成する。システムプロンプトの WM ブロックと
    # ツール実行（post_thread 等）の両方で同じインスタンスを使う。
    wm = working_memory_manager or WorkingMemoryManager(
        sqlite=sqlite, vector_store=memory_manager.vector_store
    )
    wm_all_threads, wm_fixed_threads, wm_recalled_threads = _collect_wm_blocks(
        wm, character_id, recall_query
    )

    system_prompt = build_system_prompt(
        character_system_prompt=char.system_prompt_block1 or "",
        recalled_memories=[],
        recalled_identity_memories=[],
        inner_narrative=char.inner_narrative or "",
        wm_all_threads=wm_all_threads,
        wm_fixed_threads=wm_fixed_threads,
        wm_recalled_threads=wm_recalled_threads,
        use_tools=True,
    )

    tool_executor = ToolExecutor(
        character_id=character_id,
        session_id=session_id or None,
        memory_manager=memory_manager,
        working_memory_manager=wm,
        batch_context=batch_context,
    )

    try:
        await provider.generate_with_tools(system_prompt, messages, tool_executor)
    except Exception as e:
        _log.warning(
            "ask_character_with_tools: LLMコール失敗 character_id=%s preset=%s error=%s",
            character_id, preset_id, e,
        )
        return False

    char_label = f"{char.name}@{preset.name or preset.provider}"
    _log.info(
        "ask_character_with_tools: 完了 char=%s feature=%s",
        char_label, feature_label or "(none)",
    )
    return True
