"""グループチャットサービス — 複数キャラクターのターン制御と LLM 呼び出しを担当する。

処理フロー（ユーザー発言ごとに1回実行）:
  1. 司会AIに次の発言者を問い合わせる
  2. 指名されたキャラクターの応答を（並列で）生成する
  3. 応答をDBに保存し、SSEイベントをyieldする
  4. auto_turns_used >= max_auto_turns または司会が [] を返したらユーザーターンへ戻す
  司会がNone（エラー）を返した場合もユーザーターンへ戻す。
"""

import asyncio
import uuid
from typing import Any, AsyncGenerator

from ..memory.format import format_recalled_memories
from ..memory.inscriber import carve
from ..memory.manager import MemoryManager
from ..providers.registry import create_provider
from ..system_prompt import build_system_prompt
from . import context as ctx
from .director import decide_next_speakers


def _extract_last_user_text(history: list) -> str:
    """履歴から最後のユーザーメッセージのテキストを抽出する。"""
    for msg in reversed(history):
        if msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                return content
    return ""


async def _generate_character_response(
    char_name: str,
    session_id: str,
    participants: list[dict],
    history: list,
    sqlite,
    memory_manager: MemoryManager,
    settings: dict,
) -> Any:
    """指定キャラクターの応答を生成してDBに保存し、ChatMessageオブジェクトを返す。

    generate_stream_typed を使って思考ブロック・想起記憶（reasoning）も収集し、
    DBへ保存する。

    Args:
        char_name: 応答するキャラクターの名前。
        session_id: 対象セッションID。
        participants: 参加者情報リスト（preset_name取得に使用）。
        history: 現在の会話履歴（ChatMessageオブジェクトのリスト）。
        sqlite: SQLiteStoreインスタンス。
        memory_manager: MemoryManagerインスタンス。
        settings: グローバル設定辞書。

    Returns:
        保存済みのChatMessageオブジェクト。

    Raises:
        ValueError: キャラクターまたはプリセットが見つからない場合。
        Exception: LLM呼び出しエラー。
    """
    # キャラクター情報を取得する
    char = sqlite.get_character_by_name(char_name) or sqlite.get_character(char_name)
    if not char:
        raise ValueError(f"キャラクター '{char_name}' が見つかりません")

    # 参加者情報からプリセット名を取得する
    participant = next((p for p in participants if p["char_name"] == char_name), None)
    if not participant:
        raise ValueError(f"参加者リストに '{char_name}' がありません")

    preset_name = participant["preset_name"]
    preset = sqlite.get_model_preset_by_name(preset_name) or sqlite.get_model_preset(preset_name)
    if not preset:
        raise ValueError(f"プリセット '{preset_name}' が見つかりません")

    model_config = (char.enabled_providers or {}).get(preset.id, {})

    # グループ履歴を指定キャラクター視点の messages 形式に変換する
    messages = ctx.format_group_history_for_character(history, char_name)

    # 最後のユーザーメッセージで記憶を想起する
    last_user_text = _extract_last_user_text(history)
    recalled = []
    if last_user_text:
        try:
            recalled = memory_manager.recall_memory(char.id, last_user_text)
        except Exception:
            pass

    # システムプロンプトを構築する
    system_prompt = build_system_prompt(
        character_system_prompt=char.system_prompt_block1,
        recalled_memories=recalled,
        fetched_contents=[],
        meta_instructions=char.meta_instructions,
        provider_additional_instructions=model_config.get("additional_instructions", ""),
        enable_time_awareness=False,
        current_time_str="",
        time_since_last_interaction="",
    )

    # LLM を呼び出して応答テキストと reasoning を収集する
    provider_impl = create_provider(
        preset.provider,
        preset.model_id,
        settings,
        thinking_level=preset.thinking_level or "default",
    )
    response_parts: list[str] = []
    reasoning_parts: list[str] = []
    async for chunk_type, chunk_text in provider_impl.generate_stream_typed(system_prompt, messages):
        if chunk_type == "text":
            response_parts.append(chunk_text)
        elif chunk_type in ("reasoning", "thinking"):
            reasoning_parts.append(chunk_text)

    response_text = "".join(response_parts)

    # 想起記憶テキストを reasoning の先頭に付加する（1on1 チャットと同様の表示にする）
    memory_text = format_recalled_memories(recalled)
    thinking_text = "".join(reasoning_parts)
    combined = (memory_text + thinking_text).strip()
    reasoning_text = combined if combined else None

    # [MEMORY:...] マーカーを抽出して記憶を刻み込む
    clean_text = carve(response_text, char.id, memory_manager)

    # DBに保存する（reasoning も含む）
    msg_id = str(uuid.uuid4())
    msg = sqlite.create_chat_message(
        message_id=msg_id,
        session_id=session_id,
        role="character",
        content=clean_text,
        character_name=char_name,
        reasoning=reasoning_text,
    )
    return msg


async def run_group_turn(
    session_id: str,
    group_config: dict,
    sqlite,
    memory_manager: MemoryManager,
    settings: dict,
) -> AsyncGenerator[tuple[str, Any], None]:
    """ユーザー発言後の自動ターンを実行し、SSEイベントをyieldする非同期ジェネレーター。

    Yields:
        ("speaker_decided", {"speakers": [str]})       — 次の発言者が決定した
        ("character_message", {"character": str, "message": dict}) — キャラクターの応答が完了した
        ("user_turn", {"auto_turns_used": int})         — ユーザーターンへ戻す
        ("error", {"message": str, "character": str})  — キャラクター応答エラー

    Args:
        session_id: 対象セッションID。
        group_config: グループチャット設定辞書。
        sqlite: SQLiteStoreインスタンス。
        memory_manager: MemoryManagerインスタンス。
        settings: グローバル設定辞書。
    """
    participants = group_config.get("participants", [])
    director_model_id = group_config.get("director_model_id", "")
    # "{char_name}@{preset_name}" 形式をパースする
    if "@" in director_model_id:
        director_char_name, director_preset_name = director_model_id.rsplit("@", 1)
    else:
        director_char_name, director_preset_name = "", ""
    max_auto_turns = int(group_config.get("max_auto_turns", 3))
    # グローバル設定からユーザー名を取得する
    user_name = settings.get("user_name", "ユーザ")

    auto_turn_count = 0

    while True:
        # 最新の会話履歴を取得する
        history = sqlite.list_chat_messages(session_id)

        # 司会AIに次の発言者を問い合わせる
        next_speakers = await decide_next_speakers(
            history=history,
            participants=participants,
            sqlite=sqlite,
            settings=settings,
            director_char_name=director_char_name,
            director_preset_name=director_preset_name,
            user_name=user_name,
        )

        # None はエラー（プリセット未発見・LLM障害）→ユーザーターンへ戻す
        if next_speakers is None:
            yield ("user_turn", {"auto_turns_used": auto_turn_count})
            return

        # [] は司会の意図的なユーザーターン指示、または上限超過
        if not next_speakers or auto_turn_count >= max_auto_turns:
            yield ("user_turn", {"auto_turns_used": auto_turn_count})
            return

        yield ("speaker_decided", {"speakers": next_speakers})

        # 指名されたキャラクターの応答を並列で生成する
        results = await asyncio.gather(
            *[
                _generate_character_response(
                    char_name=char_name,
                    session_id=session_id,
                    participants=participants,
                    history=history,
                    sqlite=sqlite,
                    memory_manager=memory_manager,
                    settings=settings,
                )
                for char_name in next_speakers
            ],
            return_exceptions=True,
        )

        for char_name, result in zip(next_speakers, results):
            if isinstance(result, Exception):
                # エラーはユーザーターンへ戻すがループは継続しない
                yield ("error", {"message": str(result), "character": char_name})
                yield ("user_turn", {"auto_turns_used": auto_turn_count})
                return

            # ChatMessage ORM を dict に変換してyieldする
            msg = result
            msg_dict = {
                "id": msg.id,
                "session_id": msg.session_id,
                "role": msg.role,
                "content": msg.content,
                "character_name": getattr(msg, "character_name", None),
                "reasoning": getattr(msg, "reasoning", None),
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
            yield ("character_message", {"character": char_name, "message": msg_dict})

        auto_turn_count += 1
