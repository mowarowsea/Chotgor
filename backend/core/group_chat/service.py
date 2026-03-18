"""グループチャットサービス — 複数キャラクターのターン制御と LLM 呼び出しを担当する。

処理フロー（ユーザー発言ごとに1回実行）:
  1. 司会AIに次の発言者を問い合わせる
  2. 指名されたキャラクターの応答を順番にストリーミング生成する（1on1を複数回やるのと同じ形）
  3. 各キャラクター応答をDBに保存し、SSEイベントをyieldする
  4. auto_turns_used >= max_auto_turns または司会が [] を返したらユーザーターンへ戻す
  司会がNone（エラー）を返した場合もユーザーターンへ戻す。

グループチャット固有のLLM呼び出しは ChatService.execute_stream() に委譲する。
1on1チャットと同じ記憶想起・URL取得・時刻認識・プロンプト構築・プロバイダーディスパッチ・記憶刻み込みを共用する。
"""

import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from ..chat.models import ChatRequest, Message
from ..chat.service import ChatService
from ..memory.format import format_recalled_memories
from ..utils import format_time_delta
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


async def _stream_character_response(
    char_name: str,
    session_id: str,
    participants: list[dict],
    history: list,
    sqlite,
    settings: dict,
    chat_service: ChatService,
    uploads_dir: str = "",
) -> AsyncGenerator[tuple[str, dict], None]:
    """指定キャラクターの応答をストリーミングしてDBに保存する非同期ジェネレーター。

    1on1チャットと同じフローを踏む：
    記憶想起 → URL取得 → 時刻認識 → システムプロンプト → プロバイダーストリーム → 記憶刻み込み → DB保存。

    Yields:
        ("character_reasoning", {"character": str, "content": str})  — 思考・想起記憶（リアルタイム）
        ("character_chunk",    {"character": str, "content": str})   — 応答テキスト（1チャンク）
        ("character_done",     {"character": str, "message": ORM})   — DB保存完了（ORM は呼び出し元で変換）

    Args:
        char_name: 応答するキャラクターの名前。
        session_id: 対象セッションID。
        participants: 参加者情報リスト（preset_id取得に使用）。
        history: 現在の会話履歴（ChatMessageオブジェクトのリスト）。
        sqlite: SQLiteStoreインスタンス。
        settings: グローバル設定辞書。
        chat_service: コアLLMロジックを委譲するChatServiceインスタンス。
        uploads_dir: 画像ファイルの保存ディレクトリパス（画像付きメッセージのエンコードに使用）。
    """
    # キャラクター情報を取得する
    char = sqlite.get_character_by_name(char_name) or sqlite.get_character(char_name)
    if not char:
        raise ValueError(f"キャラクター '{char_name}' が見つかりません")

    # 参加者情報からプリセットIDを取得する
    participant = next((p for p in participants if p["char_name"] == char_name), None)
    if not participant:
        raise ValueError(f"参加者リストに '{char_name}' がありません")

    preset_id = participant["preset_id"]
    preset = sqlite.get_model_preset(preset_id)
    if not preset:
        raise ValueError(f"プリセット '{preset_id}' が見つかりません")

    model_config = (char.enabled_providers or {}).get(preset.id, {})

    # グループ履歴を指定キャラクター視点の Message リストに変換する（画像も含む）
    messages = [
        Message(role=m["role"], content=m["content"])
        for m in ctx.format_group_history_for_character(history, char_name, sqlite=sqlite, uploads_dir=uploads_dir)
    ]

    # タグなしの生テキストを記憶想起クエリに使う
    last_user_text = _extract_last_user_text(history)

    # 時刻認識パラメータを計算する（1on1チャットと同様）
    enable_time_awareness = settings.get("enable_time_awareness", "true") == "true"
    now = datetime.now()
    current_time_str = ""
    time_since_last_interaction = ""
    if enable_time_awareness:
        current_time_str = now.isoformat(timespec="seconds")
        last_str = sqlite.get_setting(f"last_interaction_{char.id}")
        if last_str:
            try:
                last_dt = datetime.fromisoformat(last_str)
                time_since_last_interaction = format_time_delta(now - last_dt)
            except Exception:
                pass
    # インタラクション時刻を更新する（1on1と同様）
    sqlite.set_setting(f"last_interaction_{char.id}", now.isoformat())

    # ChatRequest を構築して ChatService に委譲する
    request = ChatRequest(
        character_id=char.id,
        character_name=char_name,
        provider=preset.provider,
        model=preset.model_id,
        messages=messages,
        character_system_prompt=char.system_prompt_block1,
        meta_instructions=char.meta_instructions,
        provider_additional_instructions=model_config.get("additional_instructions", ""),
        thinking_level=preset.thinking_level or "default",
        settings=settings,
        recall_query_override=last_user_text,
        enable_time_awareness=enable_time_awareness,
        current_time_str=current_time_str,
        time_since_last_interaction=time_since_last_interaction,
        session_id=session_id,
        current_preset_name=preset.name,
        current_preset_id=preset.id,
    )

    # execute_stream を通じてストリーミング実行しながらチャンクをリアルタイムでyieldする
    full_text = ""
    memory_text = ""
    thinking_parts: list[str] = []

    async for chunk_type, content in chat_service.execute_stream(request):
        if chunk_type == "memories":
            memory_text = format_recalled_memories(content)
            if memory_text:
                yield ("character_reasoning", {"character": char_name, "content": memory_text})
        elif chunk_type == "thinking":
            thinking_parts.append(content)
            yield ("character_reasoning", {"character": char_name, "content": content})
        elif chunk_type == "text":
            full_text = content
            if content:
                yield ("character_chunk", {"character": char_name, "content": content})

    # 1on1チャットと同様に想起記憶と思考ブロックをreasoningにまとめてDBに保存する
    combined = (memory_text + "".join(thinking_parts)).strip()
    reasoning_text = combined if combined else None

    msg_id = str(uuid.uuid4())
    msg = sqlite.create_chat_message(
        message_id=msg_id,
        session_id=session_id,
        role="character",
        content=full_text,
        character_name=char_name,
        reasoning=reasoning_text,
        preset_name=preset.name,
    )
    yield ("character_done", {"character": char_name, "message": msg})


async def run_group_turn(
    session_id: str,
    group_config: dict,
    sqlite,
    settings: dict,
    chat_service: ChatService,
    message_to_dict,
    uploads_dir: str = "",
) -> AsyncGenerator[tuple[str, Any], None]:
    """ユーザー発言後の自動ターンを実行し、SSEイベントをyieldする非同期ジェネレーター。

    各キャラクターを順番にストリーミング処理する（1on1を複数回行う形）。

    Yields:
        ("speaker_decided",     {"speakers": [str]})                  — 司会AIが発言者を一括決定
        ("character_start",     {"character": str})                   — キャラクター応答開始
        ("character_reasoning", {"character": str, "content": str})   — 思考・想起記憶（リアルタイム）
        ("character_chunk",     {"character": str, "content": str})   — 応答テキスト
        ("character_done",      {"character": str, "message": dict})  — 応答完了・DB保存済み
        ("user_turn",           {"auto_turns_used": int})             — ユーザーターンへ戻す
        ("error",               {"message": str, "character": str})   — エラー

    Args:
        session_id: 対象セッションID。
        group_config: グループチャット設定辞書。
        sqlite: SQLiteStoreインスタンス。
        settings: グローバル設定辞書。
        chat_service: 各キャラクターのLLM呼び出しに使用するChatServiceインスタンス。
        message_to_dict: ChatMessage ORM を dict に変換する関数（循環インポート回避のため外部から注入）。
        uploads_dir: 画像ファイルの保存ディレクトリパス。
    """
    participants = group_config.get("participants", [])
    director_char_name = group_config.get("director_char_name", "")
    director_preset_id = group_config.get("director_preset_id", "")
    max_auto_turns = int(group_config.get("max_auto_turns", 3))
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
            director_preset_id=director_preset_id,
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

        # 司会AIの決定を一括通知する
        yield ("speaker_decided", {"speakers": next_speakers})

        # 各キャラクターを順番にストリーミング処理する
        error_occurred = False
        for char_name in next_speakers:
            # キャラクター応答開始を通知する（フロントでスピナー表示に使う）
            yield ("character_start", {"character": char_name})
            try:
                async for chunk_type, payload in _stream_character_response(
                    char_name=char_name,
                    session_id=session_id,
                    participants=participants,
                    history=history,
                    sqlite=sqlite,
                    settings=settings,
                    chat_service=chat_service,
                    uploads_dir=uploads_dir,
                ):
                    if chunk_type == "character_done":
                        # message ORM を dict に変換してからyieldする
                        yield (chunk_type, {**payload, "message": message_to_dict(payload["message"])})
                    else:
                        yield (chunk_type, payload)
            except Exception as e:
                yield ("error", {"message": str(e), "character": char_name})
                yield ("user_turn", {"auto_turns_used": auto_turn_count})
                error_occurred = True
                break

        if error_occurred:
            return

        auto_turn_count += 1
