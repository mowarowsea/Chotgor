"""グループチャットサービス — 複数キャラクターのターン制御と LLM 呼び出しを担当する。

処理フロー（ユーザー発言ごとに1回実行）:
  1. 司会AIに次の発言者を問い合わせる（退席済みキャラクターは除外）
  2. 指名されたキャラクターの応答を順番にストリーミング生成する（1on1を複数回やるのと同じ形）
  3. 各キャラクター応答をDBに保存し、SSEイベントをyieldする
  4. キャラクターが [END_SESSION:...] タグまたは end_session ツールを使用した場合は退席を記録する
  5. auto_turns_used >= max_auto_turns または司会が [] を返したらユーザーターンへ戻す
  司会がNone（エラー）を返した場合もユーザーターンへ戻す。
  全員退席した場合は全員退席システムメッセージをyieldしてユーザーターンへ戻す。

グループチャット固有のLLM呼び出しは ChatService.execute_stream() に委譲する。
1on1チャットと同じ記憶想起・URL取得・時刻認識・プロンプト構築・プロバイダーディスパッチ・記憶刻み込みを共用する。
"""

import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from ..chat.models import ChatRequest, Message
from ..chat.service import ChatService
from ..memory.format import format_recalled_memories
from ..time_awareness import compute_time_awareness
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


def _build_exit_message(char_name: str, reason: str) -> str:
    """退席通知テキストを生成する。"""
    if reason:
        return f"{char_name}は退席しました。理由: {reason}"
    return f"{char_name}は退席しました。"


def _build_all_exited_message(exited_chars: list[dict]) -> str:
    """全員退席時の通知テキストを生成する。"""
    lines = [_build_exit_message(e["char_name"], e.get("reason", "")) for e in exited_chars]
    lines.append("（チャットを再開するには新しいセッションを作成してください）")
    return "\n".join(lines)


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
        ("session_exit",       {"char_name": str, "reason": str})    — 退席要求（character_done の後にyield）

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
    now = datetime.now()
    ta = compute_time_awareness(settings, char.id, sqlite, now)
    sqlite.set_setting(f"last_interaction_{char.id}", now.isoformat())

    # ChatRequest を構築して ChatService に委譲する
    request = ChatRequest(
        character_id=char.id,
        character_name=char_name,
        provider=preset.provider,
        model=preset.model_id,
        messages=messages,
        character_system_prompt=char.system_prompt_block1,
        inner_narrative=char.inner_narrative,
        provider_additional_instructions=model_config.get("additional_instructions", ""),
        thinking_level=preset.thinking_level or "default",
        settings=settings,
        recall_query_override=last_user_text,
        enable_time_awareness=ta.enabled,
        current_time_str=ta.current_time_str,
        time_since_last_interaction=ta.time_since_last_interaction,
        session_id=session_id,
        current_preset_name=preset.name,
        current_preset_id=preset.id,
    )

    # execute_stream を通じてストリーミング実行しながらチャンクをリアルタイムでyieldする
    full_text = ""
    memory_text = ""
    thinking_parts: list[str] = []
    exit_info: dict | None = None

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
        elif chunk_type == "session_exit":
            exit_info = content

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

    # 退席要求があれば character_done の後にyieldする
    if exit_info:
        yield ("session_exit", exit_info)


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
        ("speaker_decided",     {"speakers": [str]})                              — 司会AIが発言者を一括決定
        ("character_start",     {"character": str})                               — キャラクター応答開始
        ("character_reasoning", {"character": str, "content": str})               — 思考・想起記憶（リアルタイム）
        ("character_chunk",     {"character": str, "content": str})               — 応答テキスト
        ("character_done",      {"character": str, "message": dict})              — 応答完了・DB保存済み
        ("character_exited",    {"character": str, "system_message": dict})       — キャラクター退席・システムメッセージ保存済み
        ("all_exited",          {"system_message": dict})                         — 全員退席・システムメッセージ保存済み
        ("user_turn",           {"auto_turns_used": int})                         — ユーザーターンへ戻す
        ("error",               {"message": str, "character": str})               — エラー

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
    all_char_names = {p["char_name"] for p in participants}

    auto_turn_count = 0

    while True:
        # 最新の会話履歴・セッション退席状態を取得する
        history = sqlite.list_chat_messages(session_id)
        current_session = sqlite.get_chat_session(session_id)
        exited_chars: list[dict] = getattr(current_session, "exited_chars", None) or []
        exited_set = {e["char_name"] for e in exited_chars}

        # 全員退席済みチェック: ユーザーターンに戻すのみ（退席メッセージはキャラクター退席時に送信済み）
        if all_char_names and all_char_names.issubset(exited_set):
            yield ("user_turn", {"auto_turns_used": auto_turn_count})
            return

        # 司会AIに次の発言者を問い合わせる（退席済みキャラクターを除外して渡す）
        next_speakers = await decide_next_speakers(
            history=history,
            participants=participants,
            sqlite=sqlite,
            settings=settings,
            director_char_name=director_char_name,
            director_preset_id=director_preset_id,
            user_name=user_name,
            exited_chars=exited_chars,
        )

        # None はエラー（プリセット未発見・LLM障害）→ユーザーターンへ戻す
        if next_speakers is None:
            yield ("user_turn", {"auto_turns_used": auto_turn_count})
            return

        # [] は司会の意図的なユーザーターン指示、または上限超過
        if not next_speakers or auto_turn_count >= max_auto_turns:
            yield ("user_turn", {"auto_turns_used": auto_turn_count})
            return

        # 退席済みキャラクターを次発言者リストから除外する（司会の誤指名対策）
        next_speakers = [s for s in next_speakers if s not in exited_set]
        if not next_speakers:
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
                    elif chunk_type == "session_exit":
                        # 退席処理: DBを更新しシステムメッセージを保存・通知する
                        exit_char = payload["char_name"]
                        exit_reason = payload["reason"]

                        fresh_session = sqlite.get_chat_session(session_id)
                        fresh_exited: list[dict] = getattr(fresh_session, "exited_chars", None) or []
                        if not any(e["char_name"] == exit_char for e in fresh_exited):
                            fresh_exited = fresh_exited + [{"char_name": exit_char, "reason": exit_reason}]
                        sqlite.update_chat_session(session_id, exited_chars=fresh_exited)

                        # 退席システムメッセージをDBに保存する
                        import uuid as _uuid
                        sys_text = _build_exit_message(exit_char, exit_reason)
                        sys_msg = sqlite.create_chat_message(
                            message_id=str(_uuid.uuid4()),
                            session_id=session_id,
                            role="character",
                            content=sys_text,
                            character_name=exit_char,
                            is_system_message=True,
                        )

                        yield ("character_exited", {
                            "character": exit_char,
                            "system_message": message_to_dict(sys_msg),
                        })

                        # 全員退席チェック
                        new_exited_set = {e["char_name"] for e in fresh_exited}
                        if all_char_names and all_char_names.issubset(new_exited_set):
                            all_sys_text = _build_all_exited_message(fresh_exited)
                            all_sys_msg = sqlite.create_chat_message(
                                message_id=str(_uuid.uuid4()),
                                session_id=session_id,
                                role="character",
                                content=all_sys_text,
                                is_system_message=True,
                            )
                            yield ("all_exited", {"system_message": message_to_dict(all_sys_msg)})
                            yield ("user_turn", {"auto_turns_used": auto_turn_count})
                            return
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
