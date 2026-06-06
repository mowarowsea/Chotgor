"""シナリオ PC モード — PC（Chotgor キャラ）1 ターン分のストリーム実行。

GM ターン後にメンションで指名された Chotgor キャラを「PC」として 1 on 1 と
同じ system prompt（Block 1-10）で呼び出し、ChatService.execute_stream を
通じて応答テキストをストリーミングする。

GroupChat の `_stream_character_response`（backend/services/group_chat/service.py）
を踏襲しているが、以下の違いがある:

- 履歴ソースは ``scenario_turns`` であり ``chat_messages`` ではない。
- 自分発話の判定は ``speaker_type=="pc"`` かつ ``speaker_id == character_id``。
- ``ChatRequest.default_origin = "interlude"`` で記憶/スレッドを ``origin='interlude'``
  として保存する（``ToolExecutor.default_origin`` と Inscriber に流れる）。
- ``session_id`` は空文字列で渡す。farewell / drift / session-scoped working memory の
  ような chat 固有機構を PC モード中は無効化するため（シナリオセッション ID を
  入れると chat_session 前提の lookup が失敗するリスクがある）。
- 配役名（role_name）を system prompt へ追加注入し、キャラに「いま劇中で誰を
  演じているか」を自覚させる。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator

from backend.character_actions.anticipator import extract_anticipation
from backend.lib.log_context import (
    current_log_feature,
    current_log_target,
    new_message_id,
)
from backend.services.chat.models import Message
from backend.services.chat.request_factory import (
    build_available_presets,
    build_character_request,
)
from backend.services.chat.service import ChatService
from backend.services.memory.format import format_recalled_memories, format_recalled_threads
from backend.services.scenario_chat.mention import PcAssignment

logger = logging.getLogger(__name__)


# PC 専用に system prompt 末尾へ追加注入する「配役の自覚」テンプレ。
# Block 5（provider 追記）と同じ position で provider_additional_instructions に詰める。
# Block を新設しない理由: 1on1/GroupChat とプロンプト構造を共有したいため
# （新 Block を作ると全テンプレートに分岐が漏れる）。
_PC_ROLE_PREAMBLE_TEMPLATE = """\
## シナリオでの配役（あなた向けの状況メモ）
あなたはいま、シナリオ「{scenario_title}」のセッションに参加しており、
PC として「{role_name}」を演じています。シナリオの世界観・舞台設定は GM の
描写と直前の流れからのみ読み取ってください（あなたの普段の知識・記憶は
持ち越して構いません）。

GM 出力中の `@{role_name}` への呼びかけ、状況描写、ダイスの判定はそのまま
受け取ってください。NPC・他PC・ユーザの発言には `<タグ>` が付いています。

応答時の作法:
- 普段の 1on1 と同じく、自分の言葉で発話する。`@<名前>:` のような台詞ブロック
  プレフィックスは要りません（自分の発話の頭に `@{role_name}:` を付けない）。
- 他の PC や NPC・GM に呼びかけたいときは、本文中で `@<名前>` と書くと
  そちらに発話順が回ります。何も呼びかけずに発話を終えると、ユーザに
  ターンが戻ります。

劇中の体験を覚えるときは、本文に「（このシナリオでの出来事として）」と
劇中の出来事であることが分かる文脈を添えてください（読み返すあなた自身が
現実体験と区別できるように）。

### この配役について
{slot_description}
"""


def _format_scenario_history_for_pc(
    history: list[Any],
    self_character_id: str,
    self_role_name: str,
    user_alias: str,
    narrator_name: str = "Narrator",
) -> list[dict[str, Any]]:
    """scenario_turns を PC 視点の OpenAI messages 形式に変換する。

    ルール:
        - 自分発話（speaker_type=="pc" かつ speaker_id == self_character_id）
          → role="assistant", content=本文（タグなし）
        - 他PC発話                → role="user", content=`<role_name>本文</role_name>`
        - ユーザ発話              → role="user", content=`<user>本文</user>`
        - Narrator 地の文         → role="user", content=`<Narrator>本文</Narrator>`
        - NPC 発話               → role="user", content=`<NPC名>本文</NPC名>`
        - 連続する同 role はマージする（OpenAI/Anthropic API のロール交互制約対策）。

    Args:
        history: scenario_turns ORM のリスト（昇順）。
        self_character_id: 自分のキャラクター ID。
        self_role_name: 自分の配役名（メッセージ整形には使わないが将来拡張用）。
        user_alias: ユーザのタグ表示名。
        narrator_name: Narrator タグ名。

    Returns:
        OpenAI messages 形式の辞書リスト。
    """
    _ = self_role_name  # 将来拡張用に受け取るが現状未使用
    result: list[dict[str, Any]] = []
    for turn in history:
        stype = getattr(turn, "speaker_type", "")
        content = getattr(turn, "content", "") or ""
        if not content.strip():
            continue

        if stype == "pc" and getattr(turn, "speaker_id", None) == self_character_id:
            # 自分自身のシナリオ内発話
            oai_role = "assistant"
            tagged = content
        elif stype == "user":
            oai_role = "user"
            tagged = f"<{user_alias}>{content}</{user_alias}>"
        elif stype == "narrator":
            oai_role = "user"
            tagged = f"<{narrator_name}>{content}</{narrator_name}>"
        else:
            # npc / 他PC / 未知 → 表示名でタグ付け
            speaker_name = getattr(turn, "speaker_name", "") or "Unknown"
            oai_role = "user"
            tagged = f"<{speaker_name}>{content}</{speaker_name}>"

        if result and result[-1]["role"] == oai_role and isinstance(result[-1]["content"], str):
            result[-1]["content"] += "\n" + tagged
        else:
            result.append({"role": oai_role, "content": tagged})

    return result


async def stream_pc_response(
    pc: PcAssignment,
    scenario_title: str,
    user_alias: str,
    history: list[Any],
    preset_id: str,
    sqlite,
    settings: dict,
    chat_service: ChatService,
) -> AsyncGenerator[tuple[str, Any], None]:
    """指定 PC（Chotgor キャラ）の応答 1 ターン分をストリーミングする非同期ジェネレータ。

    ChatService.execute_stream を通じて 1on1 と同じ記憶想起・WM・inscribe フローを走らせ、
    出力は SSE 用のイベントとして yield する。記憶/スレッドは ``origin='interlude'`` で
    保存される（``default_origin`` 経由）。

    Args:
        pc: 発話させる PC の配役情報。
        scenario_title: GM プロンプトと同じシナリオタイトル。配役の自覚プロンプトに
            差し込まれる。
        user_alias: ユーザのタグ名。履歴整形と PC 自覚プロンプトに使う。
        history: scenario_turns ORM の昇順リスト（最新の GM 発話まで含む）。
        preset_id: 使用する LLMModelPreset の ID（pc_assignments 単位か、ない場合は
            キャラの enabled_providers 先頭。呼び出し側で解決して渡す）。
        sqlite: SQLiteStore。
        settings: グローバル設定辞書。
        chat_service: PC 1 ターンの LLM ディスパッチを担う ChatService インスタンス。

    Yields:
        ("pc_reasoning", {"character": role_name, "content": str})  — 想起記憶・WM・思考
        ("pc_chunk",    {"character": role_name, "content": str})   — 応答テキスト
        ("pc_done",     {"character": role_name, "speaker_id": cid,
                          "full_text": str, "anticipation": str | None}) — 完了通知
    """
    new_message_id()
    current_log_feature.set("scenario_chat_pc")
    current_log_target.set(pc.name)

    char = sqlite.get_character(pc.character_id)
    if not char:
        logger.error("PC キャラクター未発見 character_id=%s", pc.character_id)
        raise ValueError(f"PC キャラクター '{pc.character_id}' が見つかりません")

    preset = sqlite.get_model_preset(preset_id)
    if not preset:
        logger.error("PC プリセット未発見 preset_id=%s character_id=%s", preset_id, pc.character_id)
        raise ValueError(f"PC プリセット '{preset_id}' が見つかりません")

    # シナリオ履歴を PC 視点の messages に変換
    raw_messages = _format_scenario_history_for_pc(
        history,
        self_character_id=pc.character_id,
        self_role_name=pc.name,
        user_alias=user_alias,
    )
    messages = [Message(role=m["role"], content=m["content"]) for m in raw_messages]

    available_presets = build_available_presets(char, preset, sqlite)

    # ChatRequest を構築。session_id は空（chat_session 前提の機構を無効化）。
    # provider 追記欄の頭に「配役の自覚プロンプト」を差し込み、キャラ既存の追記と合成する。
    slot_desc = (getattr(pc, "description", "") or "").strip() or "（特記事項なし）"
    preamble = _PC_ROLE_PREAMBLE_TEMPLATE.format(
        scenario_title=scenario_title or "（無題のシナリオ）",
        role_name=pc.name,
        slot_description=slot_desc,
    )
    model_cfg = (char.enabled_providers or {}).get(preset.id, {})
    existing_additional = (model_cfg.get("additional_instructions", "") or "").strip()
    merged_additional = preamble.strip()
    if existing_additional:
        merged_additional = merged_additional + "\n\n" + existing_additional

    request = build_character_request(
        char, preset, messages, "", settings, sqlite,
        available_presets=available_presets,
        previous_anticipation="",
    )
    # build_character_request は固定引数で provider_additional_instructions を埋めるため、
    # PC モード用の preamble 合成と default_origin 切替は ChatRequest 構築後に直接代入する
    # （overrides に同名キーを入れると Python の重複指定エラーになるため）。
    request.provider_additional_instructions = merged_additional
    request.default_origin = "interlude"

    full_text = ""
    memory_text = ""
    recall_error_text = ""
    wm_text = ""
    thinking_parts: list[str] = []
    anticipation_text = ""

    async for chunk_type, content in chat_service.execute_stream(request):
        if chunk_type == "inscribed_memories":
            memory_text = format_recalled_memories(content)
            if memory_text:
                yield ("pc_reasoning", {"character": pc.name, "content": memory_text})
        elif chunk_type == "recall_error":
            recall_error_text = content + "\n"
            yield ("pc_reasoning", {"character": pc.name, "content": recall_error_text})
        elif chunk_type == "working_memory_threads":
            wm_text = format_recalled_threads(content)
            if wm_text:
                yield ("pc_reasoning", {"character": pc.name, "content": wm_text})
        elif chunk_type == "thinking":
            thinking_parts.append(content)
            yield ("pc_reasoning", {"character": pc.name, "content": content})
        elif chunk_type == "text":
            full_text += content
            if content:
                yield ("pc_chunk", {"character": pc.name, "content": content})
        elif chunk_type == "anticipation":
            anticipation_text = content
        elif chunk_type == "angle_switched":
            # PC モードでも switch_angle は許容（キャラ本人の判断を尊重）。
            # 呼び出し元（engine 側）でセッションの preset 反映までは行わない。
            yield ("pc_angle_switched", {
                "character": pc.name,
                "character_id": pc.character_id,
                "model_id": content["model_id"],
                "preset_id": content["preset_id"],
                "preset_name": content["preset_name"],
            })

    # 末尾 ANTICIPATE_RESPONSE タグを本文から剥がす（PC モードでも GroupChat 同様に副作用処理）。
    clean_text, parsed_anticipation = extract_anticipation(full_text)
    # ChatService 側で anticipation chunk が出ていればそちら優先。なければパースした方を採用。
    if not anticipation_text and parsed_anticipation:
        anticipation_text = parsed_anticipation

    # PC が稀に自身の発話頭に `@<role>:` を付けてくるケースがあるので剥がす
    # （プロンプトで禁止しているが、保険として）。
    role_prefix_candidates = (f"@{pc.name}:", f"@{pc.name}：")
    stripped = clean_text.lstrip()
    for prefix in role_prefix_candidates:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].lstrip()
            clean_text = stripped
            break

    yield ("pc_done", {
        "character": pc.name,
        "character_id": pc.character_id,
        "preset_name": preset.name,
        "full_text": clean_text,
        "anticipation": anticipation_text or None,
    })


__all__ = ["stream_pc_response"]
