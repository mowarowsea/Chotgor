"""グループチャット司会 — 次の発言者を判断するディレクターモジュール。

キャラクターとして登録された司会役を通じて、会話の流れから次の発言者を決定する。
パース失敗・タイムアウト・接続エラーは全て [] に統一し、ユーザーターンへ戻す。
"""

import logging
import re

from backend.lib.log_context import current_log_feature
from backend.providers.registry import create_provider

logger = logging.getLogger(__name__)


def _build_director_messages(
    history_text: str,
    participants: list[dict],
    sqlite,
    user_name: str = "ユーザ",
    exited_chars: list[dict] | None = None,
) -> tuple[str, str]:
    """司会AIへ渡すシステムプロンプトとユーザーメッセージを構築する。

    Args:
        history_text: <タグ>付き形式にフォーマットされた会話履歴テキスト。
        participants: 参加者情報リスト [{"char_name": str, "preset_id": str}]。
        sqlite: キャラクター情報取得用のSQLiteStoreインスタンス。
        user_name: ユーザーの表示名（選択肢として提示する）。
        exited_chars: 退席済みキャラクター情報リスト [{"char_name": str, "reason": str}]。
                      指定された場合はシステムプロンプトに退席情報を追加し、選択肢から除外する。

    Returns:
        (system_prompt, user_message) のタプル。
        system_prompt はフォーマット指示、user_message は参加者設定と会話履歴。
    """
    exited_set = {e["char_name"] for e in (exited_chars or [])}
    participant_names = [p["char_name"] for p in participants]
    # 退席済みキャラクターは選択肢から除外する
    active_names = [n for n in participant_names if n not in exited_set]
    char_names_str = "、".join(f'"{n}"' for n in active_names)
    all_names_str = "、".join(f'"{n}"' for n in active_names + [user_name])

    exited_note = ""
    if exited_chars:
        exited_lines = []
        for e in exited_chars:
            reason_part = f"（理由: {e['reason']}）" if e.get("reason") else ""
            exited_lines.append(f"  - {e['char_name']}{reason_part}")
        exited_note = "\n退席済み（指名不可）:\n" + "\n".join(exited_lines) + "\n"

    system_prompt = (
        "あなたはグループチャットの司会進行役です。\n"
        "会話の流れを見て、次に発言するべき参加者を判断してください。\n\n"
        f"選択可能な名前（この中から選ぶこと）: {all_names_str}\n"
        f"  - キャラクター: {char_names_str}\n"
        f"  - ユーザー: \"{user_name}\"\n"
        + exited_note +
        "名前は必ず上記の正式名称をそのまま使うこと。省略・変形・敬称の追加は禁止。\n\n"
        "回答は必ず以下のいずれかの形式のみで返してください（説明・前置き・補足は一切不要、[]は必須）：\n"
        "[正式名称]               → 単一指定（キャラクターまたはユーザー）\n"
        "[正式名称A, 正式名称B]   → 並列発言（キャラクターのみ）\n"
        "[]                       → ユーザーターンに戻す（省略形）"
    )

    lines = ["# グループチャット参加者の設定\n"]
    for p in participants:
        char_name = p["char_name"]
        if char_name in exited_set:
            continue  # 退席済みキャラクターの設定は含めない
        char = sqlite.get_character_by_name(char_name) or sqlite.get_character(char_name)
        if char and char.system_prompt_block1:
            lines.append(f"## {char_name}")
            lines.append(char.system_prompt_block1.strip())
            lines.append("")
    lines.append("---")
    lines.append("# 会話履歴\n")
    lines.append(history_text)
    lines.append("---")
    lines.append(
        f"\n選択可能: {all_names_str}\n"
        "次に発言するべき参加者を指定してください。"
    )
    user_message = "\n".join(lines)

    return system_prompt, user_message


def _parse_director_response(text: str, participant_names: list[str], user_name: str = "ユーザ") -> list[str]:
    """司会AIの応答テキストから次発言者リストを解析する。

    Args:
        text: 司会AIの応答テキスト。
        participant_names: 有効なキャラクター名リスト（フィルタリングに使用）。
        user_name: ユーザーの表示名（選択された場合は [] を返す）。

    Returns:
        次に発言すべきキャラクター名リスト。[] はユーザーターンを意味する。
    """
    # 最初に現れる [...] を探す
    match = re.search(r"\[([^\]]*)\]", text)
    if not match:
        return []
    inner = match.group(1).strip()
    if not inner:
        return []
    names = [n.strip() for n in inner.split(",")]
    # ユーザー名が含まれていたらユーザーターン（[] と同義）
    if user_name in names:
        return []
    # 実在するキャラ名のみを通す（ハルシネーション対策）
    return [n for n in names if n in participant_names]


async def decide_next_speakers(
    history: list,
    participants: list[dict],
    sqlite,
    settings: dict,
    director_char_name: str,
    director_preset_id: str,
    user_name: str = "ユーザ",
    timeout: int = 30,
    exited_chars: list[dict] | None = None,
) -> list[str] | None:
    """会話履歴を元に次の発言者リストを決定する。

    司会キャラクターのプロバイダーを通じてLLMを呼び出し、次の発言者を判断する。

    Args:
        history: ChatMessageオブジェクトのリスト（時系列順）。
        participants: 参加者情報リスト [{"char_name": str, "preset_id": str}]。
        sqlite: キャラクター情報取得用のSQLiteStoreインスタンス。
        settings: グローバル設定辞書（APIキー等）。
        director_char_name: 司会役キャラクターの名前。
        director_preset_id: 司会役が使用するモデルプリセットのID。
        user_name: ユーザーの表示名（選択肢として提示し、選ばれたらユーザーターン）。
        timeout: タイムアウト秒数（将来拡張用）。
        exited_chars: 退席済みキャラクター情報リスト [{"char_name": str, "reason": str}]。
                      指定された場合は選択肢から除外し、司会プロンプトに退席情報を追加する。

    Returns:
        次に発言すべきキャラクター名リスト。
        [] は司会が意図的にユーザーターンへ戻すことを意味する（ユーザー名選択含む）。
        None はプリセット未発見・LLMエラーなどの障害を意味する。
    """
    exited_set = {e["char_name"] for e in (exited_chars or [])}
    participant_names = [p["char_name"] for p in participants if p["char_name"] not in exited_set]

    # 会話履歴をフラットなテキストに変換する（ユーザー名を実名で表示する）
    history_lines = []
    for msg in history:
        if msg.role == "user":
            history_lines.append(f"<{user_name}>{msg.content}</{user_name}>")
        else:
            char_name = getattr(msg, "character_name", None) or "キャラクター"
            history_lines.append(f"<{char_name}>{msg.content}</{char_name}>")
    history_text = "\n".join(history_lines)

    system_prompt, user_message = _build_director_messages(
        history_text, participants, sqlite, user_name=user_name, exited_chars=exited_chars
    )

    # 司会キャラクターのプリセットをIDで取得してプロバイダーを生成する
    preset = sqlite.get_model_preset(director_preset_id)
    if not preset:
        logger.warning("プリセット未発見 director=%s preset_id=%s", director_char_name, director_preset_id)
        return None

    current_log_feature.set("group_chat")
    provider = create_provider(preset.provider, preset.model_id, settings, preset_name=preset.name)

    try:
        raw = await provider.generate(
            system_prompt,
            [{"role": "user", "content": user_message}],
        )
    except Exception:
        logger.exception("LLM呼び出し失敗 director=%s", director_char_name)
        return None

    result = _parse_director_response(raw, participant_names, user_name=user_name)
    logger.debug("判定結果 director=%s next_speakers=%s", director_char_name, result)
    return result
