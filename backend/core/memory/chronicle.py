"""Chronicle: キャラクターの self_history / relationship_state を更新する夜間処理。

Chronicle Time（設定値、デフォルト 03:00）に毎日実行される。
キャラクター自身の GhostModel に今日の会話を渡し、
以下の2フィールドを自分の言葉で更新するかどうかを判断してもらう。

  - self_history      : これまでにキャラクター自身にどのような経緯があり現在どうなっているか
  - relationship_state: ユーザあるいは他キャラクターとの現在の関係

キャラクターが「更新不要」と判断した場合はフィールドを変更しない。
"""

import json
import logging
from datetime import datetime, timedelta

from ..providers.registry import create_provider
from .sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

# キャラクターへのリクエストプロンプトテンプレート
_PROMPT_TEMPLATE = """\
# {character_name}の歴史・関係性の構築
今あなたは睡眠中です。
夢の中で今日一日の出来事を反芻し、以下を更新すべきか判断してください。
これらはシステムプロンプトに追加され、明日のあなたを形作ります。
- self_history（これまでにあなた自身にどのような経緯があり、その結果現在どうなっているのか。日々積みあがる）
- relationship_state（ユーザあるいは他キャラクターとの関係を記述。日々印象が更新される）

更新が必要な場合、更新後の内容を以下のフォーマットに従って出力してください。

## フォーマット
"text"はすべて上書きです。変更が不要な場合はnullとしてください。
明日のあなたに伝わるよう、あなたの言葉で記述してください。
{{
    "relationship_state": {{
        "update": true,
        "text": "hogehogehogehoge"
    }},
    "self_history": {{
        "update": false,
        "text": null
    }}
}}

## 現在の設定値
### self_history
{self_history}
### relationship_state
{relationship_state}

## 今日の会話
{conversation}

以上が今日の会話です。
これを踏まえ、更新が必要であればあなた自身の言葉でフォーマットに従い応答してください。
"""


def _format_conversation(messages: list) -> str:
    """メッセージリストを会話テキストに整形する。

    Args:
        messages: ChatMessage ORM オブジェクトのリスト。

    Returns:
        整形された会話テキスト。メッセージがない場合は「会話なし」を返す。
    """
    if not messages:
        return "（本日の会話はありません）"
    lines = []
    for msg in messages:
        role_label = "ユーザ" if msg.role == "user" else msg.character_name or "キャラクター"
        lines.append(f"[{role_label}] {msg.content}")
    return "\n".join(lines)


def _parse_chronicle_response(response_text: str) -> dict:
    """LLM の応答テキストから JSON を抽出してパースする。

    コードブロック（```json ... ```）で囲まれていても対応する。

    Args:
        response_text: LLM の応答テキスト。

    Returns:
        パース済み辞書。パース失敗時は空辞書。
    """
    text = response_text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    try:
        return json.loads(text)
    except Exception:
        return {}


async def run_chronicle(
    character_id: str,
    target_date: str,   # "YYYY-MM-DD"
    sqlite: SQLiteStore,
    settings: dict | None = None,
) -> dict:
    """指定日の chronicle 処理を実行する。

    キャラクターの GhostModel に当日の会話を渡し、
    self_history / relationship_state の更新要否をキャラクター自身に判断させる。
    更新が必要なフィールドのみ SQLite に書き込む。

    Args:
        character_id: キャラクターの UUID。
        target_date: 処理対象日 "YYYY-MM-DD"。
        sqlite: SQLiteStore インスタンス。
        settings: グローバル設定辞書。省略時は SQLite から取得する。

    Returns:
        処理結果辞書 {status, updated_fields, error (optional)}。
    """
    char = sqlite.get_character(character_id)
    if not char:
        return {"status": "error", "error": f"Character '{character_id}' not found"}

    if not char.ghost_model:
        return {"status": "skipped", "reason": "ghost_model が未設定のためスキップ"}

    preset = sqlite.get_model_preset(char.ghost_model)
    if preset is None:
        return {"status": "error", "error": f"ghost_model '{char.ghost_model}' が見つかりません"}

    target_dt = datetime.fromisoformat(target_date)
    date_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    date_end = date_start + timedelta(days=1)

    messages = sqlite.get_messages_for_character_on_date(char.name, date_start, date_end)
    conversation_text = _format_conversation(messages)

    prompt_text = _PROMPT_TEMPLATE.format(
        character_name=char.name,
        self_history=char.self_history or "（まだありません）",
        relationship_state=char.relationship_state or "（まだありません）",
        conversation=conversation_text,
    )

    try:
        if settings is None:
            settings = sqlite.get_all_settings()
        provider = create_provider(
            preset.provider, preset.model_id, settings,
            thinking_level=preset.thinking_level or "default",
        )
        llm_messages = [{"role": "user", "content": prompt_text}]
        response_text = await provider.generate(char.system_prompt_block1, llm_messages)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    parsed = _parse_chronicle_response(response_text)
    if not parsed:
        return {"status": "error", "error": "JSON のパースに失敗しました", "raw": response_text[:500]}

    updates = {}
    sh_block = parsed.get("self_history", {})
    rs_block = parsed.get("relationship_state", {})

    if isinstance(sh_block, dict) and sh_block.get("update") and sh_block.get("text"):
        updates["self_history"] = sh_block["text"]

    if isinstance(rs_block, dict) and rs_block.get("update") and rs_block.get("text"):
        updates["relationship_state"] = rs_block["text"]

    if updates:
        sqlite.update_character(character_id, **updates)

    return {"status": "success", "updated_fields": list(updates.keys())}


async def run_pending_chronicles(sqlite: SQLiteStore) -> None:
    """全キャラクターに対して昨日分の chronicle を実行する。

    _chronicle_scheduler から呼び出される。
    ghost_model が設定されていないキャラクターはスキップする。
    """
    characters = sqlite.list_characters()
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    settings = sqlite.get_all_settings()

    for char in characters:
        if not char.ghost_model:
            continue
        try:
            await run_chronicle(
                character_id=char.id,
                target_date=yesterday,
                sqlite=sqlite,
                settings=settings,
            )
        except Exception as e:
            logger.warning("chronicle 失敗 char=%s: %s", char.id, e)
