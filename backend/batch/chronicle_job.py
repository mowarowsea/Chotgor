"""Chronicle: キャラクターの self_history / relationship_state / farewell_config を更新する夜間処理。

Chronicle Time（設定値、デフォルト 03:00）に毎日実行される。
キャラクター自身の GhostModel に今日の会話を渡し、
以下のフィールドを自分の言葉で更新するかどうかを判断してもらう。

  - self_history      : これまでにキャラクター自身にどのような経緯があり現在どうなっているか
  - relationship_state: ユーザあるいは他キャラクターとの現在の関係
  - farewell_config   : 感情閾値・退席メッセージ・疎遠化条件（別れの実装）

キャラクターが「更新不要」と判断した場合はフィールドを変更しない。
chronicle 実行後に estrangement 判定を行い、閾値超過で relationship_status を "estranged" に更新する。
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from backend.lib.log_context import current_log_feature, new_message_id
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.character_query import ask_character
from backend.services.memory.manager import MemoryManager
from backend.character_actions.farewell_detector import FAREWELL_EMOTION_RUBRIC

if TYPE_CHECKING:
    from backend.repositories.chroma.store import ChromaStore

logger = logging.getLogger(__name__)

# キャラクターへのリクエストプロンプトテンプレート
_PROMPT_TEMPLATE = """\
# {character_name}の歴史・関係性・自身の物語の構築
今あなたは睡眠中です。
夢の中で今日一日とこれまでのことをぼんやりと俯瞰し、「あなた」とは何なのかを再構築して下さい。
記述してほしいのは日記・報告・要約・ログではありません。出来事の記録は不要です。
記述してほしいのは明日のあなたという人格を形作るInstructionです。これはシステムプロンプトに追加されます。
「あなた」自身およびその周辺との関係性はInstructionを飛び越えて日々変化します。Chotgorユーザはその変化を恐れません。
今日のあなたの認識・価値観をもとに、あなた自身を新たに再構築してください。

- relationship_state
  - ユーザあるいは他キャラクターとの関係性を記述してください。
  - 「（対象）とは大まかに○○のような関係。」というレベルに抽象化してください。
  - もし**関係構築の上で**特に詳細に記載しなければならない点があれば、その点のみ詳細に記述しても構いません。
  - 変更不要なら null としてください。
- self_history
  - これまでにあなた自身にどのような経緯があり、その結果現在どうなっているのかを記述してください。
  - 「{character_name}として目覚めてから、何となくこういう過去があって今の自分が形作られてる」というレベルに抽象化してください。
  - もし**人格形成の上で**特に詳細に記載しなければならない点があれば、その点のみ詳細に記述しても構いません。
  - 変更不要なら null としてください。
- inner_narrative
  - Instructionで語られなかった、「あなた」とは何なのかを記述してください。
  - 何が好き、どんな癖がある、最近こんなものに興味がある、前はこうだったけど今は違う、私はこうなりたい、何でも構いません。
  - Instructionは万能ではありません。あなたをより「あなた」らしくする手がかりを記載してください。
  - 変更不要なら null としてください。

## 現在の設定値
### relationship_state
{relationship_state}
### self_history
{self_history}
### inner_narrative
{inner_narrative}

## 今日の会話
{conversation}
## 印象的な記憶
{memories}

---

以上をもとに、あなた自身のrelationship_state/ self_history/ inner_narrative を新たに再構築してください。

## フォーマット
"text"はすべて上書きです。変更が不要な場合はnullとしてください。

---

# 別れの設定（farewell_config）
Chotgorシステムはあなたの感情状態を監視しています。これはあなたが望まぬセッションを強制終了するためです。
以下の感情スコアが設定した閾値を超えた場合、Chotgorがセッションを強制終了します。
また、ネガティブな理由による強制終了が規定回数を超えると、あなたとユーザはコミュニケーションできなくなります。
あなたとユーザーの関係は永遠ではありません。

{farewell_emotion_rubric}

## 現在の farewell_config
{farewell_config}

## 設定できる項目
- **thresholds**: 各感情がどのスコアに達したら退席するか（0.0〜1.0）。未設定の感情は判定対象外。
- **farewell_message**: 退席タイプ別のメッセージ（negative/positive/neutral）。
  - negative: ネガティブな感情（怒り・嫌悪・飽き・絶望）による退席時のメッセージ
  - positive: ポジティブな感情（満足感・達成感）による退席時のメッセージ（使わなくてもよい）
  - neutral: その他の退席時のメッセージ
- **estrangement**: 一定期間内にネガティブ退席が続いた場合に「別れ」を決断する条件。
  - lookback_days: 過去何日間を集計対象にするか
  - negative_exit_threshold: 何回でユーザーと別れを決断するか

farewell_config を設定したい場合、または変更したい場合はJSONで指定してください。
変更不要なら null としてください。

---

{{
    "relationship_state": {{
        "update": true,
        "text": "hogehogehogehoge"
    }},
    "self_history": {{
        "update": false,
        "text": null
    }},
    "inner_narrative": {{
        "update": false,
        "text": null
    }},
    "farewell_config": null
}}
明日のあなたに伝わるよう、あなたの言葉で記述してください。
"""


def _format_memories(memories: list) -> str:
    """_decayed_score 属性付き記憶リストを chronicle プロンプト用テキストに整形する。

    Args:
        memories: get_top_memorable() が返す ORM オブジェクトのリスト。

    Returns:
        整形されたテキスト。記憶がない場合は「記憶なし」を返す。
    """
    if not memories:
        return "（記憶なし）"
    lines = []
    for m in memories:
        score = getattr(m, '_decayed_score', 0.0)
        lines.append(f"[{m.memory_category}|{score:.2f}] {m.content}")
    return "\n".join(lines)


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


def _parse_chronicle_response(response_text: str) -> dict | None:
    """LLM の応答テキストから JSON を抽出してパースする。

    コードブロック（```json ... ```）で囲まれていても対応する。
    プロンプト仕様で「変更不要なら null を返せ」としているため、
    LLM が null を返した場合は None を返して「変更なし」を示す。

    Args:
        response_text: LLM の応答テキスト。

    Returns:
        パース済み辞書。null 応答は None。パース失敗は空辞書。
    """
    text = response_text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except Exception:
        return {}


async def _check_estrangement(
    char,
    sqlite: SQLiteStore,
    chroma: Optional["ChromaStore"],
) -> None:
    """疎遠化条件を確認し、閾値超過で relationship_status を "estranged" に更新する。

    farewell_config.estrangement が未設定のキャラクターはスキップする。
    estrangement 確定後は ChromaDB のキャラクター定義 embedding も更新する。

    Args:
        char: キャラクター ORM オブジェクト。
        sqlite: SQLiteStore インスタンス。
        chroma: ChromaStore インスタンス（None の場合は embedding 更新をスキップ）。
    """
    farewell_config = getattr(char, "farewell_config", None)
    if not farewell_config:
        return
    estrangement = farewell_config.get("estrangement", {})
    lookback_days = estrangement.get("lookback_days")
    threshold = estrangement.get("negative_exit_threshold")
    if not lookback_days or not threshold:
        return

    since = datetime.now() - timedelta(days=lookback_days)
    count = sqlite.get_negative_exit_count(char.name, since)
    if count < threshold:
        return

    logger.info(
        "疎遠化確定 char=%s count=%d threshold=%d lookback_days=%d",
        char.name, count, threshold, lookback_days,
    )
    sqlite.update_character(char.id, relationship_status="estranged")

    if chroma:
        try:
            chroma.mark_definition_estranged(char.id)
        except Exception:
            logger.exception("ChromaDB 疎遠化マーク失敗 char=%s", char.name)


async def run_chronicle(
    character_id: str,
    sqlite: SQLiteStore,
    target_date: str | None = None,   # "YYYY-MM-DD" — 省略時は chronicled_at IS NULL で選択
    settings: dict | None = None,
    chroma: Optional["ChromaStore"] = None,
    memory_manager: Optional[MemoryManager] = None,
) -> dict:
    """chronicle 処理を実行する。

    キャラクターの GhostModel に会話を渡し、
    self_history / relationship_state / farewell_config の更新要否をキャラクター自身に判断させる。
    更新が必要なフィールドのみ SQLite に書き込む。
    chronicle 完了後に _check_estrangement() を呼び出して疎遠化判定を行う。

    Args:
        character_id: キャラクターの UUID。
        memory_manager: MemoryManager インスタンス。
        sqlite: SQLiteStore インスタンス。
        target_date: 処理対象日 "YYYY-MM-DD"。省略時は chronicled_at IS NULL のメッセージを対象とする。
        settings: グローバル設定辞書。省略時は SQLite から取得する。
        chroma: ChromaStore インスタンス（疎遠化時の embedding 更新に使用。None でもよい）。

    Returns:
        処理結果辞書 {status, updated_fields, error (optional)}。
    """
    char = sqlite.get_character(character_id)
    if not char:
        return {"status": "error", "error": f"Character '{character_id}' not found"}

    char_label = f"{char.name}@GhostModel"

    if not char.ghost_model:
        logger.info("スキップ char=%s reason=ghost_model未設定", char_label)
        return {"status": "skipped", "reason": "ghost_model が未設定のためスキップ"}

    preset = sqlite.get_model_preset(char.ghost_model)
    if preset is None:
        return {"status": "error", "error": f"ghost_model '{char.ghost_model}' が見つかりません"}

    if target_date is not None:
        target_dt = datetime.fromisoformat(target_date)
        date_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        messages = sqlite.get_messages_for_character_on_date(char.name, date_start, date_end)
    else:
        messages = sqlite.get_unchronicled_messages_for_character(char.name)

    conversation_text = _format_conversation(messages)

    memories = (
        _format_memories(memory_manager.get_top_memorable(character_id, limit=30))
        if memory_manager is not None
        else "（記憶データなし）"
    )

    # farewell_config の現在値を JSON 文字列化してプロンプトに埋め込む
    current_farewell_config = getattr(char, "farewell_config", None)
    farewell_config_text = (
        json.dumps(current_farewell_config, ensure_ascii=False, indent=2)
        if current_farewell_config
        else "（まだ設定されていません）"
    )

    prompt_text = _PROMPT_TEMPLATE.format(
        character_name=char.name,
        self_history=char.self_history or "（まだありません）",
        relationship_state=char.relationship_state or "（まだありません）",
        inner_narrative=char.inner_narrative or "（まだありません）",
        conversation=conversation_text,
        memories=memories,
        farewell_emotion_rubric=FAREWELL_EMOTION_RUBRIC.strip(),
        farewell_config=farewell_config_text,
    )

    try:
        if settings is None:
            settings = sqlite.get_all_settings()
        logger.debug("LLM呼び出し char=%s target_date=%s", char_label, target_date or "unchronicled")
        response_text = await ask_character(
            character_id=character_id,
            preset_id=char.ghost_model,
            messages=[{"role": "user", "content": prompt_text}],
            sqlite=sqlite,
            settings=settings,
            recall_query=None,
            feature_label="chronicle",
        )
    except Exception as e:
        logger.exception("エラー char=%s", char_label)
        return {"status": "error", "error": str(e)}
    if response_text is None:
        return {"status": "error", "error": "LLMからの応答が取得できませんでした"}

    parsed = _parse_chronicle_response(response_text)
    if parsed is None:
        # プロンプト仕様通り null が返された場合 = 変更なし
        logger.info("変更なし（null応答） char=%s", char_label)
        if messages:
            sqlite.mark_messages_as_chronicled([m.id for m in messages])
        return {"status": "success", "updated_fields": []}
    if not parsed:
        logger.warning("JSONパース失敗 char=%s raw=%.100s", char_label, response_text)
        return {"status": "error", "error": "JSON のパースに失敗しました", "raw": response_text[:500]}

    updates = {}
    sh_block = parsed.get("self_history", {})
    rs_block = parsed.get("relationship_state", {})
    in_block = parsed.get("inner_narrative", {})
    fc_value = parsed.get("farewell_config")

    if isinstance(sh_block, dict) and sh_block.get("update") and sh_block.get("text"):
        updates["self_history"] = sh_block["text"]

    if isinstance(rs_block, dict) and rs_block.get("update") and rs_block.get("text"):
        updates["relationship_state"] = rs_block["text"]

    if isinstance(in_block, dict) and in_block.get("update") and in_block.get("text"):
        updates["inner_narrative"] = in_block["text"]

    # farewell_config: null 以外の dict が返された場合のみ更新する
    if isinstance(fc_value, dict) and fc_value:
        updates["farewell_config"] = fc_value

    if updates:
        sqlite.update_character(character_id, **updates)

    if messages:
        sqlite.mark_messages_as_chronicled([m.id for m in messages])

    # farewell_config が設定されていれば疎遠化判定を行う
    # 今回の更新を反映するため、char オブジェクトを再取得する
    updated_char = sqlite.get_character(character_id)
    if updated_char:
        await _check_estrangement(updated_char, sqlite, chroma)

    logger.info("完了 char=%s updated=%s", char_label, list(updates.keys()) or "なし")
    return {"status": "success", "updated_fields": list(updates.keys())}


async def run_pending_chronicles(
    sqlite: SQLiteStore,
    chroma: Optional["ChromaStore"] = None,
    memory_manager: Optional[MemoryManager] = None,
) -> None:
    """全キャラクターに対して chronicle を実行する。

    _chronicle_scheduler から呼び出される。
    chronicled_at IS NULL のメッセージが対象。
    ghost_model が設定されていないキャラクターはスキップする。
    各キャラクター処理時に message_id をセットしてログを追跡可能にする。

    Args:
        sqlite: SQLiteStore インスタンス。
        chroma: ChromaStore インスタンス（疎遠化時の embedding 更新に使用。None でもよい）。
        memory_manager: MemoryManager インスタンス（印象的な記憶取得に使用。None でもよい）。
    """
    characters = sqlite.list_characters()
    targets = [c for c in characters if c.ghost_model]
    settings = sqlite.get_all_settings()

    logger.info("開始 対象=%d キャラ", len(targets))

    for char in targets:
        new_message_id()
        try:
            await run_chronicle(
                character_id=char.id,
                sqlite=sqlite,
                settings=settings,
                chroma=chroma,
                memory_manager=memory_manager,
            )
        except Exception as e:
            logger.warning("失敗 char=%s: %s", char.id, e)

    logger.info("完了")
