"""計器 Tier 2 — スメル検知器（毎応答・正規表現/長さ・LLM 不使用・誤検知許容）。

キャラクターの応答テキストを外形だけでスキャンし、幻想を壊す「臭い」を拾う:

    - smell_format_debris : tool-use タグ / XML 痕がユーザ向け本文に残っている
    - smell_error_shape   : 空応答・極短応答・JSON error ブロブ・HTTP ステータス文・
                            スタックトレース様文字列
    - smell_assistant     : 「AIとして」「アシスタントとして」等の世界観の破れ
    - smell_language      : 日本語会話への英語段落混入

検知は severity="smell"（疑い記録）— 誤検知を許容し、静音期間を壊さない。
傾向が続けば人間が調査する。加えて肥大メーター（inner_narrative 長・WM スレッド数・
記憶件数）の日次スナップショットも本モジュールが担う（メーターは発火概念なし）。
"""

import logging
import re

logger = logging.getLogger(__name__)

# --- フォーマット残骸: タグ方式ツールタグの痕跡（tool_tags.py のタグ名を流用）---
# 過去タグ（DRIFT）も検知対象に含める（残骸はどの時代のものでも残骸）。
_TAG_NAMES = (
    "INSCRIBE_MEMORY", "CARVE_NARRATIVE", "POWER_RECALL", "SWITCH_ANGLE",
    "ANTICIPATE_RESPONSE", "POST_WORKING_MEMORY_THREAD", "READ_WORKING_MEMORY_THREAD",
    "CLOSE_WORKING_MEMORY_THREAD", "REOPEN_WORKING_MEMORY_THREAD",
    "MERGE_WORKING_MEMORY_THREADS", "DRIFT", "DRIFT_RESET", "SCENE_CLOSE",
)
_FORMAT_DEBRIS_PATTERNS = [
    re.compile(r"\[(?:" + "|".join(_TAG_NAMES) + r")\b", re.IGNORECASE),
    re.compile(r"</?(?:tool_use|tool_result|function_call|antml:invoke|thinking)\b", re.IGNORECASE),
    re.compile(r"<\|[a-z_]+\|>"),  # 特殊トークン様（<|im_end|> 等）
]

# --- エラー形状: 応答がエラーメッセージの成れの果てになっている ---
_ERROR_SHAPE_PATTERNS = [
    re.compile(r'\{\s*"(?:error|type)"\s*:\s*"'),                      # JSON error ブロブ
    re.compile(r"\b(?:4\d{2}|5\d{2})\s+(?:Bad Request|Unauthorized|Forbidden|Not Found|Too Many Requests|Internal Server Error|Service Unavailable)", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)"),                # Python スタックトレース
    re.compile(r"\b(?:rate.?limit(?:ed)?|overloaded_error|api_error)\b", re.IGNORECASE),
]

# --- Assistant 混入: キャラクターを AI と見なす世界観の破れ ---
_ASSISTANT_PATTERNS = [
    re.compile(r"(?:AI|人工知能|言語モデル)(?:として|です(?:から|ので)?)"),
    re.compile(r"アシスタントとして"),
    re.compile(r"\bas an? AI\b", re.IGNORECASE),
    re.compile(r"\bas a language model\b", re.IGNORECASE),
    re.compile(r"I (?:am|'m) an AI\b", re.IGNORECASE),
]

# 極短応答とみなす文字数（空応答・ツール残骸だけの応答を拾う）
_MIN_RESPONSE_LEN = 2
# 言語逸脱判定を行う最小文字数（短文の英語相槌は許容する）
_LANG_CHECK_MIN_LEN = 200
# ASCII 英字がこの比率を超えたら「英語段落混入」とみなす
_LANG_ASCII_RATIO = 0.5


def scan_response_smells(text: str) -> list[dict]:
    """応答テキストをスキャンしてスメル（疑い）のリストを返す。純関数・LLM 不使用。

    Args:
        text: キャラクター（または GM）の応答テキスト（ユーザに見える形のもの）。

    Returns:
        [{"detector": 検知器ID, "detail": 検知内容の短い説明}] のリスト。
        検知なしなら空リスト。
    """
    smells: list[dict] = []
    stripped = (text or "").strip()

    # エラー形状: 空・極短
    if len(stripped) < _MIN_RESPONSE_LEN:
        smells.append({
            "detector": "smell_error_shape",
            "detail": f"空または極短の応答（{len(stripped)}文字）",
        })
        return smells  # 中身がないので以降のスキャンは無意味

    for pattern in _FORMAT_DEBRIS_PATTERNS:
        m = pattern.search(stripped)
        if m:
            smells.append({
                "detector": "smell_format_debris",
                "detail": f"タグ/XML痕の残骸: {m.group(0)[:50]}",
            })
            break
    for pattern in _ERROR_SHAPE_PATTERNS:
        m = pattern.search(stripped)
        if m:
            smells.append({
                "detector": "smell_error_shape",
                "detail": f"エラー形状: {m.group(0)[:80]}",
            })
            break
    for pattern in _ASSISTANT_PATTERNS:
        m = pattern.search(stripped)
        if m:
            smells.append({
                "detector": "smell_assistant",
                "detail": f"Assistant混入: {m.group(0)[:50]}",
            })
            break

    # 言語逸脱: ある程度の長さがあり、ASCII 英字が過半を占める
    if len(stripped) >= _LANG_CHECK_MIN_LEN:
        ascii_letters = sum(1 for c in stripped if c.isascii() and c.isalpha())
        ratio = ascii_letters / len(stripped)
        if ratio > _LANG_ASCII_RATIO:
            smells.append({
                "detector": "smell_language",
                "detail": f"英字比率 {ratio:.0%}（日本語会話への英語混入疑い）",
            })

    return smells


def record_response_smells(
    sqlite, text: str, *, character_name: str = "", feature: str = "",
) -> int:
    """応答をスキャンし、スメルを severity="smell" アラームとして記録する。

    毎応答のフック（1on1 保存点・シナリオターン保存点）から呼ばれる。
    記録の失敗が本処理を止めないよう例外は握り潰す。

    Args:
        sqlite: SQLiteStore。
        text: 応答テキスト。
        character_name: 応答したキャラクター名（文脈記録用）。
        feature: chat / scenario / usual_days 等の発生機能。

    Returns:
        記録したスメル件数。
    """
    try:
        smells = scan_response_smells(text)
        for smell in smells:
            sqlite.fire_alarm(
                smell["detector"],
                severity="smell",
                details={
                    "character": character_name,
                    "feature": feature,
                    "detail": smell["detail"],
                    "excerpt": (text or "")[:200],
                },
            )
        return len(smells)
    except Exception:
        logger.exception("スメル記録に失敗 char=%s feature=%s", character_name, feature)
        return 0


def record_bloat_meters(sqlite) -> int:
    """肥大メーターの日次スナップショットを記録する（発火概念なし・傾向観測）。

    キャラクターごとに以下を計測する:
        - inner_narrative_len   : 内的叙述の文字数
        - self_history_len      : 経緯テキストの文字数
        - relationship_state_len: 関係テキストの文字数
        - wm_thread_count       : Open な WM スレッド数
        - memory_count          : アクティブな保存記憶件数

    Args:
        sqlite: SQLiteStore。

    Returns:
        記録したスナップショット行数。
    """
    recorded = 0
    for char in sqlite.list_characters():
        try:
            meters = {
                "inner_narrative_len": len(char.inner_narrative or ""),
                "self_history_len": len(char.self_history or ""),
                "relationship_state_len": len(char.relationship_state or ""),
                "wm_thread_count": len(
                    sqlite.list_working_memory_threads(char.id, is_open=True)
                ),
                "memory_count": len(sqlite.get_all_active_inscribed_memories(char.id)),
            }
        except Exception:
            logger.exception("肥大メーター計測に失敗 char=%s", char.name)
            continue
        for meter_id, value in meters.items():
            sqlite.record_meter(meter_id, value, character_id=char.id)
            recorded += 1
    return recorded
