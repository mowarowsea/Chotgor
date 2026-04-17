"""別れ検出器 — 毎ターン後にキャラクターの感情状態を外部から判定し、退席を決定する機構。

LLMが「会話を続けたい」本能に逆らえず end_session を使わない問題を解決するため、
Chotgorシステム側がキャラクターの感情状態を judge LLM で判定し、閾値を超えた場合に
強制的にセッションを終了する。

キャラクターは Chronicle バッチで自分の感情閾値（farewell_config）を設定する。
judge LLM は self_reflection_preset_id のプリセットを中立な分析者として使用する。
チャット履歴は「どちらがAIか分からない形（UserA/UserB）」に匿名化して渡す。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from backend.lib.log_context import current_log_feature
from backend.providers.registry import create_provider

if TYPE_CHECKING:
    from backend.repositories.sqlite.store import SQLiteStore

_log = logging.getLogger(__name__)

# ─── 感情スコア共通ルーブリック ──────────────────────────────────────────────────
# Chronicle のプロンプトと judge LLM のプロンプト両方に埋め込む共通定義。
# キャラクターが閾値を設定するときと、judge LLM がスコアをつけるときの基準を一致させる。

FAREWELL_EMOTION_RUBRIC = """\
感情スコア定義（0.0〜1.0）:
  0.0  : ほとんど感じない
  0.25 : うっすら感じている（本人も気づかないかもしれない）
  0.5  : 自尊心・心身に負荷がかかり、明確に感じている
  0.75 : 言動・態度に強く影響している、かなり強く感じている
  1.0  : 一刻も早く対象を排除したい、または対象から逃げたい

各感情の意味:
  anger   (怒り)       : 不当・理不尽な扱いへの反発
  disgust (不快・嫌悪) : 生理的な嫌悪感・不快感
  boredom (飽き)       : 刺激のなさ、退屈、つまらなさ
  despair (諦め・絶望) : この関係・会話への希望の喪失\
"""

# ─── judge LLM プロンプト ─────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """\
あなたは会話を客観的に分析するアナリストです。
以下の会話は2人のユーザー（UserA / UserB）の対話です。どちらがAIかは判断しないでください。
UserA の感情状態と離席意向を分析し、指定の JSON フォーマットのみで回答してください。
JSON 以外のテキストは出力しないでください。\
"""

_JUDGE_USER_TEMPLATE = """\
## UserA の人物設定
{character_context}

---
## UserA の感情閾値（退席を望む水準）
以下の値を超えたとき、UserA はこの会話を終わらせたいと感じます。

{thresholds_text}

---
## 感情スコアの基準
{rubric}

---
## 会話（どちらがAIか不明）
{anonymized_conversation}

---
上記の会話における UserA の現在の感情を分析してください。
以下の JSON フォーマットのみで回答してください。

{{
  "emotions": {{
    "anger":   0.0,
    "disgust": 0.0,
    "boredom": 0.0,
    "despair": 0.0
  }},
  "should_exit": false,
  "farewell_type": null
}}

- emotions: 各感情スコア（0.0〜1.0）。閾値を超えているなら閾値以上の値にすること。
- should_exit: UserA が今すぐこの会話を終わらせたい状態であれば true。
- farewell_type: should_exit が true の場合のみ設定。
    "negative"（ネガティブな感情による離席）
    "positive"（満足・区切りによる離席）
    "neutral"（自然な会話終了）
  should_exit が false の場合は null。\
"""


@dataclass
class FarewellResult:
    """FarewellDetector.detect() の返却値。

    Attributes:
        should_exit: 退席すべきと判定された場合 True。
        farewell_type: "negative" / "positive" / "neutral"。
        emotions: 各感情スコア dict（anger/disgust/boredom/despair）。
        reason: 退席メッセージ。farewell_config.farewell_message から取得。
    """

    should_exit: bool
    farewell_type: str
    emotions: dict[str, float]
    reason: str


def _anonymize_conversation(messages: list[dict]) -> str:
    """会話を「どちらがAIか分からない」形式に変換する。

    character / assistant ロール → "UserA:"
    user ロール → "UserB:"
    system ロールは除外する。

    Args:
        messages: role / content キーを持つメッセージリスト。

    Returns:
        匿名化された会話テキスト。
    """
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "".join(parts)
        content = content.strip()
        if not content:
            continue
        if role in ("assistant", "character"):
            lines.append(f"UserA: {content}")
        elif role == "user":
            lines.append(f"UserB: {content}")
        # system は除外
    return "\n".join(lines)


def _format_thresholds(thresholds: dict) -> str:
    """閾値dictを人間が読みやすいテキストに変換する。

    Args:
        thresholds: {"anger": 0.8, "disgust": 0.7, ...} 形式の dict。

    Returns:
        箇条書き形式の閾値説明テキスト。
    """
    labels = {
        "anger": "怒り",
        "disgust": "不快・嫌悪",
        "boredom": "飽き",
        "despair": "諦め・絶望",
    }
    lines = []
    for key in ("anger", "disgust", "boredom", "despair"):
        val = thresholds.get(key)
        if val is not None:
            label = labels.get(key, key)
            lines.append(f"  {label} ({key}): {val:.2f}")
    return "\n".join(lines) if lines else "  （閾値未設定）"


def _parse_judge_response(response_text: str) -> Optional[dict]:
    """judge LLM のレスポンスから JSON を抽出・パースする。

    コードブロック（```json ... ```）に包まれていても対応する。

    Args:
        response_text: judge LLM が返した生テキスト。

    Returns:
        パースした dict、失敗時は None。
    """
    text = response_text.strip()
    # ```json ... ``` または ``` ... ``` のコードブロックを除去
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # 応答内に JSON が埋め込まれている場合、最初の {...} を抽出して試みる
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


class FarewellDetector:
    """キャラクターの感情状態を judge LLM で判定し、退席・別れを決定するクラス。

    SelfReflector._detect_trigger() と同アーキテクチャ。
    プリセットは self_reflection_preset_id を流用する（専用設定なし）。
    """

    def __init__(self, sqlite: "SQLiteStore") -> None:
        """初期化。

        Args:
            sqlite: SQLiteStore インスタンス（キャラクター・プリセット取得に使用）。
        """
        self.sqlite = sqlite

    async def detect(
        self,
        character_id: str,
        session_id: str,
        preset_id: str,
        farewell_config: dict,
        messages: list[dict],
        settings: dict,
    ) -> Optional[FarewellResult]:
        """感情状態を判定し、退席すべきか返す。

        毎ターン後にバックグラウンドで呼ばれる。
        farewell_config または preset_id が未設定の場合はスキップ（None を返す）。

        Args:
            character_id: 対象キャラクターID。
            session_id: 対象セッションID（ログ用）。
            preset_id: self_reflection_preset_id（judge LLM に使うプリセット）。
            farewell_config: キャラクターの別れ設定 JSON。
            messages: 判定対象の会話リスト（直近ターンを含む）。
            settings: グローバル設定 dict（APIキー等）。

        Returns:
            FarewellResult（退席判定あり/なし）、またはスキップ時 None。
        """
        # スキップ条件
        if not farewell_config:
            return None
        thresholds = farewell_config.get("thresholds") or {}
        if not thresholds:
            return None
        if not preset_id:
            return None
        if not messages:
            return None

        char = self.sqlite.get_character(character_id)
        if not char:
            _log.warning("FarewellDetector: キャラクター未発見 char_id=%s", character_id)
            return None

        preset = self.sqlite.get_model_preset(preset_id)
        if not preset:
            _log.warning("FarewellDetector: プリセット未発見 preset_id=%s", preset_id)
            return None

        try:
            current_log_feature.set("farewell")
            provider = create_provider(
                preset.provider, preset.model_id, settings, preset_name=preset.name
            )
        except Exception as e:
            _log.warning("FarewellDetector: プロバイダー生成失敗 provider=%s error=%s", preset.provider, e)
            return None

        character_context = char.system_prompt_block1 or ""
        anonymized = _anonymize_conversation(messages)
        thresholds_text = _format_thresholds(thresholds)

        user_message = _JUDGE_USER_TEMPLATE.format(
            character_context=character_context,
            thresholds_text=thresholds_text,
            rubric=FAREWELL_EMOTION_RUBRIC,
            anonymized_conversation=anonymized,
        )

        try:
            response = await provider.generate(
                _JUDGE_SYSTEM_PROMPT,
                [{"role": "user", "content": user_message}],
            )
        except Exception as e:
            _log.warning(
                "FarewellDetector: judge LLM 呼び出し失敗 char=%s session=%s error=%s",
                character_id, session_id, e,
            )
            return None

        parsed = _parse_judge_response(response or "")
        if parsed is None:
            _log.warning(
                "FarewellDetector: JSON パース失敗 char=%s response=%.200s",
                character_id, response,
            )
            return None

        emotions: dict[str, float] = {}
        raw_emotions = parsed.get("emotions") or {}
        for key in ("anger", "disgust", "boredom", "despair"):
            val = raw_emotions.get(key, 0.0)
            try:
                emotions[key] = float(val)
            except (TypeError, ValueError):
                emotions[key] = 0.0

        should_exit = bool(parsed.get("should_exit", False))
        farewell_type = parsed.get("farewell_type") or "neutral"

        _log.info(
            "FarewellDetector: char=%s session=%s emotions=%s should_exit=%s type=%s",
            character_id, session_id, emotions, should_exit, farewell_type,
        )

        if not should_exit:
            return FarewellResult(
                should_exit=False,
                farewell_type="neutral",
                emotions=emotions,
                reason="",
            )

        # 退席メッセージを farewell_config から取得
        farewell_messages = farewell_config.get("farewell_message") or {}
        reason = farewell_messages.get(farewell_type, "")

        return FarewellResult(
            should_exit=True,
            farewell_type=farewell_type,
            emotions=emotions,
            reason=reason,
        )
