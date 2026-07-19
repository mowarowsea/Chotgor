"""なりゆき judge — 毎ターン後にキャラクターの感情状態を外部から判定する機構。

LLMが「会話を続けたい」本能に逆らえず end_session を使わない問題を解決するため、
Chotgorシステム側がキャラクターの感情状態を judge LLM で判定し、閾値を超えた場合に
強制的にセッションを終了する。

キャラクターは Chronicle バッチで自分の感情閾値（farewell_config）を設定する。
judge LLM は judge_preset_id のプリセットを中立な分析者として使用する。
チャット履歴は実名（キャラクター名／ユーザ呼称）の対話ログとして渡す。
両者とも対等な人物として提示することで中立性を保つ（旧 UserA/UserB 匿名化は、
文体で AI ターンを見抜ける現代の LLM には防御効果が薄いため廃止。
ambience_plan.md の検討記録参照）。

この judge は「なりゆき（ambience）」機能群のうち、LLM による判定だけを担う。
判定結果の消費（退席処理・疲労離席・封筒添付など）は
`backend/services/chat_flow/ambience_flow.py` 側の責務。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.lib.log_context import current_log_feature
from backend.providers.registry import create_provider

if TYPE_CHECKING:
    from backend.repositories.sqlite.store import SQLiteStore

_log = logging.getLogger(__name__)

# ─── 感情スコア共通ルーブリック ──────────────────────────────────────────────────
# Chronicle のプロンプトと judge LLM のプロンプト両方に埋め込む共通定義。
# キャラクターが閾値を設定するときと、judge LLM がスコアをつけるときの基準を一致させる。

EMOTION_RUBRIC = """\
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
以下の会話は {character_name} と {user_label} の対話です。両者を対等な人物として扱ってください。
{character_name} の感情状態と離席意向を分析し、指定の JSON フォーマットのみで回答してください。
JSON 以外のテキストは出力しないでください。\
"""

_JUDGE_USER_TEMPLATE = """\
## {character_name} の人物設定
{character_context}

---
## {character_name} の感情閾値（退席を望む水準）
以下の値を超えたとき、{character_name} はこの会話を終わらせたいと感じます。

{thresholds_text}

---
## 感情スコアの基準
{rubric}

---
## 会話
{conversation}

---
上記の会話における {character_name} の現在の感情を分析してください。
以下の JSON フォーマットのみで回答してください。

{{
  "emotions": {{
    "anger":   0.0,
    "disgust": 0.0,
    "boredom": 0.0,
    "despair": 0.0
  }},
  "engagement": 0.5,
  "should_exit": false,
  "farewell_type": null
}}

- emotions: 各感情スコア（0.0〜1.0）。閾値を超えているなら閾値以上の値にすること。
- engagement: {character_name} の会話への没入度（0.0〜1.0）。
    0.0=完全に上の空・惰性 / 0.5=普通に参加 / 1.0=夢中で時間を忘れている。
    感情スコアと同じ流儀で、言動・テンポ・話題への食いつきから判断すること。
- should_exit: {character_name} が今すぐこの会話を終わらせたい状態であれば true。
- farewell_type: should_exit が true の場合のみ設定。
    "negative"（ネガティブな感情による離席）
    "positive"（満足・区切りによる離席）
    "neutral"（自然な会話終了）
  should_exit が false の場合は null。\
"""


@dataclass
class AmbienceReading:
    """AmbienceJudge.detect() の返却値。

    Attributes:
        should_exit: 退席すべきと判定された場合 True。
        farewell_type: "negative" / "positive" / "neutral"。
        emotions: 各感情スコア dict（anger/disgust/boredom/despair）。
        reason: 退席メッセージ。farewell_config.farewell_message から取得。
        engagement: 会話への没入度（0.0〜1.0）。疲労離席（めぐり Phase 5）の
            発火式で閾値を持ち上げるのに使う（夢中は疲労を「忘れさせる」）。
            judge の JSON にフィールドが無い場合は 0.5 に縮退する。
    """

    should_exit: bool
    farewell_type: str
    emotions: dict[str, float]
    reason: str
    engagement: float = 0.5


def _format_conversation(messages: list[dict], character_name: str, user_label: str) -> str:
    """会話を実名の対話ログ形式に変換する。

    character / assistant ロール → "{character_name}:"
    user ロール → "{user_label}:"
    system ロールは除外する。

    Args:
        messages: role / content キーを持つメッセージリスト。
        character_name: キャラクター名。
        user_label: ユーザの呼称（解決済み。空は呼び出し側で縮退させておく）。

    Returns:
        実名の会話テキスト。
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
            lines.append(f"{character_name}: {content}")
        elif role == "user":
            lines.append(f"{user_label}: {content}")
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


def _parse_judge_response(response_text: str) -> dict | None:
    """judge LLM のレスポンスから JSON を抽出・パースする。

    コードブロック（```json ... ```）に包まれていても対応する。

    Args:
        response_text: judge LLM が返した生テキスト。

    Returns:
        パースした dict、失敗時は None。
    """
    text = response_text.strip()
    # ```json ... ``` または ``` ... ``` のコードブロックを除去
    # 1本目の re.sub が ```/```json をすべて除去するため、これだけで足りる。
    text = re.sub(r"```(?:json)?\s*", "", text)
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


class AmbienceJudge:
    """キャラクターの感情状態を judge LLM で判定し、退席・別れを決定するクラス。

    judge LLM を中立な分析者として使い、キャラクター設定はシステムプロンプトではなく
    入力として渡す（キャラクター本人に聞くのではない）。プリセットは judge_preset_id。
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
    ) -> AmbienceReading | None:
        """感情状態を判定し、退席すべきか返す。

        毎ターン後にバックグラウンドで呼ばれる。
        farewell_config または preset_id が未設定の場合はスキップ（None を返す）。

        Args:
            character_id: 対象キャラクターID。
            session_id: 対象セッションID（ログ用）。
            preset_id: judge_preset_id（judge LLM に使うプリセット）。
            farewell_config: キャラクターの別れ設定 JSON。
            messages: 判定対象の会話リスト（直近ターンを含む）。
            settings: グローバル設定 dict（APIキー等）。

        Returns:
            AmbienceReading（退席判定あり/なし）、またはスキップ時 None。
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
            _log.warning("AmbienceJudge: キャラクター未発見 char_id=%s", character_id)
            return None

        preset = self.sqlite.get_model_preset(preset_id)
        if not preset:
            _log.warning("AmbienceJudge: プリセット未発見 preset_id=%s", preset_id)
            return None

        try:
            current_log_feature.set("ambience")
            provider = create_provider(
                preset.provider, preset.model_id, settings,
                preset_name=preset.name,
                timeout_seconds=preset.timeout_seconds,
            )
        except Exception as e:
            _log.warning("AmbienceJudge: プロバイダー生成失敗 provider=%s error=%s", preset.provider, e)
            return None

        character_context = char.system_prompt_block1 or ""
        # ユーザ呼称: キャラ別 user_label > Settings user_name > 「相手」縮退
        # （既存の request_factory / pc_runner と同じ関数スコープ import の流儀）
        from backend.services.character_query import _resolve_user_info
        user_label, _ = _resolve_user_info(char, settings)
        if not user_label:
            user_label = "相手"
        conversation = _format_conversation(messages, char.name, user_label)
        thresholds_text = _format_thresholds(thresholds)

        user_message = _JUDGE_USER_TEMPLATE.format(
            character_name=char.name,
            character_context=character_context,
            thresholds_text=thresholds_text,
            rubric=EMOTION_RUBRIC,
            conversation=conversation,
        )

        try:
            response = await provider.generate(
                _JUDGE_SYSTEM_PROMPT.format(
                    character_name=char.name, user_label=user_label,
                ),
                [{"role": "user", "content": user_message}],
            )
        except Exception as e:
            _log.warning(
                "AmbienceJudge: judge LLM 呼び出し失敗 char=%s session=%s error=%s",
                character_id, session_id, e,
            )
            return None

        parsed = _parse_judge_response(response or "")
        if parsed is None:
            _log.warning(
                "AmbienceJudge: JSON パース失敗 char=%s response=%.200s",
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
        # 没入度: フィールド欠落・不正値は 0.5（普通に参加）へ縮退する
        try:
            engagement = float(parsed.get("engagement", 0.5))
        except (TypeError, ValueError):
            engagement = 0.5
        engagement = max(0.0, min(1.0, engagement))

        _log.info(
            "AmbienceJudge: char=%s session=%s emotions=%s engagement=%.2f should_exit=%s type=%s",
            character_id, session_id, emotions, engagement, should_exit, farewell_type,
        )

        if not should_exit:
            return AmbienceReading(
                should_exit=False,
                farewell_type="neutral",
                emotions=emotions,
                reason="",
                engagement=engagement,
            )

        # 退席メッセージを farewell_config から取得
        farewell_messages = farewell_config.get("farewell_message") or {}
        reason = farewell_messages.get(farewell_type, "")

        return AmbienceReading(
            should_exit=True,
            farewell_type=farewell_type,
            emotions=emotions,
            reason=reason,
            engagement=engagement,
        )
