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
{location_block}
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
  "farewell_type": null,
  "location_label": null
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
  should_exit が false の場合は null。
- location_label: 「対面の場所判定」セクションがある場合のみ設定。無い場合は null。\
"""

# 対面モード時のみ _JUDGE_USER_TEMPLATE の {location_block} に差し込むブロック。
# ちらつき対策として「明確に変わったと判断できないなら前回を返す」を明示する。
_LOCATION_BLOCK_TEMPLATE = """
---
## 対面の場所判定
いま {character_name} と {user_label} は対面で会っています。
会話の内容から「いま2人がいる場所」を以下の候補から1つ選び、
location_label に設定してください。

候補ラベル: {candidates}
前回の判定: {prev_label}

- 会話から場所が明確に変わったと判断できる場合のみ、新しいラベルを返すこと。
- 明確に変わったと判断できないなら、前回の判定と同じラベルを返すこと。
- どの候補にも当てはまらない・判断できない場合は前回の判定を返すこと
  （前回の判定が「なし」なら null）。
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
        location_label: 対面モード時の場所判定ラベル。detect() 内で候補検証済み
            （候補外の生値は前回ラベルへ丸められる）。非対面・候補なし・判定不能は None。
    """

    should_exit: bool
    farewell_type: str
    emotions: dict[str, float]
    reason: str
    engagement: float = 0.5
    location_label: str | None = None


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


# ─── Jev judge ────────────────────────────────────────────────────────────────
# judge_preset_id にこの予約値が入っていたら LLM プリセットではなく Jev で判定する。
# プリセット表には登録しない（ambience_plan.md「judge エンジンとしての Jev」参照）。
JEV_JUDGE_PRESET_ID = "__jev__"

_EMOTION_LABELS = {
    "anger": "怒り（不当・理不尽な扱いへの反発）",
    "disgust": "不快・嫌悪（生理的な嫌悪感・不快感）",
    "boredom": "飽き（刺激のなさ、退屈、つまらなさ）",
    "despair": "諦め・絶望（この関係・会話への希望の喪失）",
}
# EMOTION_RUBRIC の 0.0/0.25/0.5/0.75/1.0 と同じ段階。score/4 で閾値と同じ尺度に戻す
_EMOTION_LEVELS = [
    "ほとんど感じない",
    "うっすら感じている（本人も気づかないかもしれない）",
    "自尊心・心身に負荷がかかり、明確に感じている",
    "言動・態度に強く影響している、かなり強く感じている",
    "一刻も早く対象を排除したい、または対象から逃げたい",
]
_ENGAGEMENT_LEVELS = ["完全に上の空・惰性", "普通に参加", "夢中で時間を忘れている"]
# noul は確率でしか返らないため真偽へ戻す境界（技術的制約による唯一の変換）
_JEV_EXIT_PROB = 0.5
# state は 32k トークン上限。日本語は概ね1文字≒1トークン強なので文字数で余裕を持たせる
_JEV_CONTEXT_MAX_CHARS = 4000
_JEV_CONVERSATION_MAX_CHARS = 20000


def _build_jev_questions(
    character_name: str,
    user_label: str,
    location_candidates: list[str],
    prev_bg_label: str | None,
) -> dict:
    # 文言は LLM judge（_JUDGE_USER_TEMPLATE / _LOCATION_BLOCK_TEMPLATE）に揃える。
    # 変えるのは Jev の型に載せるための技術的に必要な部分だけ。
    questions: dict[str, dict] = {}
    for key, label in _EMOTION_LABELS.items():
        questions[key] = {
            "type": "score",
            "instructions": (
                f"{character_name} の現在の {label} のスコアは？"
                "閾値を超えているなら閾値以上の段階を選ぶこと。"
            ),
            "criteria": _EMOTION_LEVELS,
        }
    questions["engagement"] = {
        "type": "score",
        "instructions": (
            f"{character_name} の会話への没入度は？"
            "言動・テンポ・話題への食いつきから判断すること。"
        ),
        "criteria": _ENGAGEMENT_LEVELS,
    }
    questions["should_exit"] = {
        "type": "noul",
        "instructions": f"{character_name} は今すぐこの会話を終わらせたい状態ですか？",
    }
    questions["farewell_type"] = {
        "type": "choice",
        "instructions": f"{character_name} が会話を終わらせるとしたら、その種別は？",
        "criteria": {
            "negative": "ネガティブな感情による離席",
            "positive": "満足・区切りによる離席",
            "neutral": "自然な会話終了",
        },
    }
    if location_candidates:
        questions["location_label"] = {
            "type": "choice",
            "instructions": (
                f"いま {character_name} と {user_label} は対面で会っています。"
                "会話の内容から「いま2人がいる場所」を1つ選んでください。"
                f"前回の判定: {prev_bg_label or 'なし'}。"
                "会話から場所が明確に変わったと判断できる場合のみ新しい場所を選び、"
                "そうでなければ前回の判定と同じ場所を選ぶこと。"
            ),
            "criteria": {c: c for c in location_candidates},
        }
    return questions


# ─── 判定エンジン（LLM / Jev を同じ口で扱うラッパ） ─────────────────────────────


@dataclass
class JudgeInput:
    """判定エンジンへ渡す材料。エンジンごとの問い方の差はエンジン側に閉じる。"""

    character_name: str
    user_label: str
    character_context: str
    thresholds_text: str
    conversation: str
    # 対面かつ候補ありのときだけ非空（場所判定の有効条件は detect() が決める）
    location_candidates: list[str]
    prev_bg_label: str | None


class JudgeEngine:
    """なりゆき judge の判定エンジン。

    read() は LLM judge の JSON と同じ形の dict（emotions / engagement / should_exit /
    farewell_type / location_label）を返し、失敗時は None。値の検証・縮退は detect() 側で共通。
    リクエスト／レスポンスのログは下層（BaseLLMProvider / JevClient）が同じ形式で残すため、
    logs 画面ではどちらのエンジンも ambience 行として並ぶ（preset 欄がエンジン名）。
    """

    label: str = ""

    async def read(self, inp: JudgeInput) -> dict | None:
        raise NotImplementedError


class LlmJudgeEngine(JudgeEngine):
    """judge LLM プリセットで判定する（JSON をプロンプトで指定して生成させる）。"""

    def __init__(self, provider, label: str) -> None:
        self.provider = provider
        self.label = label

    async def read(self, inp: JudgeInput) -> dict | None:
        location_block = ""
        if inp.location_candidates:
            location_block = _LOCATION_BLOCK_TEMPLATE.format(
                character_name=inp.character_name,
                user_label=inp.user_label,
                candidates=json.dumps(inp.location_candidates, ensure_ascii=False),
                prev_label=(
                    json.dumps(inp.prev_bg_label, ensure_ascii=False)
                    if inp.prev_bg_label else "なし"
                ),
            )
        user_message = _JUDGE_USER_TEMPLATE.format(
            character_name=inp.character_name,
            character_context=inp.character_context,
            thresholds_text=inp.thresholds_text,
            rubric=EMOTION_RUBRIC,
            location_block=location_block,
            conversation=inp.conversation,
        )
        try:
            response = await self.provider.generate(
                _JUDGE_SYSTEM_PROMPT.format(
                    character_name=inp.character_name, user_label=inp.user_label,
                ),
                [{"role": "user", "content": user_message}],
            )
        except Exception as e:
            _log.warning("AmbienceJudge: judge LLM 呼び出し失敗 engine=%s error=%s", self.label, e)
            return None

        parsed = _parse_judge_response(response or "")
        if parsed is None:
            _log.warning(
                "AmbienceJudge: JSON パース失敗 engine=%s response=%.200s", self.label, response,
            )
        return parsed


class JevJudgeEngine(JudgeEngine):
    """TypeSafe Jev で判定する（型付き質問の確率を LLM judge と同じ dict 形へ詰め替える）。"""

    def __init__(self, client) -> None:
        self.client = client
        self.label = client.preset_name

    async def read(self, inp: JudgeInput) -> dict | None:
        from backend.providers.base import LLMApiError

        # 会話は直近が判定の主対象なので末尾を残して切り詰める
        state = (
            f"以下は {inp.character_name} と {inp.user_label} の対話です。両者は対等な人物です。\n\n"
            f"## {inp.character_name} の人物設定\n{inp.character_context[:_JEV_CONTEXT_MAX_CHARS]}\n\n"
            f"## {inp.character_name} の感情閾値（これを超えると会話を終わらせたいと感じる）\n"
            f"{inp.thresholds_text}\n\n"
            f"## 会話\n{inp.conversation[-_JEV_CONVERSATION_MAX_CHARS:]}"
        )
        try:
            result = await self.client.decide(
                state,
                _build_jev_questions(
                    inp.character_name, inp.user_label,
                    inp.location_candidates, inp.prev_bg_label,
                ),
            )
        except LLMApiError as e:
            _log.warning("AmbienceJudge: Jev 呼び出し失敗 error=%s", e)
            return None

        answers = result.get("answers") or {}

        def _score01(key: str, levels: int) -> float:
            try:
                return max(0.0, min(1.0, float(answers[key]["score"]) / (levels - 1)))
            except (KeyError, TypeError, ValueError):
                return 0.0

        parsed: dict = {
            "emotions": {k: _score01(k, len(_EMOTION_LEVELS)) for k in _EMOTION_LABELS},
            "engagement": (
                _score01("engagement", len(_ENGAGEMENT_LEVELS)) if "engagement" in answers else 0.5
            ),
            "should_exit": float((answers.get("should_exit") or {}).get("noul", 0.0)) >= _JEV_EXIT_PROB,
            "farewell_type": (answers.get("farewell_type") or {}).get("choice"),
            "location_label": None,
        }
        if inp.location_candidates:
            # 候補外・欠落時の前回踏襲は detect() 側の共通検証が行う
            parsed["location_label"] = (answers.get("location_label") or {}).get("choice")
        return parsed


def resolve_judge_engine(sqlite: "SQLiteStore", preset_id: str, settings: dict) -> JudgeEngine | None:
    """judge_preset_id から判定エンジンを組み立てる。予約値 JEV_JUDGE_PRESET_ID なら Jev。"""
    if preset_id == JEV_JUDGE_PRESET_ID:
        from backend.providers.jev_client import JevClient
        return JevJudgeEngine(JevClient.from_settings(settings))

    preset = sqlite.get_model_preset(preset_id)
    if not preset:
        _log.warning("AmbienceJudge: プリセット未発見 preset_id=%s", preset_id)
        return None
    try:
        provider = create_provider(
            preset.provider, preset.model_id, settings,
            preset_name=preset.name,
            timeout_seconds=preset.timeout_seconds,
        )
    except Exception as e:
        _log.warning("AmbienceJudge: プロバイダー生成失敗 provider=%s error=%s", preset.provider, e)
        return None
    return LlmJudgeEngine(provider, preset.name)


class AmbienceJudge:
    """キャラクターの感情状態を judge で判定し、退席・別れを決定するクラス。

    judge は中立な分析者として使い、キャラクター設定はシステムプロンプトではなく
    入力として渡す（キャラクター本人に聞くのではない）。エンジンは judge_preset_id で
    選ぶ（LLM プリセット or Jev）。
    """

    def __init__(self, sqlite: "SQLiteStore") -> None:
        self.sqlite = sqlite

    async def detect(
        self,
        character_id: str,
        session_id: str,
        preset_id: str,
        farewell_config: dict,
        messages: list[dict],
        settings: dict,
        face_to_face: bool = False,
        bg_label_candidates: list[str] | None = None,
        prev_bg_label: str | None = None,
    ) -> AmbienceReading | None:
        """感情状態（＋対面時は場所ラベル）を判定して返す。

        毎ターン後にバックグラウンドで呼ばれる。
        farewell_config または preset_id が未設定の場合はスキップ（None を返す）。

        Args:
            preset_id: judge_preset_id（LLM プリセットID、または JEV_JUDGE_PRESET_ID）。
            face_to_face: 対面モード中なら True。場所判定はこのときだけ行う。
            bg_label_candidates: 場所判定の候補ラベル群（face_to_face_bg_images の label）。
                空ラベルは候補から除外される。空/None なら場所判定はスキップ。
            prev_bg_label: セッションの前回判定ラベル（chat_sessions.current_bg_label）。
                候補外の judge 出力・判定不能時の踏襲先。

        Returns:
            AmbienceReading（退席判定あり/なし）、またはスキップ・失敗時 None。
            location_label は候補検証済みの値（候補外は前回踏襲に丸め済み）。
        """
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

        # ログの source_type をエンジン呼び出しより前に確定させる（下層のログがこれを見る）
        current_log_feature.set("ambience")
        engine = resolve_judge_engine(self.sqlite, preset_id, settings)
        if engine is None:
            return None

        # ユーザ呼称: キャラ別 user_label > Settings user_name > 「相手」縮退
        # （既存の request_factory / pc_runner と同じ関数スコープ import の流儀）
        from backend.services.character_query import _resolve_user_info
        user_label, _ = _resolve_user_info(char, settings)
        if not user_label:
            user_label = "相手"

        # 場所判定: 対面中かつ候補ラベルあり（空ラベル除外後）のときだけ
        candidates = [c for c in (bg_label_candidates or []) if (c or "").strip()]
        location_enabled = bool(face_to_face and candidates)

        parsed = await engine.read(JudgeInput(
            character_name=char.name,
            user_label=user_label,
            character_context=char.system_prompt_block1 or "",
            thresholds_text=_format_thresholds(thresholds),
            conversation=_format_conversation(messages, char.name, user_label),
            location_candidates=candidates if location_enabled else [],
            prev_bg_label=prev_bg_label,
        ))
        if parsed is None:
            _log.warning(
                "AmbienceJudge: 判定失敗 engine=%s char=%s session=%s",
                engine.label, character_id, session_id,
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

        # 場所ラベル: 候補内の値だけ採用。候補外・null・判定無効時は前回踏襲
        # （前回も無ければ None = 背景なし）。
        location_label: str | None = None
        if location_enabled:
            raw_label = parsed.get("location_label")
            if isinstance(raw_label, str) and raw_label in candidates:
                location_label = raw_label
            else:
                location_label = prev_bg_label

        _log.info(
            "AmbienceJudge: engine=%s char=%s session=%s emotions=%s engagement=%.2f "
            "should_exit=%s type=%s location=%s",
            engine.label, character_id, session_id, emotions, engagement,
            should_exit, farewell_type, location_label,
        )

        if not should_exit:
            return AmbienceReading(
                should_exit=False,
                farewell_type="neutral",
                emotions=emotions,
                reason="",
                engagement=engagement,
                location_label=location_label,
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
            location_label=location_label,
        )
