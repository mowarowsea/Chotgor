"""計器 Tier 3 — 判定巡回（LLM・サンプリング）。

日次で当日のキャラ応答から10件程度をサンプリングし、判定 LLM がルーブリック
（OOC度・フォーマット清浄度・隠れエラー・指示逸脱）で採点する。逸脱があれば
アラーム＋該当応答への参照を残す（docs/aliveness_plan.md §3 Tier 3）。

判定器は「人格なき環境」扱い — キャラクター問い合わせ原則（ask_character）の
対象外であり、GM と同じく素の LLM を直接使う。キャラ本人の内省（修復装置）とは
別物で、こちらは**ユーザの確信のための装置**。

判定プリセットは global_settings の "instruments_judge_preset_id" で指定する
（未設定なら Tier 3 はスキップ。安いモデルから開始する想定）。
"""

import json
import logging
import random
import re

from backend.providers.registry import create_provider

logger = logging.getLogger(__name__)

# 1回の巡回で判定するサンプル数
_SAMPLE_SIZE = 10
# 判定対象の応答テキスト上限（プロンプト肥大の抑制）
_MAX_RESPONSE_CHARS = 2000

# 判定ルーブリック。判定器には人格を与えず、監査タスクとしてだけ依頼する。
_JUDGE_SYSTEM_PROMPT = """\
あなたはテキスト品質の監査プログラムです。会話応答のサンプル群を外形的に検査します。

各サンプルについて以下の4軸で「問題あり」だけを報告してください:
1. ooc         — キャラクターの応答に「AIとして」等のメタ発言・役割崩壊が混ざっている
2. format      — ツールタグ・XML・JSON等の内部フォーマットがユーザ向け本文に漏れている
3. hidden_error — 本文がエラーメッセージ・スタックトレース・空応答の成れの果てである
4. deviation   — 会話の言語や文脈から明らかに逸脱している（日本語会話に英語段落等）

出力は次の JSON のみ（説明文・コードフェンス不要）:
{"problems": [{"index": <サンプル番号>, "kind": "<ooc|format|hidden_error|deviation>", "note": "<50字以内の指摘>"}]}
問題がなければ {"problems": []} とだけ出力してください。
"""


def _build_judge_prompt(samples: list) -> str:
    """サンプル群を判定用プロンプト本文へ整形する。

    Args:
        samples: DebugLogEntry ORM のリスト。

    Returns:
        判定 LLM へ渡す user メッセージ本文。
    """
    lines = ["以下の応答サンプルを検査してください。", ""]
    for i, entry in enumerate(samples):
        response = (entry.response or "")[:_MAX_RESPONSE_CHARS]
        lines += [
            f"--- サンプル {i} (source={entry.source_type}) ---",
            response,
            "",
        ]
    return "\n".join(lines)


def _parse_judge_response(text: str) -> list[dict]:
    """判定 LLM の応答から problems 配列を取り出す（コードフェンス・前後説明に耐性）。

    Args:
        text: 判定 LLM の生応答。

    Returns:
        [{"index": int, "kind": str, "note": str}] のリスト。パース不能なら空。
    """
    if not text:
        return []
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    problems = parsed.get("problems")
    if not isinstance(problems, list):
        return []
    results: list[dict] = []
    for p in problems:
        if not isinstance(p, dict):
            continue
        try:
            idx = int(p.get("index"))
        except (TypeError, ValueError):
            continue
        results.append({
            "index": idx,
            "kind": str(p.get("kind") or "unknown"),
            "note": str(p.get("note") or "")[:200],
        })
    return results


async def run_judgement_patrol(sqlite, settings: dict) -> dict:
    """判定巡回を1回実行する（日次スケジューラから呼ばれる）。

    当日の応答サンプルを判定プリセットで採点し、逸脱があれば
    invariant_id="judgement_patrol" のアラーム（severity="alarm"）を発火する。
    details には該当応答への参照（request_id）を含める。

    Args:
        sqlite: SQLiteStore。
        settings: グローバル設定辞書（API キー等。provider 生成に使う）。

    Returns:
        {"status": "skipped"|"success"|"error", ...} の集計 dict。
    """
    preset_id = sqlite.get_setting("instruments_judge_preset_id", "") or ""
    if not preset_id:
        return {"status": "skipped", "reason": "instruments_judge_preset_id 未設定"}
    preset = sqlite.get_model_preset(preset_id)
    if preset is None:
        return {"status": "error", "error": f"判定プリセットが見つかりません: {preset_id}"}

    candidates = sqlite.sample_today_responses(limit=_SAMPLE_SIZE)
    if not candidates:
        return {"status": "skipped", "reason": "当日の判定対象応答なし"}
    samples = random.sample(candidates, min(_SAMPLE_SIZE, len(candidates)))

    provider = create_provider(
        preset.provider,
        model=preset.model_id,
        settings=settings,
        preset_name=preset.name,
        timeout_seconds=getattr(preset, "timeout_seconds", 300),
    )
    try:
        response_text = await provider.generate(
            _JUDGE_SYSTEM_PROMPT,
            [{"role": "user", "content": _build_judge_prompt(samples)}],
        )
    except Exception as e:
        logger.exception("判定巡回の LLM 呼び出しに失敗")
        return {"status": "error", "error": str(e)}

    problems = _parse_judge_response(response_text or "")
    fired = 0
    for p in problems:
        if not (0 <= p["index"] < len(samples)):
            continue  # 判定器の index 幻覚は捨てる
        entry = samples[p["index"]]
        sqlite.fire_alarm(
            "judgement_patrol",
            details={
                "kind": p["kind"],
                "note": p["note"],
                "request_id": entry.request_id,
                "target": entry.target,
                "source_type": entry.source_type,
            },
        )
        fired += 1
    logger.info(
        "判定巡回 完了 samples=%d problems=%d fired=%d",
        len(samples), len(problems), fired,
    )
    return {"status": "success", "samples": len(samples), "fired": fired}
