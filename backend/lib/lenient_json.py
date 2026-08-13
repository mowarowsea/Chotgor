"""LLM 応答テキストから JSON を寛容に取り出すユーティリティ。

LLM は JSON 出力を指示しても、日本語文中で 強調に `"..."` を使うなどして
文字列値内に裸のダブルクォートを混入させ、`json.loads` を壊すことがある。
（実例: chronicle 応答で `"post": "感情表現の"型"の方が強い"` のようなケース）

このモジュールは、まず素直に `json.loads` で試し、失敗したら `json-repair` に
フォールバックする。`json-repair` は LLM 出力修復専用のライブラリで、
未エスケープの `"`・trailing comma・未終端・シングルクォート等を救う。

呼び出し側の契約は既存 `_parse_chronicle_response` を継承:
  - null / 空 / 非 dict → None（変更なし）
  - パース成功        → dict
  - 修復も失敗        → {} （呼び出し側で「パース失敗」として扱う）
"""
from __future__ import annotations

import json
import logging

from json_repair import repair_json

_log = logging.getLogger(__name__)


def parse_lenient_json(
    response_text: str | None,
    *,
    feature_label: str = "",
) -> dict | None:
    """LLM 応答テキストから JSON dict を寛容に取り出す。

    処理順:
      1. 応答内から最初の `{` 〜 最後の `}` を切り出す（コードフェンス除去代わり）。
      2. まず `json.loads` で試す（正常応答はここで通る、修復コスト回避）。
      3. 失敗したら `json_repair.repair_json` で修復パース。
      4. それでも dict にならなければ空 dict を返す。

    Args:
        response_text: LLM の応答テキスト。None / 空文字 / `null` は None を返す。
        feature_label: 修復発動時のログ識別用（例: "chronicle"）。

    Returns:
        - dict: 素直にパースできた、または修復パース成功
        - None: 応答が空、または **JSON として読めた上で** dict ではない
          （`null` / リスト / 文字列。LLM が明示的に「変更なし」を返したケース）
        - {}:  JSON として読めず、修復しても dict にならなかった
          （呼び出し側で「パース失敗」として扱う契約）

    None と {} の境目は重要である。chronicle は None を「変更なし＝正常終了」と
    見なして当日会話を処理済みにマークするため、パース失敗を None で返すと
    **その日の会話が二度と棚卸しされないまま失われる**。
    """
    if not response_text:
        return None
    text = response_text.strip()
    if not text:
        return None

    # コードフェンス（```json ... ```）や前後の散文を落とすため、
    # 最初の `{` から最後の `}` までを対象とする。
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    # 素直に通れば修復ライブラリを起動しない（コスト・ログノイズ回避）
    try:
        result = json.loads(text)
    except Exception:
        try:
            repaired = repair_json(text, return_objects=True)
        except Exception as e:
            _log.warning(
                "parse_lenient_json: 修復パースも失敗 feature=%s error=%s",
                feature_label or "(none)", e,
            )
            return {}
        if not isinstance(repaired, dict):
            # json-repair は救えない入力に対して例外ではなく空文字列などを返す
            # （例: "これはJSONではありません" → ""）。ここを None（＝変更なし）で
            # 返すと、呼び出し側がパース失敗を正常終了と取り違える。
            _log.warning(
                "parse_lenient_json: 修復しても dict にならず feature=%s type=%s",
                feature_label or "(none)", type(repaired).__name__,
            )
            return {}
        _log.info(
            "parse_lenient_json: json-repair で修復成功 feature=%s len=%d",
            feature_label or "(none)", len(text),
        )
        return repaired

    if isinstance(result, dict):
        return result
    # JSON としては読めたが dict ではない（`null` / リスト / 文字列）。
    # LLM が明示的に「変更なし」を返したケースとして None を返す。
    return None
