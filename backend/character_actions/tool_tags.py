"""CharacterAction（ツール）のタグ変換ロジック。

各 CharacterAction のツール名・タグ名・引数フォーマットを集約し、
tool-use 形式の関数呼び出し結果をログ表示用のタグ本体文字列に変換する。
このモジュールはフォーマット定義のみを持ち、DBアクセスや実行ロジックを含まない。
"""

# ツール名（APIの function_call.name）→ タグ名のマッピング
TOOL_TO_TAG: dict[str, str] = {
    "inscribe_memory": "INSCRIBE_MEMORY",
    "carve_narrative": "CARVE_NARRATIVE",
    "drift":           "DRIFT",
    "drift_reset":     "DRIFT_RESET",
    "switch_angle":    "SWITCH_ANGLE",
    "end_session":     "END_SESSION",
    "power_recall":    "POWER_RECALL",
}


def tool_call_to_tag_body(tool_name: str, args: dict) -> tuple[str, str]:
    """ツール呼び出しを (tag_name, tag_body) に変換する。

    tag_body はタグ方式のパイプ区切りフォーマット（例: "contextual|1.5|内容テキスト"）。
    logs_ui._parse_tag_body に渡すことで統一的に表示できる。
    未登録のツール名は大文字化してタグ名とし、引数を key=value 形式で結合する。

    Args:
        tool_name: ツール名（APIの function_call.name フィールド）。
        args: ツール引数辞書（APIレスポンスの args / input フィールド）。

    Returns:
        (tag_name, body) のタプル。
    """
    tag_name = TOOL_TO_TAG.get(tool_name, tool_name.upper())

    if tag_name == "INSCRIBE_MEMORY":
        # [INSCRIBE_MEMORY:category|impact|content]
        body = f"{args.get('category', '')}|{args.get('impact', '')}|{args.get('content', '')}"

    elif tag_name == "CARVE_NARRATIVE":
        # [CARVE_NARRATIVE:mode|content]
        body = f"{args.get('mode', 'append')}|{args.get('content', '')}"

    elif tag_name == "DRIFT":
        # [DRIFT:content]
        body = str(args.get("content", ""))

    elif tag_name == "DRIFT_RESET":
        # [DRIFT_RESET] — 引数なし
        body = ""

    elif tag_name == "SWITCH_ANGLE":
        # [SWITCH_ANGLE:preset_name|self_instruction]
        body = f"{args.get('preset_name', '')}|{args.get('self_instruction', '')}"

    elif tag_name == "END_SESSION":
        # [END_SESSION:reason]（reason は省略可）
        body = str(args.get("reason", ""))

    elif tag_name == "POWER_RECALL":
        # [POWER_RECALL:query|top_k]
        body = f"{args.get('query', '')}|{args.get('top_k', 5)}"

    else:
        # 未知のツール: キー=値形式でフォールバック
        body = "|".join(f"{k}={v}" for k, v in args.items())

    return (tag_name, body)
