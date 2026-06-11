"""CharacterAction（ツール）のタグ変換ロジック。

各 CharacterAction のツール名・タグ名・引数フォーマット、および
ログ表示用のラベル/色クラスを集約する。

- TOOL_TO_TAG:  MCP/関数呼び出しのツール名 → タグ名のマッピング
- TAG_META:     タグ名 → ログ表示用メタ情報（label / cls）
- tool_call_to_structured_tag():
    tool-use 形式の (name, args) を表示用構造化辞書
    {tag_name, meta, fields, preview} に変換する。

このモジュールはフォーマット定義のみを持ち、DBアクセスや実行ロジックを含まない。

【ANTICIPATE_RESPONSE について】
[ANTICIPATE_RESPONSE:...] は tool-use 化していない全プロバイダー一律タグ方式である
（`anticipator.py` を参照）。そのため TOOL_TO_TAG には登録しないが、ログ表示メタは
他のタグと一元管理したいので TAG_META には含める。
"""

# ツール名（APIの function_call.name）→ タグ名のマッピング
TOOL_TO_TAG: dict[str, str] = {
    "inscribe_memory": "INSCRIBE_MEMORY",
    "carve_narrative": "CARVE_NARRATIVE",
    "drift":           "DRIFT",
    "drift_reset":     "DRIFT_RESET",
    "switch_angle":    "SWITCH_ANGLE",
    "power_recall":    "POWER_RECALL",
}

# タグ名 → 表示メタ（チャットバブルの▼ログパネルや Logs 画面のバッジで使用）
TAG_META: dict[str, dict[str, str]] = {
    "INSCRIBE_MEMORY":     {"label": "記憶",            "cls": "tag-memory"},
    "CARVE_NARRATIVE":     {"label": "ナラティブ",      "cls": "tag-narrative"},
    "DRIFT":               {"label": "ドリフト",        "cls": "tag-drift"},
    "DRIFT_RESET":         {"label": "ドリフトリセット", "cls": "tag-drift"},
    "SWITCH_ANGLE":        {"label": "アングル切替",    "cls": "tag-switch"},
    "POWER_RECALL":        {"label": "強想起",          "cls": "tag-recall"},
    "END_SESSION":         {"label": "セッション終了",  "cls": "tag-end"},
    "ANTICIPATE_RESPONSE": {"label": "予想",            "cls": "tag-anticipate"},
}


def _bare_tool_name(tool_name: str) -> str:
    """MCP ネームスペースプレフィックス（"mcp__<server>__"）を除去する。

    Claude CLI が出力する stream-json では、ツール名に
    "mcp__<server_name>__<tool_name>" 形式のプレフィックスが付く。
    `__` で区切った末尾部分を返すことで素のツール名に正規化する。
    """
    return tool_name.rsplit("__", 1)[-1] if "__" in tool_name else tool_name


def _make_tag(tag_name: str, fields: dict[str, str], preview: str) -> dict:
    """タグ表示用辞書 {tag_name, meta, fields, preview} を組み立てる共通ヘルパ。

    `meta` は TAG_META からの参照に失敗した場合は tag-unknown にフォールバックする。
    """
    meta = TAG_META.get(tag_name, {"label": tag_name, "cls": "tag-unknown"})
    return {
        "tag_name": tag_name,
        "meta": meta,
        "fields": fields,
        "preview": preview,
    }


def tool_call_to_structured_tag(tool_name: str, args: dict) -> dict:
    """tool-use 形式の (name, args) を表示用構造化辞書に変換する。

    戻り値は ``{tag_name, meta, fields, preview}`` の辞書で、
    `logs_ui._parse_tag_body` がタグ方式（テキスト解析）で返すのと同じ形式。
    タグ方式と tool-use 方式の両方を同じ表示部品（フロントの ToolCallRow）で
    扱えるようにするための共通フォーマット。

    Args:
        tool_name: ツール名（function_call.name フィールド）。
                   MCP プレフィックス付き ("mcp__<server>__<name>") にも対応する。
        args: ツール引数辞書（function_call.args / input フィールド）。

    Returns:
        ``{tag_name, meta, fields, preview}`` の辞書。
    """
    bare = _bare_tool_name(tool_name)
    tag_name = TOOL_TO_TAG.get(bare, bare.upper())

    if tag_name == "INSCRIBE_MEMORY":
        fields = {
            "カテゴリ": str(args.get("category", "")),
            "重要度":   str(args.get("impact", "")),
            "内容":     str(args.get("content", "")),
        }
        return _make_tag(tag_name, fields, fields["内容"])

    if tag_name == "CARVE_NARRATIVE":
        fields = {
            "モード": str(args.get("mode", "append")),
            "内容":   str(args.get("content", "")),
        }
        return _make_tag(tag_name, fields, fields["内容"])

    if tag_name == "DRIFT":
        fields = {"内容": str(args.get("content", ""))}
        return _make_tag(tag_name, fields, fields["内容"])

    if tag_name == "DRIFT_RESET":
        # 引数なしツール。固定マーカーで表示する。
        return _make_tag(tag_name, {"内容": "(リセット)"}, "(リセット)")

    if tag_name == "SWITCH_ANGLE":
        fields = {
            "プリセット":   str(args.get("preset_name", "")),
            "コンテキスト": str(args.get("self_instruction", "")),
        }
        # preview は文脈の方（コンテキスト）を優先、なければプリセット名にフォールバック。
        preview = fields["コンテキスト"] or fields["プリセット"]
        return _make_tag(tag_name, fields, preview)

    if tag_name == "POWER_RECALL":
        fields = {
            "クエリ": str(args.get("query", "")),
            "top_k":  str(args.get("top_k", 5)),
        }
        return _make_tag(tag_name, fields, fields["クエリ"])

    if tag_name == "ANTICIPATE_RESPONSE":
        # ANTICIPATE_RESPONSE はテキストタグだが、ツール実行イベント
        # （tool_call_events、tool_name="anticipate_response"）として記録されるため、
        # タグ方式の表示（logs_ui._parse_tag_body）と同じ fields 形式で変換する。
        fields = {"予想": str(args.get("content", ""))}
        return _make_tag(tag_name, fields, fields["予想"])

    # 未知ツール: 引数の key/value をそのまま fields に並べる
    fields = {k: str(v) for k, v in args.items()}
    preview = " / ".join(f"{k}={v}" for k, v in fields.items())
    return _make_tag(tag_name, fields, preview)
