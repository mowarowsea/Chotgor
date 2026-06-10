"""ログファイルからのタグ／ツール呼び出し抽出。

debug_logger が生成した JSON / stream-json / テキストタグ形式のログを解析し、
ログ閲覧UIで表示する構造化タグ辞書を組み立てる。
"""

import json
import re
from pathlib import Path

from backend.character_actions import tool_tags
from backend.character_actions.anticipator import ANTICIPATE_RESPONSE_TAG_NAME
from backend.lib.stream_json import iter_stream_json_events
from backend.lib.tag_parser import parse_tags

# タグ方式フォールバックで認識する名前一覧（MCP/関数呼び出しから抽出されるタグ）。
# ANTICIPATE_RESPONSE は tool-use 化していない全プロバイダー一律のテキストタグのため
# ここには含めず、_last_anticipation_tag() で常に独立して抽出する
# （anticipator.py の設計意図に従う）。
_KNOWN_TAG_NAMES = list(tool_tags.TOOL_TO_TAG.values())

# debug_logger._unescape_text() が JSON 文字列値内の \\n を実際の改行に展開するため、
# バックスラッシュ直後の改行（\<LF>）が不正 JSON エスケープになる場合がある。
# このパターンで前処理し json.loads(strict=False) に渡す。
_BACKSLASH_NEWLINE_RE = re.compile(r"\\\n")


def _parse_json_safely(text: str) -> dict | list:
    """debug_logger が生成した JSON ファイルを安全にパースして返す。

    debug_logger._unescape_text() は JSON 文字列値内の \\n を実際の改行に展開するため、
    バックスラッシュ直後の改行（\\<LF>）が不正エスケープになる場合がある。
    パース前にこのパターンを JSON 改行エスケープ \\n に戻し、
    残る制御文字は strict=False で許容する。
    通常の JSON パースに失敗した場合は NDJSON（各行が独立した JSON オブジェクト、
    ただし複数行またがりあり）として解析して list を返す。
    どちらも失敗した場合は ValueError を送出する。

    Args:
        text: ログファイルのテキスト内容。

    Returns:
        パース結果の dict または list。NDJSON の場合はオブジェクトの list。

    Raises:
        ValueError: JSON・NDJSON どちらのパースにも失敗した場合。
    """
    text_safe = _BACKSLASH_NEWLINE_RE.sub(r"\\n", text)
    try:
        return json.loads(text_safe, strict=False)
    except (json.JSONDecodeError, ValueError):
        pass

    # NDJSON として解析する（LLM出力の生改行で複数行にまたがる場合も対応）
    objects: list = []
    accumulator: list[str] = []
    for line in text_safe.splitlines():
        accumulator.append(line)
        candidate = "\n".join(accumulator)
        try:
            obj = json.loads(candidate, strict=False)
            objects.append(obj)
            accumulator = []
        except (json.JSONDecodeError, ValueError):
            pass
    if objects:
        return objects
    raise ValueError("JSON/NDJSON parse failed")


def _parse_tag_body(tag_name: str, body: str) -> dict:
    """タグ本体文字列（タグ方式プロバイダーのテキストタグ）を構造化辞書に変換する。

    タグ種別ごとに `|` 区切りのフィールドを解釈し、表示用のサブフィールドを返す。
    tool-use 形式の表示には `tool_tags.tool_call_to_structured_tag()` を使う
    （こちらは引数 dict を直接受けるため文字列分割を経由しない）。

    Args:
        tag_name: タグ名 (例: "INSCRIBE_MEMORY")。
        body: タグ本体テキスト (コロン以降 ']' を除いた部分)。

    Returns:
        表示用フィールド辞書。tag_name / meta / fields / preview を含む。
    """
    meta = tool_tags.TAG_META.get(tag_name, {"label": tag_name, "cls": "tag-unknown"})
    fields: dict[str, str] = {}

    if tag_name == "INSCRIBE_MEMORY":
        # body: "category|importance|content"
        parts = body.split("|", 2)
        fields["カテゴリ"] = parts[0] if len(parts) > 0 else ""
        fields["重要度"] = parts[1] if len(parts) > 1 else ""
        fields["内容"] = parts[2] if len(parts) > 2 else ""

    elif tag_name == "CARVE_NARRATIVE":
        # body: "mode|content"
        parts = body.split("|", 1)
        fields["モード"] = parts[0] if len(parts) > 0 else ""
        fields["内容"] = parts[1] if len(parts) > 1 else ""

    elif tag_name == "SWITCH_ANGLE":
        # body: "preset|context"
        parts = body.split("|", 1)
        fields["プリセット"] = parts[0] if len(parts) > 0 else ""
        fields["コンテキスト"] = parts[1] if len(parts) > 1 else ""

    elif tag_name == "ANTICIPATE_RESPONSE":
        # body 全体がキャラクター本人の予想・期待テキスト
        fields["予想"] = body

    elif tag_name in ("DRIFT", "POWER_RECALL", "END_SESSION"):
        # body がそのまま内容
        fields["内容"] = body

    elif tag_name in ("DRIFT_RESET",):
        # 固定マーカー、body は空
        fields["内容"] = "(リセット)"

    # preview: "内容" / "予想" → SWITCH_ANGLE は "コンテキスト" → 最終フォールバックは body
    preview = (
        fields.get("内容")
        or fields.get("予想")
        or fields.get("コンテキスト")
        or body
    )

    return {
        "tag_name": tag_name,
        "meta": meta,
        "fields": fields,
        "preview": preview,
    }


def _collect_tool_calls_from_single_json(data: dict) -> list[tuple[str, dict]]:
    """単一JSONオブジェクト（Anthropic/OpenAI/Gemini形式）からツール呼び出しを収集する。

    Args:
        data: パース済みのJSONオブジェクト。

    Returns:
        (tool_name, args_dict) のリスト。
    """
    if not isinstance(data, dict):
        return []
    calls: list[tuple[str, dict]] = []

    # Gemini形式: candidates[].content.parts[].function_call
    # 注: promptFeedback でブロックされた場合など candidates が null で返ることがあるため
    # `get(..., [])` のデフォルト値は効かず明示的に `or []` でフォールバックする。
    for candidate in (data.get("candidates") or []):
        for part in ((candidate.get("content") or {}).get("parts") or []):
            fc = part.get("function_call")
            if fc and isinstance(fc, dict) and "name" in fc:
                calls.append((fc["name"], fc.get("args", {}) or {}))

    # Anthropic形式: content[].type == "tool_use"（input キー）
    content = data.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append((block.get("name", ""), block.get("input", {}) or {}))

    # OpenAI形式: choices[].message.tool_calls[].function（arguments は JSON 文字列）
    for choice in (data.get("choices") or []):
        for tc in ((choice.get("message") or {}).get("tool_calls") or []):
            fc = tc.get("function", {}) or {}
            try:
                args = json.loads(fc.get("arguments", "{}"))
            except (json.JSONDecodeError, ValueError):
                args = {}
            calls.append((fc.get("name", ""), args))

    return calls


def _collect_tool_calls_from_cli_event(event: dict) -> list[tuple[str, dict]]:
    """Claude CLI stream-json の1イベントからtool_useブロックを収集する。

    tool_use ブロックは type=assistant イベントの message.content 配列内に含まれる。

    Args:
        event: パース済みのイベント辞書。

    Returns:
        (tool_name, args_dict) のリスト。
    """
    if not isinstance(event, dict) or event.get("type") != "assistant":
        return []
    calls: list[tuple[str, dict]] = []
    for block in ((event.get("message") or {}).get("content") or []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            calls.append((block.get("name", ""), block.get("input", {}) or {}))
    return calls


def _collect_tool_calls_from_stream_json(text: str) -> list[tuple[str, dict]]:
    """stream-json（NDJSON）形式のClaude CLI出力からtool_useブロックを収集する。

    Claude CLI が --output-format stream-json --verbose で出力する形式では、
    各行が独立したJSONイベントになる。ただし debug_logger が文字列値内の \\n を
    実際の改行に展開するため、tool_use の input に改行を含むイベントは複数行に
    分断される。複数行またがり・非 JSON 行混在の解釈は
    iter_stream_json_events（backend/lib/stream_json.py）に委ねる。

    Args:
        text: stream-json形式のテキスト（1行1JSONイベント、複数行またがりあり）。

    Returns:
        (tool_name, args_dict) のリスト。
    """
    calls: list[tuple[str, dict]] = []
    for event in iter_stream_json_events(text):
        calls.extend(_collect_tool_calls_from_cli_event(event))
    return calls


def _extract_function_calls_from_json(text: str) -> list[dict]:
    """JSONレスポンスファイルからfunction_call（ツール呼び出し）をタグとして抽出する。

    Gemini、Anthropic、OpenAI等のAPIレスポンスJSON内の function_call / tool_use エントリを
    検出し、タグ構造体に変換して返す。
    Claude CLI が出力する stream-json（NDJSON）形式にも対応する。

    対応フォーマット:
        - Gemini: candidates[].content.parts[].function_call
        - Anthropic: content[].type == "tool_use"（input キー）
        - OpenAI: choices[].message.tool_calls[].function（arguments は JSON 文字列）
        - Claude CLI stream-json: assistant イベントの message.content[].type == "tool_use"

    Args:
        text: ログファイルのテキスト内容。

    Returns:
        タグ構造体のリスト。該当がなければ空リスト。
    """
    function_calls: list[tuple[str, dict]] = []
    try:
        data = _parse_json_safely(text)
        function_calls = _collect_tool_calls_from_single_json(data)
    except Exception:
        pass

    # シングルJSONでツール呼び出しが見つからなかった場合は stream-json（NDJSON）として試みる。
    # 単一行 NDJSON が正常にパースできても Claude CLI イベント形式の場合はここで抽出する。
    if not function_calls:
        try:
            function_calls = _collect_tool_calls_from_stream_json(text)
        except Exception:
            pass

    results = []
    for name, args in function_calls:
        if not name:
            continue
        # 構造化辞書を直接組み立てる（文字列 body を経由しない）。
        # タグ方式（テキストタグ）のフォールバック経路は引き続き _parse_tag_body を使う。
        results.append(tool_tags.tool_call_to_structured_tag(name, args))
    return results


def _last_anticipation_tag(text: str) -> list[dict]:
    """テキストから [ANTICIPATE_RESPONSE:...] を抽出し、最後の1件だけ構造化して返す。

    実行時の抽出（anticipator.extract_anticipation）と同じく「最後のタグを採用」する。
    Claude CLI の stream-json では同一テキストが assistant text と result に重複して
    現れるうえ、ツールエラー後の書き直しで**文面の異なる**タグが複数残ることもある。
    実際に保存・次ターン注入されるのは最後の1件だけなので、ログ表示もそれに合わせる。

    Args:
        text: ログファイルのテキスト内容。

    Returns:
        構造化タグ辞書のリスト（0件 or 1件）。
    """
    try:
        _, matches = parse_tags(text, [ANTICIPATE_RESPONSE_TAG_NAME], multiline=True)
    except Exception:
        return []
    found = matches.get(ANTICIPATE_RESPONSE_TAG_NAME, [])
    if not found:
        return []
    return [_parse_tag_body(ANTICIPATE_RESPONSE_TAG_NAME, found[-1].body)]


def _extract_tags_from_file(file_path: Path) -> list[dict]:
    """ファイルからタグ / ツール呼び出しを抽出して構造化リストで返す。

    抽出の優先順位:
      1. JSON / stream-json 内のツール呼び出し（tool-use対応プロバイダー）
      2. テキスト内の [INSCRIBE_MEMORY:...] 等のタグ（タグ方式プロバイダーへのフォールバック）

    tool-use プロバイダー（Anthropic / OpenAI / Claude CLI 等）はレスポンスに
    tool_use ブロックを含むため、それを優先表示する。
    タグ方式プロバイダー（Ollama / OpenRouter 等）はツール呼び出しが存在しないため、
    テキスト内のタグをフォールバックとして抽出する。

    ANTICIPATE_RESPONSE は tool-use 化していない全プロバイダー一律のテキストタグのため、
    どちらの経路でも本文から独立して抽出して合流させる（ツール呼び出しが見つかった
    場合に予想タグが表示されなくなる取りこぼしを防ぐ）。

    Args:
        file_path: ログファイルのパス。

    Returns:
        タグ構造化辞書のリスト。
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []

    # 予想タグは経路によらず本文から抽出する（最後の1件のみ）
    anticipation_tags = _last_anticipation_tag(text)

    # ツール呼び出し（JSON / stream-json）を優先して試みる
    tool_call_tags = _extract_function_calls_from_json(text)
    if tool_call_tags:
        return _dedupe_tags(tool_call_tags + anticipation_tags)

    # ツール呼び出しが見つからなければタグ方式（Ollama等）としてフォールバック
    try:
        _, matches = parse_tags(text, _KNOWN_TAG_NAMES, multiline=True)
    except Exception:
        return anticipation_tags

    # 全タグを (start位置, tag_name, body) フラットリストにしてから位置順にソートする。
    # タグ種別ごとに固める（INSCRIBE→DRIFT→...）のではなく、
    # ファイル内の出現順（DRIFT→INSCRIBE のような順序）を保持するため。
    flat: list[tuple[int, str, str]] = []
    for tag_name in _KNOWN_TAG_NAMES:
        for m in matches.get(tag_name, []):
            flat.append((m.start, tag_name, m.body))
    flat.sort(key=lambda x: x[0])

    # 予想タグは応答末尾に書かれる運用のため、リスト末尾に合流させる
    return _dedupe_tags([_parse_tag_body(tn, body) for _, tn, body in flat] + anticipation_tags)


def _dedupe_tags(tags: list[dict]) -> list[dict]:
    """構造化タグリストから完全一致する重複を除去する（出現順は保持）。

    Claude CLI の stream-json は type=assistant の text ブロックと type=result の
    result フィールドに同一テキストを含むため、生ログ全体をテキストとしてスキャンする
    `parse_tags` 経路では同一の ANTICIPATE_RESPONSE 等が2回検出されてしまう。
    本関数は同じ tag_name / fields の組み合わせを1件にまとめ、UI上の二重表示を防ぐ。

    フィールド辞書ごと完全一致したものだけを潰すため、別引数で同じツールを2回
    呼んだ場合のように内容が異なる重複は両方残る。
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    for tag in tags:
        key = json.dumps(tag, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tag)
    return deduped
