"""Claude CLI stream-json（NDJSON）テキストのイベント走査ヘルパー。

CLI の生 stdout は1行1イベントの NDJSON だが、debug_logger が JSON 文字列値内の
\\n を実際の改行に展開するため、ログ経由のテキストではイベントが複数行に
分断されることがある。本モジュールはどちらの形でも dict イベントを取り出せる
積み上げパーサを提供する（logs_ui のタグ抽出と claude_cli_provider の
使用量抽出で共用）。
"""

import json
from collections.abc import Iterator


def _try_parse_dict(text: str) -> dict | None:
    """text を JSON としてパースし、dict が得られればそれを、それ以外は None を返す。

    Args:
        text: パース対象のテキスト。

    Returns:
        パース済み dict。パース失敗・トップレベルが dict 以外の場合は None。
    """
    try:
        parsed = json.loads(text, strict=False)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def iter_stream_json_events(text: str) -> Iterator[dict]:
    """stream-json テキストから JSON イベント（dict）を順に取り出す。

    行単体で完結するイベントはそのまま、複数行に分断されたイベントは行を
    積み上げて完全な JSON が形成された時点で解釈する。非 JSON 行（CLI の
    警告テキスト等）が積み上げに混ざると以降のイベントを全て呑み込んで
    しまうため、積み上げのパースに失敗した場合は現在行単体でも試し、
    dict が得られたら積み上げ分をガベージとして捨てて行単体の方を採用する。

    Args:
        text: stream-json形式のテキスト（1行1JSONイベント、複数行またがりあり）。

    Yields:
        パースできた dict イベント（dict 以外のトップレベル JSON は捨てる）。
    """
    accumulator: list[str] = []
    for line in text.splitlines():
        accumulator.append(line)
        candidate = "\n".join(accumulator).strip()
        if not candidate:
            accumulator = []
            continue
        event = _try_parse_dict(candidate)
        if event is None and len(accumulator) > 1:
            event = _try_parse_dict(line.strip())
        if event is None:
            continue
        accumulator = []
        yield event
