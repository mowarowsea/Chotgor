"""シナリオチャット用 履歴整形・切り出しユーティリティ。

GM への送信時に、`zeta_turns` 全件のうち直近 N ターン（または T 文字以内）を
切り出し、`@話者名: 本文` 形式のテキストに整形する。

仕様:
    - 履歴は DB に全件保持し、送信時のみ切り出す
    - 切り出しは「ターン数上限」と「文字数上限」の **AND** で決まる
      （先に到達した方が利く）
    - ユーザ発話は `@{user_alias}: ...`
    - Narrator は `@Narrator: ...`
    - NPC（known/unknown 問わず）は `@{speaker_name}: ...`
    - ターン間は空行（\\n\\n）で区切る。`@名前:` は閉じタグを持たず
      「次の行頭 @ まで」が暗黙の終端のため、複数行本文でもターン境界が
      明確になるよう空行で分ける。
    - GM の出力規則・ScenarioChatParser と同じ `@名前: 本文` 規約に揃えることで、
      GM が履歴を見て出力フォーマットを誤学習しないようにする
      （旧 `<話者>本文</話者>` SGML 形式から移行）。
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional


# settings から読む既定値（main.py または settings UI で上書き可能）
DEFAULT_HISTORY_MAX_TURNS = 100
DEFAULT_HISTORY_MAX_CHARS = 80000


@dataclass
class TurnView:
    """履歴整形用の発話レコード簡易ビュー。

    SQLAlchemy ORM の ZetaTurn と同じフィールド名を持つが、テスト時に
    ダミーオブジェクトを差し込みやすいよう dataclass にしている。
    """

    speaker_type: str
    speaker_name: str
    content: str


def _coalesce_int(value: Any, fallback: int) -> int:
    """None・空文字列・int 変換できない値は fallback を返す。"""
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def resolve_history_limits(
    scenario: Any,
    settings: Optional[dict] = None,
) -> tuple[int, int]:
    """シナリオごとの履歴切り出し上限を解決する。

    優先順:
        1. ZetaScenario.history_max_turns / history_max_chars（NULL でなければ）
        2. settings の "scenario_chat_default_history_max_turns" / "_max_chars"
        3. ハードコード既定値 (DEFAULT_HISTORY_MAX_TURNS / DEFAULT_HISTORY_MAX_CHARS)

    Args:
        scenario: ZetaScenario ORM（もしくは同等フィールドを持つオブジェクト）。
        settings: グローバル設定辞書（None なら使わない）。

    Returns:
        (max_turns, max_chars) のタプル。
    """
    settings = settings or {}
    scen_turns = getattr(scenario, "history_max_turns", None)
    scen_chars = getattr(scenario, "history_max_chars", None)

    settings_turns = _coalesce_int(
        settings.get("scenario_chat_default_history_max_turns"),
        DEFAULT_HISTORY_MAX_TURNS,
    )
    settings_chars = _coalesce_int(
        settings.get("scenario_chat_default_history_max_chars"),
        DEFAULT_HISTORY_MAX_CHARS,
    )

    max_turns = _coalesce_int(scen_turns, settings_turns)
    max_chars = _coalesce_int(scen_chars, settings_chars)
    return max_turns, max_chars


def slice_history(
    turns: list[Any],
    max_turns: int,
    max_chars: int,
) -> list[Any]:
    """履歴を直近 max_turns ターンかつ累計 max_chars 文字以内に切り出す。

    両方の上限を **AND**（先に到達した方が利く）で適用する。
    切り出しは直近側から行い、結果は時系列昇順で返す。

    Args:
        turns: ZetaTurn 風オブジェクトのリスト（時系列昇順を前提）。
        max_turns: 最大ターン数。0 以下なら 0 件。
        max_chars: 最大文字数（content の長さの合計で計測）。0 以下なら 0 件。

    Returns:
        切り出し後のリスト（時系列昇順）。
    """
    if max_turns <= 0 or max_chars <= 0:
        return []

    # 直近側から逆順に詰めていく
    chosen: list[Any] = []
    char_sum = 0
    for t in reversed(turns):
        body = getattr(t, "content", "") or ""
        if len(chosen) >= max_turns:
            break
        if char_sum + len(body) > max_chars and chosen:
            # 1 件目だけは max_chars 超過でも入れる選択肢もあるが、
            # ここは厳密に上限を守る方針（先に入れた直近側を優先）。
            break
        chosen.append(t)
        char_sum += len(body)
    chosen.reverse()
    return chosen


def dropped_history(
    turns: list[Any],
    max_turns: int,
    max_chars: int,
) -> list[Any]:
    """`slice_history` で **送信対象から外れる** 古いターン側を返す。

    あらすじ自動更新フローで「これから LLM に渡らなくなるターン」を
    集約するための補助関数。`slice_history` が直近側を残すのに対し、
    こちらはその外側（古い側）を昇順で返す。

    Args:
        turns: ZetaTurn 風オブジェクトのリスト（時系列昇順を前提）。
        max_turns: 最大ターン数（slice_history と同じ意味）。
        max_chars: 最大文字数（slice_history と同じ意味）。

    Returns:
        送信対象に**含まれない**側のターン列（時系列昇順）。
    """
    kept = slice_history(turns, max_turns, max_chars)
    drop_count = len(turns) - len(kept)
    if drop_count <= 0:
        return []
    return list(turns[:drop_count])


def format_turn_for_gm(
    turn: Any,
    user_alias: str,
    narrator_name: str = "Narrator",
) -> str:
    """1 件の発話を `@話者名: 本文` 形式のテキストにする。

    GM の出力規則・ScenarioChatParser と同じ `@名前: 本文` 規約に揃えてある。

    Args:
        turn: ZetaTurn 風オブジェクト（speaker_type / speaker_name / content を持つ）。
        user_alias: ユーザの @タグ用エイリアス。
        narrator_name: Narrator の表示名。

    Returns:
        整形済み 1 ブロックのテキスト（末尾改行なし）。
    """
    speaker_type = getattr(turn, "speaker_type", "")
    content = getattr(turn, "content", "") or ""

    if speaker_type == "user":
        name = user_alias
    elif speaker_type == "narrator":
        name = narrator_name
    else:
        # npc / character / その他: speaker_name をそのまま使う
        name = getattr(turn, "speaker_name", "") or narrator_name

    # markdown 等の整形は加えない。本文はそのまま（複数行可）。
    return f"@{name}:\n{content}"


def format_history_for_gm(
    turns: Iterable[Any],
    user_alias: str,
    narrator_name: str = "Narrator",
) -> str:
    """履歴全体を GM へ渡す `@話者: 本文` 連結テキストに整形する。

    各ターンは空行（\\n\\n）で区切る。`@名前:` は閉じタグを持たないため、
    複数行本文でもターン境界が明確になるよう空行で分ける。

    Args:
        turns: 整形済みの ZetaTurn 風オブジェクト列（時系列昇順）。
        user_alias: ユーザの @タグ用エイリアス。
        narrator_name: Narrator の表示名。

    Returns:
        全ターンを空行区切りで連結したテキスト。空なら空文字列。
    """
    lines = [format_turn_for_gm(t, user_alias, narrator_name) for t in turns]
    return "\n\n".join(lines)
