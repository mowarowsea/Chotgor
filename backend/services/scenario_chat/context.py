"""シナリオチャット用 履歴整形・切り出しユーティリティ。

GM への送信時に、`scenario_turns` 全件のうち直近 N レスポンス（かつ T 文字以内）を
切り出し、`@話者名: 本文` 形式のテキストに整形する。

用語: 「ターン」= @話者: ブロック単位（scenario_turns 1 行）。
      「レスポンス」= 1 回の LLM 呼出で生まれたターン群（`raw_response` が同じ連続行）。
      ユーザ発話・intro（`raw_response` が NULL）は 1 行で 1 レスポンス。
      上限 `history_max_turns` は列名に反してレスポンス数で数える（2026-10-15 変更）。

仕様:
    - 履歴は DB に全件保持し、送信時のみ切り出す
    - 切り出しは「レスポンス数上限」と「文字数上限」の **AND** で決まる
      （先に到達した方が利く）。レスポンスの途中では切らない
    - 文字数上限は GM プリセットの `history_budget_chars`（空なら設定既定）
    - ユーザ発話は `@{user_alias}: ...`
    - Narrator は `@Narrator: ...`
    - NPC（known/unknown 問わず）は `@{speaker_name}: ...`
    - ターン間は空行（\\n\\n）で区切る。`@名前:` は閉じタグを持たず
      「次の行頭 @ まで」が暗黙の終端のため、複数行本文でもターン境界が
      明確になるよう空行で分ける。
    - GM の出力規則・ScenarioChatParser と同じ `@名前: 本文` 規約に揃えることで、
      GM が履歴を見て出力フォーマットを誤学習しないようにする。
"""

from dataclasses import dataclass
from typing import Any, Iterable


# settings から読む既定値（main.py または settings UI で上書き可能）
DEFAULT_HISTORY_MAX_TURNS = 100
DEFAULT_HISTORY_MAX_CHARS = 80000


@dataclass
class TurnView:
    """履歴整形用の発話レコード簡易ビュー。

    SQLAlchemy ORM の ScenarioTurn と同じフィールド名を持つが、テスト時に
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
    settings: dict | None = None,
    gm_preset: Any = None,
) -> tuple[int, int]:
    """履歴切り出し上限 (max_responses, max_chars) を解決する。

    レスポンス数: Scenario.history_max_turns → settings
        "scenario_chat_default_history_max_turns" → DEFAULT_HISTORY_MAX_TURNS
    文字数: GM プリセットの history_budget_chars → settings
        "scenario_chat_default_history_max_chars" → DEFAULT_HISTORY_MAX_CHARS

    文字数をシナリオではなく GM プリセットに持たせるのは、読める量がモデル側の性質だから。
    あらすじ蒸留に別プリセットを使っていても、基準は GM プリセット（あらすじを読むのは GM）。
    """
    settings = settings or {}
    scen_turns = getattr(scenario, "history_max_turns", None)
    preset_chars = getattr(gm_preset, "history_budget_chars", None)

    settings_turns = _coalesce_int(
        settings.get("scenario_chat_default_history_max_turns"),
        DEFAULT_HISTORY_MAX_TURNS,
    )
    settings_chars = _coalesce_int(
        settings.get("scenario_chat_default_history_max_chars"),
        DEFAULT_HISTORY_MAX_CHARS,
    )

    max_turns = _coalesce_int(scen_turns, settings_turns)
    max_chars = _coalesce_int(preset_chars, settings_chars)
    return max_turns, max_chars


def group_responses(turns: list[Any]) -> list[list[Any]]:
    """ターン列をレスポンス単位のグループ列へ束ねる（時系列順を保つ）。

    `raw_response` が同じで連続する行を 1 グループにする（フロントの `response_key` による
    レスポンスグループと同じ単位）。`raw_response` が None / 空の行は 1 行で 1 グループ。
    """
    groups: list[list[Any]] = []
    prev_raw: str | None = None
    for t in turns:
        raw = getattr(t, "raw_response", None) or None
        if groups and raw is not None and raw == prev_raw:
            groups[-1].append(t)
        else:
            groups.append([t])
        prev_raw = raw
    return groups


def count_responses(turns: list[Any]) -> int:
    """ターン列に含まれるレスポンス数を返す。"""
    return len(group_responses(turns))


def slice_history(
    turns: list[Any],
    max_turns: int,
    max_chars: int,
) -> list[Any]:
    """履歴を直近 max_turns レスポンスかつ累計 max_chars 文字以内に切り出す。

    両方の上限を **AND**（先に到達した方が利く）で適用する。切り出しは直近側から
    レスポンス単位で行い、レスポンスの途中では切らない。結果は時系列昇順で返す。
    直近 1 レスポンスだけは max_chars を超えていても入れる（何も渡らない事態を避ける）。
    """
    if max_turns <= 0 or max_chars <= 0:
        return []

    chosen_groups: list[list[Any]] = []
    char_sum = 0
    for group in reversed(group_responses(turns)):
        if len(chosen_groups) >= max_turns:
            break
        body_len = sum(len(getattr(t, "content", "") or "") for t in group)
        if char_sum + body_len > max_chars and chosen_groups:
            break
        chosen_groups.append(group)
        char_sum += body_len
    chosen_groups.reverse()
    return [t for g in chosen_groups for t in g]


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
        turns: ScenarioTurn 風オブジェクトのリスト（時系列昇順を前提）。
        max_turns: 最大レスポンス数（slice_history と同じ意味）。
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
        turn: ScenarioTurn 風オブジェクト（speaker_type / speaker_name / content を持つ）。
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
        turns: 整形済みの ScenarioTurn 風オブジェクト列（時系列昇順）。
        user_alias: ユーザの @タグ用エイリアス。
        narrator_name: Narrator の表示名。

    Returns:
        全ターンを空行区切りで連結したテキスト。空なら空文字列。
    """
    lines = [format_turn_for_gm(t, user_alias, narrator_name) for t in turns]
    return "\n\n".join(lines)
