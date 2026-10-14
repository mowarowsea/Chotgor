"""1 レスポンス全体の手動書き換え（`@話者: 本文` テキスト → 話者ブロック列）。

設計: docs/planned/scenario_turn_variants_plan.md「レスポンス単位の書き換え」。
パースは intro と同じ `parse_intro_to_turns` を流用し、ここでは話者解決の上書きと
拒否判定だけを担う。保存（行の割当・番号の繰り下げ・封筒）は store 側。
"""

from backend.services.scenario_chat.mention import (
    normalize_pc_assignments,
    normalize_pc_slots,
)
from backend.services.scenario_chat.serializers import resolve_user_speaker_name
from backend.services.scenario_chat.turns import parse_intro_to_turns


class ResponseRewriteError(ValueError):
    """書き換え内容が受け付けられない（API では 400 にする）。"""


def rewrite_scenario_response(sqlite, session_id: str, turn_id: str, text: str):
    """`turn_id` が属する 1 レスポンスを `text` で書き換える。

    話者解決: 書き換え前のレスポンスに居た話者名はその行の型・ID を引き継ぐ
    （PC 応答の `pc` 型を保つため）。それ以外は Narrator / 既知 NPC / ephemeral NPC。
    ユーザ PC 名と、このレスポンスに居ない PC 枠名のブロックは拒否する。

    Returns:
        store の `rewrite_scenario_response` の戻り値。対象が無ければ None。

    Raises:
        ResponseRewriteError: ブロックが空・ユーザ／PC の発話が混ざっている。
    """
    group = sqlite.get_scenario_response_group(session_id, turn_id)
    if not group:
        return None
    session = sqlite.get_scenario_session(session_id)
    scenario = sqlite.get_scenario(session.scenario_id) if session else None
    npcs = sqlite.list_scenario_npcs(scenario.id) if scenario else []
    known = {n.name: n.id for n in npcs if getattr(n, "name", None)}
    user_name = resolve_user_speaker_name(scenario, session, sqlite)

    blocks = parse_intro_to_turns(
        intro_text=text,
        user_alias=user_name,
        known_npc_names=known,
    )
    if not blocks:
        raise ResponseRewriteError("本文が空です（レスポンスを消すには破棄を使ってください）")

    existing = {t.speaker_name: t for t in group}
    pc_names: set[str] = set()
    if scenario is not None:
        pc_slots = normalize_pc_slots(getattr(scenario, "pc_slots", None))
        for pc in normalize_pc_assignments(
            getattr(session, "pc_assignments", None), pc_slots, sqlite,
        ):
            pc_names.add(pc.name)
            if pc.character_name:
                pc_names.add(pc.character_name)

    for b in blocks:
        prev = existing.get(b["speaker_name"])
        if prev is not None:
            b["speaker_type"] = prev.speaker_type
            b["speaker_id"] = prev.speaker_id
            continue
        if b["speaker_type"] == "user" or b["speaker_name"] in pc_names:
            raise ResponseRewriteError(
                f"@{b['speaker_name']}: はプレイヤーの発話なので、このレスポンスには書けません"
            )
    return sqlite.rewrite_scenario_response(session_id, turn_id, blocks)
