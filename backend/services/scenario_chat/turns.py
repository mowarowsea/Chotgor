"""ターン保存ヘルパと導入文（intro）のターン展開。

_save_turn はシナリオの全ターン保存（ユーザ・GM・PC・intro）の共通経路。
parse_intro_to_turns / seed_intro_turns はシナリオ開始時の導入文を
発話ターン列へ変換して保存する。
"""

import uuid

from backend.lib.log_context import current_message_id
from backend.services.scenario_chat.serializers import resolve_user_speaker_name


def _save_turn(
    sqlite,
    session_id: str,
    speaker_type: str,
    speaker_name: str,
    content: str,
    speaker_id: str | None = None,
    raw_response: str | None = None,
    attach_log_request_id: bool = False,
    anticipation: str | None = None,
):
    """ターンを次の turn_index で保存して返す共通ヘルパ。

    attach_log_request_id=True のとき、現在の current_message_id を log_request_id として保存する。
    GM ターン保存時のみ True にする（ユーザーターン・intro はログとの紐付け不要）。
    anticipation は GM がターン末尾に書いた予想（期待）。ターンに1つなので、最後の発話行にのみ渡す。
    """
    turn_id = str(uuid.uuid4())
    next_index = sqlite.get_next_scenario_turn_index(session_id)
    log_req_id = current_message_id.get() if attach_log_request_id else None
    saved = sqlite.create_scenario_turn(
        turn_id=turn_id,
        session_id=session_id,
        turn_index=next_index,
        speaker_type=speaker_type,
        speaker_name=speaker_name,
        content=content,
        speaker_id=speaker_id,
        raw_response=raw_response,
        log_request_id=log_req_id,
        anticipation=anticipation,
    )
    # 計器 Tier 2: LLM 産の発話（GM/NPC/Narrator/キャラPC）を外形スキャンする。
    # ユーザ発話・intro（人間の手書き）は対象外。誤検知許容の smell 記録であり、
    # 失敗しても保存処理は止めない（record_response_smells 内で握り潰す）。
    if speaker_type != "user" and raw_response is not None:
        from backend.services.instruments.tier2 import record_response_smells
        record_response_smells(
            sqlite, content, character_name=speaker_name, feature="scenario",
        )
    return saved
def parse_intro_to_turns(
    intro_text: str,
    user_alias: str,
    known_npc_names: dict,
    narrator_name: str = "Narrator",
) -> list[dict]:
    """導入部テキストを `@キャラ: 本文` 記法でパースしてターン辞書のリストを返す。

    GM 出力パーサ（ScenarioChatParser）と違い、`@user_alias:` ブロックも捨てずに
    user 発話として保存する。`@narrator:` は narrator 発話、既知 NPC 名なら npc、
    それ以外は ephemeral NPC として扱う。
    `@` で始まらない冒頭の地の文は Narrator に吸収する。

    Args:
        intro_text: 導入部の生テキスト（複数行可。`@名前:` ブロックを順に並べる）。
        user_alias: ユーザ表示名（@タグで一致する場合 user として扱う）。
        known_npc_names: {NPC名: NPC.id} の辞書。
        narrator_name: ナレーター表示名（@narrator は大小無視）。

    Returns:
        [{speaker_type, speaker_id, speaker_name, content}, ...] のリスト。
        本文が空のブロックはスキップする。
    """
    if not intro_text or not intro_text.strip():
        return []

    def resolve_speaker(raw_name: str) -> tuple[str, str | None, str]:
        name = (raw_name or "").strip()
        if not name:
            return ("narrator", None, narrator_name)
        if name.lower() == user_alias.lower():
            return ("user", None, user_alias)
        if name.lower() == narrator_name.lower():
            return ("narrator", None, narrator_name)
        if name in known_npc_names:
            return ("npc", known_npc_names[name], name)
        return ("npc", None, name)

    blocks: list[dict] = []
    cur_type: str = "narrator"
    cur_id: str | None = None
    cur_name: str = narrator_name
    cur_buffer: list[str] = []

    def flush_block():
        body = "".join(cur_buffer).rstrip()
        if body:
            blocks.append({
                "speaker_type": cur_type,
                "speaker_id": cur_id,
                "speaker_name": cur_name,
                "content": body,
            })

    for raw_line in intro_text.splitlines():
        line = raw_line
        if line.startswith("@"):
            colon = line.find(":", 1)
            if colon > 1:
                flush_block()
                cur_buffer = []
                cur_type, cur_id, cur_name = resolve_speaker(line[1:colon])
                rest = line[colon + 1 :]
                if rest.startswith(" "):
                    rest = rest[1:]
                if rest:
                    cur_buffer.append(rest + "\n")
                continue
        cur_buffer.append(line + "\n")

    flush_block()
    return blocks


def seed_intro_turns(sqlite, session_id: str, scenario) -> int:
    """シナリオ設定の intro をパースして当該セッションの先頭ターンとして保存する。

    `start_session` 直後に呼ぶ想定。すでに intro 由来のターンが存在しないか
    呼出側で保証すること（重複防止）。

    Args:
        sqlite: SQLiteStore インスタンス。
        session_id: 対象セッション ID。
        scenario: Scenario ORM。intro を持つ。

    Returns:
        実際に保存したターン数。
    """
    intro_text = getattr(scenario, "intro", None)
    if not intro_text or not intro_text.strip():
        return 0
    npcs = sqlite.list_scenario_npcs(scenario.id)
    known = {n.name: n.id for n in npcs if getattr(n, "name", None)}
    # ユーザPC名は user 割当スロットから解決する（旧 user_alias 廃止）。
    session = sqlite.get_scenario_session(session_id)
    user_speaker_name = resolve_user_speaker_name(scenario, session, sqlite)
    blocks = parse_intro_to_turns(
        intro_text=intro_text,
        user_alias=user_speaker_name,
        known_npc_names=known,
    )
    saved = 0
    for b in blocks:
        _save_turn(
            sqlite=sqlite,
            session_id=session_id,
            speaker_type=b["speaker_type"],
            speaker_name=b["speaker_name"],
            content=b["content"],
            speaker_id=b["speaker_id"],
        )
        saved += 1
    return saved


