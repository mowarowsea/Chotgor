"""シナリオチャット SQLite テスト群の共有ヘルパー。

scenarios / scenario_npcs / scenario_sessions / scenario_turns の
テストレコードを組み立てる関数を提供する。
test_scenario_sqlite_*.py の各分割ファイルから共通利用する。
ファイル名先頭がアンダースコアのため pytest のテスト収集対象にはならない。
"""

import uuid


def _make_scenario(
    store,
    title: str = "テストシナリオ",
    user_alias: str = "プレイヤー",
    **kwargs,
):
    """シナリオテンプレートを 1 件作成して返すユーティリティ。

    GM プリセットはセッション単位（ScenarioSession.gm_preset_id）の設定になったため、
    テンプレ作成では受け付けない。
    """
    scenario_id = str(uuid.uuid4())
    # 旧 user_alias は廃止。ユーザPCを pc_slots の 1 枠（先頭）として表現する。
    if "pc_slots" not in kwargs and user_alias:
        kwargs["pc_slots"] = [
            {"slot_id": "user", "name": user_alias, "description": ""}
        ]
    return store.create_scenario(
        scenario_id=scenario_id,
        title=title,
        **kwargs,
    )


def _make_npc(store, scenario_id: str, name: str = "レイカ", **kwargs):
    """シナリオ内 NPC を 1 件作成して返すユーティリティ。"""
    npc_id = str(uuid.uuid4())
    return store.create_scenario_npc(
        npc_id=npc_id,
        scenario_id=scenario_id,
        name=name,
        **kwargs,
    )


def _make_session(
    store,
    scenario_id: str,
    title: str = "プレイ #1",
    engine_type: str = "ensemble",
    gm_preset_id: str = "preset-test",
    synopsis_preset_id: str = None,
):
    """プレイセッションを 1 件作成して返すユーティリティ。

    gm_preset_id / synopsis_preset_id はセッション必須項目。
    synopsis_preset_id 省略時は gm_preset_id と同値で作成する（従来挙動相当）。
    """
    session_id = str(uuid.uuid4())
    if synopsis_preset_id is None:
        synopsis_preset_id = gm_preset_id
    return store.create_scenario_session(
        session_id=session_id,
        scenario_id=scenario_id,
        title=title,
        gm_preset_id=gm_preset_id,
        synopsis_preset_id=synopsis_preset_id,
        engine_type=engine_type,
    )


def _make_turn(
    store,
    session_id: str,
    speaker_type: str = "user",
    speaker_name: str = "プレイヤー",
    content: str = "こんにちは",
    speaker_id: str = None,
    raw_response: str = None,
    turn_index: int = None,
):
    """発話ターンを 1 件作成して返すユーティリティ。"""
    turn_id = str(uuid.uuid4())
    if turn_index is None:
        turn_index = store.get_next_scenario_turn_index(session_id)
    return store.create_scenario_turn(
        turn_id=turn_id,
        session_id=session_id,
        turn_index=turn_index,
        speaker_type=speaker_type,
        speaker_name=speaker_name,
        content=content,
        speaker_id=speaker_id,
        raw_response=raw_response,
    )
