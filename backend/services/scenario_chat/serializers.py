"""シナリオ関連 ORM → dict 変換とユーザ話者名解決。

API レスポンス整形に使う軽量ヘルパ群。LLM 呼び出しや DB 書き込みは行わない。
"""

from typing import Any


def scenario_to_dict(scenario: Any) -> dict:
    """Scenario ORM を JSON 化可能な dict に変換する。"""
    if scenario is None:
        return {}
    return {
        "id": scenario.id,
        "title": scenario.title,
        "scenario": scenario.scenario,
        "intro": scenario.intro,
        "history_max_turns": scenario.history_max_turns,
        "history_max_chars": scenario.history_max_chars,
        "custom_system_prompt": scenario.custom_system_prompt,
        "dice_pool_spec": getattr(scenario, "dice_pool_spec", None),
        "pc_slots": getattr(scenario, "pc_slots", None) or [],
        "banner_data": getattr(scenario, "banner_data", None),
        "created_at": scenario.created_at.isoformat() if scenario.created_at else None,
        "updated_at": scenario.updated_at.isoformat() if scenario.updated_at else None,
    }


def resolve_user_speaker_name(
    scenario: Any, session: Any, sqlite, default: str = "プレイヤー"
) -> str:
    """ユーザPCの @タグ名を pc_slots + pc_assignments から解決する。

    旧 `scenarios.user_alias` 廃止後の統一解決ロジック。engine_type に依存せず、
    セッションの pc_assignments で player_type="user" となっているスロットの name を返す。
    user 枠が見つからない異常時は default を返す。

    Args:
        scenario: Scenario ORM（pc_slots を参照）。
        session: ScenarioSession ORM（pc_assignments を参照）。
        sqlite: SQLiteStore（character 名 lookup 用に normalize へ渡す）。
        default: user 枠が無い場合のフォールバック名。

    Returns:
        ユーザPCの表示名。
    """
    from backend.services.scenario_chat.mention import (
        normalize_pc_assignments,
        normalize_pc_slots,
    )

    pc_slots = normalize_pc_slots(getattr(scenario, "pc_slots", None))
    pcs = normalize_pc_assignments(
        getattr(session, "pc_assignments", None), pc_slots, sqlite,
    )
    user_pc = next((p for p in pcs if p.is_user), None)
    return user_pc.name if user_pc else default


def scenario_session_to_dict(session: Any) -> dict:
    """ScenarioSession ORM（プレイインスタンス）を JSON 化可能な dict に変換する。

    `gm_preset_id` は同一シナリオから複数セッションを起動した際にそれぞれ別の
    GM モデルで遊ぶための、セッション固有の LLM プリセット ID。
    """
    if session is None:
        return {}
    return {
        "id": session.id,
        "scenario_id": session.scenario_id,
        "title": session.title,
        "engine_type": session.engine_type,
        "status": session.status,
        "gm_preset_id": getattr(session, "gm_preset_id", "") or "",
        "synopsis_preset_id": (
            getattr(session, "synopsis_preset_id", "")
            or getattr(session, "gm_preset_id", "")
            or ""
        ),
        "synopsis_auto": getattr(session, "synopsis_auto", "") or "",
        "synopsis_manual": getattr(session, "synopsis_manual", "") or "",
        "synopsis_last_turn_index": int(getattr(session, "synopsis_last_turn_index", -1) or -1),
        "pc_assignments": getattr(session, "pc_assignments", None) or [],
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


def scenario_npc_to_dict(npc: Any) -> dict:
    """ScenarioNpc ORM を JSON 化可能な dict に変換する。"""
    if npc is None:
        return {}
    return {
        "id": npc.id,
        "scenario_id": npc.scenario_id,
        "name": npc.name,
        "description": npc.description,
        "image_data": npc.image_data,
        "promoted_character_id": npc.promoted_character_id,
        "created_at": npc.created_at.isoformat() if npc.created_at else None,
    }


def scenario_turn_to_dict(turn: Any) -> dict:
    """ScenarioTurn ORM を JSON 化可能な dict に変換する。"""
    if turn is None:
        return {}
    return {
        "id": turn.id,
        "session_id": turn.session_id,
        "turn_index": turn.turn_index,
        "speaker_type": turn.speaker_type,
        "speaker_id": turn.speaker_id,
        "speaker_name": turn.speaker_name,
        "content": turn.content,
        "raw_response": turn.raw_response,
        "log_request_id": getattr(turn, "log_request_id", None),
        "anticipation": getattr(turn, "anticipation", None),
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
    }


# ─── ストリーム実行 ───────────────────────────────────────────────────────────


