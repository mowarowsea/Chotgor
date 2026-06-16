"""うつつ（Usual Days）のやり取りが Chronicle 当日会話へ合流することのテスト。

うつつ会話は ChatMessage ではなく ScenarioTurn（scenario_turns）別経路に保存される。
本テスト群は、その別経路に保存されたやり取りが Chronicle（夜間棚卸し）の「今日の会話」
として正しく合流・整形され、かつ chronicled_at で二重処理されないことを検証する。

検証する観点:
    - ストア層: get_unchronicled_usual_turns_for_character の抽出条件
      （owner_character_id 一致・engine_type="usual_days"・chronicled_at IS NULL・キャラ別フィルタ）
    - ストア層: get_usual_turns_for_character_on_date の日付範囲抽出
    - ストア層: mark_scenario_turns_as_chronicled のタイムスタンプ付与・空リスト no-op
    - 統合: run_chronicle がうつつターンを当日会話プロンプトへ載せること
    - 統合: run_chronicle 後にうつつターンが chronicled 化され、再実行で二重に載らないこと
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import run_chronicle

from tests._ghost_model_helpers import (  # noqa: F401
    _NO_UPDATE_RESPONSE,
    working_memory_manager,
)


def _setup_usual_world(
    sqlite_store, char_name: str = "はる", n_turns: int = 3, ghost: bool = True,
):
    """ghost_model 付きの主人公キャラ＋うつつ(usual_days)世界＋ScenarioTurn 群を作るヘルパー。

    うつつターンは GM（ナレーター）と主人公（PC）が交互に発話する想定で作る。
    ユーザ発話は無人ループでは生じないため作らない。

    Args:
        sqlite_store: テスト用 SQLiteStore。
        char_name: 主人公キャラ名。
        n_turns: 作成するうつつターン数。
        ghost: True なら ghost_model を設定する（run_chronicle が処理対象にするため）。

    Returns:
        (char_id, session_id, turn_ids, preset_id) のタプル。
    """
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(
        char_id, char_name, ghost_model=preset_id if ghost else None,
    )

    scenario_id = str(uuid.uuid4())
    sqlite_store.create_scenario(
        scenario_id=scenario_id,
        title=f"{char_name}の日常",
        owner_character_id=char_id,
        pc_slots=[{"slot_id": "pc1", "name": char_name, "description": "主人公。"}],
        usual_config={"max_turns_per_scene": 8},
    )

    session_id = str(uuid.uuid4())
    sqlite_store.create_scenario_session(
        session_id=session_id,
        scenario_id=scenario_id,
        title="うつつ",
        gm_preset_id=preset_id,
        synopsis_preset_id=preset_id,
        engine_type="usual_days",
        pc_assignments=[{
            "slot_id": "pc1",
            "player_type": "character",
            "character_id": char_id,
            "preset_id": preset_id,
        }],
    )

    turn_ids: list[str] = []
    for i in range(n_turns):
        tid = str(uuid.uuid4())
        # 偶数ターン=GM(ナレーター)、奇数ターン=主人公(PC)。
        if i % 2 == 0:
            speaker_type, speaker_name = "narrator", "ナレーター"
        else:
            speaker_type, speaker_name = "pc", char_name
        sqlite_store.create_scenario_turn(
            turn_id=tid,
            session_id=session_id,
            turn_index=i,
            speaker_type=speaker_type,
            speaker_name=speaker_name,
            content=f"うつつ出来事 {i}",
        )
        turn_ids.append(tid)

    return char_id, session_id, turn_ids, preset_id


# ---------------------------------------------------------------------------
# ストア層: get_unchronicled_usual_turns_for_character のテスト
# ---------------------------------------------------------------------------


class TestGetUnchronicledUsualTurns:
    """うつつ未処理ターン抽出（get_unchronicled_usual_turns_for_character）の検証。

    owner_character_id 一致・engine_type="usual_days"・chronicled_at IS NULL の
    3 条件と、キャラ別フィルタが正しく効くことを確認する。
    """

    def test_returns_all_unchronicled_turns(self, sqlite_store):
        """新規うつつターン（chronicled_at IS NULL）がすべて返ることを確認する。"""
        char_id, _, turn_ids, _ = _setup_usual_world(sqlite_store, n_turns=3)

        turns = sqlite_store.get_unchronicled_usual_turns_for_character(char_id)

        assert {t.id for t in turns} == set(turn_ids)

    def test_excludes_already_chronicled(self, sqlite_store):
        """chronicled_at 設定済みのターンは除外されることを確認する。"""
        char_id, _, turn_ids, _ = _setup_usual_world(sqlite_store, n_turns=3)
        sqlite_store.mark_scenario_turns_as_chronicled([turn_ids[0]])

        turns = sqlite_store.get_unchronicled_usual_turns_for_character(char_id)

        ids = {t.id for t in turns}
        assert turn_ids[0] not in ids
        assert turn_ids[1] in ids
        assert turn_ids[2] in ids

    def test_filters_by_character(self, sqlite_store):
        """別キャラのうつつ世界のターンは返らないことを確認する。"""
        haru_id, _, haru_turns, _ = _setup_usual_world(sqlite_store, char_name="はる", n_turns=2)
        _setup_usual_world(sqlite_store, char_name="なつ", n_turns=2)

        turns = sqlite_store.get_unchronicled_usual_turns_for_character(haru_id)

        assert {t.id for t in turns} == set(haru_turns)

    def test_empty_for_character_without_usual_world(self, sqlite_store):
        """うつつ世界を持たないキャラには空リストを返すことを確認する。"""
        no_world_id = str(uuid.uuid4())
        sqlite_store.create_character(no_world_id, "世界なし")

        turns = sqlite_store.get_unchronicled_usual_turns_for_character(no_world_id)

        assert turns == []


# ---------------------------------------------------------------------------
# ストア層: get_usual_turns_for_character_on_date のテスト
# ---------------------------------------------------------------------------


class TestGetUsualTurnsOnDate:
    """うつつターンの日付範囲抽出（get_usual_turns_for_character_on_date）の検証。

    created_at が指定日範囲のターンを返し、範囲外（翌日など）は返さないことを確認する。
    """

    def test_returns_turns_within_date_range(self, sqlite_store):
        """当日（now を含む範囲）に作成したターンが返ることを確認する。"""
        char_id, _, turn_ids, _ = _setup_usual_world(sqlite_store, n_turns=3)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)

        turns = sqlite_store.get_usual_turns_for_character_on_date(char_id, today, tomorrow)

        assert {t.id for t in turns} == set(turn_ids)

    def test_excludes_turns_outside_date_range(self, sqlite_store):
        """範囲外（未来日）の窓では何も返らないことを確認する。"""
        char_id, _, _, _ = _setup_usual_world(sqlite_store, n_turns=2)
        future_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=5)
        future_end = future_start + timedelta(days=1)

        turns = sqlite_store.get_usual_turns_for_character_on_date(char_id, future_start, future_end)

        assert turns == []


# ---------------------------------------------------------------------------
# ストア層: mark_scenario_turns_as_chronicled のテスト
# ---------------------------------------------------------------------------


class TestMarkScenarioTurnsAsChronicled:
    """うつつターンの処理済みマーク（mark_scenario_turns_as_chronicled）の検証。

    chronicled_at にタイムスタンプを付与すること、空リストが no-op であること、
    冪等であることを確認する。
    """

    def test_sets_timestamp(self, sqlite_store):
        """mark 後に chronicled_at が非 NULL になることを確認する。"""
        char_id, session_id, turn_ids, _ = _setup_usual_world(sqlite_store, n_turns=2)
        sqlite_store.mark_scenario_turns_as_chronicled(turn_ids)

        turns = sqlite_store.list_scenario_turns(session_id)
        assert all(t.chronicled_at is not None for t in turns)

    def test_empty_list_is_noop(self, sqlite_store):
        """空リストを渡しても例外が出ないことを確認する。"""
        sqlite_store.mark_scenario_turns_as_chronicled([])  # 例外が出なければ OK

    def test_is_idempotent(self, sqlite_store):
        """同じターンを2回 mark しても問題ないことを確認する。"""
        char_id, session_id, turn_ids, _ = _setup_usual_world(sqlite_store, n_turns=1)
        sqlite_store.mark_scenario_turns_as_chronicled(turn_ids)
        sqlite_store.mark_scenario_turns_as_chronicled(turn_ids)  # 2回目も OK

        turns = sqlite_store.list_scenario_turns(session_id)
        assert all(t.chronicled_at is not None for t in turns)


# ---------------------------------------------------------------------------
# 統合: run_chronicle がうつつターンを合流・マークすることのテスト
# ---------------------------------------------------------------------------


class TestRunChronicleMergesUsualTurns:
    """run_chronicle がうつつ ScenarioTurn を当日会話へ合流させ、二重処理しないことの検証。

    うつつのやり取りは ChatMessage を 1 件も持たないキャラでも Chronicle の入力に
    現れること、処理後に chronicled_at が立ち、再実行で同じターンが二度載らないことを
    確認する。LLM 呼び出しはモックする。
    """

    @pytest.mark.asyncio
    async def test_usual_turns_appear_in_conversation_prompt(self, sqlite_store, working_memory_manager):
        """うつつターン本文が Chronicle の当日会話プロンプトに含まれることを確認する。"""
        char_id, _, _, _ = _setup_usual_world(sqlite_store, n_turns=3)

        captured: list[str] = []

        async def fake_generate(sys_prompt, messages):
            captured.append(messages[0]["content"])
            return _NO_UPDATE_RESPONSE

        mock_provider = AsyncMock()
        mock_provider.generate = fake_generate

        with patch("backend.services.character_query.create_provider", return_value=mock_provider):
            result = await run_chronicle(
                character_id=char_id, sqlite=sqlite_store,
                working_memory_manager=working_memory_manager,
            )

        assert result["status"] == "success"
        assert len(captured) == 1
        # 3 ターンの本文がすべて当日会話に合流していること
        assert "うつつ出来事 0" in captured[0]
        assert "うつつ出来事 1" in captured[0]
        assert "うつつ出来事 2" in captured[0]

    @pytest.mark.asyncio
    async def test_usual_turns_marked_after_chronicle(self, sqlite_store, working_memory_manager):
        """run_chronicle 成功後、うつつターンの chronicled_at が設定されることを確認する。"""
        char_id, session_id, _, _ = _setup_usual_world(sqlite_store, n_turns=2)
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value=_NO_UPDATE_RESPONSE)

        with patch("backend.services.character_query.create_provider", return_value=mock_provider):
            result = await run_chronicle(
                character_id=char_id, sqlite=sqlite_store,
                working_memory_manager=working_memory_manager,
            )

        assert result["status"] == "success"
        turns = sqlite_store.list_scenario_turns(session_id)
        assert all(t.chronicled_at is not None for t in turns)

    @pytest.mark.asyncio
    async def test_chronicled_usual_turns_not_reprocessed(self, sqlite_store, working_memory_manager):
        """2 回目の run_chronicle で、処理済みうつつターンが当日会話に再登場しないことを確認する。"""
        char_id, _, _, _ = _setup_usual_world(sqlite_store, n_turns=2)

        captured: list[str] = []

        async def fake_generate(sys_prompt, messages):
            captured.append(messages[0]["content"])
            return _NO_UPDATE_RESPONSE

        mock_provider = AsyncMock()
        mock_provider.generate = fake_generate

        with patch("backend.services.character_query.create_provider", return_value=mock_provider):
            # 1 回目: うつつターンを処理して chronicled 化する
            await run_chronicle(
                character_id=char_id, sqlite=sqlite_store,
                working_memory_manager=working_memory_manager,
            )
            # 2 回目: 未処理ターンはもう無いので、当日会話に旧ターンは載らない
            await run_chronicle(
                character_id=char_id, sqlite=sqlite_store,
                working_memory_manager=working_memory_manager,
            )

        assert len(captured) == 2
        assert "うつつ出来事 0" in captured[0]
        assert "うつつ出来事 0" not in captured[1]
        assert "うつつ出来事 1" not in captured[1]


# ---------------------------------------------------------------------------
# マイグレーション: scenario_turns.chronicled_at の ALTER パスのテスト
# ---------------------------------------------------------------------------


class TestChronicledAtMigration:
    """scenario_turns.chronicled_at の冪等マイグレーション（ALTER パス）の検証。

    インメモリDBは ORM create_all で最初から列があり ALTER 分岐が走らない。ここでは
    列を一旦 DROP して「旧DB（移行前）」を再現し、マイグレーションが ALTER TABLE で列を
    追加し、既存ターンを壊さず chronicled_at=NULL で補うことを確認する。ユーザの実DBに
    再起動時に走るのと同じ経路を検証する。
    """

    def test_idempotent_when_column_exists(self, sqlite_store):
        """列が既にある状態で複数回呼んでも安全（二重 ALTER しない）ことを確認する。"""
        sqlite_store._migrate_add_scenario_turn_chronicled_at()
        sqlite_store._migrate_add_scenario_turn_chronicled_at()

        with sqlite_store.engine.begin() as conn:
            cols = {
                r[1]
                for r in conn.exec_driver_sql("PRAGMA table_info(scenario_turns)").fetchall()
            }
        assert "chronicled_at" in cols

    def test_alters_legacy_schema(self, sqlite_store):
        """列の無い旧スキーマに対し ALTER TABLE が走り、既存ターンを壊さないことを確認する。"""
        _, session_id, _, _ = _setup_usual_world(sqlite_store, n_turns=1)

        # 旧スキーマ再現: chronicled_at を物理 DROP（SQLite 3.35+ の DROP COLUMN）
        with sqlite_store.engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE scenario_turns DROP COLUMN chronicled_at")
            cols_before = {
                r[1]
                for r in conn.exec_driver_sql("PRAGMA table_info(scenario_turns)").fetchall()
            }
        assert "chronicled_at" not in cols_before

        # マイグレーション実行（ALTER パスが走る）
        sqlite_store._migrate_add_scenario_turn_chronicled_at()

        with sqlite_store.engine.begin() as conn:
            cols_after = {
                r[1]
                for r in conn.exec_driver_sql("PRAGMA table_info(scenario_turns)").fetchall()
            }
        assert "chronicled_at" in cols_after

        # 既存ターンは壊れず、chronicled_at は NULL 既定で補われている
        turns = sqlite_store.list_scenario_turns(session_id)
        assert len(turns) == 1
        assert turns[0].chronicled_at is None
