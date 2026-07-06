"""会話外行動権（actions）のテスト — めぐり（巡り / Aliveness）Phase 6。

検証対象（docs/aliveness_plan.md §5.3）:
    1. jittered_slot_time: 評価タイミングのジッターが決定論（乱数は世界に置く）
    2. evaluate_action_urge: 閾値評価が純関数・無料（LLM 不使用）で候補を返す
    3. run_action_cycle: 本人の選択で実行/見送りが分かれ、
       - push が新規セッション＋キャラ発メッセージ＋chat.message 封筒になる
       - action.performed 封筒（intent_id 参照）が載る
       - 帰還宣言 [INTENT_FULFILLED] で意図が fulfilled になる
       - 見送り（タグなし）では何も実行されない
       - コストガード（問い合わせ/実行の日次上限）が効く
    LLM（ask_character_with_tools）はモックする。
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.actions.runner import (
    _parse_action_choice,
    evaluate_action_urge,
    jittered_slot_time,
    run_action_cycle,
)


def _make_character(sqlite_store, name="はるテスト", action_menu=None):
    """テスト用キャラクター＋プリセットを作成して返すヘルパ。"""
    preset_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "anthropic", "claude-x")
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name, ghost_model=preset_id)
    if action_menu:
        sqlite_store.update_character(char_id, action_menu=action_menu)
    return char_id, name


def _make_hot_intent(sqlite_store, char_id, description="もわに話したい", target="user"):
    """意図圧が閾値を超える（古い×高圧源）active 意図を作るヘルパ。

    社会圧は封筒ゼロで 1.0（最大）なので、source_kind="social" の意図を
    10日前に巻き戻せば意図圧 ≈ 1.0 になる。
    """
    intent = sqlite_store.create_intent(
        char_id, description, target=target, source_kind="social",
    )
    from backend.repositories.sqlite.models import Intent
    with sqlite_store.get_session() as s:
        row = s.get(Intent, intent.id)
        row.created_at = datetime.now() - timedelta(days=10)
        s.commit()
    return intent


class TestJitter:
    """評価タイミングのジッター（世界の乱数）を検証するテストクラス。"""

    def test_deterministic_per_slot(self):
        """同じキャラ・同じスロットなら常に同じ時刻（決定論）。"""
        slot = datetime(2026, 7, 7, 12, 0)
        a = jittered_slot_time("char-1", slot)
        b = jittered_slot_time("char-1", slot)
        assert a == b
        assert slot <= a <= slot + timedelta(minutes=60)

    def test_differs_between_characters(self):
        """キャラが違えば（ほぼ確実に）別の時刻になる。"""
        slot = datetime(2026, 7, 7, 12, 0)
        values = {jittered_slot_time(f"char-{i}", slot) for i in range(10)}
        assert len(values) > 1


class TestEvaluateUrge:
    """閾値評価（純関数・無料）を検証するテストクラス。"""

    def test_no_intents_no_urge(self, sqlite_store):
        """active 意図が無ければ空リスト（LLM は呼ばれない）。"""
        char_id, _ = _make_character(sqlite_store)
        assert evaluate_action_urge(sqlite_store, char_id) == []

    def test_hot_intent_is_candidate(self, sqlite_store):
        """高圧の意図が候補として返る。"""
        char_id, _ = _make_character(sqlite_store)
        intent = _make_hot_intent(sqlite_store, char_id)
        candidates = evaluate_action_urge(sqlite_store, char_id)
        assert [i.id for i in candidates] == [intent.id]

    def test_fresh_intent_below_threshold(self, sqlite_store):
        """生まれたばかりの低圧意図は候補にならない。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.create_intent(char_id, "ふと思っただけ", source_kind="none")
        assert evaluate_action_urge(sqlite_store, char_id) == []


class TestParseChoice:
    """本人の選択タグのパースを検証するテストクラス。"""

    def test_parse_full_tag(self):
        """[ACTION: id | menu | 中身] をパースできる。"""
        iid = str(uuid.uuid4())
        choice = _parse_action_choice(
            f"うん、やる。\n[ACTION: {iid} | push | ねえ、いま何してる？]"
        )
        assert choice == {"intent_id": iid, "menu": "push", "body": "ねえ、いま何してる？"}

    def test_parse_without_body(self):
        """中身なし（scene 等）もパースできる。"""
        iid = str(uuid.uuid4())
        choice = _parse_action_choice(f"[ACTION: {iid} | scene]")
        assert choice == {"intent_id": iid, "menu": "scene", "body": ""}

    def test_no_tag_is_decline(self):
        """タグなし = 見送り（本人の選択として尊重）。"""
        assert _parse_action_choice("今はいいかな。そのまま心に置いておく。") is None


class TestRunActionCycle:
    """run_action_cycle の全経路を検証するテストクラス（LLM モック）。"""

    def _run(self, sqlite_store, char_id, responses, memory_manager=None, **kwargs):
        """ask_character_with_tools を応答列でモックしてサイクルを回すヘルパ。

        Args:
            responses: [選択応答, 帰還応答] の順で消費される文字列リスト。
        """
        mm = memory_manager or MagicMock()
        with patch(
            "backend.services.actions.runner.ask_character_with_tools",
            new=AsyncMock(side_effect=responses),
        ):
            return asyncio.run(run_action_cycle(
                char_id, sqlite_store, {},
                memory_manager=mm, **kwargs,
            ))

    def test_skips_when_menu_off(self, sqlite_store):
        """行動メニュー全 OFF（NULL）は評価対象外。"""
        char_id, _ = _make_character(sqlite_store)
        _make_hot_intent(sqlite_store, char_id)
        result = self._run(sqlite_store, char_id, [])
        assert result["status"] == "skipped"

    def test_skips_without_hot_intent(self, sqlite_store):
        """閾値超えの意図が無ければ LLM を呼ばずスキップ（閾値評価はゼロ円）。"""
        char_id, _ = _make_character(sqlite_store, action_menu={"push": True})
        result = self._run(sqlite_store, char_id, [])
        assert result["status"] == "skipped"

    def test_decline_executes_nothing(self, sqlite_store):
        """本人がタグを書かなければ見送り — 何も実行されない。"""
        char_id, _ = _make_character(sqlite_store, action_menu={"push": True})
        _make_hot_intent(sqlite_store, char_id)
        result = self._run(sqlite_store, char_id, ["今日はやめておく。"])
        assert result["status"] == "declined"
        assert sqlite_store.list_timeline_events(
            char_id, event_type_prefixes=["action."]
        ) == []

    def test_push_executes_and_fulfills(self, sqlite_store):
        """push 実行: 新規セッション＋キャラ発メッセージ＋封筒＋帰還で fulfilled。"""
        char_id, char_name = _make_character(
            sqlite_store, action_menu={"push": True},
        )
        intent = _make_hot_intent(sqlite_store, char_id)
        result = self._run(sqlite_store, char_id, [
            f"送ってみる。\n[ACTION: {intent.id} | push | ねえ、ふと声が聞きたくなった]",
            f"うん、満ちた。\n[INTENT_FULFILLED: {intent.id}]",
        ])
        assert result["status"] == "executed"
        assert result["fulfilled"] is True
        # 新規セッションにキャラ発メッセージが置かれている
        sessions = sqlite_store.list_chat_sessions()
        push_session = next(s for s in sessions if s.title == f"{char_name}より")
        msgs = sqlite_store.list_chat_messages(push_session.id)
        assert msgs[0].role == "character"
        assert "声が聞きたくなった" in msgs[0].content
        # chat.message（actor=character）封筒 → 社会圧の減衰源になる
        chat_events = sqlite_store.list_timeline_events(
            char_id, event_type_prefixes=["chat.message"],
        )
        assert chat_events and chat_events[-1].actor == "character"
        # action.performed 封筒（intent_id 参照）
        actions = sqlite_store.list_timeline_events(
            char_id, event_type_prefixes=["action.performed"],
        )
        assert actions[0].intent_id == intent.id
        assert actions[0].payload["menu"] == "push"
        # 意図は fulfilled ＋ intent.fulfilled 封筒
        assert sqlite_store.get_intent(intent.id).status == "fulfilled"

    def test_no_fulfilled_tag_keeps_intent_active(self, sqlite_store):
        """帰還でタグを書かなければ意図は active のまま（まだ続く）。"""
        char_id, _ = _make_character(sqlite_store, action_menu={"push": True})
        intent = _make_hot_intent(sqlite_store, char_id)
        result = self._run(sqlite_store, char_id, [
            f"[ACTION: {intent.id} | push | やあ]",
            "送ったけど、まだ足りない気がする。",
        ])
        assert result["status"] == "executed" and result["fulfilled"] is False
        assert sqlite_store.get_intent(intent.id).status == "active"

    def test_research_puts_result_in_payload(self, sqlite_store):
        """research 実行: 検索結果が action.performed の payload に残る。"""
        char_id, _ = _make_character(sqlite_store, action_menu={"research": True})
        intent = _make_hot_intent(sqlite_store, char_id, description="流星群のことを知りたい", target="self")
        with patch(
            "backend.character_actions.web_searcher.WebSearcher.search",
            return_value="【検索結果】ペルセウス座流星群は8月に極大。",
        ):
            result = self._run(sqlite_store, char_id, [
                f"[ACTION: {intent.id} | research | ペルセウス座流星群 2026]",
                "読んだ。まだ心に残しておく。",
            ])
        assert result["status"] == "executed"
        action_ev = sqlite_store.list_timeline_events(
            char_id, event_type_prefixes=["action.performed"],
        )[0]
        assert "流星群" in action_ev.payload["result_text"]

    def test_disabled_menu_choice_is_declined(self, sqlite_store):
        """OFF のメニューを選んでも執行されない（許可は設計者の領分）。"""
        char_id, _ = _make_character(sqlite_store, action_menu={"push": True})
        intent = _make_hot_intent(sqlite_store, char_id)
        result = self._run(sqlite_store, char_id, [
            f"[ACTION: {intent.id} | research | 検索したい]",
        ])
        assert result["status"] == "declined"

    def test_inquiry_cost_guard(self, sqlite_store):
        """問い合わせの日次上限に達したら LLM を呼ばずスキップ。"""
        char_id, _ = _make_character(sqlite_store, action_menu={"push": True})
        _make_hot_intent(sqlite_store, char_id)
        today = datetime.now().date().isoformat()
        sqlite_store.set_setting(f"action_inquiry_count_{today}", "6")
        result = self._run(sqlite_store, char_id, [])
        assert result["status"] == "skipped" and "上限" in result["reason"]

    def test_exec_cost_guard(self, sqlite_store):
        """実行の日次上限に達したら選択があっても執行しない。"""
        char_id, _ = _make_character(sqlite_store, action_menu={"push": True})
        intent = _make_hot_intent(sqlite_store, char_id)
        today = datetime.now().date().isoformat()
        sqlite_store.set_setting(f"action_exec_count_{today}", "3")
        result = self._run(sqlite_store, char_id, [
            f"[ACTION: {intent.id} | push | やあ]",
        ])
        assert result["status"] == "skipped" and "上限" in result["reason"]
