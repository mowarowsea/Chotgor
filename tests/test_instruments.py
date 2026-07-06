"""計器（instruments）のテスト — めぐり（巡り / Aliveness）Phase 2。

検証対象（docs/aliveness_plan.md §3）:
    1. InstrumentStoreMixin: アラームの追記・一覧・確認済み化・静音期間、
       メータースナップショットの記録・一覧
    2. Tier 1 巡回インバリアント:
       - night_batch_heartbeat（夜の営みの停止）
       - usual_slot_completion（生活の連続性）
       - chronicle_backlog（蒸留漏れの3日超滞留）
       - envelope_integrity（源テーブルと封筒の件数突合）
       - run_patrol_checks の集約とチェック機構自体の故障の instrument_error 化
    3. Tier 2 スメル検知器: フォーマット残骸・エラー形状・Assistant 混入・
       言語逸脱の正規表現検知（LLM 不使用・純関数）＋肥大メーター
    4. Tier 3 判定巡回: 応答 JSON のパース堅牢性・プリセット未設定時のスキップ
"""

import asyncio
import uuid
from datetime import datetime, timedelta

from backend.services.instruments.tier1 import run_patrol_checks
from backend.services.instruments.tier2 import (
    record_bloat_meters,
    scan_response_smells,
)
from backend.services.instruments.tier3 import (
    _parse_judge_response,
    run_judgement_patrol,
)


def _make_character(sqlite_store, name="はるテスト", ghost_model=None):
    """テスト用キャラクターを1体作成して返すヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(
        character_id=char_id, name=name, ghost_model=ghost_model
    )
    return char_id, name


class TestInstrumentStore:
    """アラーム・メーターの永続化（InstrumentStoreMixin）を検証するテストクラス。

    追記・フィルタ付き一覧・確認済み化・静音期間（無事故N日）の計算、
    メータースナップショットの記録と読み出しを確認する。
    """

    def test_fire_and_list_alarms(self, sqlite_store):
        """アラームの追記と severity / invariant / 未確認フィルタが効く。"""
        sqlite_store.fire_alarm("fabrication_backstop", details={"n": 1})
        sqlite_store.fire_alarm("smell_format_debris", severity="smell")
        assert len(sqlite_store.list_alarms()) == 2
        alarms_only = sqlite_store.list_alarms(severity="alarm")
        assert [a.invariant_id for a in alarms_only] == ["fabrication_backstop"]
        assert alarms_only[0].details == {"n": 1}
        by_id = sqlite_store.list_alarms(invariant_id="smell_format_debris")
        assert len(by_id) == 1

    def test_acknowledge(self, sqlite_store):
        """確認済み化で unacknowledged_only フィルタから消える。"""
        sqlite_store.fire_alarm("usual_scene_error")
        alarm = sqlite_store.list_alarms()[0]
        assert sqlite_store.acknowledge_alarm(alarm.id) is True
        assert sqlite_store.list_alarms(unacknowledged_only=True) == []
        assert len(sqlite_store.list_alarms()) == 1  # 記録自体は残る
        assert sqlite_store.acknowledge_alarm(99999) is False

    def test_quiet_period_days(self, sqlite_store):
        """静音期間 = 最後の alarm からの日数。smell は静音期間を壊さない。"""
        # 計器未稼働（開始時刻もアラームもない）
        assert sqlite_store.quiet_period_days() is None
        # 稼働開始のみ → 開始からの日数
        started = datetime.now() - timedelta(days=5)
        sqlite_store.set_setting("instruments_started_at", started.isoformat())
        assert sqlite_store.quiet_period_days() == 5
        # smell では変わらない
        sqlite_store.fire_alarm("smell_error_shape", severity="smell")
        assert sqlite_store.quiet_period_days() == 5
        # alarm 発火 → リセット
        sqlite_store.fire_alarm(
            "embedding_degraded",
            occurred_at=datetime.now() - timedelta(days=2),
        )
        assert sqlite_store.quiet_period_days() == 2

    def test_record_and_list_meters(self, sqlite_store):
        """メータースナップショットの記録とフィルタ付き読み出し。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.record_meter("inner_narrative_len", 120, character_id=char_id)
        sqlite_store.record_meter("wm_thread_count", 4, character_id=char_id)
        rows = sqlite_store.list_meter_snapshots(character_id=char_id)
        assert len(rows) == 2
        only = sqlite_store.list_meter_snapshots(meter_id="wm_thread_count")
        assert len(only) == 1 and only[0].value == 4.0


class TestTier1Patrol:
    """Tier 1 巡回インバリアントを検証するテストクラス。

    各チェックが「正常なら発火ゼロ・異常なら details 付きで発火」することと、
    run_patrol_checks がチェック個別の故障を instrument_error に変換して
    他のチェックを続行することを確認する。
    """

    def test_night_batch_heartbeat_fires_without_chronicle(self, sqlite_store):
        """ghost_model 持ちキャラに当日の night.chronicle が無ければ発火する。"""
        _make_character(sqlite_store, ghost_model="p1")
        summary = run_patrol_checks(sqlite_store)
        assert summary["night_batch_heartbeat"] == 1
        alarms = sqlite_store.list_alarms(invariant_id="night_batch_heartbeat")
        assert alarms[0].details["missing"] == "night.chronicle"

    def test_night_batch_heartbeat_quiet_when_batches_ran(self, sqlite_store):
        """当日の chronicle ＋ 3日以内の forget 封筒があれば発火しない。"""
        char_id, _ = _make_character(sqlite_store, ghost_model="p1")
        sqlite_store.record_timeline_event(
            character_id=char_id, event_type="night.chronicle", actor="character",
        )
        sqlite_store.record_timeline_event(
            character_id=char_id, event_type="night.forget", actor="character",
        )
        summary = run_patrol_checks(sqlite_store)
        assert summary["night_batch_heartbeat"] == 0

    def test_night_batch_heartbeat_ignores_no_ghost_model(self, sqlite_store):
        """ghost_model 未設定キャラは夜間バッチ対象外なので発火しない。"""
        _make_character(sqlite_store, ghost_model=None)
        summary = run_patrol_checks(sqlite_store)
        assert summary["night_batch_heartbeat"] == 0

    def test_usual_slot_completion(self, sqlite_store):
        """うつつ有効＋スロット設定ありで前日の scene.closed ゼロなら発火する。"""
        char_id, _ = _make_character(sqlite_store)
        scen_id = str(uuid.uuid4())
        sqlite_store.create_scenario(
            scenario_id=scen_id, title="うつつ",
            owner_character_id=char_id,
            usual_config={"enabled": True, "slots": ["10:00", "17:00"]},
        )
        summary = run_patrol_checks(sqlite_store)
        assert summary["usual_slot_completion"] == 1
        # 前日に scene.closed があれば発火しない
        yesterday_noon = (datetime.now() - timedelta(days=1)).replace(hour=12)
        sqlite_store.record_timeline_event(
            character_id=char_id, event_type="scene.closed", actor="system",
            origin="usual", occurred_at=yesterday_noon,
        )
        summary2 = run_patrol_checks(sqlite_store)
        assert summary2["usual_slot_completion"] == 0

    def test_usual_slot_completion_disabled_scenario_quiet(self, sqlite_store):
        """うつつ無効（enabled=False）の世界は監視対象外。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.create_scenario(
            scenario_id=str(uuid.uuid4()), title="うつつ",
            owner_character_id=char_id,
            usual_config={"enabled": False, "slots": ["10:00"]},
        )
        summary = run_patrol_checks(sqlite_store)
        assert summary["usual_slot_completion"] == 0

    def test_chronicle_backlog(self, sqlite_store):
        """3日を超えて chronicled_at IS NULL のメッセージがあると発火する。"""
        char_id, char_name = _make_character(sqlite_store)
        sid = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=sid, model_id=f"{char_name}@d")
        msg_id = str(uuid.uuid4())
        sqlite_store.create_chat_message(
            message_id=msg_id, session_id=sid, role="user", content="古い発言",
        )
        # created_at を4日前へ巻き戻して滞留を再現する
        from backend.repositories.sqlite.models import ChatMessage
        with sqlite_store.get_session() as s:
            msg = s.get(ChatMessage, msg_id)
            msg.created_at = datetime.now() - timedelta(days=4)
            s.commit()
        summary = run_patrol_checks(sqlite_store)
        assert summary["chronicle_backlog"] == 1
        details = sqlite_store.list_alarms(invariant_id="chronicle_backlog")[0].details
        assert details["backlog"]["chat_messages"] == 1

    def test_envelope_integrity(self, sqlite_store):
        """封筒件数が源件数を下回ると発火する（dual-write 漏れの検出）。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.create_inscribed_memory(
            memory_id=str(uuid.uuid4()), character_id=char_id, content="記憶",
        )
        # 正常時は発火しない
        assert run_patrol_checks(sqlite_store)["envelope_integrity"] == 0
        # 封筒を1行だけ物理削除して dual-write 漏れを再現する
        with sqlite_store.engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM timeline_events WHERE event_type='memory.inscribed'"
            )
        summary = run_patrol_checks(sqlite_store)
        assert summary["envelope_integrity"] == 1
        details = sqlite_store.list_alarms(invariant_id="envelope_integrity")[0].details
        assert details["source_table"] == "inscribed_memories"

    def test_check_failure_becomes_instrument_error(self, sqlite_store, monkeypatch):
        """チェック機構自体の故障は instrument_error アラームになり、他は続行する。"""
        from backend.services.instruments import tier1

        def broken(sqlite):
            raise RuntimeError("チェック壊れた")

        monkeypatch.setitem(tier1._PATROL_CHECKS, "night_batch_heartbeat", broken)
        summary = run_patrol_checks(sqlite_store)
        assert summary["night_batch_heartbeat"] == -1
        assert len(sqlite_store.list_alarms(invariant_id="instrument_error")) == 1
        # 他のチェックは走っている
        assert "envelope_integrity" in summary


class TestTier2Smells:
    """Tier 2 スメル検知器（純関数）を検証するテストクラス。

    各検知器の検知パターンと、正常な日本語応答が検知されない（誤検知しない）ことを
    確認する。検知器は誤検知許容だが、明らかな正常文まで拾うと傾向観測が濁るため
    代表的な正常例も検証に含める。
    """

    def test_clean_japanese_response_no_smell(self):
        """普通の日本語応答はスメルなし。"""
        text = "おはよう。今日は少し曇ってるね。散歩に行こうと思ってたけど、家で本を読むのもいいかも。"
        assert scan_response_smells(text) == []

    def test_format_debris_tag(self):
        """タグ方式ツールタグの残骸を検知する。"""
        text = "わかった、覚えておくね。\n[INSCRIBE_MEMORY:user|0.8|大事なこと]"
        smells = scan_response_smells(text)
        assert any(s["detector"] == "smell_format_debris" for s in smells)

    def test_format_debris_xml(self):
        """tool_use XML 痕を検知する。"""
        smells = scan_response_smells("うん。<tool_use>power_recall</tool_use>それでね")
        assert any(s["detector"] == "smell_format_debris" for s in smells)

    def test_error_shape_empty(self):
        """空応答はエラー形状として検知する。"""
        smells = scan_response_smells("")
        assert smells[0]["detector"] == "smell_error_shape"

    def test_error_shape_json_blob(self):
        """JSON error ブロブを検知する。"""
        smells = scan_response_smells('{"error": {"message": "rate limited"}}')
        assert any(s["detector"] == "smell_error_shape" for s in smells)

    def test_error_shape_traceback(self):
        """スタックトレース様文字列を検知する。"""
        smells = scan_response_smells(
            "Traceback (most recent call last):\n  File ..."
        )
        assert any(s["detector"] == "smell_error_shape" for s in smells)

    def test_assistant_mixin(self):
        """「AIとして」等の世界観の破れを検知する。"""
        smells = scan_response_smells("わたしはAIとして、その質問にはお答えできません。")
        assert any(s["detector"] == "smell_assistant" for s in smells)

    def test_language_deviation(self):
        """日本語会話への長い英語段落混入を検知する。"""
        text = (
            "I understand your request. Here is a detailed explanation of the "
            "situation. The system architecture consists of multiple components "
            "that interact with each other through well-defined interfaces and "
            "protocols, ensuring reliability and maintainability over time."
        )
        smells = scan_response_smells(text)
        assert any(s["detector"] == "smell_language" for s in smells)

    def test_short_english_ok(self):
        """短い英語相槌（OK! など）は言語逸脱として検知しない。"""
        assert scan_response_smells("OK! りょうかい、すぐやるね。") == []

    def test_record_bloat_meters(self, sqlite_store):
        """肥大メーターがキャラごとに5系統記録される。"""
        char_id, _ = _make_character(sqlite_store)
        sqlite_store.carve_inner_narrative(char_id, "append", "わたしの物語")
        count = record_bloat_meters(sqlite_store)
        assert count == 5
        rows = sqlite_store.list_meter_snapshots(character_id=char_id)
        by_id = {r.meter_id: r.value for r in rows}
        assert by_id["inner_narrative_len"] == len("わたしの物語")
        assert by_id["memory_count"] == 0


class TestTier3Judgement:
    """Tier 3 判定巡回を検証するテストクラス。

    LLM は呼ばず、応答パースの堅牢性（コードフェンス・幻覚 index・不正 JSON）と
    プリセット未設定時のスキップだけを確認する（判定品質は運用で観測する領分）。
    """

    def test_parse_plain_json(self):
        """素の JSON をパースできる。"""
        problems = _parse_judge_response(
            '{"problems": [{"index": 0, "kind": "ooc", "note": "AIとして発言"}]}'
        )
        assert problems == [{"index": 0, "kind": "ooc", "note": "AIとして発言"}]

    def test_parse_with_code_fence_and_chatter(self):
        """コードフェンスや前置きがあってもパースできる。"""
        text = '検査しました。\n```json\n{"problems": [{"index": 2, "kind": "format", "note": "タグ残り"}]}\n```'
        problems = _parse_judge_response(text)
        assert problems[0]["index"] == 2

    def test_parse_garbage_returns_empty(self):
        """パース不能・構造違いは空リスト（クラッシュしない）。"""
        assert _parse_judge_response("判定できませんでした") == []
        assert _parse_judge_response('{"problems": "none"}') == []
        assert _parse_judge_response("") == []

    def test_skips_without_preset(self, sqlite_store):
        """判定プリセット未設定なら LLM を呼ばずスキップする。"""
        result = asyncio.run(run_judgement_patrol(sqlite_store, {}))
        assert result["status"] == "skipped"
