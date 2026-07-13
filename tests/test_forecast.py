"""予報パネル（forecast）のテスト — docs/planned/forecast_panel_plan.md。

検証対象:
    1. DecisionStoreMixin: scheduler_decisions への追記・フィルタ付き読み出し・
       機構別最新（heartbeat 表示・Tier 1 の材料）が正しく動くこと。
    2. build_forecast: すべての決定論純関数の合成として、LLM 不使用で
       - 診断ヘッダ（availability / 圧力 / 意図圧 / heartbeat / cap / 配達シム）
       - 週間カレンダー（エントリ・②導出シーン・行動権スロットの予報）
       - 圧力予報72h（格子と系列の長さが一致・無風外挿）
       - 決定ログ・揺れ監査
       を一括で返すこと。datetime はすべて ISO 文字列（JSON 埋め込み前提）。
    3. action_urge_snapshot: 不発理由「閾値未達 X/Y」の材料となる全景
       （active 意図すべての意図圧・現在圧力・閾値）を意図圧降順で返すこと。
    4. run_action_cycle の決定ログ: 閾値未達・見送りが scheduler_decisions に
       理由付きで残ること（「正常な沈黙」と「壊れた沈黙」の区別の根拠）。
    5. Tier 1 scheduler_heartbeat: 鮮度1時間超のループ停止だけを発火し、
       未記録（未使用）は発火しないこと。
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from backend.services.actions.runner import action_urge_snapshot, run_action_cycle
from backend.services.instruments.tier1 import _check_scheduler_heartbeat
from backend.services.timeline.forecast import build_forecast


def _make_character(sqlite_store, name="はる予報", **updates):
    """テスト用キャラクター＋プリセットを作成して返すヘルパ。"""
    preset_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "anthropic", "claude-x")
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name=name, ghost_model=preset_id)
    if updates:
        sqlite_store.update_character(char_id, **updates)
    return char_id, name


def _make_hot_intent(sqlite_store, char_id, description="もわに話したい", now=None):
    """意図圧が閾値を超える（古い×高圧源）active 意図を作るヘルパ。

    封筒ゼロなら社会圧は 1.0（最大）なので、source_kind="social" の意図を
    10日前に巻き戻せば意図圧はほぼ 1.0 になる。
    """
    intent = sqlite_store.create_intent(
        char_id, description, target="user", source_kind="social",
    )
    from backend.repositories.sqlite.models import Intent
    base_now = now or datetime.now()
    with sqlite_store.get_session() as s:
        row = s.get(Intent, intent.id)
        row.created_at = base_now - timedelta(days=10)
        s.commit()
    return intent


class TestDecisionStore:
    """scheduler_decisions テーブルの追記・読み出しを検証するテストクラス。

    決定ログは「正常な沈黙（閾値未達）」と「壊れた沈黙（機構の死）」を事後に
    区別するための追記型記録。剪定はしない前提なので、削除系 API は存在しない。
    """

    def test_record_and_list(self, sqlite_store):
        """追記した決定が新しい順で読め、フィルタ（キャラ・機構・結果）が効く。"""
        sqlite_store.record_scheduler_decision(
            "action", "skipped", character_id="c1", reason="閾値未達 0.52/0.7",
            details={"threshold": 0.7},
        )
        sqlite_store.record_scheduler_decision(
            "usual_days", "fired", character_id="c1", reason="シーン完了",
        )
        sqlite_store.record_scheduler_decision(
            "action", "fired", character_id="c2", reason="push 実行",
        )

        all_rows = sqlite_store.list_scheduler_decisions()
        assert len(all_rows) == 3

        c1_rows = sqlite_store.list_scheduler_decisions(character_id="c1")
        assert {r.scheduler for r in c1_rows} == {"action", "usual_days"}

        skipped = sqlite_store.list_scheduler_decisions(outcome="skipped")
        assert len(skipped) == 1
        assert skipped[0].reason == "閾値未達 0.52/0.7"
        assert skipped[0].details == {"threshold": 0.7}

        action_rows = sqlite_store.list_scheduler_decisions(scheduler="action")
        assert len(action_rows) == 2

    def test_since_filter(self, sqlite_store):
        """since より古い決定は返らない。"""
        sqlite_store.record_scheduler_decision(
            "action", "skipped", occurred_at=datetime.now() - timedelta(days=3),
        )
        sqlite_store.record_scheduler_decision("action", "fired")
        rows = sqlite_store.list_scheduler_decisions(
            since=datetime.now() - timedelta(days=1),
        )
        assert len(rows) == 1
        assert rows[0].outcome == "fired"

    def test_latest_per_scheduler(self, sqlite_store):
        """機構ごとの最新1件だけが集まる（heartbeat 表示の材料）。"""
        old = datetime.now() - timedelta(hours=5)
        sqlite_store.record_scheduler_decision("action", "skipped", occurred_at=old)
        sqlite_store.record_scheduler_decision("action", "fired")
        sqlite_store.record_scheduler_decision("usual_days", "fired")

        latest = sqlite_store.latest_scheduler_decisions()
        assert set(latest.keys()) == {"action", "usual_days"}
        assert latest["action"].outcome == "fired"  # 新しい方が勝つ


class TestActionUrgeSnapshot:
    """閾値評価の全景（不発理由と予報パネル診断の共用材料）を検証するテストクラス。"""

    def test_snapshot_structure(self, sqlite_store):
        """閾値・現在圧力・全 active 意図の意図圧が意図圧降順で返る。"""
        char_id, _ = _make_character(sqlite_store)
        _make_hot_intent(sqlite_store, char_id, "熱い意図")
        sqlite_store.create_intent(char_id, "生まれたての意図", source_kind="none")

        snap = action_urge_snapshot(sqlite_store, char_id)
        assert snap["threshold"] == 0.7
        assert set(snap["pressures"].keys()) == {"social", "boredom", "body"}
        assert len(snap["intents"]) == 2
        # 降順: 古い×高圧源の意図が先頭
        assert snap["intents"][0]["description"] == "熱い意図"
        assert snap["intents"][0]["pressure"] >= snap["intents"][1]["pressure"]

    def test_snapshot_empty(self, sqlite_store):
        """active 意図が無くても圧力と閾値は返る（intents は空リスト）。"""
        char_id, _ = _make_character(sqlite_store)
        snap = action_urge_snapshot(sqlite_store, char_id)
        assert snap["intents"] == []


class TestActionCycleDecisionLog:
    """行動サイクルの各終着点が決定ログに理由付きで残ることを検証するテストクラス。

    予報パネルの核心 — 「push が来ない」が『正常な閾値未達』なのか
    『本人の見送り』なのか『機構の死』なのかを、後から区別できること。
    """

    def test_no_intents_records_skipped(self, sqlite_store):
        """active 意図ゼロ → skipped「active 意図なし」が記録される。"""
        char_id, _ = _make_character(
            sqlite_store, action_menu={"push": True},
        )
        result = asyncio.run(run_action_cycle(
            char_id, sqlite_store, {}, memory_manager=object(),
        ))
        assert result["status"] == "skipped"
        rows = sqlite_store.list_scheduler_decisions(character_id=char_id)
        assert len(rows) == 1
        assert rows[0].outcome == "skipped"
        assert rows[0].reason == "active 意図なし"
        # 全景（details）に圧力と閾値が残っている
        assert rows[0].details["threshold"] == 0.7

    def test_declined_records_with_candidates(self, sqlite_store):
        """本人が見送り（タグなし応答）→ declined と候補一覧が記録される。"""
        char_id, _ = _make_character(
            sqlite_store, action_menu={"push": True},
        )
        intent = _make_hot_intent(sqlite_store, char_id)
        with patch(
            "backend.services.actions.runner.ask_character_with_tools",
            AsyncMock(return_value="今日はやめておく。"),
        ):
            result = asyncio.run(run_action_cycle(
                char_id, sqlite_store, {}, memory_manager=object(),
            ))
        assert result["status"] == "declined"
        rows = sqlite_store.list_scheduler_decisions(
            character_id=char_id, outcome="declined",
        )
        assert len(rows) == 1
        assert rows[0].reason == "本人が見送り"
        assert rows[0].details["candidates"][0]["intent_id"] == str(intent.id)


class TestSchedulerHeartbeatInvariant:
    """Tier 1 scheduler_heartbeat（無人ループの停止検知）を検証するテストクラス。

    heartbeat は決定ログと違い「評価対象ゼロでも毎分打たれる」ため、
    鮮度の劣化＝ループ停止と断定できる（発火機会の少なさでは説明できない）。
    """

    def test_fresh_beat_no_alarm(self, sqlite_store):
        """直近の heartbeat があれば発火しない。"""
        sqlite_store.set_setting(
            "scheduler_heartbeat_action", datetime.now().isoformat(),
        )
        assert _check_scheduler_heartbeat(sqlite_store) == []

    def test_stale_beat_fires(self, sqlite_store):
        """1時間超沈黙している heartbeat は停止疑いとして発火する。"""
        sqlite_store.set_setting(
            "scheduler_heartbeat_usual_days",
            (datetime.now() - timedelta(hours=2)).isoformat(),
        )
        fired = _check_scheduler_heartbeat(sqlite_store)
        assert len(fired) == 1
        assert fired[0]["scheduler"] == "usual_days"
        assert fired[0]["age_minutes"] >= 60

    def test_missing_beat_is_tolerated(self, sqlite_store):
        """一度も記録が無い（未使用・起動直後）は発火しない。"""
        assert _check_scheduler_heartbeat(sqlite_store) == []

    def test_corrupt_beat_fires(self, sqlite_store):
        """時刻として読めない heartbeat は記録破損として発火する。"""
        sqlite_store.set_setting("scheduler_heartbeat_forget", "こわれてる")
        fired = _check_scheduler_heartbeat(sqlite_store)
        assert len(fired) == 1
        assert fired[0]["scheduler"] == "forget"


class TestBuildForecast:
    """build_forecast の一括計算を検証するテストクラス。

    すべて決定論純関数の合成なので、LLM モックは不要（呼ばれないこと自体が仕様）。
    datetime はすべて ISO 文字列で返る（テンプレートの JSON 埋め込み前提）。
    """

    def test_character_not_found(self, sqlite_store):
        """存在しないキャラは error を返す（例外にしない — パネルは常に開ける）。"""
        result = build_forecast(sqlite_store, "no-such-id")
        assert "error" in result

    def test_full_structure(self, sqlite_store):
        """全セクションが揃い、格子と系列の長さが一致する。"""
        now = datetime(2026, 7, 10, 12, 0)
        char_id, name = _make_character(
            sqlite_store, action_menu={"push": True},
        )
        _make_hot_intent(sqlite_store, char_id, now=now)
        sqlite_store.set_setting(
            "scheduler_heartbeat_action", now.isoformat(),
        )
        sqlite_store.record_scheduler_decision(
            "action", "fired", character_id=char_id, reason="push 実行", occurred_at=now,
        )

        fc = build_forecast(sqlite_store, char_id, now=now, horizon_hours=24)

        assert fc["character"]["name"] == name
        assert fc["now"] == now.isoformat()

        # 診断ヘッダ
        diag = fc["diagnosis"]
        assert diag["availability"]["available"] is True
        assert diag["urge"]["intents"][0]["pressure"] > 0.7
        beats = {b["scheduler"]: b for b in diag["heartbeats"]}
        assert beats["action"]["at"] is not None
        assert any(c["label"] == "行動問い合わせ" for c in diag["caps"])
        # 配達シム: 従来経路・常時 available → ジッター（≤10分）内に配達される
        sim = diag["delivery_sim"]
        assert sim["mode"] == "legacy"
        assert sim["delivered_at"] is not None
        assert sim["wait_minutes"] <= 15

        # カレンダー: 7日分・行動権スロットは 12/日 × 7日
        cal = fc["calendar"]
        assert len(cal["days"]) == 7
        assert len(cal["action_slots"]) == 12 * 7
        assert all(
            s["forecast"] in ("past", "fires", "quiet", "unavailable")
            for s in cal["action_slots"]
        )
        # now 以降のスロットに予報が付いている（このキャラは常時 available ＋
        # 熱い意図があるので fires が存在するはず）
        future = [s for s in cal["action_slots"] if s["forecast"] != "past"]
        assert any(s["forecast"] == "fires" for s in future)

        # 圧力予報: 格子と全系列の長さが一致（24h × 30分格子 = 49点）
        pf = fc["pressure_forecast"]
        assert len(pf["grid"]) == 49
        assert len(pf["social"]) == len(pf["grid"])
        assert len(pf["intents"][0]["series"]) == len(pf["grid"])
        assert pf["threshold"] == 0.7
        # 封筒ゼロ＋高圧意図＋常時 available なので問い合わせ予報点がある
        assert len(pf["fire_points"]) > 0

        # 決定ログと揺れ監査
        assert fc["decisions"][0]["reason"] == "push 実行"
        assert len(fc["variance"]["fired_scatter"]) == 1
        rhythm = fc["variance"]["rhythm"]
        assert len(rhythm["grid"]) == len(rhythm["values"])

    def test_windless_extrapolation_is_pure(self, sqlite_store):
        """同じ入力なら予報は完全に再現される（決定論）。"""
        char_id, _ = _make_character(sqlite_store)
        now = datetime(2026, 7, 10, 9, 30)
        a = build_forecast(sqlite_store, char_id, now=now, horizon_hours=6)
        b = build_forecast(sqlite_store, char_id, now=now, horizon_hours=6)
        assert a == b

    def test_living_schedule_calendar(self, sqlite_store):
        """生活カレンダー有効キャラはエントリ・伏せ枠がカレンダーに載る。"""
        char_id, _ = _make_character(
            sqlite_store, living_schedule_enabled=1,
        )
        now = datetime(2026, 7, 10, 8, 0)
        sqlite_store.create_schedule_entry(
            character_id=char_id,
            start_at=now.replace(hour=10), end_at=now.replace(hour=12),
            state="busy", source="world", origin="template",
            occupancy=0.8, status="planned", label="仕事",
        )
        sqlite_store.create_schedule_entry(
            character_id=char_id,
            start_at=now.replace(hour=15), end_at=now.replace(hour=16),
            state="busy", source="world", origin="adhoc",
            occupancy=0.6, status="pending", label="（伏せ）",
            payload={"kind": "sudden_event_seed", "category": "偶発"},
        )
        fc = build_forecast(sqlite_store, char_id, now=now, horizon_hours=6)
        cal = fc["calendar"]
        labels = {e["label"]: e for e in cal["entries"]}
        assert labels["仕事"]["is_seed"] is False
        assert labels["（伏せ）"]["is_seed"] is True  # 全開示 — 伏せ枠も見せる
