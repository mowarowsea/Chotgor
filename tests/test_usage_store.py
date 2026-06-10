"""Tests for UsageStoreMixin — llm_usage_events の記録と集計。"""

from datetime import datetime, timedelta

from backend.repositories.sqlite.models import LlmUsageEvent


def _backdate_latest(store, *, days: int = 0, weeks: int = 0) -> None:
    """直近に追加したイベントの created_at を過去へずらすテストヘルパー。

    add_llm_usage_event は created_at を常に now で記録するため、
    期間フィルタ・日次/週次バケットの検証にはここで日付を偽装する。
    """
    with store.get_session() as session:
        ev = session.query(LlmUsageEvent).order_by(LlmUsageEvent.id.desc()).first()
        ev.created_at = datetime.now() - timedelta(days=days, weeks=weeks)
        session.commit()


class TestAddAndRecentEvents:
    """add_llm_usage_event と get_usage_recent_events の往復を検証する。

    ダッシュボードの「直近リクエスト」表が表示する全フィールド
    （対象・feature・モデル・プリセット・トークン・コスト）が
    記録→読み出しで欠落しないことを守る。
    """

    def test_roundtrip_all_fields(self, sqlite_store):
        """全フィールドを指定して追加した行が recent_events でそのまま読めること。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli",
            model="claude-sonnet-4-6",
            preset_name="default",
            target="織羽",
            feature="chat",
            request_id="abcd1234",
            input_tokens=1200,
            output_tokens=340,
            cache_read_input_tokens=800,
            cache_creation_input_tokens=50,
            total_cost_usd=0.0123,
        )

        events = sqlite_store.get_usage_recent_events()

        assert len(events) == 1
        e = events[0]
        assert e["provider"] == "claude_cli"
        assert e["model"] == "claude-sonnet-4-6"
        assert e["preset_name"] == "default"
        assert e["target"] == "織羽"
        assert e["feature"] == "chat"
        assert e["request_id"] == "abcd1234"
        assert e["input_tokens"] == 1200
        assert e["output_tokens"] == 340
        assert e["total_cost_usd"] == 0.0123
        assert isinstance(e["created_at"], datetime)

    def test_optional_fields_default_to_empty(self, sqlite_store):
        """任意フィールド未指定（NULL）の行は表示用に空文字へ変換されること。"""
        sqlite_store.add_llm_usage_event(
            provider="google", input_tokens=10, output_tokens=5,
        )

        e = sqlite_store.get_usage_recent_events()[0]

        assert e["model"] == ""
        assert e["preset_name"] == ""
        assert e["target"] == ""
        assert e["feature"] == ""
        assert e["request_id"] == ""
        assert e["total_cost_usd"] is None

    def test_recent_events_newest_first_and_limited(self, sqlite_store):
        """直近イベントは新しい順で並び、limit 件数で打ち切られること。"""
        for i in range(5):
            sqlite_store.add_llm_usage_event(
                provider="claude_cli", request_id=f"req-{i}",
                input_tokens=1, output_tokens=1,
            )

        events = sqlite_store.get_usage_recent_events(limit=3)

        assert [e["request_id"] for e in events] == ["req-4", "req-3", "req-2"]


class TestUsageTotals:
    """get_usage_totals_since の合計と期間フィルタを検証する。

    ダッシュボードの「今日」「今週」カードの元になる集計のため、
    since 境界より古い行が混入しないことが重要。
    """

    def test_totals_aggregate_all_rows(self, sqlite_store):
        """since 以降の全行についてリクエスト数・トークン・コストが合算されること。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=100, output_tokens=20, total_cost_usd=0.01,
        )
        sqlite_store.add_llm_usage_event(
            provider="google", input_tokens=50, output_tokens=10,
        )

        totals = sqlite_store.get_usage_totals_since(datetime.now() - timedelta(hours=1))

        assert totals == {
            "requests": 2,
            "input_tokens": 150,
            "output_tokens": 30,
            "cost_usd": 0.01,
        }

    def test_totals_exclude_rows_before_since(self, sqlite_store):
        """since より古い行は集計に含まれないこと。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=999, output_tokens=999,
        )
        _backdate_latest(sqlite_store, days=2)
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=10, output_tokens=5,
        )

        totals = sqlite_store.get_usage_totals_since(datetime.now() - timedelta(hours=1))

        assert totals["requests"] == 1
        assert totals["input_tokens"] == 10
        assert totals["output_tokens"] == 5

    def test_totals_empty_table(self, sqlite_store):
        """行が1件も無いときは全てゼロが返ること（None にならない）。"""
        totals = sqlite_store.get_usage_totals_since(datetime.now() - timedelta(days=1))

        assert totals == {
            "requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        }


class TestUsageDaily:
    """get_usage_daily の日次 × プロバイダー別グルーピングを検証する。"""

    def test_groups_by_day_and_provider(self, sqlite_store):
        """同日・同プロバイダーの行が1バケットへ合算され、別プロバイダーは別行になること。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=100, output_tokens=10,
        )
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=200, output_tokens=20,
        )
        sqlite_store.add_llm_usage_event(
            provider="google", input_tokens=30, output_tokens=3,
        )

        daily = sqlite_store.get_usage_daily(days=14)

        today = datetime.now().strftime("%Y-%m-%d")
        assert {(d["day"], d["provider"]) for d in daily} == {
            (today, "claude_cli"), (today, "google"),
        }
        cli = next(d for d in daily if d["provider"] == "claude_cli")
        assert cli["requests"] == 2
        assert cli["input_tokens"] == 300
        assert cli["output_tokens"] == 30

    def test_excludes_days_outside_window(self, sqlite_store):
        """days で指定した窓より古い日の行は返らないこと。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=1, output_tokens=1,
        )
        _backdate_latest(sqlite_store, days=30)

        assert sqlite_store.get_usage_daily(days=14) == []


class TestUsageWeekly:
    """get_usage_weekly の週次 × プロバイダー別グルーピングを検証する。"""

    def test_groups_by_week(self, sqlite_store):
        """今週の行が strftime('%Y-W%W') ラベルのバケットへ合算されること。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=10, output_tokens=2,
        )
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=20, output_tokens=4,
        )

        weekly = sqlite_store.get_usage_weekly(weeks=8)

        this_week = datetime.now().strftime("%Y-W%W")
        assert len(weekly) == 1
        assert weekly[0]["week"] == this_week
        assert weekly[0]["requests"] == 2
        assert weekly[0]["input_tokens"] == 30
        assert weekly[0]["output_tokens"] == 6

    def test_excludes_weeks_outside_window(self, sqlite_store):
        """weeks で指定した窓より古い週の行は返らないこと。"""
        sqlite_store.add_llm_usage_event(
            provider="claude_cli", input_tokens=1, output_tokens=1,
        )
        _backdate_latest(sqlite_store, weeks=10)

        assert sqlite_store.get_usage_weekly(weeks=8) == []
