"""LLM 使用量イベントの記録と集計（UsageStoreMixin）。

usage_recorder（backend/lib/usage_recorder.py）が1 API 呼び出しごとに
add_llm_usage_event() で行を追加し、ダッシュボード（/ui/）が
日次・週次・直近イベントの集計クエリを使って表示する。
"""

from datetime import datetime, timedelta

from sqlalchemy import func


class UsageStoreMixin:
    """llm_usage_events テーブルへの CRUD・集計を提供する Mixin。"""

    def add_llm_usage_event(
        self,
        provider: str,
        model: str | None = None,
        preset_name: str | None = None,
        target: str | None = None,
        feature: str | None = None,
        request_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        total_cost_usd: float | None = None,
    ) -> None:
        """LLM 使用量イベントを1行追加する。"""
        from backend.repositories.sqlite.models import LlmUsageEvent

        with self.get_session() as session:
            session.add(LlmUsageEvent(
                provider=provider,
                model=model,
                preset_name=preset_name,
                target=target,
                feature=feature,
                request_id=request_id,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                cache_read_input_tokens=int(cache_read_input_tokens or 0),
                cache_creation_input_tokens=int(cache_creation_input_tokens or 0),
                total_cost_usd=total_cost_usd,
            ))
            session.commit()

    def get_usage_totals_since(self, since: datetime) -> dict:
        """since 以降の合計（リクエスト数・トークン In/Out・概算コスト）を返す。

        Returns:
            {"requests": int, "input_tokens": int, "output_tokens": int, "cost_usd": float}
        """
        from backend.repositories.sqlite.models import LlmUsageEvent

        with self.get_session() as session:
            row = (
                session.query(
                    func.count(LlmUsageEvent.id),
                    func.coalesce(func.sum(LlmUsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(LlmUsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(LlmUsageEvent.total_cost_usd), 0.0),
                )
                .filter(LlmUsageEvent.created_at >= since)
                .one()
            )
            return {
                "requests": row[0],
                "input_tokens": row[1],
                "output_tokens": row[2],
                "cost_usd": float(row[3] or 0.0),
            }

    def get_usage_daily(self, days: int = 14) -> list[dict]:
        """直近 days 日の日次 × プロバイダー別集計を新しい日付順で返す。

        Returns:
            [{"day": "2026-06-11", "provider": "claude_cli", "requests": int,
              "input_tokens": int, "output_tokens": int, "cost_usd": float}, ...]
        """
        from backend.repositories.sqlite.models import LlmUsageEvent

        today = datetime.now()
        since = datetime(today.year, today.month, today.day) - timedelta(days=days - 1)
        day = func.strftime("%Y-%m-%d", LlmUsageEvent.created_at).label("day")
        with self.get_session() as session:
            rows = (
                session.query(
                    day,
                    LlmUsageEvent.provider,
                    func.count(LlmUsageEvent.id),
                    func.coalesce(func.sum(LlmUsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(LlmUsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(LlmUsageEvent.total_cost_usd), 0.0),
                )
                .filter(LlmUsageEvent.created_at >= since)
                .group_by(day, LlmUsageEvent.provider)
                .order_by(day.desc(), LlmUsageEvent.provider)
                .all()
            )
        return [
            {
                "day": r[0],
                "provider": r[1],
                "requests": r[2],
                "input_tokens": r[3],
                "output_tokens": r[4],
                "cost_usd": float(r[5] or 0.0),
            }
            for r in rows
        ]

    def get_usage_weekly(self, weeks: int = 8) -> list[dict]:
        """直近 weeks 週の週次 × プロバイダー別集計を新しい週順で返す。

        週は SQLite strftime('%Y-W%W')（月曜起点・年内通し番号）でラベル付けする。

        Returns:
            [{"week": "2026-W23", "provider": "google", "requests": int,
              "input_tokens": int, "output_tokens": int, "cost_usd": float}, ...]
        """
        from backend.repositories.sqlite.models import LlmUsageEvent

        since = datetime.now() - timedelta(weeks=weeks)
        week = func.strftime("%Y-W%W", LlmUsageEvent.created_at).label("week")
        with self.get_session() as session:
            rows = (
                session.query(
                    week,
                    LlmUsageEvent.provider,
                    func.count(LlmUsageEvent.id),
                    func.coalesce(func.sum(LlmUsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(LlmUsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(LlmUsageEvent.total_cost_usd), 0.0),
                )
                .filter(LlmUsageEvent.created_at >= since)
                .group_by(week, LlmUsageEvent.provider)
                .order_by(week.desc(), LlmUsageEvent.provider)
                .all()
            )
        return [
            {
                "week": r[0],
                "provider": r[1],
                "requests": r[2],
                "input_tokens": r[3],
                "output_tokens": r[4],
                "cost_usd": float(r[5] or 0.0),
            }
            for r in rows
        ]

    def get_usage_monthly(self, months: int = 6) -> list[dict]:
        """直近 months ヶ月の月次 × プロバイダー別集計を新しい月順で返す。

        週次（fixed-width な timedelta）と違い、月は長さが揃わないため
        「N ヶ月前の月初」を厳密に計算して since として使う。

        Returns:
            [{"month": "2026-06", "provider": "claude_cli", "requests": int,
              "input_tokens": int, "output_tokens": int, "cost_usd": float}, ...]
        """
        from backend.repositories.sqlite.models import LlmUsageEvent

        now = datetime.now()
        year = now.year
        month = now.month - (months - 1)
        while month <= 0:
            year -= 1
            month += 12
        since = datetime(year, month, 1)
        month_label = func.strftime("%Y-%m", LlmUsageEvent.created_at).label("month")
        with self.get_session() as session:
            rows = (
                session.query(
                    month_label,
                    LlmUsageEvent.provider,
                    func.count(LlmUsageEvent.id),
                    func.coalesce(func.sum(LlmUsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(LlmUsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(LlmUsageEvent.total_cost_usd), 0.0),
                )
                .filter(LlmUsageEvent.created_at >= since)
                .group_by(month_label, LlmUsageEvent.provider)
                .order_by(month_label.desc(), LlmUsageEvent.provider)
                .all()
            )
        return [
            {
                "month": r[0],
                "provider": r[1],
                "requests": r[2],
                "input_tokens": r[3],
                "output_tokens": r[4],
                "cost_usd": float(r[5] or 0.0),
            }
            for r in rows
        ]

    def get_usage_recent_events(self, limit: int = 30) -> list[dict]:
        """直近の使用量イベント（リクエストごとの生データ）を新しい順で返す。"""
        from backend.repositories.sqlite.models import LlmUsageEvent

        with self.get_session() as session:
            rows = (
                session.query(LlmUsageEvent)
                .order_by(LlmUsageEvent.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "created_at": r.created_at,
                    "provider": r.provider,
                    "model": r.model or "",
                    "preset_name": r.preset_name or "",
                    "target": r.target or "",
                    "feature": r.feature or "",
                    "request_id": r.request_id or "",
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_cost_usd": r.total_cost_usd,
                }
                for r in rows
            ]
