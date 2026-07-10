"""スケジューラ決定ログの永続化 — SQLiteStore Mixin。

予報パネル（docs/planned/forecast_panel_plan.md）のデータ口。
無人機構（行動権・うつつ・突発・escrow配達・週次バッチ）の「評価1回ごとの
結果と理由」を追記型で記録し、『正常な沈黙』と『壊れた沈黙』を区別可能にする。
記録の呼び出しは main.py の各スケジューラと services 側にあり、ここは読み書きだけを担う。
"""

from datetime import datetime


class DecisionStoreMixin:
    """scheduler_decisions テーブルへの追記・読み出しを提供する Mixin。"""

    def record_scheduler_decision(
        self,
        scheduler: str,
        outcome: str,
        *,
        character_id: str | None = None,
        reason: str | None = None,
        details: dict | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        """決定ログを1行追記する。

        記録は決定と理由だけ — 本文・シーン内容は既存テーブルが正本のまま。
        記録失敗で本流（スケジューラ）を止めないため、例外は握らず呼び出し側の
        try に委ねる（スケジューラループは既に外側で exception を捕捉している）。

        Args:
            scheduler: 機構名（action / usual_days / sudden_event /
                escrow_delivery / weekly_schedule）。
            outcome: "fired"（発火・実行）/ "declined"（本人が見送り）/
                "skipped"（物理で流れた）/ "error"。
            character_id: 対象キャラ（全体系の決定は None）。
            reason: 不発・見送りの短い理由（「閾値未達 0.52/0.7」等）。
            details: 評価文脈（圧力値・候補意図・スロット時刻など）。
            occurred_at: 決定時刻。None なら現在時刻。
        """
        from backend.repositories.sqlite.models import SchedulerDecision

        with self.get_session() as session:
            session.add(SchedulerDecision(
                character_id=character_id,
                scheduler=scheduler,
                occurred_at=occurred_at or datetime.now(),
                outcome=outcome,
                reason=reason,
                details=details,
            ))
            session.commit()

    def list_scheduler_decisions(
        self,
        *,
        character_id: str | None = None,
        scheduler: str | None = None,
        outcome: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list:
        """決定ログを新しい順で返す（予報パネルの決定タイムライン用）。

        Args:
            character_id: 特定キャラに絞る。
            scheduler: 特定機構に絞る。
            outcome: 特定結果に絞る。
            since: この時刻以降のみ。
            limit: 最大件数。

        Returns:
            SchedulerDecision ORM オブジェクトのリスト（occurred_at 降順）。
        """
        from backend.repositories.sqlite.models import SchedulerDecision

        with self.get_session() as session:
            q = session.query(SchedulerDecision)
            if character_id:
                q = q.filter(SchedulerDecision.character_id == character_id)
            if scheduler:
                q = q.filter(SchedulerDecision.scheduler == scheduler)
            if outcome:
                q = q.filter(SchedulerDecision.outcome == outcome)
            if since is not None:
                q = q.filter(SchedulerDecision.occurred_at >= since)
            return (
                q.order_by(
                    SchedulerDecision.occurred_at.desc(), SchedulerDecision.id.desc()
                )
                .limit(limit)
                .all()
            )

    def latest_scheduler_decisions(self) -> dict:
        """機構ごとの最新決定を返す（heartbeat 表示・Tier 1 計器の材料）。

        キャラを問わず「その機構が最後に評価を行った1件」を機構別に集める。
        沈黙している機構は辞書にキー自体が現れない（一度も評価していない）か、
        occurred_at が古い（評価が止まっている）かで見分けられる。

        Returns:
            {scheduler名: SchedulerDecision} の辞書。
        """
        from backend.repositories.sqlite.models import SchedulerDecision

        with self.get_session() as session:
            rows = (
                session.query(SchedulerDecision)
                .order_by(
                    SchedulerDecision.occurred_at.desc(), SchedulerDecision.id.desc()
                )
                .limit(500)
                .all()
            )
        latest: dict = {}
        for row in rows:
            if row.scheduler not in latest:
                latest[row.scheduler] = row
        return latest
