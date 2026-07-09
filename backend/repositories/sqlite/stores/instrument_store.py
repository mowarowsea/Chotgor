"""計器（アラーム・メーター）の永続化 — SQLiteStore Mixin。

めぐり（巡り / Aliveness）の計器層（docs/planned/aliveness_plan.md §3）のデータ口。
アラーム（追記型・調査対象）とメータースナップショット（傾向・発火概念なし）を扱う。
検知ロジック自体は services/instruments/ にあり、ここは読み書きだけを担う。
"""

from datetime import datetime, timedelta


class InstrumentStoreMixin:
    """alarms / meter_snapshots テーブルへの CRUD を提供する Mixin。"""

    # --- アラーム ---

    def fire_alarm(
        self,
        invariant_id: str,
        *,
        severity: str = "alarm",
        details: dict | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        """アラームを1行追記する。

        Args:
            invariant_id: 発火したインバリアント/検知器の ID
                （fabrication_backstop / usual_scene_error / smell_format_debris 等）。
            severity: "alarm"（調査対象・静音期間の計算対象）または
                "smell"（Tier 2 の疑い記録・誤検知許容）。
            details: 発火文脈（キャラ名・対象 ID・検知内容など）。
            occurred_at: 発生時刻。None なら現在時刻。
        """
        from backend.repositories.sqlite.models import Alarm

        with self.get_session() as session:
            session.add(Alarm(
                invariant_id=invariant_id,
                severity=severity,
                occurred_at=occurred_at or datetime.now(),
                details=details,
            ))
            session.commit()

    def list_alarms(
        self,
        *,
        severity: str | None = None,
        invariant_id: str | None = None,
        unacknowledged_only: bool = False,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list:
        """アラーム一覧を新しい順で返す（計器パネル・調査用）。

        Args:
            severity: "alarm" / "smell" のフィルタ。None なら両方。
            invariant_id: 特定インバリアントに絞る。
            unacknowledged_only: True なら未確認（acknowledged_at IS NULL）のみ。
            since: この時刻以降のみ。
            limit: 最大件数。

        Returns:
            Alarm ORM オブジェクトのリスト（occurred_at 降順）。
        """
        from backend.repositories.sqlite.models import Alarm

        with self.get_session() as session:
            q = session.query(Alarm)
            if severity:
                q = q.filter(Alarm.severity == severity)
            if invariant_id:
                q = q.filter(Alarm.invariant_id == invariant_id)
            if unacknowledged_only:
                q = q.filter(Alarm.acknowledged_at.is_(None))
            if since is not None:
                q = q.filter(Alarm.occurred_at >= since)
            return q.order_by(Alarm.occurred_at.desc(), Alarm.id.desc()).limit(limit).all()

    def acknowledge_alarm(self, alarm_id: int) -> bool:
        """アラームを確認済みにする（acknowledged_at をセット）。

        Args:
            alarm_id: 対象アラーム ID。

        Returns:
            更新できたら True、存在しなければ False。
        """
        from backend.repositories.sqlite.models import Alarm

        with self.get_session() as session:
            alarm = session.get(Alarm, alarm_id)
            if not alarm:
                return False
            alarm.acknowledged_at = datetime.now()
            session.commit()
            return True

    def quiet_period_days(self) -> int | None:
        """静音期間（無事故N日）を返す。

        「既知の事故クラスすべてで無事故N日」の N。最後の severity="alarm" 発生からの
        経過日数。アラームが一度も無い場合は計器稼働開始
        （global_settings "instruments_started_at"）からの経過日数。
        稼働開始も記録されていなければ None（計器未稼働）。

        smell は誤検知許容の疑い記録なので静音期間を壊さない。

        Returns:
            無事故日数（int）。計器未稼働なら None。
        """
        from backend.repositories.sqlite.models import Alarm

        with self.get_session() as session:
            last = (
                session.query(Alarm.occurred_at)
                .filter(Alarm.severity == "alarm")
                .order_by(Alarm.occurred_at.desc())
                .first()
            )
        if last is not None:
            return max(0, (datetime.now() - last[0]).days)
        started = self.get_setting("instruments_started_at", "")
        if not started:
            return None
        try:
            started_dt = datetime.fromisoformat(started)
        except ValueError:
            return None
        return max(0, (datetime.now() - started_dt).days)

    # --- メーター ---

    def record_meter(
        self,
        meter_id: str,
        value: float,
        *,
        character_id: str | None = None,
        details: dict | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        """メータースナップショットを1行追記する。

        Args:
            meter_id: メーター ID（inner_narrative_len / pressure_social 等）。
            value: 計測値。
            character_id: 対象キャラ（全体系メーターは None）。
            details: 補助情報。
            occurred_at: 計測時刻。None なら現在時刻。
        """
        from backend.repositories.sqlite.models import MeterSnapshot

        with self.get_session() as session:
            session.add(MeterSnapshot(
                meter_id=meter_id,
                character_id=character_id,
                value=float(value),
                occurred_at=occurred_at or datetime.now(),
                details=details,
            ))
            session.commit()

    def list_meter_snapshots(
        self,
        *,
        meter_id: str | None = None,
        character_id: str | None = None,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list:
        """メータースナップショットを新しい順で返す（計器パネルの傾向表示用）。"""
        from backend.repositories.sqlite.models import MeterSnapshot

        with self.get_session() as session:
            q = session.query(MeterSnapshot)
            if meter_id:
                q = q.filter(MeterSnapshot.meter_id == meter_id)
            if character_id:
                q = q.filter(MeterSnapshot.character_id == character_id)
            if since is not None:
                q = q.filter(MeterSnapshot.occurred_at >= since)
            return (
                q.order_by(MeterSnapshot.occurred_at.desc(), MeterSnapshot.id.desc())
                .limit(limit)
                .all()
            )

    # --- 巡回チェック用の突合クエリ ---

    def count_chronicle_backlog(self, days: int = 3) -> dict:
        """蒸留漏れ（chronicled_at IS NULL の滞留）を数える。

        計器 Tier 1 `chronicle_backlog` の材料。指定日数より古いのに未処理の
        1on1 メッセージとうつつターンの件数を返す。

        Args:
            days: 滞留とみなす日数（既定3日）。

        Returns:
            {"chat_messages": int, "usual_turns": int} の辞書。
        """
        from backend.repositories.sqlite.models import (
            ChatMessage,
            ScenarioSession,
            ScenarioTurn,
        )

        cutoff = datetime.now() - timedelta(days=days)
        with self.get_session() as session:
            chat_count = (
                session.query(ChatMessage)
                .filter(
                    ChatMessage.chronicled_at.is_(None),
                    ChatMessage.created_at < cutoff,
                    (ChatMessage.is_system_message.is_(None))
                    | (ChatMessage.is_system_message == 0),
                )
                .count()
            )
            usual_count = (
                session.query(ScenarioTurn)
                .join(ScenarioSession, ScenarioTurn.session_id == ScenarioSession.id)
                .filter(
                    ScenarioSession.engine_type == "usual_days",
                    ScenarioTurn.chronicled_at.is_(None),
                    ScenarioTurn.created_at < cutoff,
                )
                .count()
            )
        return {"chat_messages": chat_count, "usual_turns": usual_count}

    def envelope_integrity_counts(self) -> dict:
        """正本性の突合材料 — 源テーブルと封筒の件数ペアを返す。

        計器 Tier 1 `envelope_integrity` の材料。ID 突合はしない（件数のみ）。
        封筒は retracted 含む全件（削除ではなくマークだから、源が消えても封筒は残る =
        封筒件数 >= 源件数 が正常。封筒件数 < 源件数 なら dual-write 漏れ）。

        突合対象は封筒の character 解決が決定的な2テーブル:
            - inscribed_memories（character_id 直付き）
            - chat_messages（model_id からの解決。ただしキャラ未解決の行は封筒を
              作らない仕様のため、突合はキャラ解決可能な行に限定する）

        Returns:
            {"inscribed_memories": {"source": n, "envelope": m},
             "chat_messages": {"source": n, "envelope": m}} の辞書。
        """
        from backend.repositories.sqlite.models import (
            Character,
            ChatMessage,
            ChatSession,
            InscribedMemory,
            TimelineEvent,
        )

        with self.get_session() as session:
            mem_source = session.query(InscribedMemory).count()
            mem_envelope = (
                session.query(TimelineEvent)
                .filter(
                    TimelineEvent.source_table == "inscribed_memories",
                    TimelineEvent.event_type == "memory.inscribed",
                )
                .count()
            )
            # キャラ解決可能な非システムメッセージ = 封筒が作られているべき行
            char_names = [
                row[0] for row in session.query(Character.name).all() if row[0]
            ]
            chat_source = 0
            if char_names:
                from sqlalchemy import or_
                chat_source = (
                    session.query(ChatMessage)
                    .join(ChatSession, ChatMessage.session_id == ChatSession.id)
                    .filter(
                        (ChatMessage.is_system_message.is_(None))
                        | (ChatMessage.is_system_message == 0),
                        or_(*[
                            ChatSession.model_id.like(f"{name}@%")
                            for name in char_names
                        ]),
                    )
                    .count()
                )
            chat_envelope = (
                session.query(TimelineEvent)
                .filter(
                    TimelineEvent.source_table == "chat_messages",
                    TimelineEvent.event_type == "chat.message",
                )
                .count()
            )
        return {
            "inscribed_memories": {"source": mem_source, "envelope": mem_envelope},
            "chat_messages": {"source": chat_source, "envelope": chat_envelope},
        }

    def sample_today_responses(self, limit: int = 10) -> list:
        """Tier 3 判定巡回のサンプル — 当日のキャラ応答ログを最大 limit 件返す。

        debug_log_entries から source_type が chat / scenario でエラーなし・
        応答本文ありの当日行を新しい順に集め、Python 側の乱択に使えるよう
        limit の3倍まで返す（呼び出し側でサンプリングする）。

        Args:
            limit: 判定対象のサンプル数（返す行数はその3倍まで）。

        Returns:
            DebugLogEntry ORM オブジェクトのリスト。
        """
        from backend.repositories.sqlite.models import DebugLogEntry

        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        with self.get_session() as session:
            return (
                session.query(DebugLogEntry)
                .filter(
                    DebugLogEntry.created_at >= today_start,
                    DebugLogEntry.source_type.in_(["chat", "scenario"]),
                    DebugLogEntry.has_error == 0,
                    DebugLogEntry.response.isnot(None),
                    DebugLogEntry.response != "",
                )
                .order_by(DebugLogEntry.created_at.desc())
                .limit(limit * 3)
                .all()
            )
