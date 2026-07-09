"""生活カレンダー（Living Schedule）実現層 CRUD — SQLiteStore Mixin。

schedule_entries テーブル（実現層・1回性の予定インスタンス）への追加・読み出し・
ステータス更新・削除を担う層（docs/planned/schedule_plan.md §2）。

設計上の要点:
    - エントリは重なりを許容し、物理分割・削除はしない。③④は insert するだけで、
      重なりの解決（占有圧最大が勝つ）は読み取り側（services/gate/availability.py）が行う。
    - 玉突き裁定（§6）で「諦める／ずらす」が起きたら status を planned→cancelled/done に
      変え、必要なら別エントリを再 insert する（本人の④として表現）。
    - 週次バッチ（Phase 3）の再生成は「対象週の template エントリを削除して入れ直す」
      流儀を取れるよう、削除は id 指定と (character_id, 期間, origin) 指定の両方を用意する。
"""

import uuid
from datetime import datetime


class ScheduleStoreMixin:
    """schedule_entries テーブルへの CRUD を提供する Mixin。"""

    # --- 書き込み ---

    def create_schedule_entry(
        self,
        *,
        character_id: str,
        start_at: datetime,
        end_at: datetime,
        state: str = "active",
        source: str = "haru",
        origin: str = "template",
        occupancy: float = 0.5,
        reply_rate: float | None = None,
        check_interval: int | None = None,
        status: str = "planned",
        label: str | None = None,
        payload: dict | None = None,
        entry_id: str | None = None,
    ):
        """実現層に予定エントリを1件追加する。

        Args:
            character_id: 誰の生活カレンダーか（characters.id）。
            start_at: 予定の開始時刻。
            end_at: 予定の終了時刻。
            state: 配達プリセット状態（OnTime / active / busy / offline）。
            source: author（world / haru）。
            origin: 固定由来 template / 随時挿入 adhoc。
            occupancy: 占有圧 0.0–1.0（§4）。
            reply_rate: 返信率の個別上書き（None = state プリセット既定）。
            check_interval: チェック間隔 [分] の個別上書き（None = state プリセット既定）。
            status: planned / cancelled / done。
            label: 表示ラベル。
            payload: 型ごとの可変属性 dict。
            entry_id: 明示指定する ID（冪等 upsert 用途）。None なら UUID を採番。

        Returns:
            作成した ScheduleEntry の ORM オブジェクト。
        """
        from backend.repositories.sqlite.models import ScheduleEntry

        with self.get_session() as session:
            entry = ScheduleEntry(
                id=entry_id or str(uuid.uuid4()),
                character_id=character_id,
                start_at=start_at,
                end_at=end_at,
                state=state,
                source=source,
                origin=origin,
                occupancy=occupancy,
                reply_rate=reply_rate,
                check_interval=check_interval,
                status=status,
                label=label,
                payload=payload,
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry

    def set_schedule_entry_status(self, entry_id: str, status: str) -> bool:
        """エントリの status を更新する（玉突き裁定の cancelled/done 反映）。

        Args:
            entry_id: 対象エントリ ID。
            status: 新しい status（planned / cancelled / done）。

        Returns:
            更新できたら True、対象が無ければ False。
        """
        from backend.repositories.sqlite.models import ScheduleEntry

        with self.get_session() as session:
            entry = session.get(ScheduleEntry, entry_id)
            if entry is None:
                return False
            entry.status = status
            entry.updated_at = datetime.now()
            session.commit()
            return True

    # --- 読み出し ---

    def list_schedule_entries(
        self,
        character_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        statuses: list[str] | None = None,
        origins: list[str] | None = None,
    ) -> list:
        """キャラクターの予定エントリを開始時刻昇順で返す。

        期間フィルタは「エントリが [since, until) と重なるか」で判定する
        （start_at < until かつ end_at > since）。start_at だけで切ると、期間を跨ぐ
        長い予定（就寝・仕事）が since 直前開始だと取り逃すため。

        Args:
            character_id: 対象キャラクター ID。
            since: この時刻以降に重なるエントリ（inclusive 側）。None なら下限なし。
            until: この時刻より前に重なるエントリ（exclusive 側）。None なら上限なし。
            statuses: status フィルタ（["planned"] 等）。None なら全 status。
            origins: origin フィルタ（["template"] 等）。None なら全 origin。

        Returns:
            ScheduleEntry の ORM オブジェクトリスト（start_at 昇順）。
        """
        from backend.repositories.sqlite.models import ScheduleEntry

        with self.get_session() as session:
            q = session.query(ScheduleEntry).filter(
                ScheduleEntry.character_id == character_id
            )
            if until is not None:
                q = q.filter(ScheduleEntry.start_at < until)
            if since is not None:
                q = q.filter(ScheduleEntry.end_at > since)
            if statuses:
                q = q.filter(ScheduleEntry.status.in_(statuses))
            if origins:
                q = q.filter(ScheduleEntry.origin.in_(origins))
            return q.order_by(
                ScheduleEntry.start_at.asc(), ScheduleEntry.created_at.asc()
            ).all()

    def get_active_schedule_entries(
        self, character_id: str, now: datetime
    ) -> list:
        """now を含む planned エントリを返す（availability 純関数の入力）。

        条件: start_at <= now < end_at かつ status = "planned"。占有圧最大の解決は
        呼び出し側（services/gate/availability.py）が行うため、ここでは絞り込みだけする。

        Args:
            character_id: 対象キャラクター ID。
            now: 基準時刻。

        Returns:
            now を含む planned な ScheduleEntry のリスト（占有圧降順）。
        """
        from backend.repositories.sqlite.models import ScheduleEntry

        with self.get_session() as session:
            return (
                session.query(ScheduleEntry)
                .filter(
                    ScheduleEntry.character_id == character_id,
                    ScheduleEntry.status == "planned",
                    ScheduleEntry.start_at <= now,
                    ScheduleEntry.end_at > now,
                )
                .order_by(ScheduleEntry.occupancy.desc())
                .all()
            )

    # --- 削除 ---

    def delete_schedule_entries(
        self,
        *,
        entry_ids: list[str] | None = None,
        character_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        origins: list[str] | None = None,
        statuses: list[str] | None = None,
    ) -> int:
        """エントリを物理削除する（週次バッチの再生成・掃除用）。

        id 指定と (character_id, 期間, origin, status) 指定の両方を受ける。両方を渡した
        場合は AND で絞り込む。character_id も entry_ids も無い呼び出しは、全キャラの一括
        削除事故を避けるため何もしない。

        Args:
            entry_ids: 削除対象 ID のリスト。
            character_id: このキャラのエントリに限定。
            since: この時刻以降に開始するエントリに限定（start_at >= since）。
            until: この時刻より前に開始するエントリに限定（start_at < until）。
            origins: origin フィルタ（["template"] 等）。
            statuses: status フィルタ（["pending"] 等。③伏せ枠の再配置で未発火のみ消す用途）。

        Returns:
            削除した件数。
        """
        from backend.repositories.sqlite.models import ScheduleEntry

        if not entry_ids and character_id is None:
            return 0  # 無条件全削除の事故防止
        with self.get_session() as session:
            q = session.query(ScheduleEntry)
            if entry_ids:
                q = q.filter(ScheduleEntry.id.in_(entry_ids))
            if character_id is not None:
                q = q.filter(ScheduleEntry.character_id == character_id)
            if since is not None:
                q = q.filter(ScheduleEntry.start_at >= since)
            if until is not None:
                q = q.filter(ScheduleEntry.start_at < until)
            if origins:
                q = q.filter(ScheduleEntry.origin.in_(origins))
            if statuses:
                q = q.filter(ScheduleEntry.status.in_(statuses))
            deleted = q.delete(synchronize_session=False)
            session.commit()
            return deleted
