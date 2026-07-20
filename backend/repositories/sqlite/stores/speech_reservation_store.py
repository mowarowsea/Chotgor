"""発話予約（speak_later）CRUD — SQLiteStore Mixin。

speech_reservations テーブル（docs/planned/speak_later_plan.md）への
追加・読み出し・ステータス遷移を担う層。

設計上の要点:
    - pending は同一セッションに1件。置き直しは呼び出し側（LaterSpeaker）が
      旧行を superseded に倒してから新行を insert する（履歴は行として残る）。
    - 行は消さない — fired / expired / cancelled / superseded への遷移だけで表現する
      （ログ的データの肥大は許容する方針）。
"""

import uuid
from datetime import datetime


class SpeechReservationStoreMixin:
    """speech_reservations テーブルへの CRUD を提供する Mixin。"""

    def create_speech_reservation(
        self,
        *,
        character_id: str,
        session_id: str,
        speak_at: datetime,
        note: str,
        reservation_id: str | None = None,
    ):
        """発話予約を1件追加する（status=pending）。

        Args:
            character_id: 誰の予約か（characters.id）。
            session_id: 発話先の 1on1 セッション ID。
            speak_at: 発火予定時刻。
            note: 何を話そうとしているかの短いメモ（本人の言葉のまま）。
            reservation_id: 明示指定する ID。None なら UUID を採番。

        Returns:
            作成した SpeechReservation の ORM オブジェクト。
        """
        from backend.repositories.sqlite.models import SpeechReservation

        with self.get_session() as session:
            row = SpeechReservation(
                id=reservation_id or str(uuid.uuid4()),
                character_id=character_id,
                session_id=session_id,
                speak_at=speak_at,
                note=note,
                status="pending",
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def get_pending_speech_reservation(self, session_id: str):
        """セッションの pending 予約を返す（pending は1件/セッションの前提）。

        万一複数残っていても最新（created_at 降順の先頭）を返す。無ければ None。
        """
        from backend.repositories.sqlite.models import SpeechReservation

        with self.get_session() as session:
            return (
                session.query(SpeechReservation)
                .filter(
                    SpeechReservation.session_id == session_id,
                    SpeechReservation.status == "pending",
                )
                .order_by(SpeechReservation.created_at.desc())
                .first()
            )

    def list_pending_speech_reservations(
        self,
        *,
        character_id: str | None = None,
        due_before: datetime | None = None,
    ) -> list:
        """pending の予約を speak_at 昇順で返す。

        Args:
            character_id: このキャラの予約に限定（プロンプト注入用）。None なら全キャラ。
            due_before: speak_at がこの時刻以前のものに限定（発火スケジューラ用）。

        Returns:
            SpeechReservation ORM オブジェクトのリスト（speak_at 昇順）。
        """
        from backend.repositories.sqlite.models import SpeechReservation

        with self.get_session() as session:
            q = session.query(SpeechReservation).filter(
                SpeechReservation.status == "pending"
            )
            if character_id is not None:
                q = q.filter(SpeechReservation.character_id == character_id)
            if due_before is not None:
                q = q.filter(SpeechReservation.speak_at <= due_before)
            return q.order_by(
                SpeechReservation.speak_at.asc(), SpeechReservation.created_at.asc()
            ).all()

    def set_speech_reservation_status(
        self,
        reservation_id: str,
        status: str,
        *,
        fired_at: datetime | None = None,
    ) -> bool:
        """予約の status を遷移させる（fired / expired / cancelled / superseded）。

        Args:
            reservation_id: 対象予約 ID。
            status: 新しい status。
            fired_at: fired 遷移時の発火時刻（fired 以外では渡さない）。

        Returns:
            更新できたら True、対象が無ければ False。
        """
        from backend.repositories.sqlite.models import SpeechReservation

        with self.get_session() as session:
            row = session.get(SpeechReservation, reservation_id)
            if row is None:
                return False
            row.status = status
            if fired_at is not None:
                row.fired_at = fired_at
            session.commit()
            return True
