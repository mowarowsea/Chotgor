"""ダイジェストログ CRUD — SQLiteStore Mixin。"""

from typing import Optional


class DigestStoreMixin:
    """ダイジェスト実行履歴の記録・参照を担う Mixin。"""

    def has_digest(self, character_id: str, date_str: str) -> bool:
        """指定キャラクター・日付のダイジストログが存在するか返す。"""
        with self.get_session() as session:
            from ..sqlite_store import DigestLog
            count = (
                session.query(DigestLog)
                .filter(
                    DigestLog.character_id == character_id,
                    DigestLog.digest_date == date_str,
                )
                .count()
            )
            return count > 0

    def record_digest(
        self,
        character_id: str,
        date_str: str,
        status: str,
        memory_id: Optional[str] = None,
        memory_count: int = 0,
        message: Optional[str] = None,
    ) -> None:
        """ダイジストログを1行追記する。"""
        with self.get_session() as session:
            from ..sqlite_store import DigestLog
            log = DigestLog(
                character_id=character_id,
                digest_date=date_str,
                status=status,
                memory_id=memory_id,
                memory_count=memory_count,
                message=message,
            )
            session.add(log)
            session.commit()

    def get_digest_logs(self, character_id: str, limit: int = 50) -> list:
        """キャラクターのダイジストログを新しい順で返す。"""
        with self.get_session() as session:
            from ..sqlite_store import DigestLog
            return (
                session.query(DigestLog)
                .filter(DigestLog.character_id == character_id)
                .order_by(DigestLog.created_at.desc())
                .limit(limit)
                .all()
            )
