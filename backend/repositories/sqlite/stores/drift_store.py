"""SELF_DRIFT CRUD — SQLiteStore Mixin。"""

import uuid as _uuid_module


class DriftStoreMixin:
    """SessionDrift（SELF_DRIFT指針）の作成・取得・更新・削除を担う Mixin。"""

    def add_session_drift(self, session_id: str, character_id: str, content: str):
        """SELF_DRIFT指針を追加する。同キャラの上限3件を超えた場合は最古を削除してから追加する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import SessionDrift
            existing = (
                session.query(SessionDrift)
                .filter(
                    SessionDrift.session_id == session_id,
                    SessionDrift.character_id == character_id,
                )
                .order_by(SessionDrift.created_at.asc())
                .all()
            )
            while len(existing) >= 3:
                oldest = existing.pop(0)
                session.delete(oldest)
            drift = SessionDrift(
                id=str(_uuid_module.uuid4()),
                session_id=session_id,
                character_id=character_id,
                content=content,
            )
            session.add(drift)
            session.commit()
            session.refresh(drift)
            return drift

    def list_session_drifts(self, session_id: str) -> list:
        """セッションの全キャラのdrift一覧を作成日時順で返す（UI表示用）。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import SessionDrift
            return (
                session.query(SessionDrift)
                .filter(SessionDrift.session_id == session_id)
                .order_by(SessionDrift.created_at.asc())
                .all()
            )

    def list_active_session_drifts(self, session_id: str, character_id: str) -> list[str]:
        """指定キャラの有効（enabled=1）なdrift内容テキスト一覧を返す（システムプロンプト注入用）。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import SessionDrift
            rows = (
                session.query(SessionDrift)
                .filter(
                    SessionDrift.session_id == session_id,
                    SessionDrift.character_id == character_id,
                    SessionDrift.enabled == 1,
                )
                .order_by(SessionDrift.created_at.asc())
                .all()
            )
            return [r.content for r in rows]

    def toggle_session_drift(self, drift_id: str):
        """drift の enabled フラグを反転する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import SessionDrift
            drift = session.get(SessionDrift, drift_id)
            if not drift:
                return None
            drift.enabled = 0 if drift.enabled else 1
            session.commit()
            session.refresh(drift)
            return drift

    def reset_session_drifts(self, session_id: str, character_id: str) -> int:
        """指定キャラのdriftを全件物理削除する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import SessionDrift
            deleted = (
                session.query(SessionDrift)
                .filter(
                    SessionDrift.session_id == session_id,
                    SessionDrift.character_id == character_id,
                )
                .delete()
            )
            session.commit()
            return deleted
