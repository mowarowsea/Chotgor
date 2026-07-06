"""意図（intents）CRUD — SQLiteStore Mixin。

めぐり（巡り / Aliveness）の動機経済・意図レコードの永続化層
（docs/aliveness_plan.md §4.3）。

タイムライン封筒との関係:
    - 作成時に intent.created、遷移時に intent.fulfilled / expired / soured の
      封筒を **同一トランザクション** で直書きする（遷移だけがイベント。
      意図圧の増減は連続量なので封筒に載せない）。
    - 封筒の intent_id 列がこのテーブルへの FK になる。
"""

import uuid
from datetime import datetime

# 意図の終端状態（active 以外）。遷移はこのいずれかへ一度だけ起きる。
_TERMINAL_STATUSES = ("fulfilled", "expired", "soured")


class IntentStoreMixin:
    """intents テーブルへの CRUD と封筒 dual-write を提供する Mixin。"""

    def create_intent(
        self,
        character_id: str,
        description: str,
        *,
        target: str | None = None,
        source_kind: str = "none",
        born_from: str = "night_chronicle",
        payload: dict | None = None,
    ):
        """意図を作成し、intent.created 封筒を同一トランザクションで残す。

        Args:
            character_id: 意図の持ち主。
            description: 本人の言葉のままの「〜したい」（丸めない）。
            target: user / npc:<名前> / self / None。
            source_kind: 源になった圧（social / boredom / body / none）。
            born_from: 拾い上げ地点（night_chronicle / usual_scene）。
            payload: 補助情報。

        Returns:
            作成された Intent。
        """
        from backend.repositories.sqlite.models import Intent

        intent_id = str(uuid.uuid4())
        with self.get_session() as session:
            intent = Intent(
                id=intent_id,
                character_id=character_id,
                description=description,
                target=target,
                status="active",
                source_kind=source_kind,
                born_from=born_from,
                payload=payload,
            )
            session.add(intent)
            self._append_timeline_event(
                session,
                character_id=character_id,
                event_type="intent.created",
                actor="character",
                counterpart=target if target and target != "self" else None,
                origin="real",
                source_table="intents",
                source_id=intent_id,
                intent_id=intent_id,
            )
            session.commit()
            session.refresh(intent)
            return intent

    def get_intent(self, intent_id: str):
        """ID で意図を取得する。"""
        from backend.repositories.sqlite.models import Intent

        with self.get_session() as session:
            return session.get(Intent, intent_id)

    def list_intents(
        self,
        character_id: str,
        *,
        status: str | None = "active",
        limit: int = 100,
    ) -> list:
        """キャラクターの意図一覧を新しい順で返す。

        Args:
            character_id: 対象キャラクター。
            status: 状態フィルタ（既定 "active"）。None なら全状態。
            limit: 最大件数。

        Returns:
            Intent の ORM オブジェクトリスト（created_at 降順）。
        """
        from backend.repositories.sqlite.models import Intent

        with self.get_session() as session:
            q = session.query(Intent).filter(Intent.character_id == character_id)
            if status is not None:
                q = q.filter(Intent.status == status)
            return q.order_by(Intent.created_at.desc()).limit(limit).all()

    def resolve_intent(
        self,
        intent_id: str,
        status: str,
        *,
        words: str | None = None,
    ):
        """意図を終端状態へ遷移させ、対応する封筒を同一トランザクションで残す。

        遷移は本人の裁定によってのみ起きる（機械は候補を挙げるだけ）。
        soured の場合は本人が言語化した不満の言葉を payload に凍結する
        （記憶への刻み込みは呼び出し側 = pickup 層の責務）。

        Args:
            intent_id: 対象意図 ID。
            status: "fulfilled" / "expired" / "soured"。
            words: 遷移にあたっての本人の言葉（soured の不満など）。

        Returns:
            更新後の Intent。存在しない・すでに終端済みなら None。
        """
        from backend.repositories.sqlite.models import Intent

        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"不正な遷移先: {status}")
        with self.get_session() as session:
            intent = session.get(Intent, intent_id)
            if intent is None or intent.status != "active":
                return None
            intent.status = status
            intent.resolved_at = datetime.now()
            if words:
                payload = dict(intent.payload or {})
                payload["resolution_words"] = words
                intent.payload = payload
            self._append_timeline_event(
                session,
                character_id=intent.character_id,
                event_type=f"intent.{status}",
                actor="character",
                counterpart=(
                    intent.target if intent.target and intent.target != "self" else None
                ),
                origin="real",
                source_table="intents",
                source_id=intent_id,
                intent_id=intent_id,
                payload={"words": words} if words else None,
            )
            session.commit()
            session.refresh(intent)
            return intent
