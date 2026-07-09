"""タイムライン封筒（timeline_events）CRUD — SQLiteStore Mixin。

めぐり（巡り / Aliveness）の正本テーブルへの追記・巻き戻しマーク・読み出しを担う層
（docs/planned/aliveness_plan.md §2）。

設計上の要点:
    - 封筒の追記は、中身の書き込みと **同一トランザクション** で行う
      （`_append_timeline_event` に呼び出し元の Session を渡す）。
      payload 完結型イベント（night.* / scene.closed / chat.farewell 等）は
      `record_timeline_event` が独立トランザクションで直書きする。
    - 巻き戻し（再生成・編集による履歴削除）は封筒を削除せず `retracted_at` を
      マークする（不可逆性の担保）。retracted 行は既定の読み出しから除外される。
    - 可視性の判定はここでは行わない（読み取り時ポリシー = services/timeline/projector）。
"""

import uuid
from datetime import datetime


class TimelineStoreMixin:
    """timeline_events テーブルへの追記・巻き戻し・読み出しを提供する Mixin。"""

    # --- 書き込み ---

    def _append_timeline_event(
        self,
        session,
        *,
        character_id: str,
        event_type: str,
        occurred_at: datetime | None = None,
        actor: str | None = None,
        counterpart: str | None = None,
        origin: str = "real",
        modality: str | None = None,
        session_id: str | None = None,
        source_table: str | None = None,
        source_id: str | None = None,
        intent_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        """開いている ORM Session に封筒1行を追加する（コミットは呼び出し元が行う）。

        中身テーブルへの書き込みと同一トランザクションで封筒を足すための内部口。
        store mixin の各書き込みメソッド（create_chat_message 等）から呼ばれる。

        Args:
            session: 呼び出し元が開いている SQLAlchemy Session。
            character_id: 誰のタイムラインか（characters.id）。
            event_type: イベントカタログのドット記法（例 "chat.message"）。
            occurred_at: 出来事の時刻。None なら現在時刻。
            actor: user / character / narrator / npc:<名前> / system。
            counterpart: 封筒の「相手」（user / npc:<名前> / None）。
            origin: real / usual / interlude。
            modality: text / face（chat.message のみ）。
            session_id: chat/scenario セッション ID（投影の封筒集約キー）。
            source_table: 中身テーブル名（payload 完結型は None）。
            source_id: 中身行の ID（payload 完結型は None）。
            intent_id: intent.* / action.* が張る intents.id。
            payload: 型ごとの可変属性 dict。
        """
        from backend.repositories.sqlite.models import TimelineEvent

        session.add(TimelineEvent(
            id=str(uuid.uuid4()),
            character_id=character_id,
            event_type=event_type,
            occurred_at=occurred_at or datetime.now(),
            actor=actor,
            counterpart=counterpart,
            origin=origin,
            modality=modality,
            session_id=session_id,
            source_table=source_table,
            source_id=source_id,
            intent_id=intent_id,
            payload=payload,
            created_at=datetime.now(),
        ))

    def record_timeline_event(
        self,
        *,
        character_id: str,
        event_type: str,
        occurred_at: datetime | None = None,
        actor: str | None = None,
        counterpart: str | None = None,
        origin: str = "real",
        modality: str | None = None,
        session_id: str | None = None,
        source_table: str | None = None,
        source_id: str | None = None,
        intent_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        """封筒1行を独立トランザクションで直書きする（payload 完結型イベント用）。

        night.chronicle / night.forget / scene.closed / chat.farewell /
        intent.* / action.performed など、中身テーブルを持たない（または中身の
        書き込みポイントと切り離されている）イベントの追記口。
        引数は `_append_timeline_event` と同じ。
        """
        with self.get_session() as session:
            self._append_timeline_event(
                session,
                character_id=character_id,
                event_type=event_type,
                occurred_at=occurred_at,
                actor=actor,
                counterpart=counterpart,
                origin=origin,
                modality=modality,
                session_id=session_id,
                source_table=source_table,
                source_id=source_id,
                intent_id=intent_id,
                payload=payload,
            )
            session.commit()

    # --- 巻き戻し（retract） ---

    def _retract_timeline_events_in_session(
        self,
        session,
        source_table: str,
        source_ids: list[str],
        event_type: str | None = None,
    ) -> None:
        """開いている Session 内で、指定ソース行群の封筒に retracted_at をマークする。

        履歴の巻き戻し（delete_chat_messages_from / delete_scenario_turns_from 等）と
        同一トランザクションで呼ぶための内部口。既に retracted の行は触らない。

        Args:
            session: 呼び出し元が開いている SQLAlchemy Session。
            source_table: 中身テーブル名（"chat_messages" / "scenario_turns" 等）。
            source_ids: 巻き戻し対象の中身行 ID リスト。
            event_type: 指定時はこの event_type の封筒だけマークする。
                同一ソース行に複数種の封筒が張られているケース
                （inscribed_memories の memory.inscribed / memory.forgotten）で使う。
        """
        from backend.repositories.sqlite.models import TimelineEvent

        if not source_ids:
            return
        q = session.query(TimelineEvent).filter(
            TimelineEvent.source_table == source_table,
            TimelineEvent.source_id.in_(source_ids),
            TimelineEvent.retracted_at.is_(None),
        )
        if event_type is not None:
            q = q.filter(TimelineEvent.event_type == event_type)
        q.update({"retracted_at": datetime.now()}, synchronize_session=False)

    def retract_timeline_events(self, source_table: str, source_ids: list[str]) -> None:
        """指定ソース行群の封筒に retracted_at をマークする（独立トランザクション版）。

        Args:
            source_table: 中身テーブル名。
            source_ids: 巻き戻し対象の中身行 ID リスト。
        """
        if not source_ids:
            return
        with self.get_session() as session:
            self._retract_timeline_events_in_session(session, source_table, source_ids)
            session.commit()

    def unretract_timeline_events(self, source_table: str, source_ids: list[str]) -> None:
        """指定ソース行群の封筒の retracted_at を解除する（復元操作用）。

        restore_inscribed_memory のような「取り消しの取り消し」で使う。
        通常の巻き戻しフローでは使わないこと。

        Args:
            source_table: 中身テーブル名。
            source_ids: 復元対象の中身行 ID リスト。
        """
        from backend.repositories.sqlite.models import TimelineEvent

        if not source_ids:
            return
        with self.get_session() as session:
            session.query(TimelineEvent).filter(
                TimelineEvent.source_table == source_table,
                TimelineEvent.source_id.in_(source_ids),
                TimelineEvent.retracted_at.isnot(None),
            ).update({"retracted_at": None}, synchronize_session=False)
            session.commit()

    # --- 読み出し ---

    def list_timeline_events(
        self,
        character_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        origins: list[str] | None = None,
        event_type_prefixes: list[str] | None = None,
        include_retracted: bool = False,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list:
        """キャラクターの封筒を時系列で返す（投影・計器・圧力計算の共通読み出し口）。

        Args:
            character_id: 対象キャラクター ID。
            since: この時刻以降（inclusive）。None なら制限なし。
            until: この時刻以前（exclusive）。None なら制限なし。
            origins: origin のフィルタ（["real", "usual"] 等）。None なら全 origin。
            event_type_prefixes: event_type の前方一致フィルタ。
                "chat." のような名前空間指定と "chat.message" のような完全指定の両方を受ける。
            include_retracted: True なら retracted 行も含める（計器の監査用）。
                既定 False（retracted は全観測者から hidden、の実装）。
            limit: 最大件数。None なら無制限。
            newest_first: True なら新しい順で返す（limit と併用して「直近N件」を取る用途）。

        Returns:
            TimelineEvent の ORM オブジェクトリスト（既定は occurred_at 昇順）。
        """
        from sqlalchemy import or_
        from backend.repositories.sqlite.models import TimelineEvent

        with self.get_session() as session:
            q = session.query(TimelineEvent).filter(
                TimelineEvent.character_id == character_id
            )
            if not include_retracted:
                q = q.filter(TimelineEvent.retracted_at.is_(None))
            if since is not None:
                q = q.filter(TimelineEvent.occurred_at >= since)
            if until is not None:
                q = q.filter(TimelineEvent.occurred_at < until)
            if origins:
                q = q.filter(TimelineEvent.origin.in_(origins))
            if event_type_prefixes:
                q = q.filter(or_(*[
                    TimelineEvent.event_type.like(f"{p}%")
                    for p in event_type_prefixes
                ]))
            if newest_first:
                q = q.order_by(TimelineEvent.occurred_at.desc(), TimelineEvent.created_at.desc())
            else:
                q = q.order_by(TimelineEvent.occurred_at.asc(), TimelineEvent.created_at.asc())
            if limit is not None:
                q = q.limit(limit)
            return q.all()

    def attach_payload_to_latest_chat_event(
        self, character_id: str, session_id: str, patch: dict
    ) -> bool:
        """セッション内の最新キャラ発話封筒（chat.message）へ payload をマージする。

        farewell judge の採点結果（emotions / engagement）を該当ターンの封筒に
        残すための口（Tier 3 サンプリングの材料を兼ねる。docs/planned/aliveness_plan.md §5.2）。
        judge はバックグラウンドで走るため「最新のキャラ発話」への best-effort 添付。

        Args:
            character_id: 対象キャラクター。
            session_id: 対象チャットセッション。
            patch: payload にマージする dict。

        Returns:
            添付できたら True（該当封筒なしなら False）。
        """
        from backend.repositories.sqlite.models import TimelineEvent

        with self.get_session() as session:
            event = (
                session.query(TimelineEvent)
                .filter(
                    TimelineEvent.character_id == character_id,
                    TimelineEvent.session_id == session_id,
                    TimelineEvent.event_type == "chat.message",
                    TimelineEvent.actor == "character",
                    TimelineEvent.retracted_at.is_(None),
                )
                .order_by(TimelineEvent.occurred_at.desc(), TimelineEvent.created_at.desc())
                .first()
            )
            if event is None:
                return False
            event.payload = {**(event.payload or {}), **patch}
            session.commit()
            return True

    def count_timeline_events_by_source(self, source_table: str) -> int:
        """指定ソーステーブル由来の封筒件数を返す（retracted 含む）。

        計器 Tier 1 `envelope_integrity`（源テーブルと封筒の件数突合）の材料。
        ID 突合はしない（docs/planned/aliveness_plan.md §3）。

        Args:
            source_table: 中身テーブル名。

        Returns:
            封筒件数。
        """
        from backend.repositories.sqlite.models import TimelineEvent

        with self.get_session() as session:
            return (
                session.query(TimelineEvent)
                .filter(TimelineEvent.source_table == source_table)
                .count()
            )

    def fetch_timeline_source_contents(
        self, source_table: str, source_ids: list[str]
    ) -> dict[str, str]:
        """封筒の source 参照から中身テキストを一括取得する（投影の content 開示用）。

        観測者ポリシーが disclosure="content" を返した封筒についてだけ呼ばれる想定。
        テーブルごとに「中身」として意味のあるテキストへ整形して返す:

            - chat_messages      → 発言本文
            - scenario_turns     → "話者名: 本文"
            - inscribed_memories → 記憶本文
            - tool_call_events   → 引数 JSON（power_recall のクエリ等）

        Args:
            source_table: 中身テーブル名。
            source_ids: 取得する中身行 ID のリスト。

        Returns:
            {source_id: 中身テキスト} の辞書。見つからない ID はキー自体を含まない。
        """
        if not source_ids:
            return {}
        from backend.repositories.sqlite.models import (
            ChatMessage,
            InscribedMemory,
            ScenarioTurn,
            ToolCallEvent,
        )

        with self.get_session() as session:
            if source_table == "chat_messages":
                rows = (
                    session.query(ChatMessage.id, ChatMessage.content)
                    .filter(ChatMessage.id.in_(source_ids))
                    .all()
                )
                return {str(i): str(c or "") for i, c in rows}
            if source_table == "scenario_turns":
                rows = (
                    session.query(
                        ScenarioTurn.id, ScenarioTurn.speaker_name, ScenarioTurn.content
                    )
                    .filter(ScenarioTurn.id.in_(source_ids))
                    .all()
                )
                return {str(i): f"{n}: {c or ''}" for i, n, c in rows}
            if source_table == "inscribed_memories":
                rows = (
                    session.query(InscribedMemory.id, InscribedMemory.content)
                    .filter(InscribedMemory.id.in_(source_ids))
                    .all()
                )
                return {str(i): str(c or "") for i, c in rows}
            if source_table == "tool_call_events":
                # autoincrement 整数 ID を文字列として保持しているため int へ変換して引く
                int_ids = [int(i) for i in source_ids if str(i).isdigit()]
                rows = (
                    session.query(ToolCallEvent.id, ToolCallEvent.arguments_json)
                    .filter(ToolCallEvent.id.in_(int_ids))
                    .all()
                )
                return {str(i): str(a or "") for i, a in rows}
            return {}

    # --- 封筒付与のための共通ヘルパ ---

    def _resolve_character_id_by_name_in_session(self, session, name: str) -> str | None:
        """開いている Session 内でキャラクター名から ID を引く。

        chat_sessions.model_id（"{char_name}@{preset_name}"）のように
        名前しか持たない書き込み点で、封筒の character_id を解決するためのヘルパ。
        同名キャラが複数いる場合は最初の1件を返す（名前はアプリ運用上ユニーク前提）。

        Args:
            session: 呼び出し元が開いている SQLAlchemy Session。
            name: キャラクター名。

        Returns:
            characters.id。見つからなければ None（封筒はスキップされる）。
        """
        from backend.repositories.sqlite.models import Character

        if not name:
            return None
        row = (
            session.query(Character.id)
            .filter(Character.name == name)
            .first()
        )
        return row[0] if row else None
