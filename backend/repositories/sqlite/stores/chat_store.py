"""チャットセッション・メッセージ・画像 CRUD — SQLiteStore Mixin。"""

import uuid
from datetime import datetime

from sqlalchemy import false, or_

class ChatStoreMixin:
    """チャットセッション・メッセージ・添付画像の作成・取得・更新・削除を担う Mixin。"""

    # --- Chat Sessions ---

    def create_chat_session(
        self,
        session_id: str,
        model_id: str,
        title: str = "新しいチャット",
        session_type: str = "1on1",
    ):
        """チャットセッションを作成する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatSession
            obj = ChatSession(
                id=session_id,
                model_id=model_id,
                title=title,
                session_type=session_type,
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj

    def get_chat_session(self, session_id: str):
        """IDでチャットセッションを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatSession
            return session.get(ChatSession, session_id)

    def list_chat_sessions(self, limit: int = 100) -> list:
        """チャットセッション一覧を新しい順で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatSession
            return (
                session.query(ChatSession)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
                .all()
            )

    def update_chat_session(self, session_id: str, **kwargs):
        """チャットセッションを更新する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatSession
            obj = session.get(ChatSession, session_id)
            if not obj:
                return None
            for k, v in kwargs.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            obj.updated_at = datetime.now()
            session.commit()
            session.refresh(obj)
            return obj

    def delete_chat_session(self, session_id: str) -> bool:
        """チャットセッションとそのメッセージ・添付レコードを削除する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import (
                ChatAttachment, ChatMessage, ChatMessageVariant, ChatSession,
            )
            # 封筒は削除せず retracted マーク（セッションごと消してもタイムラインの
            # 存在記録は残す。retracted は全観測者から hidden）
            message_ids = [
                row[0]
                for row in session.query(ChatMessage.id)
                .filter(ChatMessage.session_id == session_id)
                .all()
            ]
            self._retract_timeline_events_in_session(
                session, "chat_messages", message_ids
            )
            session.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            session.query(ChatMessageVariant).filter(
                ChatMessageVariant.session_id == session_id
            ).delete()
            session.query(ChatAttachment).filter(ChatAttachment.session_id == session_id).delete()
            obj = session.get(ChatSession, session_id)
            if not obj:
                session.commit()
                return False
            session.delete(obj)
            session.commit()
            return True

    # --- Chat Messages ---

    def create_chat_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        reasoning: str | None = None,
        attachments: list | None = None,
        character_name: str | None = None,
        preset_name: str | None = None,
        is_system_message: bool = False,
        log_message_id: str | None = None,
        anticipation: str | None = None,
        face_to_face: int = 0,
        delivered: bool = True,
    ):
        """チャットメッセージを作成する。

        Args:
            is_system_message: True の場合、退席通知などのシステムメッセージとしてフラグを立てる。
            log_message_id: デバッグログフォルダ名（8桁hex）。CHOTGOR_DEBUG=1 時のみ設定される。
            anticipation: キャラクターが本文末尾に書いた予想（期待）。次ターンのシステムプロンプトに注入される。
            face_to_face: 当該メッセージが交わされた時点のチャットモード（0=テキスト / 1=対面）。
                キャラクタースコープの face_to_face_mode を送信時に焼き付ける。
            delivered: False なら「預かり（escrow）」として delivered_at=NULL で保存する
                （キャラ未読・LLM 未到達。availability が戻ったら配達される）。
                既定 True（即配達＝従来挙動）。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage
            msg = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                reasoning=reasoning,
                attachments=attachments or None,
                character_name=character_name,
                preset_name=preset_name,
                is_system_message=1 if is_system_message else None,
                log_message_id=log_message_id or None,
                anticipation=anticipation or None,
                face_to_face=1 if face_to_face else 0,
                delivered_at=datetime.now() if delivered else None,
            )
            session.add(msg)
            # タイムライン封筒 dual-write（chat.message）— 同一トランザクション。
            # システムメッセージ（退席通知等）は「キャラの身に起きたこと」ではない
            # 表示上の掲示なので封筒に載せない（退席自体は chat.farewell が載る）。
            # 預かり（delivered=False）のメッセージもまだキャラの身に起きていないため
            # ここでは載せず、配達時（mark_messages_delivered）に封筒を作る。
            if not is_system_message and delivered:
                self._append_chat_message_envelope(session, msg)
            session.commit()
            session.refresh(msg)
            return msg

    def list_chat_messages(self, session_id: str) -> list:
        """セッション内のメッセージを時系列順で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage
            return (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .all()
            )

    def get_messages_for_character_on_date(
        self, character_name: str, date_start: datetime, date_end: datetime
    ) -> list:
        """chronicle 用: 指定日に対象キャラクターが参加したセッションのメッセージを時系列で返す。

        character_name が model_id の前半（`{char}@{preset}` の `{char}`）に含まれるセッションを対象とする。

        Args:
            character_name: キャラクター名。
            date_start: 対象日の開始 datetime（inclusive）。
            date_end: 対象日の終了 datetime（exclusive）。

        Returns:
            当日のメッセージ一覧（時系列昇順）。is_system_message=1 は除外。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage, ChatSession
            return (
                session.query(ChatMessage)
                .join(ChatSession, ChatMessage.session_id == ChatSession.id)
                .filter(
                    ChatSession.model_id.like(f"{character_name}@%"),
                    ChatMessage.created_at >= date_start,
                    ChatMessage.created_at < date_end,
                    (ChatMessage.is_system_message == None) | (ChatMessage.is_system_message == 0),  # noqa: E711
                )
                .order_by(ChatMessage.created_at.asc())
                .all()
            )

    @staticmethod
    def unchronicled_message_filters(character_names: list[str] | None = None) -> list:
        """「未蒸留の 1on1 メッセージ」を選ぶ WHERE 条件を返す — 唯一の定義。

        Chronicle 実処理（``get_unchronicled_messages_for_character``）と計器
        ``chronicle_backlog``（``count_chronicle_backlog``）が同じ条件を共有するための
        共通部品。両者の条件がズレると「実処理が一生触らないレコードを滞留として
        数え続ける」永久誤検知になるため、条件を足すときは必ずここへ足すこと。

        呼び出し側は ``ChatMessage`` に ``ChatSession`` を JOIN 済みであること
        （キャラの紐付けが ``ChatSession.model_id`` にしか無いため）。

        Args:
            character_names: 対象キャラ名のリスト。None なら全キャラ（呼び出し側で
                対象を絞る前提）。空リストは「対象キャラなし」として常に偽を返す。

        Returns:
            SQLAlchemy 式のリスト（``.filter(*ret)`` で使う）。
        """
        from backend.repositories.sqlite.store import ChatMessage, ChatSession
        filters = [
            ChatMessage.chronicled_at == None,  # noqa: E711
            (ChatMessage.is_system_message == None) | (ChatMessage.is_system_message == 0),  # noqa: E711
        ]
        if character_names is None:
            return filters
        if not character_names:
            return [false()]
        filters.append(
            or_(*[ChatSession.model_id.like(f"{n}@%") for n in character_names])
        )
        return filters

    def get_unchronicled_messages_for_character(self, character_name: str) -> list:
        """chronicle 用: chronicled_at が NULL のメッセージを時系列で返す（スケジューラー用）。

        Args:
            character_name: キャラクター名。

        Returns:
            未処理メッセージ一覧（時系列昇順）。is_system_message=1 は除外。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage, ChatSession
            return (
                session.query(ChatMessage)
                .join(ChatSession, ChatMessage.session_id == ChatSession.id)
                .filter(*self.unchronicled_message_filters([character_name]))
                .order_by(ChatMessage.created_at.asc())
                .all()
            )

    def mark_messages_as_chronicled(self, message_ids: list[str]) -> None:
        """指定IDのメッセージの chronicled_at を現在日時にセットする。

        Args:
            message_ids: 処理済みにするメッセージ ID のリスト。
        """
        if not message_ids:
            return
        from datetime import datetime as dt
        now = dt.utcnow()
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage
            session.query(ChatMessage).filter(
                ChatMessage.id.in_(message_ids)
            ).update({"chronicled_at": now}, synchronize_session=False)
            session.commit()

    def list_undelivered_messages(self, session_id: str) -> list:
        """セッション内の預かり中（delivered_at IS NULL）メッセージを時系列で返す。

        availability が戻った際の配達（時間差注釈付きのまとめ渡し）に使う。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage
            return (
                session.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == session_id,
                    ChatMessage.delivered_at.is_(None),
                )
                .order_by(ChatMessage.created_at.asc())
                .all()
            )

    def mark_messages_delivered(self, message_ids: list[str]) -> None:
        """預かり中メッセージを配達済みにし、chat.message 封筒を同時に作る。

        封筒の occurred_at は配達時刻（キャラの身に起きた瞬間）。送信時刻は
        中身（chat_messages.created_at）が持っているので封筒には複製しない。

        Args:
            message_ids: 配達するメッセージ ID のリスト。
        """
        if not message_ids:
            return
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage
            now = datetime.now()
            msgs = (
                session.query(ChatMessage)
                .filter(
                    ChatMessage.id.in_(message_ids),
                    ChatMessage.delivered_at.is_(None),
                )
                .all()
            )
            for msg in msgs:
                msg.delivered_at = now
                self._append_chat_message_envelope(session, msg, occurred_at=now)
            session.commit()

    def list_sessions_with_undelivered(self) -> list:
        """預かり中メッセージを持つセッション一覧を返す（配達スケジューラ用）。

        Returns:
            (ChatSession, 預かり件数, 最古の未配達 created_at) のタプルのリスト。
            最古時刻は生活カレンダー経路のチェック間隔格子（schedule_plan.md §5）が
            「メッセージ到着後の最初のチェック点」を求めるのに使う。
        """
        with self.get_session() as session:
            from sqlalchemy import func
            from backend.repositories.sqlite.store import ChatMessage, ChatSession
            rows = (
                session.query(
                    ChatSession,
                    func.count(ChatMessage.id),
                    func.min(ChatMessage.created_at),
                )
                .join(ChatMessage, ChatMessage.session_id == ChatSession.id)
                .filter(ChatMessage.delivered_at.is_(None))
                .group_by(ChatSession.id)
                .all()
            )
            return rows

    def delete_chat_messages_from(self, session_id: str, message_id: str) -> bool:
        """指定メッセージ以降（自身を含む）をすべて削除する。

        消えるユーザ発話を起点とする退避応答（chat_message_variants）も一緒に消す。
        発話が変わる／なくなる以上、それに対して引いた応答は無効なため。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage, ChatMessageVariant
            all_ids: list[str] = [
                row[0]
                for row in (
                    session.query(ChatMessage.id)
                    .filter(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.asc())
                    .all()
                )
            ]
            if message_id not in all_ids:
                return False
            pivot_idx = all_ids.index(message_id)
            ids_to_delete = all_ids[pivot_idx:]
            session.query(ChatMessage).filter(
                ChatMessage.id.in_(ids_to_delete)
            ).delete(synchronize_session=False)
            session.query(ChatMessageVariant).filter(
                ChatMessageVariant.anchor_message_id.in_(ids_to_delete)
            ).delete(synchronize_session=False)
            # 封筒は削除せず retracted マーク（不可逆性の担保。データは残す）
            self._retract_timeline_events_in_session(
                session, "chat_messages", ids_to_delete
            )
            session.commit()
            return True

    # --- Chat Message Variants（1on1 レスポンスガチャ）---
    # docs/planned/chat_turn_variants_plan.md。選ばれなかった応答は chat_messages から
    # 別テーブルへ行ごと移すので、chat_messages を読む既存経路からは一切見えない。

    def _append_chat_message_envelope(self, session, msg, occurred_at=None) -> None:
        """chat.message 封筒を1件積む（同一トランザクション内で呼ぶこと）。"""
        from backend.repositories.sqlite.store import ChatSession
        chat_session = session.get(ChatSession, msg.session_id)
        char_name = None
        if chat_session and chat_session.model_id:
            # model_id = "{char_name}@{preset_name}" からキャラ名を取り出す
            char_name = chat_session.model_id.rsplit("@", 1)[0]
        char_id = self._resolve_character_id_by_name_in_session(session, char_name or "")
        if not char_id:
            return
        self._append_timeline_event(
            session,
            character_id=char_id,
            event_type="chat.message",
            occurred_at=occurred_at,
            actor="user" if msg.role == "user" else "character",
            counterpart="user",
            origin="real",
            modality="face" if msg.face_to_face else "text",
            session_id=msg.session_id,
            source_table="chat_messages",
            source_id=msg.id,
        )

    @staticmethod
    def _copy_chat_message_columns(src, dst_cls):
        """src の列のうち dst_cls にもある列を名前で写した dst_cls インスタンスを返す。"""
        dst_cols = {c.name for c in dst_cls.__table__.columns}
        return dst_cls(**{
            c.name: getattr(src, c.name)
            for c in src.__table__.columns
            if c.name in dst_cols
        })

    def _archive_messages_in_session(self, session, anchor_message_id: str, msgs: list) -> None:
        """msgs を1つの退避グループとして chat_message_variants へ移す。封筒は retract。"""
        from backend.repositories.sqlite.store import ChatMessageVariant
        if not msgs:
            return
        group_id = str(uuid.uuid4())
        now = datetime.now()
        for m in msgs:
            v = self._copy_chat_message_columns(m, ChatMessageVariant)
            v.anchor_message_id = anchor_message_id
            v.variant_group_id = group_id
            v.archived_at = now
            session.add(v)
            session.delete(m)
        self._retract_timeline_events_in_session(
            session, "chat_messages", [m.id for m in msgs]
        )

    def _tail_after_anchor(self, session, session_id: str, anchor_message_id: str) -> list | None:
        """起点ユーザ発話より後の本線メッセージを返す。

        起点が本線の最新ユーザ発話でない（＝後ろにユーザ発話がある）なら None。
        レスポンスガチャは最新ターン限定のため。
        """
        from backend.repositories.sqlite.store import ChatMessage
        msgs = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        idx = next((i for i, m in enumerate(msgs) if m.id == anchor_message_id), None)
        if idx is None or msgs[idx].role != "user":
            return None
        tail = msgs[idx + 1:]
        if any(m.role == "user" for m in tail):
            return None
        return tail

    def archive_chat_messages_as_variant(
        self, session_id: str, anchor_message_id: str
    ) -> list[str] | None:
        """最新ターンの応答（起点ユーザ発話より後の行）を退避グループとして移す。

        Returns:
            退避したメッセージ ID のリスト（LanceDB からの削除に使う）。
            起点が最新ユーザ発話でないなど退避できないときは None（呼び出し側は
            従来どおり物理削除に切り替える）。
        """
        with self.get_session() as session:
            tail = self._tail_after_anchor(session, session_id, anchor_message_id)
            if tail is None:
                return None
            archived_ids = [m.id for m in tail]
            self._archive_messages_in_session(session, anchor_message_id, tail)
            session.commit()
            return archived_ids

    def _variant_groups_in_session(
        self, session, session_id: str, anchor_message_id: str, tail: list
    ) -> list[tuple[str | None, list]]:
        """兄弟の応答グループを枝番号順に返す。本線にいるグループの group_id は None。

        並びは各グループ先頭行の created_at 昇順。復元しても created_at は元の値の
        ままなので、往復しても番号は変わらない。
        """
        from backend.repositories.sqlite.store import ChatMessageVariant
        rows = (
            session.query(ChatMessageVariant)
            .filter(
                ChatMessageVariant.session_id == session_id,
                ChatMessageVariant.anchor_message_id == anchor_message_id,
            )
            .order_by(ChatMessageVariant.created_at.asc())
            .all()
        )
        groups: dict[str | None, list] = {}
        for r in rows:
            groups.setdefault(r.variant_group_id, []).append(r)
        if tail:
            groups[None] = tail
        return sorted(
            groups.items(),
            key=lambda kv: kv[1][0].created_at or datetime.min,
        )

    def get_chat_variant_info(self, session_id: str, anchor_message_id: str) -> dict | None:
        """最新ターンの枝ナビ情報 `{"index": n, "count": m}` を返す（1 始まり）。

        起点が最新ユーザ発話でない・退避グループがないときは None。
        """
        with self.get_session() as session:
            tail = self._tail_after_anchor(session, session_id, anchor_message_id)
            if not tail:
                return None
            groups = self._variant_groups_in_session(
                session, session_id, anchor_message_id, tail
            )
            if len(groups) < 2:
                return None
            index = next(i for i, (gid, _) in enumerate(groups) if gid is None) + 1
            return {"index": index, "count": len(groups)}

    def activate_chat_variant(
        self, session_id: str, anchor_message_id: str, index: int
    ) -> tuple[list[str], list] | None:
        """最新ターンの応答を index 番目（1 始まり）の枝へ切り替える。

        本線にいる応答を退避し、選んだグループを同じ ID のまま chat_messages へ戻す。
        戻した行には新しい chat.message 封筒を積む（retract の取り消しはしない。
        「もう一度本線に戻った」という新しい事実として扱う — めぐりの不可逆性）。

        Returns:
            `(退避したID群, 復元した ChatMessage 群)`。起点が不正・index 範囲外なら None。
            すでに本線の枝を指定したときは `([], 現在の本線)`。
        """
        from backend.repositories.sqlite.store import ChatMessage
        with self.get_session() as session:
            tail = self._tail_after_anchor(session, session_id, anchor_message_id)
            if tail is None:
                return None
            groups = self._variant_groups_in_session(
                session, session_id, anchor_message_id, tail
            )
            if not 1 <= index <= len(groups):
                return None
            target_gid, target_rows = groups[index - 1]
            if target_gid is None:
                return [], tail
            archived_ids = [m.id for m in tail]
            self._archive_messages_in_session(session, anchor_message_id, tail)
            restored = []
            for v in target_rows:
                m = self._copy_chat_message_columns(v, ChatMessage)
                session.delete(v)
                session.add(m)
                if not m.is_system_message and m.delivered_at is not None:
                    self._append_chat_message_envelope(session, m)
                restored.append(m)
            session.commit()
            for m in restored:
                session.refresh(m)
            return archived_ids, restored

    # --- Chat Attachments ---

    def create_chat_attachment(
        self,
        attachment_id: str,
        session_id: str,
        mime_type: str,
        message_id: str | None = None,
        filename: str | None = None,
    ):
        """添付レコード（画像・音声・テキスト）を作成する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatAttachment
            att = ChatAttachment(
                id=attachment_id,
                session_id=session_id,
                message_id=message_id,
                mime_type=mime_type,
                filename=filename,
            )
            session.add(att)
            session.commit()
            session.refresh(att)
            return att

    def get_chat_attachment(self, attachment_id: str):
        """添付IDでレコードを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatAttachment
            return session.get(ChatAttachment, attachment_id)

    def list_chat_attachments_by_session(self, session_id: str) -> list:
        """セッションに紐づく全添付レコードを返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatAttachment
            return (
                session.query(ChatAttachment)
                .filter(ChatAttachment.session_id == session_id)
                .all()
            )
