"""チャットセッション・メッセージ・画像 CRUD — SQLiteStore Mixin。"""

from datetime import datetime

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
        """チャットセッションとそのメッセージ・画像レコードを削除する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatSession, ChatMessage, ChatImage
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
            session.query(ChatImage).filter(ChatImage.session_id == session_id).delete()
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
        images: list | None = None,
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
            from backend.repositories.sqlite.store import ChatMessage, ChatSession
            msg = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                reasoning=reasoning,
                images=images or None,
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
                chat_session = session.get(ChatSession, session_id)
                char_name = None
                if chat_session and chat_session.model_id:
                    # model_id = "{char_name}@{preset_name}" からキャラ名を取り出す
                    char_name = chat_session.model_id.rsplit("@", 1)[0]
                char_id = self._resolve_character_id_by_name_in_session(
                    session, char_name or ""
                )
                if char_id:
                    self._append_timeline_event(
                        session,
                        character_id=char_id,
                        event_type="chat.message",
                        actor="user" if role == "user" else "character",
                        counterpart="user",
                        origin="real",
                        modality="face" if face_to_face else "text",
                        session_id=session_id,
                        source_table="chat_messages",
                        source_id=message_id,
                    )
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
                .filter(
                    ChatSession.model_id.like(f"{character_name}@%"),
                    ChatMessage.chronicled_at == None,  # noqa: E711
                    (ChatMessage.is_system_message == None) | (ChatMessage.is_system_message == 0),  # noqa: E711
                )
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
            from backend.repositories.sqlite.store import ChatMessage, ChatSession
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
                chat_session = session.get(ChatSession, msg.session_id)
                char_name = None
                if chat_session and chat_session.model_id:
                    char_name = chat_session.model_id.rsplit("@", 1)[0]
                char_id = self._resolve_character_id_by_name_in_session(
                    session, char_name or ""
                )
                if char_id:
                    self._append_timeline_event(
                        session,
                        character_id=char_id,
                        event_type="chat.message",
                        occurred_at=now,
                        actor="user" if msg.role == "user" else "character",
                        counterpart="user",
                        origin="real",
                        modality="face" if msg.face_to_face else "text",
                        session_id=msg.session_id,
                        source_table="chat_messages",
                        source_id=msg.id,
                    )
            session.commit()

    def list_sessions_with_undelivered(self) -> list:
        """預かり中メッセージを持つセッション一覧を返す（配達スケジューラ用）。

        Returns:
            (ChatSession, 預かり件数) のタプルのリスト。
        """
        with self.get_session() as session:
            from sqlalchemy import func
            from backend.repositories.sqlite.store import ChatMessage, ChatSession
            rows = (
                session.query(ChatSession, func.count(ChatMessage.id))
                .join(ChatMessage, ChatMessage.session_id == ChatSession.id)
                .filter(ChatMessage.delivered_at.is_(None))
                .group_by(ChatSession.id)
                .all()
            )
            return rows

    def delete_chat_messages_from(self, session_id: str, message_id: str) -> bool:
        """指定メッセージ以降（自身を含む）をすべて削除する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage
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
            # 封筒は削除せず retracted マーク（不可逆性の担保。データは残す）
            self._retract_timeline_events_in_session(
                session, "chat_messages", ids_to_delete
            )
            session.commit()
            return True

    # --- Chat Images ---

    def create_chat_image(
        self,
        image_id: str,
        session_id: str,
        mime_type: str,
        message_id: str | None = None,
    ):
        """添付画像レコードを作成する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatImage
            img = ChatImage(
                id=image_id,
                session_id=session_id,
                message_id=message_id,
                mime_type=mime_type,
            )
            session.add(img)
            session.commit()
            session.refresh(img)
            return img

    def get_chat_image(self, image_id: str):
        """画像IDでレコードを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatImage
            return session.get(ChatImage, image_id)

    def list_chat_images_by_session(self, session_id: str) -> list:
        """セッションに紐づく全画像レコードを返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatImage
            return (
                session.query(ChatImage)
                .filter(ChatImage.session_id == session_id)
                .all()
            )
