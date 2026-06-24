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
    ):
        """チャットメッセージを作成する。

        Args:
            is_system_message: True の場合、退席通知などのシステムメッセージとしてフラグを立てる。
            log_message_id: デバッグログフォルダ名（8桁hex）。CHOTGOR_DEBUG=1 時のみ設定される。
            anticipation: キャラクターが本文末尾に書いた予想（期待）。次ターンのシステムプロンプトに注入される。
            face_to_face: 当該メッセージが交わされた時点のチャットモード（0=テキスト / 1=対面）。
                キャラクタースコープの face_to_face_mode を送信時に焼き付ける。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatMessage
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
            )
            session.add(msg)
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
