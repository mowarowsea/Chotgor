"""チャットセッション・メッセージ・画像 CRUD — SQLiteStore Mixin。"""

from datetime import datetime
from typing import Optional


class ChatStoreMixin:
    """チャットセッション・メッセージ・添付画像の作成・取得・更新・削除を担う Mixin。"""

    # --- Chat Sessions ---

    def create_chat_session(
        self,
        session_id: str,
        model_id: str,
        title: str = "新しいチャット",
        session_type: str = "1on1",
        group_config: Optional[str] = None,
    ):
        """チャットセッションを作成する。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatSession
            obj = ChatSession(
                id=session_id,
                model_id=model_id,
                title=title,
                session_type=session_type,
                group_config=group_config,
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj

    def get_chat_session(self, session_id: str):
        """IDでチャットセッションを取得する。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatSession
            return session.get(ChatSession, session_id)

    def list_chat_sessions(self, limit: int = 100) -> list:
        """チャットセッション一覧を新しい順で返す。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatSession
            return (
                session.query(ChatSession)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
                .all()
            )

    def update_chat_session(self, session_id: str, **kwargs):
        """チャットセッションを更新する。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatSession
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
            from ..sqlite_store import ChatSession, ChatMessage, ChatImage, SessionDrift
            session.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            session.query(ChatImage).filter(ChatImage.session_id == session_id).delete()
            session.query(SessionDrift).filter(SessionDrift.session_id == session_id).delete()
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
        reasoning: Optional[str] = None,
        images: Optional[list] = None,
        character_name: Optional[str] = None,
        preset_name: Optional[str] = None,
    ):
        """チャットメッセージを作成する。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatMessage
            msg = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                reasoning=reasoning,
                images=images or None,
                character_name=character_name,
                preset_name=preset_name,
            )
            session.add(msg)
            session.commit()
            session.refresh(msg)
            return msg

    def list_chat_messages(self, session_id: str) -> list:
        """セッション内のメッセージを時系列順で返す。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatMessage
            return (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .all()
            )

    def delete_chat_messages_from(self, session_id: str, message_id: str) -> bool:
        """指定メッセージ以降（自身を含む）をすべて削除する。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatMessage
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
        message_id: Optional[str] = None,
    ):
        """添付画像レコードを作成する。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatImage
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
            from ..sqlite_store import ChatImage
            return session.get(ChatImage, image_id)

    def list_chat_images_by_session(self, session_id: str) -> list:
        """セッションに紐づく全画像レコードを返す。"""
        with self.get_session() as session:
            from ..sqlite_store import ChatImage
            return (
                session.query(ChatImage)
                .filter(ChatImage.session_id == session_id)
                .all()
            )
