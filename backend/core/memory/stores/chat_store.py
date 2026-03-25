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
        afterglow_session_id: Optional[str] = None,
    ):
        """チャットセッションを作成する。

        Args:
            afterglow_session_id: Afterglow（感情継続機構）で引き継ぐ元セッションID。
                                  NULLなら引き継ぎなし。
        """
        with self.get_session() as session:
            from ..sqlite_store import ChatSession
            obj = ChatSession(
                id=session_id,
                model_id=model_id,
                title=title,
                session_type=session_type,
                group_config=group_config,
                afterglow_session_id=afterglow_session_id,
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
        is_system_message: bool = False,
    ):
        """チャットメッセージを作成する。

        Args:
            is_system_message: True の場合、退席通知などのシステムメッセージとしてフラグを立てる。
        """
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
                is_system_message=1 if is_system_message else None,
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

    def get_recent_turns(self, session_id: str, n_turns: int = 5) -> list:
        """Afterglow用: 指定セッションの末尾 n_turns ターン分のメッセージを返す。

        1ターン = user発言 + character応答のペア（最大 n_turns * 2 件）。
        引き継ぎ元セッションの直近の流れをプリペンドする際に使用する。

        Args:
            session_id: 引き継ぎ元のセッションID。
            n_turns: 引き継ぐターン数（デフォルト: 5ターン = 最大10メッセージ）。

        Returns:
            時系列順のメッセージリスト（最大 n_turns * 2 件）。
        """
        all_msgs = self.list_chat_messages(session_id)
        return all_msgs[-(n_turns * 2):]

    def find_latest_session_for_character(
        self, character_name: str, exclude_session_id: Optional[str] = None
    ) -> Optional[str]:
        """Afterglow用: 指定キャラクターの最新1on1セッションIDを返す。

        同キャラクターの新規セッション作成時に引き継ぎ元セッションを自動特定するために使用する。

        Args:
            character_name: キャラクター名（model_id の "@" より前の部分）。
            exclude_session_id: 除外するセッションID（作成中のセッション自身を除くために使用）。

        Returns:
            最新セッションのID。見つからない場合は None。
        """
        with self.get_session() as session:
            from ..sqlite_store import ChatSession
            q = (
                session.query(ChatSession)
                .filter(
                    ChatSession.session_type == "1on1",
                    ChatSession.model_id.like(f"{character_name}@%"),
                )
                .order_by(ChatSession.updated_at.desc())
            )
            if exclude_session_id:
                q = q.filter(ChatSession.id != exclude_session_id)
            result = q.first()
            return result.id if result else None

    def get_messages_for_character_on_date(
        self, character_name: str, date_start: datetime, date_end: datetime
    ) -> list:
        """chronicle 用: 指定日に対象キャラクターが参加したセッションのメッセージを時系列で返す。

        1on1・グループチャットを問わず character_name が model_id の前半に含まれるセッションを対象とする。

        Args:
            character_name: キャラクター名。
            date_start: 対象日の開始 datetime（inclusive）。
            date_end: 対象日の終了 datetime（exclusive）。

        Returns:
            当日のメッセージ一覧（時系列昇順）。is_system_message=1 は除外。
        """
        with self.get_session() as session:
            from ..sqlite_store import ChatMessage, ChatSession
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
