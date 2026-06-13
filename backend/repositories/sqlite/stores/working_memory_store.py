"""ワーキングメモリ（短期記憶スレッド・ポスト）CRUD — SQLiteStore Mixin。"""

from datetime import datetime

class WorkingMemoryStoreMixin:
    """WorkingMemoryThread / WorkingMemoryPost の作成・取得・更新・削除を担う Mixin。

    ワーキングメモリのスレッド（並行する短期記憶ストリーム）と、その内部に時系列で連なる
    ポストの永続化を担当する。type 別の本数制約はこの層では検証せず、上位の
    WorkingMemoryManager がアプリ層のルールとして担保する。
    """

    # ------------------------------------------------------------------
    # スレッド
    # ------------------------------------------------------------------

    def add_working_memory_thread(
        self,
        thread_id: str,
        character_id: str,
        type: str,
        summary: str = "",
        atmosphere_tag: str = "",
        importance: float = 0.5,
        relation_target: str | None = None,
        origin: str = "real",
    ):
        """ワーキングメモリスレッドを新規作成する。

        Args:
            origin: 記憶のソース識別（3値）。"real"=日常、"usual"=うつつ（ユーザ未共有の自分の生活体験）、"interlude"=シナリオPCモードの幕間体験。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import WorkingMemoryThread
            now = datetime.now()
            thread = WorkingMemoryThread(
                id=thread_id,
                character_id=character_id,
                type=type,
                summary=summary,
                atmosphere_tag=atmosphere_tag,
                importance=importance,
                is_open=1,
                relation_target=relation_target,
                last_touched_at=now,
                origin=origin,
            )
            session.add(thread)
            session.commit()
            session.refresh(thread)
            return thread

    def get_working_memory_thread(self, thread_id: str):
        """ID でスレッドを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import WorkingMemoryThread
            return session.get(WorkingMemoryThread, thread_id)

    def list_working_memory_threads(
        self,
        character_id: str,
        type: str | None = None,
        is_open: bool | None = None,
    ) -> list:
        """キャラクターのスレッド一覧を返す（updated_at 降順）。

        Args:
            character_id: 対象キャラクターID。
            type: 指定時はその type のみに絞り込む。
            is_open: True=Openのみ／False=Archivedのみ／None=全件。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import WorkingMemoryThread
            q = session.query(WorkingMemoryThread).filter(
                WorkingMemoryThread.character_id == character_id
            )
            if type is not None:
                q = q.filter(WorkingMemoryThread.type == type)
            if is_open is not None:
                q = q.filter(WorkingMemoryThread.is_open == (1 if is_open else 0))
            return q.order_by(WorkingMemoryThread.updated_at.desc()).all()

    def get_working_memory_thread_by_relation(self, character_id: str, relation_target: str):
        """relation 型スレッドを相手識別子で取得する（重複作成防止用）。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import WorkingMemoryThread
            return (
                session.query(WorkingMemoryThread)
                .filter(
                    WorkingMemoryThread.character_id == character_id,
                    WorkingMemoryThread.type == "relation",
                    WorkingMemoryThread.relation_target == relation_target,
                )
                .first()
            )

    def update_working_memory_thread(
        self,
        thread_id: str,
        summary: str | None = None,
        atmosphere_tag: str | None = None,
        importance: float | None = None,
        is_open: bool | None = None,
        touch: bool = False,
    ) -> bool:
        """スレッドのフィールドを部分更新する。None の引数は変更しない。

        Args:
            thread_id: 更新対象スレッドID。
            summary / atmosphere_tag / importance / is_open: 非None のものだけ更新する。
            touch: True なら last_touched_at を現在時刻に更新する（heat の decay 起点リセット）。

        Returns:
            更新できたら True、対象が存在しなければ False。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import WorkingMemoryThread
            thread = session.get(WorkingMemoryThread, thread_id)
            if not thread:
                return False
            if summary is not None:
                thread.summary = summary
            if atmosphere_tag is not None:
                thread.atmosphere_tag = atmosphere_tag
            if importance is not None:
                thread.importance = importance
            if is_open is not None:
                thread.is_open = 1 if is_open else 0
            if touch:
                thread.last_touched_at = datetime.now()
            session.commit()
            return True

    def delete_working_memory_thread(self, thread_id: str) -> bool:
        """スレッドと配下の全ポストを物理削除する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import (
                WorkingMemoryPost,
                WorkingMemoryThread,
            )
            thread = session.get(WorkingMemoryThread, thread_id)
            if not thread:
                return False
            session.query(WorkingMemoryPost).filter(
                WorkingMemoryPost.thread_id == thread_id
            ).delete()
            session.delete(thread)
            session.commit()
            return True

    # ------------------------------------------------------------------
    # ポスト
    # ------------------------------------------------------------------

    def add_working_memory_post(self, post_id: str, thread_id: str, content: str):
        """スレッドにポストを追加し、親スレッドの last_touched_at を更新する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import (
                WorkingMemoryPost,
                WorkingMemoryThread,
            )
            post = WorkingMemoryPost(
                id=post_id,
                thread_id=thread_id,
                content=content,
            )
            session.add(post)
            thread = session.get(WorkingMemoryThread, thread_id)
            if thread:
                thread.last_touched_at = datetime.now()
            session.commit()
            session.refresh(post)
            return post

    def list_working_memory_posts(self, thread_id: str) -> list:
        """スレッド内の全ポストを作成日時昇順で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import WorkingMemoryPost
            return (
                session.query(WorkingMemoryPost)
                .filter(WorkingMemoryPost.thread_id == thread_id)
                .order_by(WorkingMemoryPost.created_at.asc())
                .all()
            )

    def get_latest_working_memory_post(self, thread_id: str):
        """スレッド内の最新ポストを返す（なければ None）。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import WorkingMemoryPost
            return (
                session.query(WorkingMemoryPost)
                .filter(WorkingMemoryPost.thread_id == thread_id)
                .order_by(WorkingMemoryPost.created_at.desc())
                .first()
            )
