"""キャラクター CRUD — SQLiteStore Mixin。"""

from datetime import datetime

class CharacterStoreMixin:
    """キャラクターの作成・取得・更新・削除を担う Mixin。"""

    def create_character(
        self,
        character_id: str,
        name: str,
        system_prompt_block1: str = "",
        inner_narrative: str = "",
        self_history: str = "",
        relationship_state: str = "",
        cleanup_config: dict | None = None,
        enabled_providers: dict | None = None,
        ghost_model: str | None = None,
        image_data: str | None = None,
        switch_angle_enabled: bool = False,
        self_reflection_mode: str = "disabled",
        self_reflection_preset_id: str | None = None,
        self_reflection_n_turns: int = 5,
        farewell_config: dict | None = None,
        relationship_status: str = "active",
        allowed_tools: dict | None = None,
        user_label: str = "",
        user_position: str = "",
    ):
        """キャラクターを新規作成する。

        Args:
            self_history: chronicle で更新されるキャラクターの歴史・経緯。
            relationship_state: chronicle で更新されるユーザ・他キャラとの現在の関係。
            self_reflection_mode: 自己参照ループの動作モード。disabled/local_trigger/always。
            self_reflection_preset_id: 契機判断モデルプリセットID（local_trigger 時に使用）。
            self_reflection_n_turns: 自己参照に使う直近ターン数。
            farewell_config: chronicle で更新される感情閾値・退席設定JSON。
            relationship_status: 関係ステータス。"active" または "estranged"。
            allowed_tools: 外部ツール許可設定。{google_calendar, gmail, google_drive} の bool dict。
            user_label: このキャラがユーザを呼ぶ呼称。空なら Settings の user_name フォールバック。
            user_position: このキャラから見たユーザの位置づけ（役職・関係など短文）。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Character
            char = Character(
                id=character_id,
                name=name,
                system_prompt_block1=system_prompt_block1,
                inner_narrative=inner_narrative,
                self_history=self_history,
                relationship_state=relationship_state,
                cleanup_config=cleanup_config or {},
                enabled_providers=enabled_providers or {},
                ghost_model=ghost_model,
                image_data=image_data,
                switch_angle_enabled=1 if switch_angle_enabled else 0,
                self_reflection_mode=self_reflection_mode,
                self_reflection_preset_id=self_reflection_preset_id,
                self_reflection_n_turns=self_reflection_n_turns,
                farewell_config=farewell_config,
                relationship_status=relationship_status,
                allowed_tools=allowed_tools or {},
                user_label=user_label,
                user_position=user_position,
            )
            session.add(char)
            session.commit()
            session.refresh(char)
            return char

    def get_character(self, character_id: str):
        """IDでキャラクターを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Character
            return session.get(Character, character_id)

    def get_character_by_name(self, name: str):
        """名前でキャラクターを取得する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Character
            return session.query(Character).filter(Character.name == name).first()

    def list_characters(self) -> list:
        """全キャラクターを返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Character
            return session.query(Character).all()

    def update_character(self, character_id: str, **kwargs):
        """キャラクターの指定フィールドを更新する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Character
            char = session.get(Character, character_id)
            if not char:
                return None
            for k, v in kwargs.items():
                if hasattr(char, k):
                    setattr(char, k, v)
            char.updated_at = datetime.now()
            session.commit()
            session.refresh(char)
            return char

    def delete_character_cascade(self, character_id: str) -> bool:
        """キャラクターと、SQLite 上で紐づく全レコードを1トランザクションで削除する。

        SQLite は FK カスケードが効かないため、関連テーブルを明示的に削除する。
        削除対象:
        - inscribed_memories（character_id）
        - working_memory_threads ＋ 配下の working_memory_posts（character_id / thread_id）
        - 1on1 chat_sessions（model_id が "キャラ名@..."）＋ 配下の
          chat_messages / chat_images（session_id）
        - characters 本体

        保持するもの:
        - debug_log_entries（監査記録として残す）

        LanceDB 側のベクトル（inscribed_memories / chat_turns / definitions /
        working_memory_threads）は呼び出し側（MemoryManager）が別途削除する。

        Args:
            character_id: 削除対象のキャラクターID。

        Returns:
            削除成否。False の場合はキャラクターが存在しない。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import (
                Character,
                InscribedMemory,
                WorkingMemoryThread,
                WorkingMemoryPost,
                ChatSession,
                ChatMessage,
                ChatImage,
            )

            char = session.get(Character, character_id)
            if not char:
                return False
            character_name = char.name

            # --- ワーキングメモリ（スレッド＋ポスト） ---
            thread_ids = [
                row[0]
                for row in session.query(WorkingMemoryThread.id)
                .filter(WorkingMemoryThread.character_id == character_id)
                .all()
            ]
            if thread_ids:
                session.query(WorkingMemoryPost).filter(
                    WorkingMemoryPost.thread_id.in_(thread_ids)
                ).delete(synchronize_session=False)
                session.query(WorkingMemoryThread).filter(
                    WorkingMemoryThread.character_id == character_id
                ).delete(synchronize_session=False)

            # --- 保存記憶（SQLite 側。ハード削除） ---
            session.query(InscribedMemory).filter(
                InscribedMemory.character_id == character_id
            ).delete(synchronize_session=False)

            # --- 1on1 チャット（セッション＋メッセージ＋画像） ---
            # model_id は "{character_name}@{preset_name}" 形式。グループチャット
            # （model_id="group"）は対象外。
            session_ids = [
                row[0]
                for row in session.query(ChatSession.id)
                .filter(ChatSession.model_id.like(f"{character_name}@%"))
                .all()
            ]
            if session_ids:
                session.query(ChatMessage).filter(
                    ChatMessage.session_id.in_(session_ids)
                ).delete(synchronize_session=False)
                session.query(ChatImage).filter(
                    ChatImage.session_id.in_(session_ids)
                ).delete(synchronize_session=False)
                session.query(ChatSession).filter(
                    ChatSession.id.in_(session_ids)
                ).delete(synchronize_session=False)

            # --- キャラクター本体 ---
            session.delete(char)
            session.commit()
            return True

    def get_negative_exit_count(self, character_name: str, since: datetime) -> int:
        """指定日以降にネガティブ退席したセッション数を返す。

        exited_chars JSON 内の各エントリを Python 側でフィルタし、
        farewell_type="negative" かつ char_name が一致するものを数える。
        1セッションにつき最大1カウント。

        Args:
            character_name: 集計対象のキャラクター名。
            since: 集計開始日時。これ以降に更新されたセッションが対象。

        Returns:
            ネガティブ退席セッション数。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import ChatSession
            sessions = (
                session.query(ChatSession)
                .filter(
                    ChatSession.model_id.like(f"{character_name}@%"),
                    ChatSession.updated_at >= since,
                    ChatSession.exited_chars.isnot(None),
                )
                .all()
            )
        count = 0
        for s in sessions:
            if not s.exited_chars:
                continue
            for entry in s.exited_chars:
                if (
                    isinstance(entry, dict)
                    and entry.get("char_name") == character_name
                    and entry.get("farewell_type") == "negative"
                ):
                    count += 1
                    break  # 1セッションにつき1カウント
        return count
