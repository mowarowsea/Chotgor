"""キャラクター CRUD — SQLiteStore Mixin。"""

from datetime import datetime
from typing import Optional, Union


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
        cleanup_config: Optional[dict] = None,
        enabled_providers: Optional[dict] = None,
        ghost_model: Optional[str] = None,
        image_data: Optional[str] = None,
        switch_angle_enabled: bool = False,
        afterglow_default: int = 0,
        self_reflection_mode: str = "disabled",
        self_reflection_preset_id: Optional[str] = None,
        self_reflection_n_turns: int = 5,
        farewell_config: Optional[dict] = None,
        relationship_status: str = "active",
    ):
        """キャラクターを新規作成する。

        Args:
            afterglow_default: Afterglow（感情継続機構）の新規チャット作成時デフォルト値。1=ON, 0=OFF。
            self_history: chronicle で更新されるキャラクターの歴史・経緯。
            relationship_state: chronicle で更新されるユーザ・他キャラとの現在の関係。
            self_reflection_mode: 自己参照ループの動作モード。disabled/local_trigger/always。
            self_reflection_preset_id: 契機判断モデルプリセットID（local_trigger 時に使用）。
            self_reflection_n_turns: 自己参照に使う直近ターン数。
            farewell_config: chronicle で更新される感情閾値・退席設定JSON。
            relationship_status: 関係ステータス。"active" または "estranged"。
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
                afterglow_default=afterglow_default,
                self_reflection_mode=self_reflection_mode,
                self_reflection_preset_id=self_reflection_preset_id,
                self_reflection_n_turns=self_reflection_n_turns,
                farewell_config=farewell_config,
                relationship_status=relationship_status,
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

    def delete_character(self, character_id: str) -> bool:
        """キャラクターを削除する。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Character
            char = session.get(Character, character_id)
            if not char:
                return False
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

    def list_estranged_characters(self) -> list:
        """relationship_status が 'estranged' のキャラクター一覧を返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import Character
            return session.query(Character).filter(
                Character.relationship_status == "estranged"
            ).all()
