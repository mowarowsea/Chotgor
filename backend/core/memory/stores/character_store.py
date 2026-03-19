"""キャラクター CRUD — SQLiteStore Mixin。"""

from datetime import datetime
from typing import Optional


class CharacterStoreMixin:
    """キャラクターの作成・取得・更新・削除を担う Mixin。"""

    def create_character(
        self,
        character_id: str,
        name: str,
        system_prompt_block1: str = "",
        meta_instructions: str = "",
        cleanup_config: Optional[dict] = None,
        enabled_providers: Optional[dict] = None,
        ghost_model: Optional[str] = None,
        image_data: Optional[str] = None,
        switch_angle_enabled: bool = False,
    ):
        """キャラクターを新規作成する。"""
        with self.get_session() as session:
            from ..sqlite_store import Character
            char = Character(
                id=character_id,
                name=name,
                system_prompt_block1=system_prompt_block1,
                meta_instructions=meta_instructions,
                cleanup_config=cleanup_config or {},
                enabled_providers=enabled_providers or {},
                ghost_model=ghost_model,
                image_data=image_data,
                switch_angle_enabled=1 if switch_angle_enabled else 0,
            )
            session.add(char)
            session.commit()
            session.refresh(char)
            return char

    def get_character(self, character_id: str):
        """IDでキャラクターを取得する。"""
        with self.get_session() as session:
            from ..sqlite_store import Character
            return session.get(Character, character_id)

    def get_character_by_name(self, name: str):
        """名前でキャラクターを取得する。"""
        with self.get_session() as session:
            from ..sqlite_store import Character
            return session.query(Character).filter(Character.name == name).first()

    def list_characters(self) -> list:
        """全キャラクターを返す。"""
        with self.get_session() as session:
            from ..sqlite_store import Character
            return session.query(Character).all()

    def update_character(self, character_id: str, **kwargs):
        """キャラクターの指定フィールドを更新する。"""
        with self.get_session() as session:
            from ..sqlite_store import Character
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
            from ..sqlite_store import Character
            char = session.get(Character, character_id)
            if not char:
                return False
            session.delete(char)
            session.commit()
            return True
