"""LLMモデルプリセット CRUD — SQLiteStore Mixin。"""

from typing import Optional


class PresetStoreMixin:
    """LLMモデルプリセットの作成・取得・更新・削除を担う Mixin。"""

    def create_model_preset(
        self, preset_id: str, name: str, provider: str, model_id: str, thinking_level: str = "default"
    ):
        """LLMモデルプリセットを新規作成する。"""
        with self.get_session() as session:
            from ..sqlite_store import LLMModelPreset
            preset = LLMModelPreset(
                id=preset_id, name=name, provider=provider, model_id=model_id, thinking_level=thinking_level
            )
            session.add(preset)
            session.commit()
            session.refresh(preset)
            return preset

    def list_model_presets(self) -> list:
        """LLMモデルプリセット一覧を作成日順で返す。"""
        with self.get_session() as session:
            from ..sqlite_store import LLMModelPreset
            return session.query(LLMModelPreset).order_by(LLMModelPreset.created_at).all()

    def get_model_preset(self, preset_id: str):
        """IDでLLMモデルプリセットを取得する。"""
        with self.get_session() as session:
            from ..sqlite_store import LLMModelPreset
            return session.get(LLMModelPreset, preset_id)

    def get_model_preset_by_name(self, name: str):
        """名前でLLMモデルプリセットを取得する。"""
        with self.get_session() as session:
            from ..sqlite_store import LLMModelPreset
            return (
                session.query(LLMModelPreset)
                .filter(LLMModelPreset.name == name)
                .first()
            )

    def update_model_preset(self, preset_id: str, **kwargs):
        """LLMモデルプリセットの指定フィールドを更新する。"""
        with self.get_session() as session:
            from ..sqlite_store import LLMModelPreset
            preset = session.get(LLMModelPreset, preset_id)
            if not preset:
                return None
            for k, v in kwargs.items():
                if hasattr(preset, k):
                    setattr(preset, k, v)
            session.commit()
            session.refresh(preset)
            return preset

    def delete_model_preset(self, preset_id: str) -> bool:
        """LLMモデルプリセットを削除する。"""
        with self.get_session() as session:
            from ..sqlite_store import LLMModelPreset
            preset = session.get(LLMModelPreset, preset_id)
            if not preset:
                return False
            session.delete(preset)
            session.commit()
            return True
