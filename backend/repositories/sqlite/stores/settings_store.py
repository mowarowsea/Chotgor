"""グローバル設定 CRUD — SQLiteStore Mixin。"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# APIキー・シークレット系のキー名に含まれるキーワード（マスク対象）
_SENSITIVE_KEYS = ("api_key", "secret", "token", "password")


class SettingsStoreMixin:
    """グローバル設定（key/value）の読み書きを担う Mixin。"""

    def get_setting(self, key: str, default: Any = None) -> Any:
        """キーで設定値を取得する。JSON文字列は自動的にパースする。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import GlobalSetting
            row = session.get(GlobalSetting, key)
            if row is None:
                return default
            try:
                return json.loads(row.value)
            except (json.JSONDecodeError, TypeError):
                return row.value

    def set_setting(self, key: str, value: Any) -> None:
        """設定値をupsertする。文字列以外はJSONシリアライズする。

        APIキー・シークレット系のキーは値をマスクしてログ出力する。
        """
        with self.get_session() as session:
            from backend.repositories.sqlite.store import GlobalSetting
            serialized = json.dumps(value) if not isinstance(value, str) else value
            row = session.get(GlobalSetting, key)
            if row:
                row.value = serialized
            else:
                session.add(GlobalSetting(key=key, value=serialized))
            session.commit()
        # センシティブなキーは値をマスクして記録する
        masked = "***" if any(s in key for s in _SENSITIVE_KEYS) else str(value)[:80]
        logger.debug("設定更新 key=%s value=%s", key, masked)

    def get_all_settings(self) -> dict[str, Any]:
        """全設定をdict形式で返す。"""
        with self.get_session() as session:
            from backend.repositories.sqlite.store import GlobalSetting
            rows = session.query(GlobalSetting).all()
            result = {}
            for row in rows:
                try:
                    result[row.key] = json.loads(row.value)
                except (json.JSONDecodeError, TypeError):
                    result[row.key] = row.value
            return result
