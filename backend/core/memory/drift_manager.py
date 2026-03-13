"""DriftManager — SELF_DRIFT指針の読み書きを統括する。

MemoryManager が記憶を管理するように、DriftManager はチャット内一時指針（SELF_DRIFT）を管理する。
DriftはSQLiteのみに保存され、ChromaDBへの永続化は不要。
"""

from .sqlite_store import SQLiteStore


class DriftManager:
    """SELF_DRIFT指針の永続化・取得を担うマネージャー。"""

    def __init__(self, sqlite: SQLiteStore) -> None:
        """DriftManager を初期化する。

        Args:
            sqlite: drift の読み書きに使うSQLiteストア。
        """
        self.sqlite = sqlite

    def add_drift(self, session_id: str, character_id: str, content: str) -> object:
        """SELF_DRIFT指針を追加する。

        同キャラの上限3件を超えた場合は最古を自動削除する。

        Args:
            session_id: 所属セッションのID。
            character_id: キャラクターID。
            content: drift内容テキスト。

        Returns:
            作成された SessionDrift レコード。
        """
        return self.sqlite.add_session_drift(session_id, character_id, content)

    def list_active_drifts(self, session_id: str, character_id: str) -> list[str]:
        """有効（ON）なdrift内容テキスト一覧を返す。

        システムプロンプト注入用。キャラクターごとに独立して取得する。

        Args:
            session_id: セッションID。
            character_id: キャラクターID。

        Returns:
            有効なdrift内容テキストのリスト（作成日時昇順）。
        """
        return self.sqlite.list_active_session_drifts(session_id, character_id)

    def list_drifts(self, session_id: str) -> list:
        """セッション内の全キャラのdrift一覧を返す（UI表示用）。

        Args:
            session_id: セッションID。

        Returns:
            SessionDrift レコードのリスト（作成日時昇順）。
        """
        return self.sqlite.list_session_drifts(session_id)

    def toggle_drift(self, drift_id: str) -> object | None:
        """drift の enabled フラグを反転する。

        Args:
            drift_id: トグルするdriftのID。

        Returns:
            更新後の SessionDrift レコード。存在しない場合は None。
        """
        return self.sqlite.toggle_session_drift(drift_id)

    def reset_drifts(self, session_id: str, character_id: str) -> int:
        """指定キャラのdriftを全件物理削除する。

        [DRIFT_RESET] マーカー処理用。他キャラのdriftには影響しない。

        Args:
            session_id: セッションID。
            character_id: リセット対象のキャラクターID。

        Returns:
            削除件数。
        """
        return self.sqlite.reset_session_drifts(session_id, character_id)
