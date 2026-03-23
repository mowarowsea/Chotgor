"""既存チャット履歴を ChromaDB の chat_ コレクションへバックフィルするスクリプト。

新規メッセージは保存時に自動でインデックス登録されるが、
導入前の既存メッセージはこのスクリプトで一括登録する。

使い方:
    python -m scripts.backfill_chat_index

環境変数（.env または実行環境で設定）:
    SQLITE_DB_PATH  : SQLite DB のパス（デフォルト: ./data/chotgor.db）
    CHROMA_DB_PATH  : ChromaDB のパス（デフォルト: ./data/chroma）
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# プロジェクトルートを sys.path に追加する（-m 経由での実行に対応）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.chat.indexer import get_participant_char_ids, index_message_sync
from backend.core.memory.chroma_store import ChromaStore
from backend.core.memory.sqlite_store import SQLiteStore

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/chotgor.db")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")


def run() -> None:
    """全セッションのメッセージを chat_ コレクションへインデックス登録する。

    処理フロー:
      1. 全チャットセッションを取得する
      2. セッションごとに参加キャラIDを解決する
      3. セッション内の全メッセージを取得し、is_system_message=1 を除外する
      4. 各メッセージを参加キャラ全員の chat_ コレクションへupsertする
      5. 最後に登録件数をレポートする
    """
    print(f"SQLite: {SQLITE_DB_PATH}")
    print(f"ChromaDB: {CHROMA_DB_PATH}")

    sqlite = SQLiteStore(SQLITE_DB_PATH)

    # embeddingモデル設定を SQLite から取得して ChromaStore に渡す
    all_settings = sqlite.get_all_settings()
    chroma = ChromaStore(
        CHROMA_DB_PATH,
        embedding_provider=all_settings.get("embedding_provider", "default"),
        embedding_model=all_settings.get("embedding_model", ""),
        api_key=all_settings.get("google_api_key", ""),
    )

    user_name = sqlite.get_setting("user_name", "ユーザ")
    print(f"ユーザ名: {user_name}")

    sessions = sqlite.list_chat_sessions(limit=100000)
    print(f"\nセッション数: {len(sessions)}")

    total_indexed = 0
    total_skipped = 0
    error_sessions = 0

    for session in sessions:
        # セッション参加キャラIDを解決する
        char_ids = get_participant_char_ids(session, sqlite)
        if not char_ids:
            # キャラが解決できないセッション（削除済みキャラなど）はスキップする
            print(f"  [SKIP] session={session.id[:8]}... キャラ未解決")
            error_sessions += 1
            continue

        messages = sqlite.list_chat_messages(session.id)
        session_indexed = 0
        session_skipped = 0

        for msg in messages:
            # システムメッセージはインデックス対象外
            if getattr(msg, "is_system_message", None):
                session_skipped += 1
                continue

            try:
                index_message_sync(msg, char_ids, chroma, user_name=user_name)
                session_indexed += 1
            except Exception as e:
                print(f"    [ERROR] msg={msg.id[:8]}... {e}")
                session_skipped += 1

        total_indexed += session_indexed
        total_skipped += session_skipped

        session_label = getattr(session, "title", session.id[:8])
        print(f"  session={session.id[:8]}... '{session_label}' "
              f"chars={len(char_ids)} indexed={session_indexed} skipped={session_skipped}")

    print(f"\n--- 完了 ---")
    print(f"登録済み: {total_indexed} 件")
    print(f"スキップ: {total_skipped} 件")
    print(f"キャラ未解決セッション: {error_sessions} 件")


if __name__ == "__main__":
    run()
