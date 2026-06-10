"""ログ閲覧UIパッケージ。

旧 `backend/api/logs_ui.py`（単一ファイル）を責務単位で分割したパッケージ。

- `config.py`      — 共有状態（DEBUG_BASE / CHOTGOR_LOG / templates / SQLiteStore）
- `tag_extract.py` — JSON解析＋タグ／ツール呼び出し抽出
- `entries.py`     — ログエントリ構築＋一覧ロード
- `json_view.py`   — JSON整形HTMLレンダリング
- `routes.py`      — HTML/JSONルーター定義

main.py からは従来通り `router` / `json_router` / `set_sqlite_store` /
`set_templates` を参照できる（互換ファサード）。
パス類（DEBUG_BASE 等）を差し替えるテストは `config` モジュールを直接 monkeypatch すること。
"""

from backend.api.logs_ui.config import (  # noqa: F401
    get_templates,
    set_sqlite_store,
    set_templates,
)
from backend.api.logs_ui.routes import json_router, router  # noqa: F401
