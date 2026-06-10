"""logs_ui パッケージの共有状態。

ログ閲覧UIが参照するパス・テンプレート・SQLiteStore を一元管理する。
モジュール変数を直接書き換えれば挙動を差し替えられる（テストの monkeypatch 対象）。
パスを参照する側は必ず `config.DEBUG_BASE` のように属性アクセスで読むこと
（`from ... import DEBUG_BASE` で値をコピーすると差し替えが効かなくなる）。
"""

from pathlib import Path

from fastapi.templating import Jinja2Templates

# debug/ フォルダのベースパス（サーバ起動ディレクトリ基準）
DEBUG_BASE = Path("debug")
# chotgor.log のパス（RotatingFileHandler の primary ファイル）
CHOTGOR_LOG = Path("logs/chotgor.log")

# UIテンプレートインスタンス（main.py から注入される）
templates: Jinja2Templates | None = None

# SQLiteStore インスタンス（main.py の lifespan から注入される）
_store = None


def set_templates(t: Jinja2Templates) -> None:
    """テンプレートインスタンスをセットする。main.py の起動時に呼ぶ。

    Args:
        t: Jinja2Templates インスタンス。
    """
    global templates
    templates = t


def get_templates() -> Jinja2Templates:
    """テンプレートインスタンスを取得する。未初期化の場合は例外を送出する。"""
    if templates is None:
        raise RuntimeError("Templates not initialized")
    return templates


def set_sqlite_store(sqlite) -> None:
    """SQLiteStore をセットする。main.py の lifespan から起動時に呼ぶ。

    Args:
        sqlite: SQLiteStore インスタンス。
    """
    global _store
    _store = sqlite


def get_sqlite_store():
    """セット済みの SQLiteStore を返す。

    logs_ui はルーターなので app.state に直接アクセスできないため、
    モジュール変数 `_store` に起動時にセットする方式を取る。
    セットされていない場合（テスト時等）は None を返す。
    """
    return _store
