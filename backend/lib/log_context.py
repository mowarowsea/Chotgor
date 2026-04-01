"""ログ用コンテキスト変数とロギング設定。

アプリケーション全体で使用するログフォーマット・ハンドラを一元管理する。
リクエストごとに message_id をセットして、ログ行にトレース情報を付与する。
"""

import logging
import logging.handlers
import os
import uuid as _uuid_mod
from contextvars import ContextVar

# リクエスト識別子（チャット1回ごと、chronicle/forgetキャラごとにセット）
current_message_id: ContextVar[str] = ContextVar("current_message_id", default="--------")

# 現在のLLM呼び出し機能名（chat / power_recall / trigger / reflection / forget / chronicle / group_chat）
current_log_feature: ContextVar[str] = ContextVar("current_log_feature", default="chat")

# デバッグログファイルの通し番号（リクエストごとに new_message_id() でリセット）
_log_call_counter: ContextVar[int] = ContextVar("_log_call_counter", default=0)


def new_message_id() -> str:
    """ログ追跡用の短縮IDを生成して current_message_id にセットして返す。

    チャットリクエスト開始時・chronicle/forget の各キャラ処理開始時に呼び出す。
    デバッグログの通し番号もここでリセットする。

    Returns:
        生成した8文字の16進数ID。
    """
    msg_id = _uuid_mod.uuid4().hex[:8]
    current_message_id.set(msg_id)
    _log_call_counter.set(0)
    return msg_id


def next_log_index() -> int:
    """デバッグログファイルの通し番号を1増やして返す。

    _write_log() から呼び出すことで、リクエスト内の全ログが時系列順に並ぶ。

    Returns:
        インクリメント後の番号（1始まり）。
    """
    n = _log_call_counter.get() + 1
    _log_call_counter.set(n)
    return n


class _ChotgorFormatter(logging.Formatter):
    """message_id を自動付与し、backend. プレフィックスを除去するカスタムフォーマッタ。

    ContextVar から現在の message_id を取得して %(msg_id)s に埋め込む。
    logger 名の "backend." プレフィックスを除去して %(short_name)s に格納する。
    """

    def format(self, record: logging.LogRecord) -> str:
        """レコードに msg_id と short_name を付与してフォーマットする。

        Args:
            record: ロギングレコード。

        Returns:
            フォーマット済み文字列。
        """
        record.msg_id = current_message_id.get()
        record.short_name = record.name.removeprefix("backend.")
        return super().format(record)


def setup_logging() -> None:
    """アプリケーション起動時に一度だけ呼び出すロギング設定。

    以下の2つのハンドラを root ロガーに追加する:
    - コンソール出力 (INFO以上): stdout に常時出力
    - ファイル出力 (DEBUG以上): logs/chotgor.log に RotatingFileHandler で出力

    uvicorn / httpx / chromadb の過剰な DEBUG ログは WARNING レベルに制限する。
    """
    fmt = _ChotgorFormatter(
        "%(asctime)s %(levelname)-7s [%(msg_id)8s] %(short_name)s:%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # コンソール出力（INFO以上）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # ファイル出力（DEBUG以上、RotatingFileHandler: 10MB × 5世代）
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/chotgor.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # uvicorn/fastapi/httpx/chromadb の過剰なDEBUGログを抑制
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
