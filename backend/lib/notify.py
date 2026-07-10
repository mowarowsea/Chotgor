"""ntfy プッシュ通知ユーティリティ。

キャラクターがユーザー宛に「発話」を届けたタイミング（＝キャラクター回答到着完了）で、
docker で立てた ntfy サーバーの特定トピックへプッシュ通知を送る薄い層。

設計方針:
- **ベストエフォート**。通知の失敗はチャット本流を絶対に止めない（例外は握り潰す）。
- **fire-and-forget**。呼び出し元（同期・非同期どちらの文脈でも）をブロックしないよう、
  送信はデーモンスレッドへ逃がす。backend の「ブロッキング I/O はスレッドへ」方針に沿う。
- **設定は環境変数で上書き可能**。既定値は WSL/docker の ntfy（8085:80 マップ）に合わせている。
"""

import logging
import os
import threading

import httpx

_log = logging.getLogger("chotgor.notify")

# ntfy サーバーの発行先ベース URL。docker-compose で 8085:80 にマップした ntfy コンテナ。
# backend は Windows ホスト直実行で、WSL2 の localhost フォワードにより 8085 に到達できる。
NTFY_BASE_URL = os.environ.get("NTFY_URL", "http://localhost:8085").rstrip("/")
# 通知先トピック。
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "Chotgor")
# 通知機能の有効/無効トグル。"0" / "false" / 空文字で無効化できる。
NTFY_ENABLED = os.environ.get("NTFY_ENABLED", "1").strip().lower() not in ("", "0", "false", "no")


def _send(message: str) -> None:
    """ntfy トピックへ 1 件の通知を同期 POST する（ワーカースレッド上で実行）。

    失敗しても本流に影響を与えないよう、あらゆる例外を握り潰して警告ログに留める。

    Args:
        message: 通知本文。日本語を含むため UTF-8 バイト列で送る。
    """
    url = f"{NTFY_BASE_URL}/{NTFY_TOPIC}"
    try:
        httpx.post(url, content=message.encode("utf-8"), timeout=5.0)
    except Exception as e:
        # 通知はベストエフォート。ntfy 停止中でもチャット応答は止めない。
        _log.warning("ntfy 通知の送信に失敗: %s (%s)", e, url)


def notify_character_spoke(character_name: str) -> None:
    """キャラクターがユーザー宛に発話したことを ntfy へ非同期通知する。

    「キャラクター回答到着完了」の各経路（1on1 同期返信・預かり配達の遅延返信・
    能動 push・シナリオ/うつつの本人発話）から共通で呼ぶ入口。送信はデーモンスレッドへ
    投げるため、呼び出し元（同期・非同期どちらでも）をブロックしない。

    Args:
        character_name: 発話したキャラクター名。通知本文に埋め込む。
    """
    if not NTFY_ENABLED or not character_name:
        return
    message = f"Chotgor:{character_name}が発話"
    threading.Thread(target=_send, args=(message,), daemon=True).start()
