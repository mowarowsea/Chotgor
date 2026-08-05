"""SSE の送出と生成処理を切り離す共通ランナー。

1on1（`api/chat.py`）とシナリオ（`api/scenario_chat/stream.py`）が共有する。
仕様・経緯は `docs/planned/sse_disconnect_resilience_plan.md` を参照。

設計の要点:
    - **生成本体は独立した asyncio.Task で走らせる**。HTTP 接続が切れても
      タスクはキャンセルせず完走させるため、LLM の応答が「課金だけされて
      DB に残らない」事故が起きない。素朴に SSE ジェネレータの中で生成すると、
      保存コードが `yield` より後ろにある限り切断で巻き戻される。
    - **無通信が続く間は heartbeat（コメント行）を流す**。claude_cli は
      ブロック完成時にしか chunk を出さないため、数十秒の沈黙が普通に発生し、
      モバイル回線や中間装置に「死んだ接続」と見なされて切られる。
    - **切断は例外ではなく記録対象**。warn として残し、あとから頻度を測れるようにする。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable

from backend.lib.debug_logger import logger as debug_logger

logger = logging.getLogger(__name__)

# 無通信がこの秒数続いたら heartbeat を 1 行流す。
# LLM の初回チャンクまで数十秒かかるのが常態なので、それより十分短く取る。
HEARTBEAT_SECONDS = 15.0

# SSE のコメント行。`data:` で始まらないためフロントのパーサは読み飛ばす。
_HEARTBEAT_LINE = ": ping\n\n"

# 生成タスクの強参照を保持する。asyncio はタスクへの参照を持たないと
# 実行中でも GC される可能性があるため、完了までここで握っておく
# （切断後は誰も await しないので、この参照だけが命綱になる）。
_running_tasks: set[asyncio.Task] = set()


class _StreamEnd:
    """生成タスクの終了を送出側へ伝える番兵。"""


async def stream_sse(
    make_source: Callable[[], AsyncIterator[tuple[str, dict[str, Any]]]],
    *,
    label: str,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> AsyncIterator[str]:
    """`(event_type, payload)` を吐く生成器を SSE 行へ変換して送出する。

    生成は独立タスクで走るため、クライアントが切断してもそのまま完走する
    （保存・ログ記録は最後まで行われる）。

    Args:
        make_source: 呼ぶと `(event_type, payload)` を yield する非同期イテレータを返す
            ファクトリ。タスク内で呼ばれるので、生成処理はすべてこの中に閉じること。
        label: ログ用の識別名（"scenario" / "chat" など）。
        heartbeat_seconds: 無通信時に heartbeat を流す間隔（秒）。

    Yields:
        `data: {...}\\n\\n` 形式の SSE 行、または heartbeat のコメント行。
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def _pump() -> None:
        """生成器を回してキューへ流す。例外は error イベントに変換して握り潰さない。"""
        try:
            async for event_type, payload in make_source():
                queue.put_nowait((event_type, payload))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # ここで潰さないと ASGI 例外になり、traceback がファイルログに残らない
            # （uvicorn.error は root へ propagate しない）。
            logger.exception("SSE 生成タスクが例外で終了 label=%s", label)
            queue.put_nowait(("error", {"message": str(e)}))
        finally:
            queue.put_nowait(_StreamEnd)

    task = asyncio.create_task(_pump())
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)

    disconnected = False
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield _HEARTBEAT_LINE
                continue
            if item is _StreamEnd:
                break
            event_type, payload = item
            data = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
            yield f"data: {data}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        # クライアント切断。生成タスクは**あえて cancel しない**（完走させて保存を守る）。
        disconnected = True
        raise
    finally:
        if disconnected and not task.done():
            _log_disconnect(label)


def _log_disconnect(label: str) -> None:
    """切断を warn として記録する（生成は裏で継続中である旨も残す）。

    `debug_logger.log_warning` は現リクエストの MAIN 行へ `has_error` と理由を書くため、
    Logs 画面から「どのリクエストで切れたか」を追える。
    """
    message = (
        f"SSE 接続がレスポンス完了前に切断された（label={label}）。"
        "生成タスクはバックグラウンドで継続し、保存まで完走する。"
    )
    try:
        debug_logger.log_warning("sse_disconnect", message)
    except Exception:
        # 記録の失敗で切断処理を壊さない（この時点で応答経路は既に無い）。
        logger.warning("sse_disconnect の記録に失敗 label=%s", label)


__all__ = ["stream_sse", "HEARTBEAT_SECONDS"]
