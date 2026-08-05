"""lib/sse_runner.stream_sse — SSE 切断耐性ランナーのユニットテスト。

背景:
    生成の永続化が SSE 接続の生存に依存していたため、クライアントが切断すると
    「LLM は完走して課金もされたのに応答が DB に一行も残らない」事故が起きていた
    （debug_log_entries.request_id=c799f37d ほか）。経緯と設計判断は
    docs/planned/sse_disconnect_resilience_plan.md を参照。

検証対象:
    - イベントタプル → SSE 行への変換（正常系・done まで）
    - 生成器が例外で落ちたときの error イベントへの変換（ASGI 例外にしない）
    - 無通信時の heartbeat 送出（沈黙で切られるのを防ぐ予防線）
    - **切断されても生成タスクが完走すること**（このモジュールの存在理由）
    - 切断時に警告が記録されること（あとから頻度を測るための計器）
"""

import asyncio
import json

import pytest

from backend.lib import sse_runner
from backend.lib.sse_runner import stream_sse


def _parse(line: str) -> dict:
    """`data: {...}` 形式の SSE 行を dict に戻すヘルパ。"""
    assert line.startswith("data: "), line
    assert line.endswith("\n\n"), line
    return json.loads(line[len("data: "):].strip())


async def _collect(agen) -> list[str]:
    """非同期ジェネレータを最後まで消費して行のリストを返すヘルパ。"""
    return [line async for line in agen]


# ─── 正常系 ──────────────────────────────────────────────────────────────────


class TestStreamSseHappyPath:
    """切断が起きない通常フローの送出内容を検証するテストクラス。

    生成器が yield した `(event_type, payload)` が、順序を保ったまま
    `data: {"type": ..., ...payload}` の SSE 行へ 1 対 1 で変換されることを確認する。
    payload のキーが type と同階層へ展開される点（フロントが `ev.content` の形で
    読む前提）もここで担保する。
    """

    @pytest.mark.asyncio
    async def test_events_are_converted_to_sse_lines(self):
        """イベントタプルが SSE 行へ順序どおり変換されること。"""
        async def source():
            yield ("chunk", {"content": "こんにちは"})
            yield ("chunk", {"content": "、はる"})
            yield ("done", {"log_message_id": "abc123"})

        lines = await _collect(stream_sse(source, label="test"))

        assert [_parse(x) for x in lines] == [
            {"type": "chunk", "content": "こんにちは"},
            {"type": "chunk", "content": "、はる"},
            {"type": "done", "log_message_id": "abc123"},
        ]

    @pytest.mark.asyncio
    async def test_non_ascii_is_not_escaped(self):
        """日本語が \\uXXXX へエスケープされずそのまま流れること。"""
        async def source():
            yield ("chunk", {"content": "紅音"})

        lines = await _collect(stream_sse(source, label="test"))

        assert "紅音" in lines[0]

    @pytest.mark.asyncio
    async def test_empty_source_emits_nothing(self):
        """1 件も yield しない生成器では SSE 行が出ないこと。"""
        async def source():
            return
            yield  # pragma: no cover — 空の非同期ジェネレータにするための番人

        assert await _collect(stream_sse(source, label="test")) == []


# ─── 生成側の例外 ────────────────────────────────────────────────────────────


class TestStreamSseSourceError:
    """生成器が例外で落ちた場合の扱いを検証するテストクラス。

    例外をそのまま外へ投げると ASGI 例外になり、uvicorn.error が root へ
    propagate しない設定のためファイルログに traceback が残らない
    （事故調査時に「何も記録がない」状態になる）。error イベントへ変換して
    フロントに届けたうえで、python logger 側に例外を残すのが期待動作。
    """

    @pytest.mark.asyncio
    async def test_exception_becomes_error_event(self):
        """例外が error イベントへ変換され、それまでの出力は保たれること。"""
        async def source():
            yield ("chunk", {"content": "途中まで"})
            raise RuntimeError("provider が落ちた")

        lines = await _collect(stream_sse(source, label="test"))

        assert _parse(lines[0]) == {"type": "chunk", "content": "途中まで"}
        assert _parse(lines[1]) == {"type": "error", "message": "provider が落ちた"}


# ─── heartbeat ───────────────────────────────────────────────────────────────


class TestStreamSseHeartbeat:
    """無通信時の heartbeat 送出を検証するテストクラス。

    claude_cli はブロック完成時にしか chunk を出さないため、数十秒の沈黙が
    常態として発生する（実測 42 秒）。その沈黙で接続を切られるのを防ぐため、
    一定間隔で SSE コメント行を流す。コメント行は `data:` で始まらないので、
    フロントの parseSSEStream に読み飛ばされる（パーサ改修は不要）。
    """

    @pytest.mark.asyncio
    async def test_ping_is_sent_while_source_is_silent(self):
        """生成器が黙っている間、heartbeat のコメント行が流れること。"""
        async def source():
            await asyncio.sleep(0.25)
            yield ("done", {})

        lines = await _collect(
            stream_sse(source, label="test", heartbeat_seconds=0.05)
        )

        assert lines.count(": ping\n\n") >= 2
        assert _parse(lines[-1]) == {"type": "done"}

    @pytest.mark.asyncio
    async def test_ping_is_not_data_line(self):
        """heartbeat 行がフロントのパーサ条件（`data: ` 始まり）に一致しないこと。"""
        assert not sse_runner._HEARTBEAT_LINE.startswith("data: ")


# ─── 切断耐性（このモジュールの存在理由） ────────────────────────────────────


class TestStreamSseDisconnect:
    """クライアント切断時の振る舞いを検証するテストクラス。

    このランナーの中核。送出側が途中で閉じられても（＝ブラウザが切れても）、
    生成タスクは cancel されずに最後まで走りきり、DB 保存に相当する副作用が
    実行されることを保証する。旧実装ではここで巻き戻され、応答が丸ごと
    消えていた（課金だけ残る）。あわせて、切断が warn として記録されることも
    確認する（記録がないと事故が再発しても気づけないため）。
    """

    @pytest.mark.asyncio
    async def test_source_completes_after_client_disconnect(self):
        """送出を途中で閉じても、生成器は最後まで走って副作用を残すこと。"""
        saved: list[str] = []

        async def source():
            yield ("chunk", {"content": "最初のチャンク"})
            await asyncio.sleep(0.05)
            # 実運用ではここが _save_turn / create_chat_message にあたる
            saved.append("保存された")
            yield ("done", {})

        agen = stream_sse(source, label="test")
        first = await agen.__anext__()
        assert _parse(first)["content"] == "最初のチャンク"

        # クライアント切断相当: 送出側だけを閉じる
        await agen.aclose()

        # 生成タスクは道連れにされず完走する
        for _ in range(100):
            if saved:
                break
            await asyncio.sleep(0.01)
        assert saved == ["保存された"]

    @pytest.mark.asyncio
    async def test_disconnect_is_recorded_as_warning(self, monkeypatch):
        """切断時に sse_disconnect の警告が記録されること。"""
        recorded: list[tuple[str, str]] = []
        monkeypatch.setattr(
            sse_runner.debug_logger,
            "log_warning",
            lambda tag, message: recorded.append((tag, message)),
        )

        started = asyncio.Event()

        async def source():
            started.set()
            await asyncio.sleep(0.2)
            yield ("done", {})

        agen = stream_sse(source, label="scenario", heartbeat_seconds=0.01)
        # heartbeat を 1 本受け取った時点＝生成器はまだ走っている
        assert await agen.__anext__() == ": ping\n\n"
        await started.wait()
        await agen.aclose()

        assert recorded and recorded[0][0] == "sse_disconnect"
        assert "scenario" in recorded[0][1]

    @pytest.mark.asyncio
    async def test_no_warning_when_stream_completes(self, monkeypatch):
        """完走したときは切断警告を出さないこと（誤検知の防止）。"""
        recorded: list[tuple[str, str]] = []
        monkeypatch.setattr(
            sse_runner.debug_logger,
            "log_warning",
            lambda tag, message: recorded.append((tag, message)),
        )

        async def source():
            yield ("done", {})

        await _collect(stream_sse(source, label="chat"))

        assert recorded == []
