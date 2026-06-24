"""SceneLoop の単体テスト。

ループ骨格（停止判定 → 次の話者問い合わせ → 実行 → イベント転送 → state 反映）が
仕様どおりに動作するかを、ダミーの TurnRouter / TurnExecutor を差し込んで検証する。
1on1 / Scenario 等の具体的なフローは別途各サービス層のテストで担保するので、
ここでは抽象骨そのものの振る舞いだけに集中する。
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from backend.services.chat_flow.scene_loop import (
    LoopState,
    SceneLoop,
    SpeakerInfo,
    TurnResult,
)


class _ScriptedRouter:
    """テスト用 Router。あらかじめ与えた話者列を順に返し、列が尽きると None を返す。

    停止条件は外部から ``stop_after`` で指定できる（その回数の iteration に達したら
    Router 側でループ停止を要求する）。
    """

    def __init__(
        self,
        speakers: list[SpeakerInfo | None],
        stop_after: int | None = None,
        stop_reason: str = "router_stop",
    ) -> None:
        self._speakers = list(speakers)
        self._index = 0
        self._stop_after = stop_after
        self._stop_reason = stop_reason

    async def stop_condition(self, state: LoopState) -> tuple[bool, str]:
        if self._stop_after is not None and state.iteration >= self._stop_after:
            return True, self._stop_reason
        return False, ""

    async def next_speaker(self, state: LoopState) -> SpeakerInfo | None:
        if self._index >= len(self._speakers):
            return None
        spk = self._speakers[self._index]
        self._index += 1
        return spk


class _RecordingExecutor:
    """テスト用 Executor。話者ごとに固定イベント列を yield し、最後に TurnResult を返す。

    ``results`` を引数で渡すと話者順に対応する TurnResult を返す。
    省略時はテキスト ``"<name>:resp"`` で正常応答を組み立てる。
    """

    def __init__(self, results: list[TurnResult] | None = None) -> None:
        self._results = results or []
        self._call_count = 0

    async def execute(
        self,
        speaker: SpeakerInfo,
        state: LoopState,
    ) -> AsyncIterator[tuple[str, object]]:
        # 1 ターンあたり 2 つの中間イベントを yield してから turn_result を返す。
        yield ("speaker_start", {"name": speaker.name})
        yield ("chunk", {"text": f"hello from {speaker.name}"})
        if self._call_count < len(self._results):
            result = self._results[self._call_count]
        else:
            result = TurnResult(text=f"{speaker.name}:resp")
        self._call_count += 1
        yield ("turn_result", result)


class TestSceneLoopBasic:
    """SceneLoop の基本動作（イベント転送・state 反映・上限保護）を確認するテスト群。"""

    @pytest.mark.asyncio
    async def test_single_turn_yields_events_and_advances_state(self):
        """1 ターンを実行すると、Executor の中間イベントが上位へ転送され、
        最終 ``turn_result`` は state.last_result に反映されること。"""
        router = _ScriptedRouter([SpeakerInfo(kind="character", name="Alice")])
        executor = _RecordingExecutor()
        loop = SceneLoop(router=router, executor=executor, max_iterations=5)

        events = [ev async for ev in loop.run()]

        # 期待: speaker_start, chunk, loop_complete（turn_result は転送されない）
        types = [t for t, _ in events]
        assert types == ["speaker_start", "chunk", "loop_complete"]
        last = events[-1][1]
        assert last["iterations"] == 1
        # speaker 列が尽きたので no_more_speakers で終了する
        assert last["reason"] == "no_more_speakers"

    @pytest.mark.asyncio
    async def test_multiple_speakers_iterate_in_order(self):
        """複数話者を順に Router が返す場合、iteration ごとに対応する話者で実行されること。"""
        router = _ScriptedRouter([
            SpeakerInfo(kind="gm", name="GM"),
            SpeakerInfo(kind="pc", name="はる"),
            SpeakerInfo(kind="gm", name="GM"),
        ])
        executor = _RecordingExecutor()
        loop = SceneLoop(router=router, executor=executor, max_iterations=10)

        events = [ev async for ev in loop.run()]

        # 各 iteration で speaker_start が 1 度ずつ流れる
        starts = [p["name"] for t, p in events if t == "speaker_start"]
        assert starts == ["GM", "はる", "GM"]
        last = events[-1][1]
        assert last["iterations"] == 3
        assert last["reason"] == "no_more_speakers"

    @pytest.mark.asyncio
    async def test_max_iterations_caps_loop(self):
        """話者列が無限にあっても、max_iterations で確実に止まること。"""
        # 同じ話者を 10 回返す Router（実質無限）
        router = _ScriptedRouter([SpeakerInfo(kind="pc", name="X")] * 10)
        executor = _RecordingExecutor()
        loop = SceneLoop(router=router, executor=executor, max_iterations=3)

        events = [ev async for ev in loop.run()]

        starts = [p["name"] for t, p in events if t == "speaker_start"]
        assert len(starts) == 3
        last = events[-1][1]
        assert last["iterations"] == 3
        assert last["reason"] == "max_iterations"

    @pytest.mark.asyncio
    async def test_stop_condition_terminates_loop(self):
        """Router が stop_condition で True を返すと、その iteration で即終了すること。"""
        router = _ScriptedRouter(
            [SpeakerInfo(kind="pc", name="A"), SpeakerInfo(kind="pc", name="B")],
            stop_after=1,
            stop_reason="scene_close",
        )
        executor = _RecordingExecutor()
        loop = SceneLoop(router=router, executor=executor, max_iterations=10)

        events = [ev async for ev in loop.run()]

        starts = [p["name"] for t, p in events if t == "speaker_start"]
        # 1 ターン目は実行、2 ターン目に入る前に stop_condition で打ち切り
        assert starts == ["A"]
        last = events[-1][1]
        assert last["iterations"] == 1
        assert last["reason"] == "scene_close"

    @pytest.mark.asyncio
    async def test_executor_error_terminates_loop(self):
        """Executor が TurnResult.error を返したら、次ターンへ進まずループを止めること。"""
        router = _ScriptedRouter([
            SpeakerInfo(kind="pc", name="A"),
            SpeakerInfo(kind="pc", name="B"),
        ])
        executor = _RecordingExecutor(results=[
            TurnResult(text="", error="provider down"),
            TurnResult(text="never reached"),
        ])
        loop = SceneLoop(router=router, executor=executor, max_iterations=10)

        events = [ev async for ev in loop.run()]

        starts = [p["name"] for t, p in events if t == "speaker_start"]
        # 1 ターン目だけ実行され、エラーで停止する
        assert starts == ["A"]
        last = events[-1][1]
        assert last["reason"] == "executor_error"

    @pytest.mark.asyncio
    async def test_router_returns_none_terminates_loop(self):
        """Router が next_speaker で None を返したらユーザ待ちとしてループ終了。"""
        router = _ScriptedRouter([
            SpeakerInfo(kind="pc", name="A"),
            None,  # 2 ターン目は None
        ])
        executor = _RecordingExecutor()
        loop = SceneLoop(router=router, executor=executor, max_iterations=10)

        events = [ev async for ev in loop.run()]

        starts = [p["name"] for t, p in events if t == "speaker_start"]
        assert starts == ["A"]
        last = events[-1][1]
        assert last["iterations"] == 1
        assert last["reason"] == "no_more_speakers"

    @pytest.mark.asyncio
    async def test_initial_state_is_carried_through(self):
        """initial_state を渡すとループ全体で同じ state を使い回せる
        （context を介して Router/Executor が情報共有できる）。"""
        state = LoopState(context={"shared": "scenario-X"})

        seen_context: list[dict] = []

        class _ContextSpyExecutor:
            async def execute(self, speaker, st):
                seen_context.append(dict(st.context))
                yield ("turn_result", TurnResult(text="ok"))

        router = _ScriptedRouter([SpeakerInfo(kind="pc", name="A")])
        loop = SceneLoop(router=router, executor=_ContextSpyExecutor(), max_iterations=5)

        events = [ev async for ev in loop.run(initial_state=state)]

        assert seen_context == [{"shared": "scenario-X"}]
        assert events[-1][0] == "loop_complete"

    @pytest.mark.asyncio
    async def test_turn_result_updates_last_result(self):
        """Executor が返した TurnResult が次ターンの Router からも参照できること。"""
        observed_last_text: list[str] = []

        class _LastTextSpyRouter:
            def __init__(self) -> None:
                self._calls = 0

            async def stop_condition(self, state):
                observed_last_text.append(
                    state.last_result.text if state.last_result else ""
                )
                return False, ""

            async def next_speaker(self, state):
                self._calls += 1
                if self._calls > 2:
                    return None
                return SpeakerInfo(kind="pc", name=f"call{self._calls}")

        executor = _RecordingExecutor(results=[
            TurnResult(text="first-text"),
            TurnResult(text="second-text"),
        ])
        loop = SceneLoop(router=_LastTextSpyRouter(), executor=executor, max_iterations=5)

        _ = [ev async for ev in loop.run()]

        # 1 回目はまだ last_result が無いので空、2 回目は first-text、3 回目は second-text
        assert observed_last_text == ["", "first-text", "second-text"]
