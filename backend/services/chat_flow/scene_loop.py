"""SceneLoop — 「次の話者を決めて 1 ターン実行する」を繰り返す共通骨。

1on1（max_iterations=1 で character ターン 1 回）と Scenario（メンション解析・
SCENE_CLOSE・headless 等で連鎖）を同じインターフェースに乗せる。

責務分担:
    - SceneLoop: ループ制御（停止判定 + 次の話者問い合わせ + 実行 + イベント転送）
    - TurnRouter: 「次の話者は誰か」「ループ停止条件か」を判断する Strategy
    - TurnExecutor: 「指定された話者の 1 ターン」を実行する Strategy

TurnExecutor は最後に ``("turn_result", TurnResult)`` を yield することで、
ループへ実行結果（生テキスト・最終応答）を返す。SceneLoop はそれを LoopState に
反映して次のループへ進む。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


@dataclass
class SpeakerInfo:
    """発話者の同定情報。"""

    # "user" | "character" | "gm" | "pc" 等。Router/Executor の合意で決める。
    kind: str
    # 表示用名前（character_name / role_name / "Narrator" など）
    name: str
    # 関連 ID（character_id / slot_id 等）。Executor が必要に応じて参照する。
    id: str | None = None
    # 自由メタデータ（PC モードの slot 情報、GM の preset_id 等）。
    metadata: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    """1 ターン完了時に Executor が SceneLoop へ返す結果。

    text:
        最終応答テキスト（後続ターンのメンション解析や last_text 参照に使う）。
    raw:
        プロバイダ生応答（SCENE_CLOSE 検出等で使う）。text と異なる場合のみ
        非空にする。空なら text と同じものとみなしてよい。
    error:
        例外/プロバイダエラー文字列。非空ならループ側がそれを終端事由として
        記録する。次ターンは進めない（Router 側で is_done 判定する）。
    """

    text: str = ""
    raw: str = ""
    error: str | None = None


@dataclass
class LoopState:
    """ループ進行中の可変状態。Router / Executor / 呼び出し元が共有する。

    iteration:
        次のターンが何回目か（0 始まり）。SceneLoop が毎ターン後にインクリメント。
    last_speaker:
        直前ターンの話者。Router が次の話者判定に使う（GM の次は PC、等）。
    last_result:
        直前ターンの TurnResult。Router が SCENE_CLOSE 検出・メンション解析等に使う。
    done / done_reason:
        Router が stop_condition で True を返すと SceneLoop がここをセットしてループ脱出。
    context:
        Router/Executor が自由に使える共有メモリ。call site が初期データを詰めてよい
        （Scenario なら scenario / session / npcs / pc_slots 等）。
    """

    iteration: int = 0
    last_speaker: SpeakerInfo | None = None
    last_result: TurnResult | None = None
    done: bool = False
    done_reason: str = ""
    context: dict = field(default_factory=dict)


class TurnRouter(Protocol):
    """次の話者を決める Strategy。"""

    async def next_speaker(self, state: LoopState) -> SpeakerInfo | None:
        """次の話者を返す。``None`` ならループ終了（ユーザ待ちなど）。

        ループ停止の主な判断はこのメソッドの返り値 ``None`` と
        ``stop_condition`` の組み合わせで行う。
        """
        ...

    async def stop_condition(self, state: LoopState) -> tuple[bool, str]:
        """ループ継続前のガード判定。

        Returns:
            ``(stop, reason)``。``stop=True`` ならその場で終了し、
            ``reason`` は ``LoopState.done_reason`` にセットされる。
        """
        ...


class TurnExecutor(Protocol):
    """指定話者の 1 ターンを実行する Strategy。"""

    async def execute(
        self,
        speaker: SpeakerInfo,
        state: LoopState,
    ) -> AsyncIterator[tuple[str, Any]]:
        """1 ターン分のイベントを yield する。

        最後に必ず ``("turn_result", TurnResult)`` を yield すること。
        SceneLoop はそれを ``state.last_result`` に格納し、それ以外のイベントは
        そのまま上位へ転送する。
        """
        ...


class SceneLoop:
    """1 シーン分のターン連鎖を回す薄い骨。"""

    def __init__(
        self,
        router: TurnRouter,
        executor: TurnExecutor,
        max_iterations: int = 10,
    ) -> None:
        """SceneLoop を初期化する。

        Args:
            router: 次の話者と停止条件を判断する Strategy。
            executor: 1 ターン分の実行を担う Strategy。
            max_iterations: ループの保険上限。これに達したら ``done_reason="max_iterations"``
                で終了する。1on1 は 1、Scenario は usual_config / シナリオ既定上限を渡す。
        """
        self._router = router
        self._executor = executor
        self._max_iterations = max_iterations

    async def run(
        self,
        initial_state: LoopState | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """ループを回し、各ターンのイベントを上位へ転送する。

        終了時に ``("loop_complete", {"iterations": int, "reason": str})`` を 1 度 yield する。
        Executor の ``("turn_result", TurnResult)`` イベントは外へは転送せず、
        ``state.last_result`` への反映のみに使う。
        """
        state = initial_state if initial_state is not None else LoopState()

        while True:
            # 保険上限
            if state.iteration >= self._max_iterations:
                state.done = True
                state.done_reason = "max_iterations"
                break

            # Router の事前停止判定（SCENE_CLOSE 検出後の幕引き等）
            stop, reason = await self._router.stop_condition(state)
            if stop:
                state.done = True
                state.done_reason = reason
                break

            # 次の話者を確定
            speaker = await self._router.next_speaker(state)
            if speaker is None:
                state.done = True
                state.done_reason = "no_more_speakers"
                break

            # 実行。turn_result イベントは内部消費、それ以外は素通し。
            async for event in self._executor.execute(speaker, state):
                if isinstance(event, tuple) and len(event) == 2 and event[0] == "turn_result":
                    payload = event[1]
                    if isinstance(payload, TurnResult):
                        state.last_result = payload
                    continue
                yield event

            state.iteration += 1
            state.last_speaker = speaker

            # Executor がエラーを返した場合は次ターンへ進まずここで終了する。
            if state.last_result is not None and state.last_result.error:
                state.done = True
                state.done_reason = "executor_error"
                break

        yield ("loop_complete", {"iterations": state.iteration, "reason": state.done_reason})


__all__ = [
    "SceneLoop",
    "LoopState",
    "SpeakerInfo",
    "TurnResult",
    "TurnRouter",
    "TurnExecutor",
]
