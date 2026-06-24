"""1on1 チャット用の SceneLoop Strategy。

1on1 はループ制御を必要としない（ユーザの 1 メッセージに対しキャラが 1 回応答するだけ）
が、SceneLoop の枠組みに乗せることで Scenario と共通の骨で動かす意義がある。

- OneOnOneRouter: 最初の 1 ターンだけ character speaker を返し、その後は終了。
- OneOnOneExecutor: ChatFlow.execute_stream をそのまま回し、最後に TurnResult を返す。

呼び出し例:
    loop = SceneLoop(
        router=OneOnOneRouter(),
        executor=OneOnOneExecutor(chat_flow),
        max_iterations=1,
    )
    initial = LoopState(context={"pending_request": chat_request})
    async for event in loop.run(initial_state=initial):
        ...
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from backend.services.chat_flow.flow import ChatFlow
from backend.services.chat_flow.scene_loop import (
    LoopState,
    SpeakerInfo,
    TurnResult,
)


class OneOnOneRouter:
    """1on1 用 TurnRouter。最初の 1 ターンだけ character を返す。

    LoopState.context["pending_request"] に ChatRequest を入れて呼び出す前提。
    """

    async def stop_condition(self, state: LoopState) -> tuple[bool, str]:
        """1on1 は外部停止条件を持たない。max_iterations と next_speaker で制御する。"""
        return False, ""

    async def next_speaker(self, state: LoopState) -> SpeakerInfo | None:
        """初回のみ context["pending_request"] から character speaker を組み立てて返す。

        2 回目以降は ``None``。1on1 は max_iterations=1 で運用する想定だが、
        万一上限が大きい設定でも 1 ターンで自然停止する。
        """
        if state.iteration > 0:
            return None
        request = state.context.get("pending_request")
        if request is None:
            return None
        return SpeakerInfo(
            kind="character",
            name=getattr(request, "character_name", "") or "",
            id=getattr(request, "character_id", None),
            metadata={"request": request},
        )


class OneOnOneExecutor:
    """1on1 用 TurnExecutor。ChatFlow.execute_stream を素通しで回す。

    SceneLoop は最後に ``("turn_result", TurnResult)`` を期待するため、ストリーミング
    終端で text を集計したものを TurnResult として返す。``angle_switched`` 経由で
    第 2 プロバイダーへ再ディスパッチした場合も最終 clean_text を text として渡す。
    """

    def __init__(self, chat_flow: ChatFlow) -> None:
        self._flow = chat_flow

    async def execute(
        self,
        speaker: SpeakerInfo,
        state: LoopState,
    ) -> AsyncIterator[tuple[str, Any]]:
        """ChatFlow.execute_stream を回し、結果を turn_result として返す。"""
        request = speaker.metadata.get("request")
        if request is None:
            yield ("turn_result", TurnResult(text="", error="missing pending_request"))
            return

        full_text = ""
        async for event in self._flow.execute_stream(request):
            yield event
            # 1on1 経路で「最終応答テキスト」とみなせるのは ``text`` チャンクの累計。
            # power_recall / switch_angle の再帰先で出された text もすべて連結される。
            if isinstance(event, tuple) and len(event) == 2 and event[0] == "text":
                content = event[1]
                if isinstance(content, str):
                    full_text += content

        yield ("turn_result", TurnResult(text=full_text))


__all__ = ["OneOnOneRouter", "OneOnOneExecutor"]
