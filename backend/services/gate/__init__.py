"""応答可能性ゲート（gate）— めぐり（巡り / Aliveness）の動機経済・行動側の土台。

前提となる不信（docs/planned/aliveness_plan.md §5）:
    LLM の会話継続本能は、キャラクター性を押しのけて会話を続けようとする。
    したがって終了系の権利は本人のツール呼び出しに頼らず、**物理（外部ゲート）が
    終わりを決め、本人は終わり方（意味づけ）だけを決める**。

availability は純関数・LLM 不使用。unavailable 中のユーザ発言は預かり（escrow）
され、**LLM を呼ばない** — 呼ばれなければ継続できない（会話継続本能と最初から
戦わない）。
"""

from backend.services.gate.availability import (  # noqa: F401
    Availability,
    check_availability,
    format_escrow_annotation,
    is_usual_scene_running,
    mark_usual_scene_running,
)
from backend.services.gate.fatigue import (  # noqa: F401
    check_fatigue_leave,
)
