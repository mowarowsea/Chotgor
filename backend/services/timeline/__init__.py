"""タイムライン投影層 — めぐり（巡り / Aliveness）の読み取り側。

正本（timeline_events 封筒）を観測者クラス別の可視性ポリシーでフィルタし、
「誰に・どこまで見せるか」を一元管理する（docs/planned/aliveness_plan.md §2.4〜2.5）。
"""

from backend.services.timeline.projector import (  # noqa: F401
    Budget,
    ProjectedEvent,
    format_real_contact_block,
    project,
    resolve_disclosure,
)
