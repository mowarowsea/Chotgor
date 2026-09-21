"""意図（intents）— めぐり（巡り / Aliveness）の動機経済・意図層。

欲求は行動に**先行**し、事後に遡って発見できねばならない（docs/planned/aliveness_plan.md §4.3）。
「システムがキャラにプッシュを命じる」方式は却下済み — 行動の瞬間に欲求を
捏造させるため。意図は夜のChronicleとうつつシーン完走後の「拾い上げ」で
本人の言葉のまま記録され、終端遷移（手放す／満ちた／不満）も機械は候補を挙げるだけで本人が裁く。

意図圧の唯一の減衰源は「一区切り（settled）」で、封筒 intent.settled が起点を今へ移す。
受け口は3箇所 — 行動権の帰還・夜の拾い上げ・1on1本文タグ（character_actions/intent_settler.py）。
"""

from backend.services.intents.lifecycle import (  # noqa: F401
    intent_pressure,
    pressure_origin,
    record_intent_meters,
    stale_candidates,
)
from backend.services.intents.pickup import (  # noqa: F401
    run_intent_pickup,
)
