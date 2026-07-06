"""意図（intents）— めぐり（巡り / Aliveness）の動機経済・意図層。

欲求は行動に**先行**し、事後に遡って発見できねばならない（docs/aliveness_plan.md §4.3）。
「システムがキャラにプッシュを命じる」方式は却下済み — 行動の瞬間に欲求を
捏造させるため。意図は夜のChronicleとうつつシーン完走後の「拾い上げ」で
本人の言葉のまま記録され、失効・不満化も機械は候補を挙げるだけで本人が裁く。
"""

from backend.services.intents.lifecycle import (  # noqa: F401
    expired_candidates,
    intent_pressure,
    soured_candidates,
)
from backend.services.intents.pickup import (  # noqa: F401
    run_intent_pickup,
)
