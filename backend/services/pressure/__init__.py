"""圧力（pressure）— めぐり（巡り / Aliveness）の動機経済・状態側。

圧力は保存しない。**タイムライン封筒の導関数として毎回計算する純関数**
（docs/aliveness_plan.md §4.1）。保存値と履歴の不整合という事故クラスが
原理的に存在せず、計器からいつでも監査できる。

圧＝物理、WM＝意味、の分業:
    圧力は「いつ聞くか」だけを決め、意味はキャラクターが与える。
    プロンプトへは解釈済みの言葉ではなく生に近い淡白な一行で渡す。
"""

from backend.services.pressure.engine import (  # noqa: F401
    DEFAULT_PROFILE,
    compute_pressures,
    pressure_plain_lines,
    record_pressure_meters,
)
from backend.services.pressure.interview import (  # noqa: F401
    run_constitution_interview,
)
