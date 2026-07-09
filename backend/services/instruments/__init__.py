"""計器（instruments）— めぐり（巡り / Aliveness）の監査層。

キャラクターの世界には一切現れない監査者（docs/planned/aliveness_plan.md §3）。
内容を見ずに外形だけで健全性を監視し、「窓（内容観測）を閉じる」ための
ユーザの確信を数字で支える。

3層構造:
    - Tier 1（tier1.py）: インバリアント（機械・真偽確定）。即時＋巡回。
    - Tier 2（tier2.py）: スメル検知器（毎応答・正規表現/長さ・LLM 不使用・誤検知許容）。
    - Tier 3（tier3.py）: 判定巡回（LLM・サンプリング。判定器は「人格なき環境」）。

ラチェット原則: 網を抜けた事故は、必ず新しい検知器になる。
静音期間の意味は「異常ゼロの証明」ではなく「既知の事故クラスすべてで無事故N日」。
"""

from backend.services.instruments.tier1 import run_patrol_checks  # noqa: F401
from backend.services.instruments.tier2 import (  # noqa: F401
    record_bloat_meters,
    record_response_smells,
    scan_response_smells,
)
from backend.services.instruments.tier3 import run_judgement_patrol  # noqa: F401
