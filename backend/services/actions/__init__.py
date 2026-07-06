"""会話外行動権（actions）— めぐり（巡り / Aliveness）の自発性の完成。

圧力→意図→表出権→帰還のループを閉じる層（docs/aliveness_plan.md §5.3）。

構造:
    availability 内で周期評価＋ジッター（乱数は世界に置く）
      → 閾値評価（純関数・無料）: active intents の意図圧
      → 閾値超えがあるときだけ本人に問い合わせ（WM込み）
         「いま、これをする？　しないならしないでいい。」
      → 本人の選択（意志に乱数なし）で実行 or 見送り
      → 帰還: 「これで満ちた？　まだ？」を本人が宣言（fulfilled / active 継続）

「調べるかどうかも本人任せ」がアシスタントとの決定的な差。
"""

from backend.services.actions.runner import (  # noqa: F401
    evaluate_action_urge,
    jittered_slot_time,
    run_action_cycle,
)
