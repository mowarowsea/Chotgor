"""chat_flow.strategies — SceneLoop に差し込む Router / Executor の具体実装群。

1on1（one_on_one）／シナリオ PC・GM（scenario）の Strategy をここに集約する。
各 Strategy は SceneLoop に依存せず、TurnRouter / TurnExecutor の protocol だけを
満たす形に閉じる。
"""

from backend.services.chat_flow.strategies.one_on_one import (
    OneOnOneRouter,
    OneOnOneExecutor,
)

__all__ = ["OneOnOneRouter", "OneOnOneExecutor"]
