"""chat_flow — 1キャラ1ターン分のLLM呼び出しを担う共通骨。

1on1（services/chat）／シナリオ PC モード（services/scenario_chat/pc_runner）／
うつつ PC ターンから共通利用される。「次のターンは誰か」「履歴の管理」など
ターン制御の責務は呼び出し側に残し、本パッケージは 1 ターンの実行に集中する。
"""

from backend.services.chat_flow.flow import (
    ChatFlow,
    extract_text_content,
    _run_farewell_detection,
)

__all__ = ["ChatFlow", "extract_text_content", "_run_farewell_detection"]
