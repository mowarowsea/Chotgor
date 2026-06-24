"""1on1 チャット用 ChatService — chat_flow.ChatFlow の薄い再エクスポート。

旧 ChatService の処理本体は backend.services.chat_flow.flow.ChatFlow に
移動された。既存の import を壊さないため、ここでは ChatFlow を ChatService
として再エクスポートする。新規コードは backend.services.chat_flow から直接
import すること。
"""

from backend.services.chat_flow.flow import (
    ChatFlow as ChatService,
    extract_text_content,
    _run_farewell_detection,
)

__all__ = ["ChatService", "extract_text_content", "_run_farewell_detection"]
