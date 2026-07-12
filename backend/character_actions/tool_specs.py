"""ツール定義の単一ソース — キャラクターへ公開する全ツールの ToolSpec 台帳。

新しいツールを追加する手順:
    1. ツールモジュール（inscriber.py 等）に SCHEMA / TOOL_DESCRIPTION と実装を書く
    2. ここの BASE_TOOL_SPECS（全文脈共通）か CONTEXT_TOOL_SPECS（文脈別出し分け）へ1件足す
    3. executor._dispatch に実行分岐を足す

ここに載せた定義は以下すべてへ自動反映される:
    - tool-use プロバイダーの定義リスト（executor.ANTHROPIC_TOOLS / OPENAI_TOOLS）
    - MCP プロキシの tools/list（api/mcp_tools.py）
    - コンテキスト別出し分けとプロンプトヒント（context_tools.py）
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.character_actions.carver import (
    CARVE_NARRATIVE_SCHEMA,
    CARVE_NARRATIVE_TOOL_DESCRIPTION,
)
from backend.character_actions.inscriber import (
    INSCRIBE_MEMORY_SCHEMA,
    INSCRIBE_MEMORY_TOOL_DESCRIPTION,
)
from backend.character_actions.leaver import (
    TAKE_LEAVE_SCHEMA,
    TAKE_LEAVE_TOOL_DESCRIPTION,
)
from backend.character_actions.messenger import (
    REACH_OUT_SCHEMA,
    REACH_OUT_TOOL_DESCRIPTION,
    REACH_OUT_TOOLS_HINT,
    VISIT_USER_SCHEMA,
    VISIT_USER_TOOL_DESCRIPTION,
    VISIT_USER_TOOLS_HINT,
)
from backend.character_actions.recaller import (
    POWER_RECALL_SCHEMA,
    POWER_RECALL_TOOL_DESCRIPTION,
)
from backend.character_actions.rescheduler import (
    OVERRIDE_SCHEDULE_SCHEMA,
    OVERRIDE_SCHEDULE_TOOL_DESCRIPTION,
    OVERRIDE_SCHEDULE_TOOLS_HINT,
)
from backend.character_actions.switcher import (
    SWITCH_ANGLE_SCHEMA,
    SWITCH_ANGLE_TOOL_DESCRIPTION,
)
from backend.character_actions.threader import (
    CLOSE_WORKING_MEMORY_THREAD_SCHEMA,
    CLOSE_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
    MERGE_WORKING_MEMORY_THREADS_SCHEMA,
    MERGE_WORKING_MEMORY_THREADS_TOOL_DESCRIPTION,
    POST_WORKING_MEMORY_THREAD_SCHEMA,
    POST_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
    READ_WORKING_MEMORY_THREAD_SCHEMA,
    READ_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
    REOPEN_WORKING_MEMORY_THREAD_SCHEMA,
    REOPEN_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
)
from backend.character_actions.web_searcher import (
    WEB_SEARCH_SCHEMA,
    WEB_SEARCH_TOOL_DESCRIPTION,
)


@dataclass(frozen=True)
class ToolSpec:
    """キャラクターへ公開するツール1件の定義。

    Attributes:
        name: ツール名。
        description: ツール説明文（tool-use プロバイダーではモデルに渡る実プロンプト）。
        input_schema: 引数の JSONSchema（Anthropic の input_schema 形式）。
        hint: システムプロンプトへ注入する操作ガイド。コンテキストツールのみ使用
            （基本ツールのガイドは request_builder が組み立てるため空）。
    """

    name: str
    description: str
    input_schema: dict
    hint: str = ""

    def as_anthropic(self) -> dict:
        """Anthropic 形式（name / description / input_schema）の dict を返す。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# 全文脈共通で露出する基本ツール（露出順のまま列挙する）
BASE_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("inscribe_memory", INSCRIBE_MEMORY_TOOL_DESCRIPTION, INSCRIBE_MEMORY_SCHEMA),
    ToolSpec(
        "post_working_memory_thread",
        POST_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
        POST_WORKING_MEMORY_THREAD_SCHEMA,
    ),
    ToolSpec(
        "read_working_memory_thread",
        READ_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
        READ_WORKING_MEMORY_THREAD_SCHEMA,
    ),
    ToolSpec(
        "close_working_memory_thread",
        CLOSE_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
        CLOSE_WORKING_MEMORY_THREAD_SCHEMA,
    ),
    ToolSpec(
        "reopen_working_memory_thread",
        REOPEN_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
        REOPEN_WORKING_MEMORY_THREAD_SCHEMA,
    ),
    ToolSpec(
        "merge_working_memory_threads",
        MERGE_WORKING_MEMORY_THREADS_TOOL_DESCRIPTION,
        MERGE_WORKING_MEMORY_THREADS_SCHEMA,
    ),
    ToolSpec("carve_narrative", CARVE_NARRATIVE_TOOL_DESCRIPTION, CARVE_NARRATIVE_SCHEMA),
    ToolSpec("switch_angle", SWITCH_ANGLE_TOOL_DESCRIPTION, SWITCH_ANGLE_SCHEMA),
    ToolSpec("power_recall", POWER_RECALL_TOOL_DESCRIPTION, POWER_RECALL_SCHEMA),
    ToolSpec("web_search", WEB_SEARCH_TOOL_DESCRIPTION, WEB_SEARCH_SCHEMA),
    ToolSpec("take_leave", TAKE_LEAVE_TOOL_DESCRIPTION, TAKE_LEAVE_SCHEMA),
)

# 文脈によって露出が変わるツール（出し分けの判定は context_tools.py）
CONTEXT_TOOL_SPECS: dict[str, ToolSpec] = {
    "reach_out": ToolSpec(
        "reach_out", REACH_OUT_TOOL_DESCRIPTION, REACH_OUT_SCHEMA,
        hint=REACH_OUT_TOOLS_HINT,
    ),
    "visit_user": ToolSpec(
        "visit_user", VISIT_USER_TOOL_DESCRIPTION, VISIT_USER_SCHEMA,
        hint=VISIT_USER_TOOLS_HINT,
    ),
    "override_schedule": ToolSpec(
        "override_schedule", OVERRIDE_SCHEDULE_TOOL_DESCRIPTION, OVERRIDE_SCHEDULE_SCHEMA,
        hint=OVERRIDE_SCHEDULE_TOOLS_HINT,
    ),
}
