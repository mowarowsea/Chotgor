"""MCP tools プロキシ用 内部 API。

mcp_server.py（独立子プロセスとして Claude CLI が起動する）がベクトルストアを
backend と二重に開かないよう、backend に集約された ToolExecutor を HTTP 経由で
実行するための内部 API。

ベクトルストア自体は multi-process safe だが、InscribedMemoryManager / DriftManager の
単一インスタンス集約という設計上の利点を保つため、MCP プロセスが直接ストアを開かず
backend に集約された ToolExecutor を HTTP 経由で呼び出す構成にしている。

セキュリティ: localhost (127.0.0.1 / ::1) からのリクエストのみ受け付ける。
backend は 0.0.0.0:8000 にバインドするため、ガードを入れないと LAN 内の他端末から
キャラクターの記憶を改変できてしまう。MCP プロセスは必ず同一ホストで動作する前提。
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.character_actions.executor import ToolExecutor
from backend.character_actions.inscriber import (
    INSCRIBE_MEMORY_SCHEMA,
    INSCRIBE_MEMORY_TOOL_DESCRIPTION,
)
from backend.character_actions.threader import (
    POST_WORKING_MEMORY_THREAD_SCHEMA,
    POST_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
    OPEN_WORKING_MEMORY_THREAD_SCHEMA,
    OPEN_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
)
from backend.character_actions.carver import (
    CARVE_NARRATIVE_SCHEMA,
    CARVE_NARRATIVE_TOOL_DESCRIPTION,
)
from backend.character_actions.recaller import (
    POWER_RECALL_SCHEMA,
    POWER_RECALL_TOOL_DESCRIPTION,
)
from backend.character_actions.switcher import (
    SWITCH_ANGLE_SCHEMA,
    SWITCH_ANGLE_TOOL_DESCRIPTION,
)


router = APIRouter(prefix="/api/mcp", tags=["mcp"])


# localhost と判定するクライアントホスト集合
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _ensure_local(request: Request) -> None:
    """リクエストが localhost からのものであることを保証する。違えば 403。

    backend が 0.0.0.0 にバインドする運用を前提に、MCP API への外部からの
    アクセスを物理的に拒絶する。MCP プロセスは常に同一ホスト上で動作する。
    """
    client = request.client
    host = client.host if client else None
    if host not in _LOCAL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail=f"MCP API は localhost からのみアクセス可能です（来訪元: {host!r}）",
        )


class ToolCallRequest(BaseModel):
    """``POST /api/mcp/tools/call`` のリクエストボディ。"""

    character_id: str
    session_id: str | None = None
    name: str
    arguments: dict


class ToolCallResponse(BaseModel):
    """``POST /api/mcp/tools/call`` のレスポンス。"""

    result: str
    is_error: bool


class ToolDefinition(BaseModel):
    """MCP 公開ツール 1 件の定義。"""

    name: str
    description: str
    inputSchema: dict


class ToolListResponse(BaseModel):
    """``GET /api/mcp/tools`` のレスポンス。"""

    tools: list[ToolDefinition]


def _build_tool_definitions() -> list[ToolDefinition]:
    """現在公開する MCP ツール定義のリストを構築する。

    スキーマは各 character_action モジュールに定義された定数をそのまま参照する。
    """
    return [
        ToolDefinition(
            name="inscribe_memory",
            description=INSCRIBE_MEMORY_TOOL_DESCRIPTION,
            inputSchema=INSCRIBE_MEMORY_SCHEMA,
        ),
        ToolDefinition(
            name="post_working_memory_thread",
            description=POST_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
            inputSchema=POST_WORKING_MEMORY_THREAD_SCHEMA,
        ),
        ToolDefinition(
            name="open_working_memory_thread",
            description=OPEN_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION,
            inputSchema=OPEN_WORKING_MEMORY_THREAD_SCHEMA,
        ),
        ToolDefinition(
            name="carve_narrative",
            description=CARVE_NARRATIVE_TOOL_DESCRIPTION,
            inputSchema=CARVE_NARRATIVE_SCHEMA,
        ),
        ToolDefinition(
            name="power_recall",
            description=POWER_RECALL_TOOL_DESCRIPTION,
            inputSchema=POWER_RECALL_SCHEMA,
        ),
        ToolDefinition(
            name="switch_angle",
            description=SWITCH_ANGLE_TOOL_DESCRIPTION,
            inputSchema=SWITCH_ANGLE_SCHEMA,
        ),
    ]


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(request: Request) -> ToolListResponse:
    """公開 MCP ツール定義を返す。MCP プロセスからの ``tools/list`` をプロキシする。"""
    _ensure_local(request)
    return ToolListResponse(tools=_build_tool_definitions())


@router.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(request: Request, payload: ToolCallRequest) -> ToolCallResponse:
    """ToolExecutor.execute() を HTTP 経由で実行する。

    backend の app.state に保持された InscribedMemoryManager / WorkingMemoryManager を流用し、
    リクエストごとにキャラ／セッションを束ねた ToolExecutor を生成する。
    ``execute`` 自体が文字列を返す素直なインターフェースなので、
    そのまま ``result`` フィールドに乗せる。
    """
    _ensure_local(request)
    state = request.app.state
    executor = ToolExecutor(
        character_id=payload.character_id,
        session_id=payload.session_id,
        memory_manager=state.memory_manager,
        working_memory_manager=state.working_memory_manager,
    )
    try:
        result_text = executor.execute(payload.name, payload.arguments)
        return ToolCallResponse(result=result_text, is_error=False)
    except Exception as e:
        return ToolCallResponse(
            result=f"[Error: {type(e).__name__}: {e}]",
            is_error=True,
        )
