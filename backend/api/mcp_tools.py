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
from backend.lib.log_context import (
    current_log_dir_id,
    current_log_feature,
    current_log_target,
    current_message_id,
)
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
from backend.character_actions.web_searcher import (
    WEB_SEARCH_SCHEMA,
    WEB_SEARCH_TOOL_DESCRIPTION,
)


router = APIRouter(prefix="/api/mcp", tags=["mcp"])


# localhost と判定するクライアントホスト集合
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _restore_log_context(log_context: dict | None) -> None:
    """env→HTTP リレーで運ばれた元リクエストのログ文脈を ContextVar へ復元する。

    Claude CLI の MCP プロセスは backend と別プロセスのため、チャットリクエスト時の
    ContextVar（current_message_id 等）はこの HTTP リクエストには引き継がれない。
    tool_event_recorder がツール実行イベントを元リクエストへ紐付けられるよう、
    リレーされた値をここで復元する。値が欠けているキーは触らない
    （デフォルトの "--------" のままなら recorder 側で None に正規化される）。

    Args:
        log_context: mcp_server.py がリレーした {"request_id", "dir_id", "feature",
            "target"} の辞書。None / 空なら何もしない。
    """
    if not log_context:
        return
    request_id = str(log_context.get("request_id") or "").strip()
    dir_id = str(log_context.get("dir_id") or "").strip()
    feature = str(log_context.get("feature") or "").strip()
    target = str(log_context.get("target") or "").strip()
    if request_id:
        current_message_id.set(request_id)
    if dir_id:
        current_log_dir_id.set(dir_id)
    if feature:
        current_log_feature.set(feature)
    if target:
        current_log_target.set(target)


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
    """``POST /api/mcp/tools/call`` のリクエストボディ。

    ``batch_context`` は forget 蒸留などの内部バッチが ``inscribe_memory`` の挙動
    （例: ``force_insert_memory=True``）を切り替えるためのフラグ群。Claude CLI 経由の
    MCP 呼び出しでは Python 側 ``ToolExecutor`` インスタンスが共有されないため、
    HTTP ペイロード上で明示的に伝搬しないと in-process プロバイダーとの間に
    挙動差分が出てしまう（forget 蒸留物が in-place 上書きされて道連れ消失する等）。

    ``default_origin`` は inscribe_memory / post_working_memory_thread の保存時に
    付与する origin ラベル（"real" / "interlude"）。シナリオ PC モードからキャラを
    動かす経路では mcp_server.py が CHOTGOR_DEFAULT_ORIGIN env から拾って渡す。
    省略時は "real"。同じく env→HTTP リレーが必要な理由は batch_context と同じ。

    ``log_context`` は元リクエストのログ文脈（request_id / dir_id / feature / target）。
    ツール実行イベント（tool_call_events）を元のチャットリクエストに紐付けるために
    使う。MCP プロセスは別プロセスのため backend 側 ContextVar が届かず、
    CHOTGOR_LOG_CONTEXT env → HTTP のリレーで明示的に運ぶ必要がある
    （batch_context / default_origin と同じ理由）。
    """

    character_id: str
    session_id: str | None = None
    name: str
    arguments: dict
    batch_context: dict | None = None
    default_origin: str = "real"
    log_context: dict | None = None


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
        ToolDefinition(
            name="web_search",
            description=WEB_SEARCH_TOOL_DESCRIPTION,
            inputSchema=WEB_SEARCH_SCHEMA,
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

    ``log_context`` が渡された場合は、実行前に元リクエストのログ文脈を本 HTTP
    リクエストの ContextVar へ復元する。これにより ToolExecutor 内の
    tool_event_recorder が正しい request_id / dir_id でイベントを記録でき、
    backend ログ行の [msg_id] も元のチャットリクエストと揃う。
    ContextVar は ASGI リクエストごとに独立しているため、他リクエストへ漏れない。
    """
    _ensure_local(request)
    _restore_log_context(payload.log_context)
    state = request.app.state
    executor = ToolExecutor(
        character_id=payload.character_id,
        session_id=payload.session_id,
        memory_manager=state.memory_manager,
        working_memory_manager=state.working_memory_manager,
        batch_context=payload.batch_context,
        default_origin=payload.default_origin,
    )
    try:
        result_text = executor.execute(payload.name, payload.arguments)
        return ToolCallResponse(result=result_text, is_error=False)
    except Exception as e:
        return ToolCallResponse(
            result=f"[Error: {type(e).__name__}: {e}]",
            is_error=True,
        )
