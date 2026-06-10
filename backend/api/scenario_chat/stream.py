"""シナリオチャット API — SSE ストリーム実行とデフォルトプロンプト取得。"""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.api.scenario_chat.schemas import StreamRequest
from backend.lib.debug_logger import logger as debug_logger
from backend.lib.log_context import (
    current_log_feature,
    current_log_session_id,
    current_log_target,
    current_log_turn_sequence,
    current_message_id,
    new_message_id,
)
from backend.services.scenario_chat.service import run_scenario_turn

router = APIRouter(prefix="/api/scenario_chat", tags=["scenario_chat"])

@router.post("/sessions/{session_id}/stream")
async def stream_turn(request: Request, session_id: str, body: StreamRequest):
    """プレイヤー発話を入力としてシナリオ 1 ターン分を SSE で返す。

    既存 chat と同じく、リクエストごとに log_message_id を発行して
    `debug/<8 桁hex>/` フォルダ内に各種ログを保存できるようにする。
    """
    # リクエスト識別子を発行。再生成時は前ターンの log_request_id を引き継ぐ。
    new_message_id()
    current_log_session_id.set(session_id)
    current_log_feature.set("scenario")

    state = request.app.state
    sqlite = state.sqlite
    sess = sqlite.get_scenario_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    if sess.status != "active":
        raise HTTPException(status_code=400, detail="セッションは終了しています")

    # シナリオタイトルとターン番号をログコンテキストに設定する
    _scenario = sqlite.get_scenario(sess.scenario_id)
    if _scenario:
        current_log_target.set(_scenario.title)
    _next_turn = sqlite.get_next_scenario_turn_index(session_id)
    current_log_turn_sequence.set(_next_turn)

    # 再生成時は前ターンの log_request_id を引き継いで同一ログエントリにまとめる。
    # 過去ターン編集や新規ターンの場合は引き継がない（新規 ID のまま）。
    if body.regenerate_request_id:
        current_message_id.set(body.regenerate_request_id)

    debug_logger.log_front_input(body.model_dump())

    settings = sqlite.get_all_settings()

    async def sse_generator():
        # GM の発話内容を収集して最後に log_front_output() に渡す
        _gm_parts: list[str] = []
        # chat_service は app.state に必ず存在する想定だが、テスト用 fixture では未注入のことが
        # あるため getattr で防御する。None でも ensemble モードは動作する（ensemble_pc 時のみ
        # PC ターンがスキップされる）。
        _chat_service = getattr(state, "chat_service", None)
        async for event_type, payload in run_scenario_turn(
            session_id=session_id,
            user_message=body.content,
            sqlite=sqlite,
            settings=settings,
            auto_advance=body.auto_advance,
            chat_service=_chat_service,
        ):
            data = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
            yield f"data: {data}\n\n"
            if event_type == "speaker_end":
                turn = payload.get("turn", {})
                if turn.get("speaker_type") != "user":
                    name = turn.get("speaker_name", "")
                    content = turn.get("content", "")
                    if content:
                        _gm_parts.append(content)
        # ストリーム完了後に DB の response カラムを更新する
        if _gm_parts:
            debug_logger.log_front_output("\n\n".join(_gm_parts))
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/default-system-prompt-template")
async def get_default_system_prompt_template():
    """デフォルトのGMシステムプロンプト（テンプレートタグ版）を返す。

    「デフォルトに戻す」ボタンで custom_system_prompt 欄に自動入力される。
    実行時にはテンプレートタグが実際の値に置き換えられる。
    """
    from backend.services.scenario_chat.prompt_builder import DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE
    return {
        "template": DEFAULT_GM_SYSTEM_PROMPT_TEMPLATE
    }
