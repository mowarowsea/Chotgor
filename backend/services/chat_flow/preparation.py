"""ChatFlow のターン前処理 — 想起・WM取得・URL fetch・プロンプト構築を PreparedContext に束ねる。

flow.py（経路のオーケストレーション）から分離した「LLM を呼ぶ前に揃えるもの」の層。
タグ方式 / tool-use 方式のどちらの経路もこの前処理を共有する。

Flow:
  1. 記憶を想起 (長期記憶 RAG)
  1b. ワーキングメモリ（スレッド）を取得
  2. URL自動fetch
  3. コンテキスト別追加ツール・動機ブロック・予定コンテキストの収集
  4. プロバイダー決定とシステムプロンプト構築・ターン注釈の付加
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from backend.services.memory.manager import InscribedMemoryManager
    from backend.services.memory.working_memory_manager import WorkingMemoryManager

from backend.lib.log_context import current_log_feature
from backend.lib.web_fetch import fetch_urls, find_urls
from backend.providers.registry import create_provider
from backend.repositories.lance.store import EmbeddingError
from backend.services.chat.models import ChatRequest
from backend.services.chat.request_builder import (
    append_turn_annotation,
    build_system_prompt,
    build_turn_annotation,
)


# 自動想起（pre-recall）クエリの最大文字数。超過分は**先頭を捨てて末尾を残す**。
#
# 末尾を残す理由: 会話でも統合履歴でも「いま応答すべき最新の発話」は末尾にある。
# embedding モデル（ruri-v3-310m 等）はモデル最大長を超えた入力を先頭優先で切り捨てるため、
# 長いクエリを素通しすると「一番古い発話」でクエリしてしまう。
#
# 上限が必要な理由: うつつPC は1日ぶんの全シーンを単一 user メッセージとして渡す設計
# （external_scenes.build_unified_pc_messages）で、クエリが数十万文字になる。infinity の
# 入力上限 122,880 文字を超えると HTTP 422 で保存記憶想起と WM heat 想起が両方落ちる。
# 1on1 の通常メッセージはこの値を下回るので挙動は変わらない。
RECALL_QUERY_MAX_CHARS = 2000


def extract_text_content(content: str | list | None) -> str:
    """メッセージの content (str or list) からプレーンテキストのみを抽出する。"""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


@dataclass
class PreparedContext:
    """prepare_context() が返す前処理済みデータ。"""

    messages: list[dict]
    recalled_identity: list[dict]
    recalled: list[dict]
    wm_fixed: list[dict]
    wm_recalled: list[dict]
    provider_impl: object
    system_prompt: str
    # 記憶想起に失敗したときの UI 表示用メッセージ（成功時は None）。
    recall_error: str | None = None


async def prepare_context(
    memory_manager: "InscribedMemoryManager",
    working_memory_manager: "WorkingMemoryManager | None",
    request: ChatRequest,
) -> PreparedContext:
    """execute() / execute_stream() 共通の前処理を実行する。

    以下を順番に行い、PreparedContext にまとめて返す:
      1. 長期記憶の想起
      1b. ワーキングメモリ（スレッド）の取得
      2. URLの自動fetch
      3（4）. プロバイダー決定とシステムプロンプト構築

    Returns:
        PreparedContext: 以降の処理で必要な全データ。
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # --- 1. 記憶の想起 ---
    # XML タグ（<user>...</user> 等）を除去してから想起クエリにする。
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            raw = extract_text_content(m.get("content"))
            last_user_msg = re.sub(r"<[^>]+>", "", raw).strip()
            break

    # 想起クエリは末尾 RECALL_QUERY_MAX_CHARS 文字に絞る（理由は定数の説明を参照）。
    # URL 自動fetch は全文を対象にしたいので、last_user_msg 自体は切らずに別変数へ分ける。
    recall_query = last_user_msg[-RECALL_QUERY_MAX_CHARS:]

    recalled_identity: list[dict] = []
    recalled: list[dict] = []
    recall_error: str | None = None
    # 記憶系（長期記憶・WM）の読み出しがこのターンで縮退しているか。
    # True なら build_system_prompt が運用告知ブロックをキャラクター本人へ注入する
    # （recall_error はユーザ向け UI 通知、こちらはキャラクター向け通知と役割を分ける）。
    memory_degraded = False
    if recall_query:
        try:
            recalled_identity, recalled = memory_manager.recall_with_identity(
                request.character_id, recall_query
            )
        except EmbeddingError as e:
            # embedding サーバ（infinity 等）に接続できないケース。原因が分かるよう専用メッセージにする。
            _log.warning("記憶想起失敗（embedding）char=%s error=%s", request.character_id, e)
            recall_error = "記憶の想起に失敗しました（embedding接続エラー）"
            memory_degraded = True
            # 計器 Tier 1（embedding_degraded）: 記憶の縮退は幻想の穴（既存の
            # 縮退通知二系統 = recall_error / memory_notice に加え、監査記録を残す）。
            from backend.lib.instrument_recorder import fire_alarm
            fire_alarm("embedding_degraded", details={
                "where": "recall_with_identity",
                "character_id": request.character_id,
                "error": str(e),
            })
        except Exception as e:
            _log.warning("記憶想起失敗 char=%s error=%s", request.character_id, e)
            recall_error = "記憶の想起に失敗しました"
            memory_degraded = True

    # --- 1b. ワーキングメモリ（スレッド）を取得 ---
    # 全スレッド一覧（self_history 代替）／emotion・body・relation の固定注入／
    # task・topic の heat 上位想起、の3系統を取得する。
    wm_all_threads: list[dict] | None = None
    wm_fixed_threads: list[dict] | None = None
    wm_recalled_threads: list[dict] | None = None
    if working_memory_manager:
        try:
            wm_all_threads = (
                working_memory_manager.list_all_threads(request.character_id) or None
            )
            wm_fixed_threads = (
                working_memory_manager.get_fixed_threads(request.character_id) or None
            )
            if recall_query:
                wm_recalled_threads = (
                    working_memory_manager.recall_threads(
                        request.character_id, recall_query
                    )
                    or None
                )
        except EmbeddingError as e:
            # heat 想起（recall_threads）の embedding 失敗。一覧・固定注入は取得済みのまま残る。
            _log.warning("ワーキングメモリ取得失敗（embedding）char=%s error=%s", request.character_id, e)
            memory_degraded = True
            if recall_error is None:
                recall_error = "ワーキングメモリの取得に失敗しました（embedding接続エラー）"
            # 計器 Tier 1（embedding_degraded）: 監査記録（縮退通知二系統とは別枠）。
            from backend.lib.instrument_recorder import fire_alarm
            fire_alarm("embedding_degraded", details={
                "where": "recall_threads",
                "character_id": request.character_id,
                "error": str(e),
            })
        except Exception as e:
            _log.warning("ワーキングメモリ取得失敗 char=%s error=%s", request.character_id, e)
            memory_degraded = True
            if recall_error is None:
                recall_error = "ワーキングメモリの取得に失敗しました"

    # --- 2. URLの自動fetch ---
    # 1on1（default_origin=="real"）限定。シナリオ/うつつでは last_user_msg が
    # 実ユーザではなくNPC/Narratorの台詞であり、会話文中のURL"らしき"文字列
    # （例: 技術相談での「https://」か「http://」かという言い回し）まで誤fetchしてしまうため。
    fetched_contents = []
    if last_user_msg and request.default_origin == "real":
        urls = find_urls(last_user_msg)
        if urls:
            try:
                fetched_contents = await fetch_urls(urls)
            except Exception:
                pass

    # --- コンテキスト別追加ツール（reach_out / visit_user / override_schedule）---
    # 文脈（origin / session_id）に応じてキャラへ露出するツールと、その操作ガイドを
    # 単一判定点（character_actions/context_tools.py）から取得する。
    # in-process tool-use プロバイダーには extra_tools として渡し、claude_cli は
    # MCP tools/list（env→クエリ経路）側で同じ判定が効く。失敗しても会話は止めない。
    context_extra_tools: list[dict] = []
    context_tool_hints: list[str] = []
    try:
        from backend.character_actions.context_tools import (
            resolve_context_tool_hints,
            resolve_context_tools,
        )
        _ct_sqlite = getattr(memory_manager, "sqlite", None)
        context_extra_tools = resolve_context_tools(
            _ct_sqlite, request.character_id,
            origin=request.default_origin, session_id=request.session_id or None,
        )
        context_tool_hints = resolve_context_tool_hints(
            _ct_sqlite, request.character_id,
            origin=request.default_origin, session_id=request.session_id or None,
        )
    except Exception:
        _log.exception("コンテキストツール判定に失敗 char=%s", request.character_id)

    # --- プロバイダーを決定（use_tools フラグに使用） ---
    # PowerRecall 再帰呼び出し中は power_recalled が非空
    feature = "power_recall" if request.power_recalled else "chat"
    current_log_feature.set(feature)
    provider_impl = create_provider(
        request.provider, request.model, request.settings,
        thinking_level=request.thinking_level,
        preset_name=request.current_preset_name or "",
        # character_name は claude_cli の conversation 整形（<キャラ名>...</キャラ名>）に使う。
        # 未指定だと <character> へフォールバックして「自分の発話」感が薄れるため必ず渡す。
        character_name=request.character_name or "",
        character_id=request.character_id,
        session_id=request.session_id or "",
        allowed_tools=request.allowed_tools,
        timeout_seconds=request.timeout_seconds,
        extra_tools=context_extra_tools,
    )

    # --- 動機ブロック（めぐり）: 圧力の淡白な一行＋active な意図を毎ターン
    # 読み取り時計算する。圧力は保存されない純関数（封筒の導関数）。
    # 失敗しても会話は止めない。
    motive_lines: list[str] | None = None
    active_intents: list[dict] | None = None
    try:
        from backend.services.pressure import compute_pressures, pressure_plain_lines
        sqlite_store = getattr(memory_manager, "sqlite", None)
        if sqlite_store is not None and request.character_id:
            pressures = compute_pressures(sqlite_store, request.character_id)
            motive_lines = pressure_plain_lines(pressures)
            active_intents = [
                {"description": i.description, "target": i.target}
                for i in sqlite_store.list_intents(
                    request.character_id, status="active"
                )
            ] or None
    except Exception:
        _log.exception("圧力計算に失敗 char=%s", request.character_id)

    # --- 予定コンテキスト（生活カレンダー）: 現在の予定・次の予定・意志上書きの超過 ---
    # 1on1 とうつつ PC の両方で、本人が「この後の予定」を認知できるようにする。
    # ランダムイベント（伏せ枠）はネタバレ防止のため含まれない
    # （awareness 側で構造的に除外）。失敗しても会話は止めない。
    schedule_lines: list[str] | None = None
    try:
        from backend.services.schedule.awareness import build_schedule_lines
        _sched_sqlite = getattr(memory_manager, "sqlite", None)
        if _sched_sqlite is not None and request.character_id:
            schedule_lines = build_schedule_lines(
                _sched_sqlite, request.character_id
            ) or None
    except Exception:
        _log.exception("予定コンテキスト構築に失敗 char=%s", request.character_id)

    # --- システムプロンプト構築（安定ブロックのみ） ---
    # 毎ターン変動する情報はターン注釈として最新 user メッセージ側へ付加する
    # （プロンプトキャッシュ対応）。
    system_prompt = build_system_prompt(
        character_system_prompt=request.character_system_prompt,
        inner_narrative=request.inner_narrative,
        session_frame_instruction=request.session_frame_instruction,
        wm_all_threads=wm_all_threads,
        wm_fixed_threads=wm_fixed_threads,
        use_tools=provider_impl.SUPPORTS_TOOLS,
        memory_degraded=memory_degraded,
        usual_days_enabled=request.usual_days_enabled,
        user_label=request.user_label,
        user_position=request.user_position,
        face_to_face=request.face_to_face,
        context_tool_hints=context_tool_hints or None,
    )

    # --- ターン注釈（変動ブロック）を最新 user メッセージへ付加 ---
    # 注釈は LLM リクエストにのみ乗り、DB 保存の履歴には残らない
    # （messages はここでコピーへ差し替わるため、元リストは汚れない）。
    annotation = build_turn_annotation(
        recalled_memories=recalled or None,
        recalled_identity_memories=recalled_identity or None,
        enable_time_awareness=request.enable_time_awareness,
        current_time_str=request.current_time_str,
        time_since_last_interaction=request.time_since_last_interaction,
        fetched_contents=fetched_contents,
        wm_recalled_threads=wm_recalled_threads,
        motive_lines=motive_lines,
        active_intents=active_intents,
        schedule_lines=schedule_lines,
        previous_anticipation=request.previous_anticipation,
    )
    messages = append_turn_annotation(messages, annotation)

    return PreparedContext(
        messages=messages,
        recalled_identity=recalled_identity,
        recalled=recalled,
        wm_fixed=wm_fixed_threads or [],
        wm_recalled=wm_recalled_threads or [],
        provider_impl=provider_impl,
        system_prompt=system_prompt,
        recall_error=recall_error,
    )
