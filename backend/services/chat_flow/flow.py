"""ChatFlow — 1キャラ1ターン分のLLM呼び出し+ツール実行の経路オーケストレーション。

旧 ChatService の処理本体を 1on1 / シナリオ PC モード / うつつ PC ターンで
共通利用できる形に抽出した。「次のターンが誰か」「履歴をどう持つか」は
呼び出し側（services/chat、services/scenario_chat）の責務であり、本クラスは
1 ターン分の LLM 呼び出しと付帯処理だけに集中する。

Architecture:
  Adapter / Scenario Loop → ChatFlow → LLM provider

パッケージ内の分業:
  - preparation.py  : ターン前処理（想起・WM・URL fetch・プロンプト構築）→ PreparedContext
  - flow.py（本体） : tool-use 経路／タグ経路のディスパッチ、switch_angle / power_recall の再帰
  - farewell_flow.py: ターン完了後の別れ検出・疲労離席の起動

後方互換: `backend.services.chat.service.ChatService` は本クラスの別名として
そのまま使える（services/chat/service.py が再エクスポート）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

_log = logging.getLogger(__name__)

from backend.lib.debug_logger import logger
from backend.character_actions.anticipator import extract_anticipation
from backend.services.memory.manager import InscribedMemoryManager
from backend.services.memory.working_memory_manager import WorkingMemoryManager
from backend.providers.base import LLMApiError
from backend.lib.tag_parser import StreamingTagStripper
from backend.lib.tool_event_recorder import record_tool_event
from backend.character_actions.executor import ToolExecutor
from backend.services.chat.models import ChatRequest, Message
from backend.character_actions.recaller import Recaller, format_power_recall_turn
from backend.services.chat_flow.preparation import (
    PreparedContext,
    extract_text_content,
    prepare_context,
)
from backend.services.chat_flow.farewell_flow import (
    launch_farewell_tasks,
    run_farewell_detection,
)

# 後方互換の別名（services/chat/service.py・既存テストが参照する旧名）
_run_farewell_detection = run_farewell_detection

__all__ = [
    "ChatFlow",
    "PreparedContext",
    "extract_text_content",
    "prepare_context",
    "_run_farewell_detection",
]


class ChatFlow:
    def __init__(
        self,
        memory_manager: InscribedMemoryManager,
        working_memory_manager: WorkingMemoryManager | None = None,
    ) -> None:
        """ChatFlow を初期化する。

        Args:
            memory_manager: 長期記憶の読み書きを担うマネージャー。
            working_memory_manager: ワーキングメモリ（スレッド）の読み書きを担うマネージャー。
                                    None の場合はワーキングメモリ注入をスキップする。
        """
        self.memory_manager = memory_manager
        self.working_memory_manager = working_memory_manager

    # --- 内部ヘルパー ---

    def _extract_switch_info(
        self, tool_executor: "ToolExecutor", clean_text: str, has_angle_presets: bool
    ) -> tuple[str, tuple[str, str] | None]:
        """switch_angle リクエストを tool_executor.switch_request から取り出す。

        available_presets が空のときは無視する。SUPPORTS_TOOLS プロバイダーは常に
        switch_angle ツールを LLM に渡すため、presets が未設定でも LLM が誤呼び出しする
        可能性があり、ここでガードする。

        タグ方式・tool-use 方式 のどちらの経路でも、SWITCH_ANGLE は ToolExecutor.execute()
        経由で _switcher.switch_angle() が呼ばれて switch_request にセットされる
        （タグ方式は事前に apply_switch_angle_tags() を呼んでおく）。本メソッドは
        単に「セット済みの switch_request を読むだけ」になる。
        """
        if not has_angle_presets:
            return clean_text, None
        return clean_text, tool_executor.switch_request

    def _build_switched_request(
        self, original: ChatRequest, preset_name: str, self_instruction: str,
        first_response_text: str = "",
    ) -> ChatRequest | None:
        """switch_angle 後の再ディスパッチ用 ChatRequest を構築する。

        Args:
            original: 元のリクエスト。
            preset_name: 切り替え先プリセット名。
            self_instruction: 切り替え後モデルへの自己指針。
                プロバイダー固有追記（Block 5）に畳み込んで切り替え先に伝える。
            first_response_text: 第1プロバイダーが生成したテキスト。
                空でなければ assistant ターンとして messages に追加し、
                第2プロバイダーが会話の流れを引き継げるようにする。
        """
        preset = next(
            (p for p in original.available_presets if p.get("preset_name") == preset_name),
            None,
        )
        if preset is None:
            _log.warning("switch_angle プリセット未発見 char=%s@%s preset=%s", original.character_name, original.current_preset_name, preset_name)
            return None

        # 切り替え後モデルへの自己指針は、プロバイダー固有追記の末尾に畳み込む。
        extra_instructions = preset.get("additional_instructions", "") or ""
        if self_instruction:
            extra_instructions = (
                f"{extra_instructions}\n\n{self_instruction}".strip()
                if extra_instructions.strip()
                else self_instruction
            )

        # 第1プロバイダーの応答を assistant ターンとして追加し、第2プロバイダーへ文脈を引き継ぐ
        new_messages = list(original.messages)
        if first_response_text:
            new_messages.append(Message(role="assistant", content=first_response_text))

        return replace(
            original,
            provider=preset["provider"],
            model=preset.get("model_id", ""),
            provider_additional_instructions=extra_instructions,
            thinking_level=preset.get("thinking_level", "default"),
            current_preset_name=preset_name,
            current_preset_id=preset.get("preset_id", ""),
            available_presets=[],
            messages=new_messages,
            timeout_seconds=preset.get("timeout_seconds", 300),
        )

    def _log_debug(self, label: str, request: ChatRequest, messages: list[dict], clean_text: str) -> None:
        """LLM呼び出しの操作ログを出力する。char_label は {name}@{preset} 形式で出力する。"""
        char_label = f"{request.character_name}@{request.current_preset_name or request.provider}"
        _log.info(
            "%s char=%s provider=%s model=%s messages=%d response_chars=%d",
            label, char_label, request.provider, request.model or "(default)",
            len(messages), len(clean_text),
        )

    # --- 公開メソッド ---

    async def execute(self, request: ChatRequest) -> str:
        """LLMにディスパッチして応答テキストを返す。SSEを知らない。

        退席タグ・ツールが検出された場合も、退席メッセージをテキストに含めてそのまま返す。
        （OpenWebUI non-stream 向け: セッション終了の永続化はここでは行わない）
        """
        ctx = await prepare_context(self.memory_manager, self.working_memory_manager, request)

        # ツール実行の唯一の関門。SUPPORTS_TOOLS=True なら provider 内 tool-use ループから、
        # False ならタグ抽出後の apply_* メソッドから呼ばれ、どちらも同じ execute() を通って
        # 記録 (tool_call_events) も実装 (_dispatch) も一元化される。
        tool_executor = ToolExecutor(
            character_id=request.character_id,
            session_id=request.session_id,
            memory_manager=self.memory_manager,
            working_memory_manager=self.working_memory_manager,
            default_origin=request.default_origin,
            source_preset_id=request.current_preset_id,
        )
        if ctx.provider_impl.SUPPORTS_TOOLS:
            try:
                # thinking は非ストリーミングパスでは捨てる（execute はテキスト返却のみ）
                clean_text, _ = await ctx.provider_impl.generate_with_tools(ctx.system_prompt, ctx.messages, tool_executor)
            except LLMApiError as e:
                # str(e) は既に "[Error: ...]" 形式なのでそのまま返す
                _log.warning("LLM APIエラー（ツール方式）char=%s@%s: %s", request.character_name, request.current_preset_name or request.provider, e)
                return str(e)
            except Exception as e:
                _log.exception("LLM呼び出し失敗（ツール方式）char=%s@%s", request.character_name, request.current_preset_name or request.provider)
                return f"[Error: {type(e).__name__}: {e}]"
        else:
            try:
                response_text = await ctx.provider_impl.generate(ctx.system_prompt, ctx.messages)
            except Exception as e:
                _log.exception("LLM呼び出し失敗（タグ方式）char=%s@%s", request.character_name, request.current_preset_name or request.provider)
                return f"[Error: {type(e).__name__}: {e}]"

            # タグ方式は抽出だけして、実行は tool_executor.execute() 経由で記録までまとめて行う。
            clean_text = tool_executor.apply_all_tags(response_text)

        clean_text, switch_info = self._extract_switch_info(
            tool_executor, clean_text, bool(request.available_presets)
        )
        if switch_info:
            switched = self._build_switched_request(request, *switch_info, first_response_text=clean_text)
            if switched is not None:
                return await self.execute(switched)

        # 予想（ANTICIPATE_RESPONSE）タグは本文から除去する（この非ストリーミング経路では
        # 保存先が無いため抽出値は捨てるが、ユーザー向けテキストにタグを残さない）。
        clean_text, _ = extract_anticipation(clean_text)

        logger.log_front_output(clean_text)
        self._log_debug("CHAT", request, ctx.messages, clean_text)

        return clean_text

    async def execute_stream(self, request: ChatRequest):
        """ストリーミングでLLMにディスパッチし、型付きチャンクをyieldする。

        Yields:
            tuple[str, Any]:
                ("inscribed_memories",      list[dict])          : 想起した記憶リスト（最初に1回だけyield）
                ("recall_error",  str)                  : 想起失敗時のUI表示メッセージ（失敗時のみ1回）
                ("thinking",      str)                  : 思考ブロック（リアルタイム）
                ("text",          str)                  : クリーンな応答テキスト（最後に1回）
        """
        ctx = await prepare_context(self.memory_manager, self.working_memory_manager, request)

        # 想起した記憶を最初にyield
        all_recalled = ctx.recalled_identity + ctx.recalled
        if all_recalled:
            yield ("inscribed_memories", all_recalled)
        # 想起に失敗していたら、その旨を想起欄に表示するためのメッセージをyieldする
        if ctx.recall_error:
            yield ("recall_error", ctx.recall_error)
        # ワーキングメモリスレッドをyield。
        # 固定注入（emotion/body/relation）と heat 想起（task/topic）の両方を
        # まとめて送り、フロントの想起セクションに全スレッドを表示させる。id で重複排除する。
        working_memory_threads_payload: list[dict] = []
        _seen_thread_ids: set = set()
        for t in [*ctx.wm_fixed, *ctx.wm_recalled]:
            tid = t.get("id")
            if tid in _seen_thread_ids:
                continue
            _seen_thread_ids.add(tid)
            working_memory_threads_payload.append(t)
        if working_memory_threads_payload:
            yield ("working_memory_threads", working_memory_threads_payload)

        # タグ方式でリアルタイムストリーミングした場合 True。
        # True の場合、最終 yield ("text", clean_text) をスキップする。
        text_already_streamed = False

        # ツール実行の唯一の関門。SUPPORTS_TOOLS=True なら provider 内 tool-use ループから、
        # False ならタグ抽出後の apply_* メソッドから呼ばれ、どちらも同じ execute() を通る。
        tool_executor = ToolExecutor(
            character_id=request.character_id,
            session_id=request.session_id,
            memory_manager=self.memory_manager,
            working_memory_manager=self.working_memory_manager,
            default_origin=request.default_origin,
            source_preset_id=request.current_preset_id,
        )
        if ctx.provider_impl.SUPPORTS_TOOLS:
            try:
                clean_text, thinking_text = await ctx.provider_impl.generate_with_tools(ctx.system_prompt, ctx.messages, tool_executor)
            except LLMApiError as e:
                # str(e) は既に "[Error: ...]" 形式なのでそのまま yield する
                _log.warning("LLM APIエラー（ツール方式）char=%s@%s: %s", request.character_name, request.current_preset_name or request.provider, e)
                yield ("text", str(e))
                return
            except Exception as e:
                _log.exception("LLM呼び出し失敗（ツール方式）char=%s@%s", request.character_name, request.current_preset_name or request.provider)
                yield ("text", f"[Error: {type(e).__name__}: {e}]")
                return
            # 思考ブロックがあればテキスト本体より先にyieldする
            if thinking_text:
                yield ("thinking", thinking_text)
        else:
            full_text = ""
            stripper = StreamingTagStripper()

            try:
                async for chunk_type, content in ctx.provider_impl.generate_stream_typed(ctx.system_prompt, ctx.messages):
                    if chunk_type == "thinking":
                        yield ("thinking", content)
                    elif chunk_type == "text":
                        full_text += content
                        # マーカーを除去しながらリアルタイムでチャンクをyieldする
                        safe_chunk = stripper.feed(content)
                        if safe_chunk:
                            yield ("text", safe_chunk)
                    elif chunk_type == "error":
                        # プロバイダ由来エラー（APIキー未設定・SDK 例外・safety filter 等）。
                        # 1on1 チャットでは UX 維持のためエラー文言を従来どおりキャラ発話
                        # として表示・保存する（履歴から消すと「無言の応答」になり追跡不能）。
                        # text として上位へ転送するが、後段の inscribe/carve/recall の
                        # マーカー解析は走らせるとエラー文字列を誤抽出するため、ここで stream を
                        # 終了させる（remaining flush と clean_text は通常経路を通る）。
                        full_text += content
                        safe_chunk = stripper.feed(content)
                        if safe_chunk:
                            yield ("text", safe_chunk)
                        _log.warning(
                            "LLM provider エラー（タグ方式）char=%s@%s: %s",
                            request.character_name,
                            request.current_preset_name or request.provider,
                            (content or "")[:300],
                        )
                        break
            except Exception as e:
                _log.exception("LLM呼び出し失敗（タグ方式）char=%s@%s", request.character_name, request.current_preset_name or request.provider)
                yield ("text", f"[Error: {type(e).__name__}: {e}]")
                return

            # ストリーム終了後、バッファに残ったテキストを流す
            remaining = stripper.flush()
            if remaining:
                yield ("text", remaining)
            text_already_streamed = True

            # PowerRecall: Inscriber 等の後処理より先に実行し、検知したら即 return する。
            # request.power_recalled が非空の場合は再呼び出し中なので処理しない（ループ防止）。
            recaller = Recaller()
            pre_recall_text = recaller.power_recall_from_text(full_text)
            if recaller.recall_request is not None and not request.power_recalled:
                query, top_k = recaller.recall_request
                try:
                    power_recalled = await asyncio.to_thread(
                        self.memory_manager.power_recall,
                        request.character_id,
                        query,
                        top_k,
                    )
                    # タグ方式の power_recall はここが実際の実行箇所のため、
                    # （Recaller はタグ検出のみ）実行成否ごとここで記録する。
                    record_tool_event(
                        "power_recall", {"query": query, "top_k": top_k}, source="tag",
                    )
                except Exception as e:
                    logger.log_warning("PowerRecall", f"検索失敗 character={request.character_id}: {e}")
                    power_recalled = {}
                    record_tool_event(
                        "power_recall", {"query": query, "top_k": top_k},
                        status="error", error_message=f"{type(e).__name__}: {e}", source="tag",
                    )

                # キャラクターのターン終了 → Chotgor からの新ターンとして再呼び出しする。
                # assistant: pre_recall_text（キャラクターの発話途中）
                # user: Chotgor が検索結果を注入（新しいターン）
                chotgor_turn = format_power_recall_turn(power_recalled, query)
                new_messages = list(request.messages) + [
                    Message(role="assistant", content=pre_recall_text),
                    Message(role="user", content=chotgor_turn),
                ]
                recalled_request = replace(
                    request,
                    messages=new_messages,
                    power_recalled=power_recalled,  # ループ防止フラグとして使用
                )
                async for event in self.execute_stream(recalled_request):
                    yield event
                return

            # 後処理: マーカーの側効果処理（記憶保存・narrative彫り込み・アングル切替）
            # テキスト表示は済んでいるため、clean_text は副作用処理とログ用途にのみ使う。
            # タグ抽出後の実行は tool_executor 経由（記録まで一元化）。
            clean_text = tool_executor.apply_all_tags(full_text)

        clean_text, switch_info = self._extract_switch_info(
            tool_executor, clean_text, bool(request.available_presets)
        )
        if switch_info:
            switched = self._build_switched_request(request, *switch_info, first_response_text=clean_text)
            if switched is not None:
                # SUPPORTS_TOOLS 方式: 第1プロバイダーのテキストはまだUIに流れていないため先にyieldする。
                # タグ方式: text_already_streamed=True のためすでにUIに流れている（何もしない）。
                if not text_already_streamed and clean_text:
                    yield ("text", clean_text)
                async for event in self.execute_stream(switched):
                    yield event
                yield ("angle_switched", {
                    "model_id": f"{request.character_name}@{switch_info[0]}",
                    "preset_id": switched.current_preset_id,
                    "preset_name": switch_info[0],
                })
                return

        # 予想（ANTICIPATE_RESPONSE）タグを抽出する。全プロバイダー一律タグのため、
        # tool-use 方式・タグ方式どちらの clean_text からも、この共通地点で1回だけ取り出す。
        # （switch_angle / power_recall の再帰時は再帰先で抽出済みのため、ここには到達しない）
        clean_text, anticipation = extract_anticipation(clean_text)
        # 予想はここで採用（末尾で yield → API 層が保存・次ターン注入）されるため、
        # 採用が確定したこの地点でツール実行イベントとして記録する。
        # 抽出だけして捨てる経路（非ストリーミング execute() 等）では記録しない。
        if anticipation:
            record_tool_event(
                "anticipate_response", {"content": anticipation}, source="anticipation",
            )

        # FrontOutput は farewell タスク起動より前にログする。
        # asyncio.create_task はコンテキストをコピーするため、
        # FrontOutput のカウンターインクリメントが反映された状態で farewell を起動する。
        logger.log_front_output(clean_text)
        self._log_debug("CHAT stream", request, ctx.messages, clean_text)

        # 別れ検出: ストリーム完了後にバックグラウンドタスクとして起動する。
        # 結果はDBに保存し、次リクエスト時の already_exited チェックで検知される。
        launch_farewell_tasks(self.memory_manager, request, ctx.messages, clean_text)

        if clean_text and not text_already_streamed:
            yield ("text", clean_text)

        # 予想（期待）は本文の後にyieldする。API層がこれを anticipation カラムへ保存し、
        # 次ターンのシステムプロンプトに「前回のあなたの予想」として注入する。
        if anticipation:
            yield ("anticipation", anticipation)
