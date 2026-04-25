"""ChatService — リクエスト受付から応答テキスト返却までのサービス層。

Architecture:
  Adapter → ChatService → LLM provider

Flow:
  1. 記憶を想起 (ChromaDB RAG)
  1b. SELF_DRIFT指針をDBからロード
  2. URL自動fetch
  3. システムプロンプト構築
  4. プロバイダーへディスパッチ
  5. 応答から記憶を刻み込み (Inscriber.inscribe_memory_from_text)
  5b. SELF_DRIFTマーカーを処理してDBに保存
  5c. inner_narrativeマーカーを処理してDBに保存 (Carver.carve_narrative_from_text)
  6. デバッグログ
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Union

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from backend.character_actions.executor import ToolExecutor

from backend.lib.debug_logger import logger
from backend.lib.log_context import current_log_feature
from backend.services.memory.drift_manager import DriftManager
from backend.character_actions.carver import Carver
from backend.character_actions.inscriber import Inscriber
from backend.services.memory.manager import MemoryManager
from backend.providers.base import LLMApiError
from backend.providers.registry import create_provider
from backend.services.chat.request_builder import build_system_prompt
from backend.lib.tag_parser import StreamingTagStripper
from backend.character_actions.executor import ToolExecutor
from backend.lib.web_fetch import fetch_urls, find_urls
from backend.character_actions.drifter import Drifter
from backend.services.chat.models import ChatRequest, Message
from backend.character_actions.recaller import Recaller, format_power_recall_turn
from backend.character_actions.reflector import SelfReflector
from backend.character_actions.switcher import Switcher
from backend.character_actions.farewell_detector import FarewellDetector


def extract_text_content(content: Union[str, list, None]) -> str:
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


async def _run_farewell_detection(
    detector: "FarewellDetector",
    character_id: str,
    character_name: str,
    session_id: str,
    preset_id: str,
    farewell_config: dict,
    messages: list[dict],
    settings: dict,
    chroma=None,
) -> None:
    """FarewellDetectorをバックグラウンドで実行し、退席判定をDBに保存するコルーチン。

    退席確定の場合のみセッションの exited_chars を更新する。
    疎遠化確定時は SQLite に加えて ChromaDB の定義 embedding も "estranged" に更新する。
    SSEストリームは既に終了しているため、イベント送信は行わない。
    次リクエスト時の already_exited チェックで自動検知される。

    Args:
        detector: FarewellDetectorインスタンス。
        character_id: キャラクターID。
        character_name: キャラクター名。
        session_id: 対象セッションID。
        preset_id: judge LLM に使用するプリセットID。
        farewell_config: キャラクターのfarewell_config辞書。
        messages: 判定に使用する会話履歴（最新の応答を含む）。
        settings: グローバル設定辞書。
        chroma: ChromaStore インスタンス（疎遠化時の embedding 更新に使用。None でもよい）。
    """
    try:
        result = await detector.detect(
            character_id=character_id,
            session_id=session_id,
            preset_id=preset_id,
            farewell_config=farewell_config,
            messages=messages,
            settings=settings,
        )
    except Exception:
        _log.exception("FarewellDetector実行エラー char=%s session=%s", character_name, session_id)
        return

    if result is None or not result.should_exit:
        return

    _log.info(
        "別れ検出: セッション退席 char=%s session=%s type=%s emotions=%s",
        character_name, session_id, result.farewell_type, result.emotions,
    )

    # セッションの exited_chars に退席エントリを追記する
    try:
        sqlite = detector.sqlite
        session = sqlite.get_chat_session(session_id)
        if session is None:
            return
        exited_chars: list[dict] = getattr(session, "exited_chars", None) or []
        # 重複チェック: 既に退席済みなら何もしない
        if any(e.get("char_name") == character_name for e in exited_chars):
            return

        # ネガティブ退席時: 累積回数を確認し、閾値超過なら estranged に移行する
        reason = result.reason
        if result.farewell_type == "negative":
            estrangement = farewell_config.get("estrangement", {})
            lookback_days = estrangement.get("lookback_days", 30)
            threshold = estrangement.get("negative_exit_threshold", 5)
            from datetime import timedelta
            since = datetime.now() - timedelta(days=lookback_days)
            prev_count = sqlite.get_negative_exit_count(character_name, since)
            total_count = prev_count + 1  # 今回の退席を含む合計

            if total_count >= threshold:
                # 閾値到達: relationship_status を estranged に変更する
                sqlite.update_character(character_id, relationship_status="estranged")
                _log.info(
                    "別れ決断: 閾値到達 char=%s total=%d threshold=%d → estranged",
                    character_name, total_count, threshold,
                )
                # ChromaDB の定義 embedding も estranged に更新する（類似キャラ登録ブロックに必要）
                if chroma is not None:
                    try:
                        chroma.mark_definition_estranged(character_id)
                    except Exception:
                        _log.exception("ChromaDB 疎遠化マーク失敗 char=%s", character_name)
                farewell_messages = farewell_config.get("farewell_message") or {}
                estranged_msg = farewell_messages.get("estranged", "")
                reason = estranged_msg if estranged_msg else reason
            else:
                warning = (
                    f"過去{lookback_days}日間で{total_count}回、"
                    f"{character_name}は嫌がって会話を打ち切りました。\n"
                    f"{lookback_days}日間に{threshold}回これが続けば、"
                    f"{character_name}はあなたとの別れを決断するでしょう。"
                )
                reason = f"{reason}\n{warning}" if reason else warning

        exited_chars = [*exited_chars, {
            "char_name": character_name,
            "reason": reason,
            "farewell_type": result.farewell_type,
        }]
        sqlite.update_chat_session(session_id, exited_chars=exited_chars)
    except Exception:
        _log.exception("退席DB保存エラー char=%s session=%s", character_name, session_id)


@dataclass
class _Context:
    """_prepare_context() が返す前処理済みデータ。"""

    messages: list[dict]
    recalled_identity: list[dict]
    recalled: list[dict]
    provider_impl: object
    system_prompt: str


class ChatService:
    def __init__(self, memory_manager: MemoryManager, drift_manager: DriftManager | None = None) -> None:
        """ChatService を初期化する。

        Args:
            memory_manager: 記憶の読み書きを担うマネージャー。
            drift_manager: SELF_DRIFT指針の読み書きを担うマネージャー。
                           None の場合は SELF_DRIFT 処理をスキップする。
        """
        self.memory_manager = memory_manager
        self.drift_manager = drift_manager

    # --- 内部ヘルパー ---

    def _apply_drifts(self, text: str, request: ChatRequest) -> str:
        """テキストから [DRIFT:...] / [DRIFT_RESET] マーカーを抽出しDBに反映する。

        Args:
            text: [INSCRIBE_MEMORY:...] 除去済みの応答テキスト。
            request: セッションID・キャラクターIDを含むリクエスト。

        Returns:
            マーカーを除去したクリーンなテキスト。
        """
        drifter = Drifter(request.session_id, request.character_id, self.drift_manager)
        return drifter.drift_from_text(text)

    def _extract_switch_info(
        self, tool_executor: "ToolExecutor | None", clean_text: str, has_angle_presets: bool
    ) -> tuple[str, tuple[str, str] | None]:
        """switch_angle リクエストを tool_executor またはタグから抽出する。

        available_presets が空のとき両方式とも無視する。
        SUPPORTS_TOOLS プロバイダーは常に switch_angle ツールを LLM に渡すため、
        presets が未設定でも LLM が誤呼び出しする可能性があり、ここでガードする。
        """
        if not has_angle_presets:
            return clean_text, None
        if tool_executor is not None:
            return clean_text, tool_executor.switch_request
        switcher = Switcher()
        clean_text = switcher.switch_from_text(clean_text)
        return clean_text, switcher.switch_request

    def _build_switched_request(
        self, original: ChatRequest, preset_name: str, self_instruction: str,
        first_response_text: str = "",
    ) -> ChatRequest | None:
        """switch_angle 後の再ディスパッチ用 ChatRequest を構築する。

        Args:
            original: 元のリクエスト。
            preset_name: 切り替え先プリセット名。
            self_instruction: 切り替え後モデルへの自己指針。
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

        if self.drift_manager and original.session_id:
            try:
                self.drift_manager.reset_drifts(original.session_id, original.character_id)
                if self_instruction:
                    self.drift_manager.add_drift(
                        original.session_id, original.character_id, self_instruction
                    )
            except Exception:
                pass

        # 第1プロバイダーの応答を assistant ターンとして追加し、第2プロバイダーへ文脈を引き継ぐ
        new_messages = list(original.messages)
        if first_response_text:
            new_messages.append(Message(role="assistant", content=first_response_text))

        return replace(
            original,
            provider=preset["provider"],
            model=preset.get("model_id", ""),
            provider_additional_instructions=preset.get("additional_instructions", ""),
            thinking_level=preset.get("thinking_level", "default"),
            current_preset_name=preset_name,
            current_preset_id=preset.get("preset_id", ""),
            active_drifts=[self_instruction] if self_instruction else [],
            available_presets=[],
            messages=new_messages,
        )

    async def _prepare_context(self, request: ChatRequest) -> _Context:
        """execute() / execute_stream() 共通の前処理を実行する。

        以下を順番に行い、_Context にまとめて返す:
          1. 記憶の想起
          1b. SELF_DRIFT指針のロード
          2. URLの自動fetch
          3（4）. プロバイダー決定とシステムプロンプト構築

        Returns:
            _Context: 以降の処理で必要な全データ。
        """
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # --- 1. 記憶の想起 ---
        if request.recall_query_override:
            last_user_msg = request.recall_query_override
        else:
            last_user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_msg = extract_text_content(m.get("content"))
                    break

        recalled_identity: list[dict] = []
        recalled: list[dict] = []
        if last_user_msg:
            try:
                recalled_identity, recalled = self.memory_manager.recall_with_identity(
                    request.character_id, last_user_msg
                )
            except Exception as e:
                _log.warning("記憶想起失敗 char=%s error=%s", request.character_id, e)

        # --- 1b. SELF_DRIFT指針をDBからロード ---
        active_drifts = request.active_drifts or []
        if self.drift_manager and request.session_id and not active_drifts:
            try:
                active_drifts = self.drift_manager.list_active_drifts(request.session_id, request.character_id)
            except Exception:
                pass

        # --- 2. URLの自動fetch ---
        fetched_contents = []
        if last_user_msg:
            urls = find_urls(last_user_msg)
            if urls:
                try:
                    fetched_contents = await fetch_urls(urls)
                except Exception:
                    pass

        # --- プロバイダーを決定（use_tools フラグに使用） ---
        # PowerRecall 再帰呼び出し中は power_recalled が非空
        feature = "power_recall" if request.power_recalled else "chat"
        current_log_feature.set(feature)
        provider_impl = create_provider(
            request.provider, request.model, request.settings,
            thinking_level=request.thinking_level,
            preset_name=request.current_preset_name or "",
            character_id=request.character_id,
            session_id=request.session_id or "",
            allowed_tools=request.allowed_tools,
        )

        # --- システムプロンプト構築 ---
        system_prompt = build_system_prompt(
            character_system_prompt=request.character_system_prompt,
            recalled_identity_memories=recalled_identity or None,
            recalled_memories=recalled or None,
            fetched_contents=fetched_contents,
            self_history=request.self_history,
            relationship_state=request.relationship_state,
            inner_narrative=request.inner_narrative,
            provider_additional_instructions=request.provider_additional_instructions,
            enable_time_awareness=request.enable_time_awareness,
            current_time_str=request.current_time_str,
            time_since_last_interaction=request.time_since_last_interaction,
            active_drifts=active_drifts or None,
            use_tools=provider_impl.SUPPORTS_TOOLS,
            available_presets=request.available_presets or None,
            current_preset_name=request.current_preset_name,
        )

        return _Context(
            messages=messages,
            recalled_identity=recalled_identity,
            recalled=recalled,
            provider_impl=provider_impl,
            system_prompt=system_prompt,
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
        ctx = await self._prepare_context(request)

        if ctx.provider_impl.SUPPORTS_TOOLS:
            tool_executor = ToolExecutor(
                character_id=request.character_id,
                session_id=request.session_id,
                memory_manager=self.memory_manager,
                drift_manager=self.drift_manager,
            )
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
            tool_executor = None
            try:
                response_text = await ctx.provider_impl.generate(ctx.system_prompt, ctx.messages)
            except Exception as e:
                _log.exception("LLM呼び出し失敗（タグ方式）char=%s@%s", request.character_name, request.current_preset_name or request.provider)
                return f"[Error: {type(e).__name__}: {e}]"

            inscriber = Inscriber(request.character_id, self.memory_manager)
            clean_text = inscriber.inscribe_memory_from_text(response_text, request.current_preset_id)
            clean_text = Carver(request.character_id, self.memory_manager.sqlite).carve_narrative_from_text(clean_text)
            clean_text = self._apply_drifts(clean_text, request)

        clean_text, switch_info = self._extract_switch_info(
            tool_executor, clean_text, bool(request.available_presets)
        )
        if switch_info:
            switched = self._build_switched_request(request, *switch_info, first_response_text=clean_text)
            if switched is not None:
                return await self.execute(switched)

        logger.log_front_output(clean_text)
        self._log_debug("CHAT", request, ctx.messages, clean_text)

        return clean_text

    async def execute_stream(self, request: ChatRequest):
        """ストリーミングでLLMにディスパッチし、型付きチャンクをyieldする。

        Yields:
            tuple[str, Any]:
                ("memories",      list[dict])          : 想起した記憶リスト（最初に1回だけyield）
                ("thinking",      str)                  : 思考ブロック（リアルタイム）
                ("text",          str)                  : クリーンな応答テキスト（最後に1回）
        """
        ctx = await self._prepare_context(request)

        # 想起した記憶を最初にyield
        all_recalled = ctx.recalled_identity + ctx.recalled
        if all_recalled:
            yield ("memories", all_recalled)

        # タグ方式でリアルタイムストリーミングした場合 True。
        # True の場合、最終 yield ("text", clean_text) をスキップする。
        text_already_streamed = False

        if ctx.provider_impl.SUPPORTS_TOOLS:
            tool_executor = ToolExecutor(
                character_id=request.character_id,
                session_id=request.session_id,
                memory_manager=self.memory_manager,
                drift_manager=self.drift_manager,
            )
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
            tool_executor = None
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
                except Exception as e:
                    logger.log_warning("PowerRecall", f"検索失敗 character={request.character_id}: {e}")
                    power_recalled = {}

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

            # 後処理: マーカーの側効果処理（記憶保存・narrative彫り込み・drift適用）
            # テキスト表示は済んでいるため、clean_text は副作用処理とログ用途にのみ使う。
            inscriber = Inscriber(request.character_id, self.memory_manager)
            clean_text = inscriber.inscribe_memory_from_text(full_text, request.current_preset_id)
            clean_text = Carver(request.character_id, self.memory_manager.sqlite).carve_narrative_from_text(clean_text)
            clean_text = self._apply_drifts(clean_text, request)

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
                yield ("angle_switched", f"{request.character_name}@{switch_info[0]}")
                return

        # FrontOutput は reflector タスク起動より前にログする。
        # asyncio.create_task はコンテキストをコピーするため、
        # FrontOutput のカウンターインクリメントが反映された状態で reflector を起動する。
        logger.log_front_output(clean_text)
        self._log_debug("CHAT stream", request, ctx.messages, clean_text)

        # 自己参照ループ: ストリーム完了後にバックグラウンドタスクとして起動する。
        # ユーザーへの応答は既に完了しているため、SSE接続をブロックしない。
        if request.self_reflection_mode != "disabled" and request.session_id:
            # キャラクター応答を末尾に追加して渡す。ウィンドウ切り出しは reflector 内で行う。
            reflection_messages = [*ctx.messages, {"role": "assistant", "content": clean_text}]
            reflector = SelfReflector(self.memory_manager, self.drift_manager)
            asyncio.create_task(
                reflector.run(
                    request_mode=request.self_reflection_mode,
                    trigger_preset_id=request.self_reflection_preset_id,
                    n_turns=request.self_reflection_n_turns,
                    settings=request.settings,
                    messages=reflection_messages,
                    character_id=request.character_id,
                    session_id=request.session_id,
                    current_preset_id=request.current_preset_id,
                )
            )

        # 別れ検出: ストリーム完了後にバックグラウンドタスクとして起動する。
        # 結果はDBに保存し、次リクエスト時の already_exited チェックで検知される。
        if request.session_id:
            char_for_farewell = self.memory_manager.sqlite.get_character(request.character_id)
            if (
                char_for_farewell
                and getattr(char_for_farewell, "farewell_config", None)
                and getattr(char_for_farewell, "self_reflection_preset_id", None)
                and getattr(char_for_farewell, "relationship_status", "active") != "estranged"
            ):
                farewell_messages = [*ctx.messages, {"role": "assistant", "content": clean_text}]
                detector = FarewellDetector(self.memory_manager.sqlite)
                asyncio.create_task(
                    _run_farewell_detection(
                        detector=detector,
                        character_id=request.character_id,
                        character_name=request.character_name,
                        session_id=request.session_id,
                        preset_id=char_for_farewell.self_reflection_preset_id,
                        farewell_config=char_for_farewell.farewell_config,
                        messages=farewell_messages,
                        settings=request.settings,
                        chroma=self.memory_manager.chroma,
                    )
                )

        if clean_text and not text_already_streamed:
            yield ("text", clean_text)
