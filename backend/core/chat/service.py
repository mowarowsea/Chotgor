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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..tools import ToolExecutor

from ..debug_logger import log_front_output
from ..memory.drift_manager import DriftManager
from ..memory.carver import Carver
from ..memory.inscriber import Inscriber
from ..memory.manager import MemoryManager
from ..providers.registry import create_provider
from ..system_prompt import build_system_prompt
from ..tools import ToolExecutor
from ..web_fetch import fetch_urls, find_urls
from .drifter import extract as drift_extract
from .models import ChatRequest
from .switcher import extract as switch_extract


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
        clean, drifts, reset = drift_extract(text)
        if not self.drift_manager or not request.session_id:
            return clean
        if reset:
            self.drift_manager.reset_drifts(request.session_id, request.character_id)
        for content in drifts:
            try:
                self.drift_manager.add_drift(request.session_id, request.character_id, content)
            except Exception:
                pass
        return clean

    def _extract_switch_info(
        self, tool_executor: "ToolExecutor | None", clean_text: str, has_angle_presets: bool
    ) -> tuple[str, tuple[str, str] | None]:
        """switch_angle リクエストを tool_executor またはタグから抽出する。"""
        if tool_executor is not None:
            return clean_text, tool_executor.switch_request
        if not has_angle_presets:
            return clean_text, None
        return switch_extract(clean_text)

    def _build_switched_request(
        self, original: ChatRequest, preset_name: str, self_instruction: str
    ) -> ChatRequest | None:
        """switch_angle 後の再ディスパッチ用 ChatRequest を構築する。"""
        preset = next(
            (p for p in original.available_presets if p.get("preset_name") == preset_name),
            None,
        )
        if preset is None:
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
            except Exception:
                pass

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
        provider_impl = create_provider(
            request.provider, request.model, request.settings, thinking_level=request.thinking_level
        )

        # --- システムプロンプト構築 ---
        system_prompt = build_system_prompt(
            character_system_prompt=request.character_system_prompt,
            recalled_identity_memories=recalled_identity or None,
            recalled_memories=recalled or None,
            fetched_contents=fetched_contents,
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
        """デバッグログを出力する。"""
        sep = "-" * 60
        print(f"\n{sep}")
        print(
            f"[{label}] character={request.character_id}"
            f" provider={request.provider}"
            f" model={request.model or '(default)'}"
        )
        for m in messages:
            role = m.get("role", "?").upper()
            content = extract_text_content(m.get("content"))
            print(f"  [{role}] {content[:300]}{'...' if len(content) > 300 else ''}")
        print(f"  [RESPONSE] {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")
        print(sep, flush=True)

    # --- 公開メソッド ---

    async def execute(self, request: ChatRequest) -> str:
        """LLMにディスパッチして応答テキストを返す。SSEを知らない。"""
        ctx = await self._prepare_context(request)

        if ctx.provider_impl.SUPPORTS_TOOLS:
            tool_executor = ToolExecutor(
                character_id=request.character_id,
                session_id=request.session_id,
                memory_manager=self.memory_manager,
                drift_manager=self.drift_manager,
            )
            try:
                clean_text = await ctx.provider_impl.generate_with_tools(ctx.system_prompt, ctx.messages, tool_executor)
            except Exception as e:
                import traceback
                return f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]"
        else:
            tool_executor = None
            try:
                response_text = await ctx.provider_impl.generate(ctx.system_prompt, ctx.messages)
            except Exception as e:
                import traceback
                return f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]"

            inscriber = Inscriber(request.character_id, self.memory_manager)
            clean_text = inscriber.inscribe_memory_from_text(response_text, request.current_preset_id)
            clean_text = Carver(request.character_id, self.memory_manager.sqlite).carve_narrative_from_text(clean_text)
            clean_text = self._apply_drifts(clean_text, request)

        clean_text, switch_info = self._extract_switch_info(
            tool_executor, clean_text, bool(request.available_presets)
        )
        if switch_info:
            switched = self._build_switched_request(request, *switch_info)
            if switched is not None:
                return await self.execute(switched)

        log_front_output(clean_text)
        self._log_debug("CHAT", request, ctx.messages, clean_text)

        return clean_text

    async def execute_stream(self, request: ChatRequest):
        """ストリーミングでLLMにディスパッチし、型付きチャンクをyieldする。

        Yields:
            tuple[str, Any]:
                ("memories", list[dict]) : 想起した記憶リスト（最初に1回だけyield）
                ("thinking", str)        : 思考ブロック（リアルタイム）
                ("text",     str)        : クリーンな応答テキスト（最後に1回）
        """
        import traceback

        ctx = await self._prepare_context(request)

        # 想起した記憶を最初にyield
        all_recalled = ctx.recalled_identity + ctx.recalled
        if all_recalled:
            yield ("memories", all_recalled)

        if ctx.provider_impl.SUPPORTS_TOOLS:
            tool_executor = ToolExecutor(
                character_id=request.character_id,
                session_id=request.session_id,
                memory_manager=self.memory_manager,
                drift_manager=self.drift_manager,
            )
            try:
                clean_text = await ctx.provider_impl.generate_with_tools(ctx.system_prompt, ctx.messages, tool_executor)
            except Exception as e:
                yield ("text", f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]")
                return
        else:
            tool_executor = None
            full_text = ""
            try:
                async for chunk_type, content in ctx.provider_impl.generate_stream_typed(ctx.system_prompt, ctx.messages):
                    if chunk_type == "thinking":
                        yield ("thinking", content)
                    elif chunk_type == "text":
                        full_text += content
            except Exception as e:
                yield ("text", f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]")
                return

            inscriber = Inscriber(request.character_id, self.memory_manager)
            clean_text = inscriber.inscribe_memory_from_text(full_text, request.current_preset_id)
            clean_text = Carver(request.character_id, self.memory_manager.sqlite).carve_narrative_from_text(clean_text)
            clean_text = self._apply_drifts(clean_text, request)

        clean_text, switch_info = self._extract_switch_info(
            tool_executor, clean_text, bool(request.available_presets)
        )
        if switch_info:
            switched = self._build_switched_request(request, *switch_info)
            if switched is not None:
                async for event in self.execute_stream(switched):
                    yield event
                yield ("angle_switched", f"{request.character_name}@{switch_info[0]}")
                return

        log_front_output(clean_text)
        self._log_debug("CHAT stream", request, ctx.messages, clean_text)

        if clean_text:
            yield ("text", clean_text)
