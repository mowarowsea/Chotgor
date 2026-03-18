"""ChatService — リクエスト受付から応答テキスト返却までのサービス層。

Architecture:
  Adapter → ChatService → LLM provider

Flow:
  1. 記憶を想起 (ChromaDB RAG)
  1b. SELF_DRIFT指針をDBからロード
  2. URL自動fetch
  3. システムプロンプト構築
  4. プロバイダーへディスパッチ
  5. 応答から記憶を刻み込み (Inscriber.carve)
  5b. SELF_DRIFTマーカーを処理してDBに保存
  6. デバッグログ
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..tools import ToolExecutor

from ..debug_logger import log_front_output
from ..memory.drift_manager import DriftManager
from ..memory.inscriber import carve
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

    def _apply_drifts(self, text: str, request: ChatRequest) -> str:
        """テキストから [DRIFT:...] / [DRIFT_RESET] マーカーを抽出しDBに反映する。

        Args:
            text: [MEMORY:...] 除去済みの応答テキスト。
            request: セッションID・キャラクターIDを含むリクエスト。

        Returns:
            マーカーを除去したクリーンなテキスト。
        """
        # session_id がない場合はDB書き込みをスキップするが、マーカーのストリップは必ず行う
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
        """switch_angle リクエストを tool_executor またはタグから抽出する。

        tool-use 方式（SUPPORTS_TOOLS=True）は ToolExecutor.switch_request を確認する。
        タグ方式（SUPPORTS_TOOLS=False）は [SWITCH_ANGLE:...] マーカーをスキャンする。
        available_presets が空の場合はスキャンをスキップする。

        Args:
            tool_executor: tool-use 方式の場合は ToolExecutor インスタンス、タグ方式は None。
            clean_text: carve/drift 処理済みの応答テキスト。
            has_angle_presets: available_presets が非空かどうか。

        Returns:
            tuple:
                clean_text (str): タグを除去したテキスト（タグ方式のみ変化する）。
                switch_info (tuple[str, str] | None): (preset_name, self_instruction) または None。
        """
        if tool_executor is not None:
            return clean_text, tool_executor.switch_request
        # タグ方式: available_presets が空なら parse をスキップする
        if not has_angle_presets:
            return clean_text, None
        return switch_extract(clean_text)

    def _build_switched_request(
        self, original: ChatRequest, preset_name: str, self_instruction: str
    ) -> ChatRequest | None:
        """switch_angle 後の再ディスパッチ用 ChatRequest を構築する。

        available_presets からプリセット名で一致するエントリを検索し、
        プロバイダー・モデル・指針を差し替えた新しい ChatRequest を返す。
        再ディスパッチでは無限ループ防止のため available_presets を空にする。

        Args:
            original: 元のリクエスト。
            preset_name: 切り替え先プリセット名。
            self_instruction: 切り替え後モデルへの自己指針。

        Returns:
            新しい ChatRequest。preset_name が見つからない場合は None。
        """
        preset = next(
            (p for p in original.available_presets if p.get("preset_name") == preset_name),
            None,
        )
        if preset is None:
            return None

        # セッションの SELF_DRIFT をリセットし、新たな self_instruction を設定する
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
            # 切り替え後の自己指針を active_drifts に直接セットする
            active_drifts=[self_instruction] if self_instruction else [],
            # 再ディスパッチでは switch_angle を禁止する（無限ループ防止）
            available_presets=[],
        )

    async def execute(self, request: ChatRequest) -> str:
        """LLMにディスパッチして応答テキストを返す。SSEを知らない。"""
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # --- 1. 記憶の想起 ---
        # recall_query_override が指定されている場合はそれを使う（グループチャット用）
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

        # --- 4. プロバイダーを決定（use_tools フラグに使用） ---
        provider_impl = create_provider(
            request.provider, request.model, request.settings, thinking_level=request.thinking_level
        )

        # --- 3. システムプロンプト構築 ---
        system_prompt = build_system_prompt(
            character_system_prompt=request.character_system_prompt,
            recalled_identity_memories=recalled_identity or None,
            recalled_memories=recalled or None,
            fetched_contents=fetched_contents,
            meta_instructions=request.meta_instructions,
            provider_additional_instructions=request.provider_additional_instructions,
            enable_time_awareness=request.enable_time_awareness,
            current_time_str=request.current_time_str,
            time_since_last_interaction=request.time_since_last_interaction,
            active_drifts=active_drifts or None,
            use_tools=provider_impl.SUPPORTS_TOOLS,
            available_presets=request.available_presets or None,
            current_preset_name=request.current_preset_name,
        )

        # --- 4. プロバイダーへディスパッチ ---
        # SUPPORTS_TOOLS=True のプロバイダーはtool-useで記憶・DRIFTを操作する。
        # それ以外（Claude CLI等）は従来のマーカー方式にフォールバックする。
        tool_executor = None
        if provider_impl.SUPPORTS_TOOLS:
            tool_executor = ToolExecutor(
                character_id=request.character_id,
                session_id=request.session_id,
                memory_manager=self.memory_manager,
                drift_manager=self.drift_manager,
            )
            try:
                clean_text = await provider_impl.generate_with_tools(system_prompt, messages, tool_executor)
            except Exception as e:
                import traceback
                return f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]"
        else:
            try:
                response_text = await provider_impl.generate(system_prompt, messages)
            except Exception as e:
                import traceback
                return f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]"

            # --- 5. 記憶を刻み込む（マーカー方式フォールバック） ---
            clean_text = carve(response_text, request.character_id, self.memory_manager, request.current_preset_id)

            # --- 5b. SELF_DRIFT マーカーを処理する ---
            clean_text = self._apply_drifts(clean_text, request)

        # --- 5c. switch_angle の処理 ---
        clean_text, switch_info = self._extract_switch_info(
            tool_executor, clean_text, bool(request.available_presets)
        )
        if switch_info:
            switched = self._build_switched_request(request, *switch_info)
            if switched is not None:
                return await self.execute(switched)

        log_front_output(clean_text)

        # --- 6. デバッグログ ---
        sep = "-" * 60
        print(f"\n{sep}")
        print(
            f"[CHAT] character={request.character_id}"
            f" provider={request.provider}"
            f" model={request.model or '(default)'}"
        )
        for m in messages:
            role = m.get("role", "?").upper()
            content = extract_text_content(m.get("content"))
            print(f"  [{role}] {content[:300]}{'...' if len(content) > 300 else ''}")
        print(f"  [ASSISTANT] {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")
        print(sep, flush=True)

        return clean_text

    async def execute_stream(self, request: ChatRequest):
        """ストリーミングでLLMにディスパッチし、型付きチャンクをyieldする。

        記憶想起・思考ブロック・応答テキストをそれぞれ別の型でyieldする。
        テキストは全チャンクを内部バッファで受け取った後に carve() を実行し、
        [MEMORY:...] マーカーを取り除いたクリーンなテキストをまとめてyieldする。

        Yields:
            tuple[str, Any]:
                ("memories", list[dict]) : 想起した記憶リスト（最初に1回だけyield）
                ("thinking", str)        : 思考ブロック（リアルタイム）
                ("text",     str)        : クリーンな応答テキスト（最後に1回）
        """
        import traceback

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # --- 1. 記憶の想起 ---
        # recall_query_override が指定されている場合はそれを使う（グループチャット用）
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

        # 想起した記憶を最初にyield（フロントに表示するため）identity + others を合わせて渡す
        all_recalled = recalled_identity + recalled
        if all_recalled:
            yield ("memories", all_recalled)

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

        # --- 4. プロバイダーを決定（use_tools フラグに使用） ---
        provider_impl = create_provider(
            request.provider, request.model, request.settings,
            thinking_level=request.thinking_level
        )

        # --- 3. システムプロンプト構築 ---
        system_prompt = build_system_prompt(
            character_system_prompt=request.character_system_prompt,
            recalled_identity_memories=recalled_identity or None,
            recalled_memories=recalled or None,
            fetched_contents=fetched_contents,
            meta_instructions=request.meta_instructions,
            provider_additional_instructions=request.provider_additional_instructions,
            enable_time_awareness=request.enable_time_awareness,
            current_time_str=request.current_time_str,
            time_since_last_interaction=request.time_since_last_interaction,
            active_drifts=active_drifts or None,
            use_tools=provider_impl.SUPPORTS_TOOLS,
            available_presets=request.available_presets or None,
            current_preset_name=request.current_preset_name,
        )

        # --- 4. プロバイダーへストリーミングディスパッチ ---
        # SUPPORTS_TOOLS=True のプロバイダーはtool-useで記憶・DRIFTを操作する。
        # ストリーミングモードでもtool-useはバッファリング（思考ブロックのリアルタイム送信は非対応）。
        # それ以外（Claude CLI等）は従来のストリーミング＋マーカー方式を維持する。
        tool_executor = None
        if provider_impl.SUPPORTS_TOOLS:
            tool_executor = ToolExecutor(
                character_id=request.character_id,
                session_id=request.session_id,
                memory_manager=self.memory_manager,
                drift_manager=self.drift_manager,
            )
            try:
                clean_text = await provider_impl.generate_with_tools(system_prompt, messages, tool_executor)
            except Exception as e:
                yield ("text", f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]")
                return
        else:
            # テキストはバッファリングして [MEMORY:...] マーカーをcarveしてからyield
            full_text = ""
            try:
                async for chunk_type, content in provider_impl.generate_stream_typed(system_prompt, messages):
                    if chunk_type == "thinking":
                        # 思考ブロックはリアルタイムでyield
                        yield ("thinking", content)
                    elif chunk_type == "text":
                        full_text += content
            except Exception as e:
                yield ("text", f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]")
                return

            # --- 5. 記憶を刻み込む（マーカー方式フォールバック） ---
            clean_text = carve(full_text, request.character_id, self.memory_manager, request.current_preset_id)

            # --- 5b. SELF_DRIFT マーカーを処理する ---
            clean_text = self._apply_drifts(clean_text, request)

        # --- 5c. switch_angle の処理 ---
        clean_text, switch_info = self._extract_switch_info(
            tool_executor, clean_text, bool(request.available_presets)
        )
        if switch_info:
            switched = self._build_switched_request(request, *switch_info)
            if switched is not None:
                # 切り替え後のプロバイダーで再ディスパッチし、そのイベントをすべて転送する
                async for event in self.execute_stream(switched):
                    yield event
                # API 層がセッションの model_id を更新できるよう切り替え情報を通知する
                yield ("angle_switched", f"{request.character_name}@{switch_info[0]}")
                return

        log_front_output(clean_text)

        # --- 6. デバッグログ ---
        sep = "-" * 60
        print(f"\n{sep}")
        print(
            f"[CHAT stream] character={request.character_id}"
            f" provider={request.provider}"
            f" model={request.model or '(default)'}"
        )
        print(f"  [RESPONSE] {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")
        print(sep, flush=True)

        if clean_text:
            yield ("text", clean_text)
