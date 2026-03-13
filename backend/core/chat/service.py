"""ChatService — リクエスト受付から応答テキスト返却までのサービス層。

Architecture:
  Adapter → ChatService → LLM provider

Flow:
  1. 記憶を想起 (ChromaDB RAG)
  2. URL自動fetch
  3. システムプロンプト構築
  4. プロバイダーへディスパッチ
  5. 応答から記憶を刻み込み (Inscriber.carve)
  6. デバッグログ
"""

from typing import Union

from ..debug_logger import log_front_output
from ..memory.drift_manager import DriftManager
from ..memory.inscriber import carve
from ..memory.manager import MemoryManager
from ..providers.registry import create_provider
from ..system_prompt import build_system_prompt
from ..web_fetch import fetch_urls, find_urls
from .drifter import extract as drift_extract
from .models import ChatRequest


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
        if not self.drift_manager or not request.session_id:
            return text
        clean, drifts, reset = drift_extract(text)
        if reset:
            self.drift_manager.reset_drifts(request.session_id, request.character_id)
        for content in drifts:
            try:
                self.drift_manager.add_drift(request.session_id, request.character_id, content)
            except Exception:
                pass
        return clean

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

        recalled = []
        if last_user_msg:
            try:
                recalled = self.memory_manager.recall_memory(request.character_id, last_user_msg)
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

        # --- 3. システムプロンプト構築 ---
        system_prompt = build_system_prompt(
            character_system_prompt=request.character_system_prompt,
            recalled_memories=recalled,
            fetched_contents=fetched_contents,
            meta_instructions=request.meta_instructions,
            provider_additional_instructions=request.provider_additional_instructions,
            enable_time_awareness=request.enable_time_awareness,
            current_time_str=request.current_time_str,
            time_since_last_interaction=request.time_since_last_interaction,
            active_drifts=request.active_drifts or None,
        )

        # --- 4. プロバイダーへディスパッチ ---
        provider_impl = create_provider(request.provider, request.model, request.settings, thinking_level=request.thinking_level)
        try:
            response_text = await provider_impl.generate(system_prompt, messages)
        except Exception as e:
            import traceback
            return f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]"

        # --- 5. 記憶を刻み込む ---
        clean_text = carve(response_text, request.character_id, self.memory_manager)

        # --- 5b. SELF_DRIFT マーカーを処理する ---
        clean_text = self._apply_drifts(clean_text, request)

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

        recalled = []
        if last_user_msg:
            try:
                recalled = self.memory_manager.recall_memory(request.character_id, last_user_msg)
            except Exception:
                pass

        # 想起した記憶を最初にyield（フロントに表示するため）
        if recalled:
            yield ("memories", recalled)

        # --- 2. URLの自動fetch ---
        fetched_contents = []
        if last_user_msg:
            urls = find_urls(last_user_msg)
            if urls:
                try:
                    fetched_contents = await fetch_urls(urls)
                except Exception:
                    pass

        # --- 3. システムプロンプト構築 ---
        system_prompt = build_system_prompt(
            character_system_prompt=request.character_system_prompt,
            recalled_memories=recalled,
            fetched_contents=fetched_contents,
            meta_instructions=request.meta_instructions,
            provider_additional_instructions=request.provider_additional_instructions,
            enable_time_awareness=request.enable_time_awareness,
            current_time_str=request.current_time_str,
            time_since_last_interaction=request.time_since_last_interaction,
            active_drifts=request.active_drifts or None,
        )

        # --- 4. プロバイダーへストリーミングディスパッチ ---
        provider_impl = create_provider(
            request.provider, request.model, request.settings,
            thinking_level=request.thinking_level
        )

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

        # --- 5. 記憶を刻み込む ---
        clean_text = carve(full_text, request.character_id, self.memory_manager)

        # --- 5b. SELF_DRIFT マーカーを処理する ---
        clean_text = self._apply_drifts(clean_text, request)

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
