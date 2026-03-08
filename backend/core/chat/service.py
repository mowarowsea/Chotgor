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
from ..memory.inscriber import carve
from ..memory.manager import MemoryManager
from ..providers.registry import create_provider
from ..system_prompt import build_system_prompt
from ..web_fetch import fetch_urls, find_urls
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
    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager

    async def execute(self, request: ChatRequest) -> str:
        """LLMにディスパッチして応答テキストを返す。SSEを知らない。"""
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # --- 1. 記憶の想起 ---
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
