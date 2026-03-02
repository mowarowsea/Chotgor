"""Claude Code CLI wrapper.

Claude Code (claude CLI) をサブプロセスとして呼び出し、
OpenAI互換のSSEストリームとして返す。

アーキテクチャ:
  OpenWebUI → Chotgor(ホスト) → claude CLI(ホスト)

記憶の仕組み:
  - 会話前: ChromaDBから関連記憶を検索してシステムプロンプトに注入 (Block 2)
  - 会話後: Claudeが [MEMORY:category|content] 形式で返答内に埋め込んだ記憶を抽出・保存
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import AsyncIterator, Optional

from .memory.manager import MemoryManager
from .system_prompt import build_system_prompt

# [MEMORY:category|content] パターン
MEMORY_PATTERN = re.compile(r"\[MEMORY:(\w+)\|([^\]]+)\]", re.DOTALL)


def _find_claude() -> str:
    """claude CLI のパスを返す。見つからなければ 'claude' を返す。"""
    for name in ("claude", "claude.cmd", "claude.ps1"):
        path = shutil.which(name)
        if path:
            return path
    return "claude"


CLAUDE_BIN = _find_claude()


async def stream_chat(
    messages: list[dict],
    character_id: str,
    character_system_prompt: str,
    meta_instructions: str,
    memory_manager: MemoryManager,
    tavily_api_key: Optional[str] = None,
    **kwargs,  # anthropic_api_key / model / max_tokens は無視
) -> AsyncIterator[str]:
    """Claude Code CLI を呼び出してSSEチャンクをyieldする。"""

    # --- 1. 関連記憶を事前に検索 ---
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    recalled = []
    if last_user_msg:
        try:
            recalled = memory_manager.recall_memory(character_id, last_user_msg)
        except Exception:
            pass

    # --- 2. システムプロンプト構築 (3ブロック) ---
    system_prompt = build_system_prompt(
        character_system_prompt=character_system_prompt,
        recalled_memories=recalled,
        meta_instructions=meta_instructions,
    )

    # --- 3. 会話履歴をテキストに整形 ---
    conversation = _format_conversation(messages)

    # --- 4. 一時ファイルに書き出し (改行・特殊文字を安全に渡すため) ---
    sys_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    sys_file.write(system_prompt)
    sys_file.close()

    msg_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    msg_file.write(conversation)
    msg_file.close()

    collected_text = ""

    try:
        result = await _run_claude(sys_file.name, msg_file.name)

        if result.returncode != 0:
            err_msg = result.stderr.decode("utf-8", errors="replace")
            out_msg = result.stdout.decode("utf-8", errors="replace")
            yield _sse_chunk(f"[Claude Code error (code {result.returncode})\nSTDERR: {err_msg[:1000]}\nSTDOUT: {out_msg[:2000]}]")
            yield "data: [DONE]\n\n"
            return

        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        collected_text += block["text"]

            elif etype == "result":
                result_text = event.get("result", "")
                if result_text and not collected_text:
                    collected_text = result_text

    except FileNotFoundError:
        yield _sse_chunk(
            "[Error: claude CLI が見つかりません。Claude Code がインストール・認証済みか確認してください]"
        )
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        import traceback
        yield _sse_chunk(f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]")
        yield "data: [DONE]\n\n"
        return
    finally:
        for path in (sys_file.name, msg_file.name):
            try:
                os.unlink(path)
            except Exception:
                pass

    # --- 5. [MEMORY:...] マーカーを抽出して保存、テキストから削除 ---
    clean_text, memories = _extract_memories(collected_text)

    for category, content in memories:
        try:
            memory_manager.write_memory(
                character_id=character_id,
                content=content.strip(),
                category=category.strip(),
            )
        except Exception:
            pass

    # --- 6. テキストをSSEとして返す ---
    # Console log for debugging
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"[CHAT] character={character_id}")
    for m in messages:
        role = m.get("role", "?").upper()
        content = m.get("content", "")
        print(f"  [{role}] {content[:300]}{'...' if len(content) > 300 else ''}")
    print(f"  [ASSISTANT] {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")
    print(sep, flush=True)

    if clean_text:
        yield _sse_chunk(clean_text)
    yield "data: [DONE]\n\n"


async def _run_claude(sys_path: str, msg_path: str) -> subprocess.CompletedProcess:
    """claude CLI をスレッド内で同期実行して結果を返す。

    Windows の asyncio SelectorEventLoop は create_subprocess_exec 非対応のため
    subprocess.run を asyncio.to_thread でラップして使用する。
    """
    # CLAUDECODE: ネストセッション禁止チェックを回避
    # ANTHROPIC_API_KEY: 無効なキーがあるとOAuth認証より優先されてしまうため除去
    _exclude = {"CLAUDECODE", "ANTHROPIC_API_KEY"}
    env = {k: v for k, v in os.environ.items() if k not in _exclude}

    with open(sys_path, encoding="utf-8") as f:
        system_content = f.read()
    with open(msg_path, encoding="utf-8") as f:
        msg_content = f.read()

    def run():
        return subprocess.run(
            [CLAUDE_BIN, "--output-format", "stream-json", "--verbose",
             "--print", "--tools", "", "--no-session-persistence",
             "--system-prompt", system_content],
            input=msg_content.encode("utf-8"),
            capture_output=True,
            env=env,
        )

    return await asyncio.to_thread(run)


def _extract_memories(text: str) -> tuple[str, list[tuple[str, str]]]:
    """テキストから [MEMORY:category|content] を抽出し、削除したテキストと一緒に返す。"""
    memories = MEMORY_PATTERN.findall(text)
    clean = MEMORY_PATTERN.sub("", text).strip()
    return clean, memories


def _format_conversation(messages: list[dict]) -> str:
    """OpenAI形式のmessage配列をClaudeへのプロンプトテキストに変換する。"""
    if not messages:
        return ""

    # メッセージが1つだけならそのまま返す
    if len(messages) == 1:
        return messages[0].get("content", "")

    # 複数ターン: 最後のユーザーメッセージを「現在のメッセージ」として分離
    history_parts = []
    for msg in messages[:-1]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        if role == "System":
            continue
        history_parts.append(f"{role}: {content}")

    last = messages[-1].get("content", "")

    if history_parts:
        history = "\n".join(history_parts)
        return f"[Previous conversation]\n{history}\n\n[Current message]\n{last}"
    return last


def _sse_chunk(text: str) -> str:
    payload = {
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": text}, "finish_reason": None}
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
