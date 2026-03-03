"""Claude Code CLI provider.

Calls the `claude` CLI as a subprocess using asyncio.to_thread (Windows-safe).
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile

from .base import BaseLLMProvider


def _find_claude() -> str:
    for name in ("claude", "claude.cmd", "claude.ps1"):
        path = shutil.which(name)
        if path:
            return path
    return "claude"


CLAUDE_BIN = _find_claude()


class ClaudeCliProvider(BaseLLMProvider):
    def __init__(self, model: str = ""):
        self.model = model  # CLI model is configured via Claude Code settings, not flags

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        conversation = _format_conversation(messages)

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

        try:
            result = await _run_claude(sys_file.name, msg_file.name)

            if result.returncode != 0:
                err_msg = result.stderr.decode("utf-8", errors="replace")
                out_msg = result.stdout.decode("utf-8", errors="replace")
                return (
                    f"[Claude Code error (code {result.returncode})\n"
                    f"STDERR: {err_msg[:1000]}\nSTDOUT: {out_msg[:2000]}]"
                )

            collected_text = ""
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

            return collected_text

        except FileNotFoundError:
            return (
                "[Error: claude CLI が見つかりません。"
                "Claude Code がインストール・認証済みか確認してください]"
            )
        except Exception as e:
            import traceback
            return f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]"
        finally:
            for path in (sys_file.name, msg_file.name):
                try:
                    os.unlink(path)
                except Exception:
                    pass


async def _run_claude(sys_path: str, msg_path: str) -> subprocess.CompletedProcess:
    """Run claude CLI in a thread (Windows asyncio SelectorEventLoop workaround)."""
    # CLAUDECODE: remove to avoid nested session error
    # ANTHROPIC_API_KEY: remove so Claude falls back to OAuth instead of invalid key
    _exclude = {"CLAUDECODE", "ANTHROPIC_API_KEY"}
    env = {k: v for k, v in os.environ.items() if k not in _exclude}

    with open(sys_path, encoding="utf-8") as f:
        system_content = f.read()
    with open(msg_path, encoding="utf-8") as f:
        msg_content = f.read()

    def run():
        return subprocess.run(
            [
                CLAUDE_BIN,
                "--output-format", "stream-json",
                "--verbose",
                "--print",
                "--tools", "",
                "--no-session-persistence",
                "--system-prompt", system_content,
            ],
            input=msg_content.encode("utf-8"),
            capture_output=True,
            env=env,
        )

    return await asyncio.to_thread(run)


def _format_conversation(messages: list[dict]) -> str:
    """Convert OpenAI-format messages to a text prompt for the Claude CLI.

    XMLタグ形式を使う。"User: / Assistant:" のようなプレーンテキスト形式だと
    LLMがそのパターンを応答の中で継続してしまうため。
    """
    if not messages:
        return ""

    if len(messages) == 1:
        return messages[0].get("content", "")

    history_parts = []
    for msg in messages[:-1]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        if role == "user":
            history_parts.append(f"<human>{content}</human>")
        elif role == "assistant":
            history_parts.append(f"<ai>{content}</ai>")

    last = messages[-1].get("content", "")

    if history_parts:
        history = "\n".join(history_parts)
        return f"<history>\n{history}\n</history>\n\n{last}"
    return last
