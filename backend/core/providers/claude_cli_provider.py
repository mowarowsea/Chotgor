"""Claude Code CLI provider.

Calls the `claude` CLI as a subprocess using asyncio.to_thread (Windows-safe).

Public helpers
--------------
invoke_claude_cli(system_prompt, input_text) -> str
    Generic single-turn Claude CLI call shared by digest.py / forget.py.
    Raises RuntimeError on non-zero exit.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile

from ..debug_logger import log_provider_request, log_provider_response
from .base import BaseLLMProvider


def _find_claude() -> str:
    for name in ("claude", "claude.cmd", "claude.ps1"):
        path = shutil.which(name)
        if path:
            return path
    return "claude"


CLAUDE_BIN = _find_claude()

# Env vars that must be stripped before spawning the Claude subprocess.
# CLAUDECODE  : prevents nested-session error.
# ANTHROPIC_API_KEY : forces OAuth fallback instead of using a possibly invalid key.
_CLAUDE_ENV_EXCLUDES = {"CLAUDECODE", "ANTHROPIC_API_KEY"}

_THINKING_TOKENS = {
    "low": 1024,
    "medium": 5000,
    "high": 16000,
}


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _CLAUDE_ENV_EXCLUDES}


def _parse_stream_json(raw: str) -> str:
    """Extract assistant text from Claude CLI stream-json output."""
    collected = ""
    for line in raw.splitlines():
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
                    collected += block["text"]
        elif etype == "result":
            result_text = event.get("result", "")
            if result_text and not collected:
                collected = result_text
    return collected


async def invoke_claude_cli(system_prompt: str, input_text: str) -> str:
    """Invoke Claude CLI with a system prompt and stdin input; return assistant text.

    Used by digest.py and forget.py for batch / scheduled tasks.
    Raises RuntimeError if the CLI exits with a non-zero return code.
    """
    env = _clean_env()

    def run():
        return subprocess.run(
            [
                CLAUDE_BIN,
                "--output-format", "stream-json",
                "--verbose",
                "--print",
                "--tools", "",
                "--no-session-persistence",
                "--system-prompt", system_prompt,
            ],
            input=input_text.encode("utf-8"),
            capture_output=True,
            env=env,
        )

    result = await asyncio.to_thread(run)

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Claude CLI exited with code {result.returncode}: {err[:500]}"
        )

    return _parse_stream_json(result.stdout.decode("utf-8", errors="replace"))


class ClaudeCliProvider(BaseLLMProvider):
    PROVIDER_ID = "claude_cli"
    DEFAULT_MODEL = ""
    REQUIRES_API_KEY = False

    def __init__(self, model: str = "", character_name: str = "", thinking_level: str = "default"):
        self.model = model  # CLI model is configured via Claude Code settings, not flags
        self.character_name = character_name
        self.thinking_level = thinking_level

    @classmethod
    def from_config(cls, model: str, settings: dict, character_name: str = "", thinking_level: str = "default", **kwargs):
        return cls(model=model, character_name=character_name, thinking_level=thinking_level)

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        # Detect if any image exists in messages
        has_images = any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for m in messages
            for item in (m.get("content") if isinstance(m.get("content"), list) else [])
        )

        if has_images:
            system_prompt += (
                "\n\n[SYSTEM NOTE: The user has provided one or more images, "
                "but you currently cannot 'see' them because of the current connection mode (Claude CLI). "
                "Please inform the user naturally that you cannot see the images right now.]"
            )

        conversation = _format_conversation(messages, self.character_name)

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

        extra_env = {}
        if self.thinking_level != "default":
            extra_env["MAX_THINKING_TOKENS"] = str(_THINKING_TOKENS[self.thinking_level])

        log_provider_request("claude_cli", {
            "system_prompt": system_prompt,
            "conversation": conversation,
            "extra_env": extra_env,
        })

        try:
            result = await _run_claude(sys_file.name, msg_file.name, extra_env=extra_env or None)

            if result.returncode != 0:
                err_msg = result.stderr.decode("utf-8", errors="replace")
                out_msg = result.stdout.decode("utf-8", errors="replace")
                return (
                    f"[Claude Code error (code {result.returncode})\n"
                    f"STDERR: {err_msg[:1000]}\nSTDOUT: {out_msg[:2000]}]"
                )

            raw_stdout = result.stdout.decode("utf-8", errors="replace")
            log_provider_response("claude_cli", raw_stdout)
            return _parse_stream_json(raw_stdout)

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

    async def generate_stream(self, system_prompt: str, messages: list[dict]):
        """Claude CLI からテキストチャンクをストリーミングで取得する。

        subprocess.Popen で stdout を行単位で読み取り、逐次yieldする。
        """
        # 画像が含まれる場合はCLIでは見えない旨をシステムプロンプトに追記
        has_images = any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for m in messages
            for item in (m.get("content") if isinstance(m.get("content"), list) else [])
        )
        if has_images:
            system_prompt += (
                "\n\n[SYSTEM NOTE: The user has provided one or more images, "
                "but you currently cannot 'see' them because of the current connection mode (Claude CLI). "
                "Please inform the user naturally that you cannot see the images right now.]"
            )

        conversation = _format_conversation(messages, self.character_name)
        extra_env = {}
        if self.thinking_level != "default":
            extra_env["MAX_THINKING_TOKENS"] = str(_THINKING_TOKENS[self.thinking_level])

        env = _clean_env()
        if extra_env:
            env.update(extra_env)

        import threading

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        log_provider_request("claude_cli", {
            "system_prompt": system_prompt,
            "conversation": conversation,
            "extra_env": extra_env,
        })

        def run():
            """subprocess.Popenでストリーミング出力を行単位で読み、キューへ送信する。"""
            try:
                proc = subprocess.Popen(
                    [
                        CLAUDE_BIN,
                        "--output-format", "stream-json",
                        "--verbose",
                        "--print",
                        "--tools", "",
                        "--no-session-persistence",
                        "--system-prompt", system_prompt,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                proc.stdin.write(conversation.encode("utf-8"))
                proc.stdin.close()

                for line_bytes in iter(proc.stdout.readline, b""):
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("type") == "assistant":
                            for block in event.get("message", {}).get("content", []):
                                if block.get("type") == "text" and block["text"]:
                                    loop.call_soon_threadsafe(queue.put_nowait, block["text"])
                    except json.JSONDecodeError:
                        pass

                proc.wait()
                if proc.returncode != 0:
                    err = proc.stderr.read().decode("utf-8", errors="replace")
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        RuntimeError(f"Claude CLI exited with code {proc.returncode}: {err[:500]}")
                    )
            except FileNotFoundError:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    RuntimeError("claude CLI が見つかりません。Claude Code がインストール・認証済みか確認してください")
                )
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield f"[Claude CLI error: {item}]"
                break
            yield item


async def _run_claude(sys_path: str, msg_path: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run claude CLI in a thread (Windows asyncio SelectorEventLoop workaround)."""
    env = _clean_env()
    if extra_env:
        env.update(extra_env)

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


def _format_conversation(messages: list[dict], character_name: str = "") -> str:
    """Convert OpenAI-format messages to a text prompt for the Claude CLI.

    XMLタグ形式を使う。"User: / Assistant:" のようなプレーンテキスト形式だと
    LLMがそのパターンを応答の中で継続してしまうため。
    キャラクターのロールはキャラクター名（なければ 'character'）で表現し、
    'assistant' という汎用ロール名をLLMに見せない（Chotgor哲学）。
    """
    char_tag = character_name.strip() if character_name.strip() else "character"

    if not messages:
        return ""

    if len(messages) == 1:
        content = messages[0].get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(content or "")

    history_parts = []
    for msg in messages[:-1]:
        role = msg.get("role", "")
        content = msg.get("content")

        text_content = ""
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            text_content = "".join(parts)

        if role == "system":
            continue
        if role == "user":
            history_parts.append(f"<human>{text_content}</human>")
        elif role == "assistant":
            history_parts.append(f"<{char_tag}>{text_content}</{char_tag}>")

    last_content = messages[-1].get("content", "")
    if isinstance(last_content, list):
        parts = []
        for item in last_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        last_text = "".join(parts)
    else:
        last_text = str(last_content)

    if history_parts:
        history = "\n".join(history_parts)
        return f"<history>\n{history}\n</history>\n\n{last_text}"
    return last_text
