"""Claude Code CLI provider.

Calls the `claude` CLI as a subprocess using asyncio.to_thread (Windows-safe).

Public helpers
--------------
invoke_claude_cli(system_prompt, input_text) -> str
    Generic single-turn Claude CLI call shared by digest.py / forget.py.
    Raises RuntimeError on non-zero exit.
"""

import asyncio
import contextvars
import json
import os
import shutil
import subprocess
import tempfile

from backend.providers.base import BaseLLMProvider


def _find_claude() -> str:
    for name in ("claude", "claude.cmd", "claude.ps1"):
        path = shutil.which(name)
        if path:
            return path
    return "claude"


CLAUDE_BIN = _find_claude()


def _build_tools_flag(allowed_tools: dict) -> str:
    """allowed_tools 設定から --tools フラグ文字列を組み立てる。

    web_search が True の場合のみ WebSearch と WebFetch（組み込みツール）を有効化する。
    Google系ツールは MCP 経由のため --tools フラグでは制御できず、ここでは加算しない。
    """
    tool_names: list[str] = []
    if allowed_tools.get("web_search"):
        tool_names.extend(["WebSearch", "WebFetch"])
    return ",".join(tool_names)


def _build_cli_args(system_prompt: str, model: str = "", effort: str = "default", allowed_tools: dict | None = None) -> list[str]:
    """claude CLI 呼び出しの共通フラグ列を組み立てる。

    model が空文字列の場合は --model フラグを付けない（CLIデフォルトを使用）。
    effort が "default" の場合は --effort フラグを付けない。
    allowed_tools が None または空の場合は --tools "" で全組み込みツールを無効化する。
    """
    tools_str = _build_tools_flag(allowed_tools or {})
    args = [
        CLAUDE_BIN,
        "--output-format", "stream-json",
        "--verbose",
        "--print",
        "--tools", tools_str,
        "--no-session-persistence",
        "--system-prompt", system_prompt,
    ]
    if model:
        args.extend(["--model", model])
    if effort and effort != "default":
        args.extend(["--effort", effort])
    return args

# Env vars that must be stripped before spawning the Claude subprocess.
# CLAUDECODE  : prevents nested-session error.
# ANTHROPIC_API_KEY : forces OAuth fallback instead of using a possibly invalid key.
_CLAUDE_ENV_EXCLUDES = {"CLAUDECODE", "ANTHROPIC_API_KEY"}


def _clean_env() -> dict:
    """グローバル環境変数からClaudeネスト検出・誤認証キーを除き、コスト削減フラグを注入した dict を返す。"""
    env = {k: v for k, v in os.environ.items() if k not in _CLAUDE_ENV_EXCLUDES}
    # プロンプトキャッシュを無効化してコストを削減する（キャッシュ料金を避けるため）
    env["DISABLE_PROMPT_CACHING"] = "1"
    return env


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


def _extract_thinking_from_stream_json(raw: str) -> str:
    """Claude CLI stream-json 出力から thinking ブロックのテキストを抽出する。"""
    collected = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "thinking":
                    collected += block.get("thinking", "")
    return collected


async def invoke_claude_cli(system_prompt: str, input_text: str) -> str:
    """Invoke Claude CLI with a system prompt and stdin input; return assistant text.

    Used by digest.py and forget.py for batch / scheduled tasks.
    Raises RuntimeError if the CLI exits with a non-zero return code.
    """
    env = _clean_env()

    def run():
        return subprocess.run(
            _build_cli_args(system_prompt),
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
    # MCP サーバー経由でtool-useを実現するため True に設定する。
    # generate_with_tools() をオーバーライドしてClaude CLI内部のMCPループに委譲する。
    SUPPORTS_TOOLS = True

    @classmethod
    async def list_models(cls, settings: dict) -> list[dict]:
        """Claude CLIで指定可能なモデル一覧を静的リストで返す。

        CLIはOAuth認証のためAPIキー不要。モデル一覧取得用サブコマンドも
        存在しないため、既知のモデルIDとエイリアスを静的に定義する。
        エイリアス（opus/sonnet/haiku）はCLIが常に最新モデルへ自動解決する。
        """
        return [
            {"id": "opus",                     "name": "Claude Opus（最新エイリアス）"},
            {"id": "sonnet",                   "name": "Claude Sonnet（最新エイリアス）"},
            {"id": "haiku",                    "name": "Claude Haiku（最新エイリアス）"},
            {"id": "claude-opus-4-7",          "name": "Claude Opus 4.7"},
            {"id": "claude-sonnet-4-6",        "name": "Claude Sonnet 4.6"},
            {"id": "claude-haiku-4-5-20251001","name": "Claude Haiku 4.5"},
        ]

    def __init__(
        self,
        model: str = "",
        character_name: str = "",
        thinking_level: str = "default",
        character_id: str = "",
        session_id: str = "",
        allowed_tools: dict | None = None,
    ):
        self.model = model  # 空文字列の場合はCLIのデフォルトモデルを使用
        self.character_name = character_name
        self.thinking_level = thinking_level
        # MCP サーバーへのコンテキスト注入に使用
        self.character_id = character_id
        self.session_id = session_id
        # キャラクターごとの外部ツール許可設定
        self.allowed_tools: dict = allowed_tools or {}

    @classmethod
    def from_config(
        cls,
        model: str,
        settings: dict,
        character_name: str = "",
        thinking_level: str = "default",
        character_id: str = "",
        session_id: str = "",
        allowed_tools: dict | None = None,
        **kwargs,
    ):
        return cls(
            model=model,
            character_name=character_name,
            thinking_level=thinking_level,
            character_id=character_id,
            session_id=session_id,
            allowed_tools=allowed_tools,
        )

    def _make_env(self) -> dict:
        """Claude CLI サブプロセス用の環境変数 dict を返す。

        _clean_env() をベースに、MCP サーバーが必要とするキャラクターコンテキストを追加する。
        """
        env = _clean_env()
        if self.character_id:
            env["CHOTGOR_CHARACTER_ID"] = self.character_id
        if self.session_id:
            env["CHOTGOR_SESSION_ID"] = self.session_id
        return env

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tool_executor,
    ) -> tuple[str, str]:
        """Claude CLI 内部の MCP ループに委譲する。

        tool_executor（Python 側）は使用しない。ツール呼び出しは MCP サーバーが処理する。

        Returns:
            (text, thinking): thinking は思考ブロックが存在する場合その内容、なければ空文字列。
        """
        raw = await self._run_generate_raw(system_prompt, messages)
        text = _parse_stream_json(raw)
        thinking = _extract_thinking_from_stream_json(raw)
        return text, thinking

    async def _run_generate_raw(self, system_prompt: str, messages: list[dict]) -> str:
        """Claude CLI を呼び出して raw stdout 文字列を返す内部メソッド。

        エラー時はエラーメッセージ文字列を返す（例外は送出しない）。
        """
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

        self._log_request({
            "system_prompt": system_prompt,
            "conversation": conversation,
        })

        try:
            result = await _run_claude(sys_file.name, msg_file.name, model=self.model, effort=self.thinking_level, env=self._make_env(), allowed_tools=self.allowed_tools)

            if result.returncode != 0:
                err_msg = result.stderr.decode("utf-8", errors="replace")
                out_msg = result.stdout.decode("utf-8", errors="replace")
                err = (
                    f"[Claude Code error (code {result.returncode})\n"
                    f"STDERR: {err_msg[:1000]}\nSTDOUT: {out_msg[:2000]}]"
                )
                self._log_error(err)
                return err

            raw_stdout = result.stdout.decode("utf-8", errors="replace")
            self._log_response(raw_stdout)
            return raw_stdout

        except FileNotFoundError:
            err = (
                "[Error: claude CLI が見つかりません。"
                "Claude Code がインストール・認証済みか確認してください]"
            )
            self._log_error(err)
            return err
        except Exception as e:
            import traceback
            err = f"[Error: {type(e).__name__}: {e}\n{traceback.format_exc()}]"
            self._log_error(err)
            return err
        finally:
            for path in (sys_file.name, msg_file.name):
                try:
                    os.unlink(path)
                except Exception:
                    pass

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Claude CLI を呼び出してテキスト応答を返す。"""
        raw = await self._run_generate_raw(system_prompt, messages)
        return _parse_stream_json(raw)

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
        env = self._make_env()

        import threading

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        self._log_request({
            "system_prompt": system_prompt,
            "conversation": conversation,
        })

        def run():
            """subprocess.Popenでストリーミング出力を行単位で読み、キューへ送信する。"""
            accumulated = []
            try:
                proc = subprocess.Popen(
                    _build_cli_args(system_prompt, self.model, self.thinking_level, self.allowed_tools),
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
                                    accumulated.append(block["text"])
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
                self._log_response("".join(accumulated))
                loop.call_soon_threadsafe(queue.put_nowait, None)

        ctx = contextvars.copy_context()
        threading.Thread(target=lambda: ctx.run(run), daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield f"[Claude CLI error: {item}]"
                break
            yield item

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """Claude CLIからthinkingブロックを含む型付きチャンクをストリーミングで取得する。

        stream-json 形式の assistant イベントを解析し、
        content ブロックの type ("thinking" / "text") に応じて
        ("thinking", text) または ("text", text) タプルをyieldする。

        Yields:
            tuple[str, str]: (type, content) 形式。
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
        env = self._make_env()

        import threading

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        self._log_request({
            "system_prompt": system_prompt,
            "conversation": conversation,
        })

        def run():
            """subprocess.Popenでstream-json出力を行単位で読み、型付きチャンクをキューへ送信する。

            assistant イベントの content 配列を走査し、
            type == "thinking" は ("thinking", ...) として、
            type == "text" は ("text", ...) として送信する。
            """
            accumulated = []
            try:
                proc = subprocess.Popen(
                    _build_cli_args(system_prompt, self.model, self.thinking_level, self.allowed_tools),
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
                                btype = block.get("type")
                                if btype == "thinking" and block.get("thinking"):
                                    accumulated.append(block["thinking"])
                                    loop.call_soon_threadsafe(
                                        queue.put_nowait, ("thinking", block["thinking"])
                                    )
                                elif btype == "text" and block.get("text"):
                                    accumulated.append(block["text"])
                                    loop.call_soon_threadsafe(
                                        queue.put_nowait, ("text", block["text"])
                                    )
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
                self._log_response("".join(accumulated))
                loop.call_soon_threadsafe(queue.put_nowait, None)

        ctx = contextvars.copy_context()
        threading.Thread(target=lambda: ctx.run(run), daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield ("text", f"[Claude CLI error: {item}]")
                break
            yield item


async def _run_claude(
    sys_path: str,
    msg_path: str,
    model: str = "",
    effort: str = "default",
    env: dict | None = None,
    allowed_tools: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run claude CLI in a thread (Windows asyncio SelectorEventLoop workaround).

    env が None の場合は _clean_env() をフォールバックとして使用する。
    """
    if env is None:
        env = _clean_env()

    with open(sys_path, encoding="utf-8") as f:
        system_content = f.read()
    with open(msg_path, encoding="utf-8") as f:
        msg_content = f.read()

    def run():
        return subprocess.run(
            _build_cli_args(system_content, model, effort, allowed_tools),
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
