"""Claude Code CLI プロバイダー。

`claude` CLI を asyncio.to_thread 経由のサブプロセスとして呼び出す（Windows 対応）。

パブリックヘルパー
------------------
invoke_claude_cli(system_prompt, input_text) -> str
    digest.py / forget.py が共用するシングルターン CLI 呼び出し。
    終了コードが非ゼロの場合は RuntimeError を送出する。
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

    現状、組み込みツール（WebSearch/WebFetch 等）は一切有効化しない方針のため
    常に空文字列を返す。外部情報の取得は Chotgor MCP の web_search ツール
    （Tavily 経由）に一本化されている。Google 系ツールは MCP 経由で
    --tools フラグでは制御されないため、ここでは何も加算しない。
    """
    return ""


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

# Claude サブプロセス起動前に除去すべき環境変数。
# CLAUDECODE          : ネストされたセッションエラーを防ぐ。
# ANTHROPIC_API_KEY   : 無効なキーによるOAuth迂回を防ぎ、OAuth認証を優先させる。
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


def _extract_switch_angle_from_stream_json(raw: str) -> tuple[str, str] | None:
    """Claude CLI stream-json 出力から switch_angle ツール呼び出しを抽出する。

    Claude CLI は MCP サーバー（別プロセス）経由で switch_angle を実行するため、
    ChatService 側の tool_executor には反映されない。raw stream-json に含まれる
    mcp__chotgor__switch_angle の tool_use ブロックを解析して、(preset_name, self_instruction)
    として返すことで、呼び出し側が tool_executor.switch_request に反映できるようにする。

    複数回呼ばれている場合は最初の1件のみ返す。

    Returns:
        (preset_name, self_instruction) のタプル。見つからなければ None。
    """
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []):
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            # MCP 経由のツール名は "mcp__{server}__{tool}" 形式
            if not name.endswith("__switch_angle"):
                continue
            tool_input = block.get("input", {}) or {}
            preset_name = str(tool_input.get("preset_name", ""))
            self_instruction = str(tool_input.get("self_instruction", ""))
            if preset_name:
                return preset_name, self_instruction
    return None


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

    def _make_env(
        self,
        batch_context: dict | None = None,
        default_origin: str = "real",
    ) -> dict:
        """Claude CLI サブプロセス用の環境変数 dict を返す。

        _clean_env() をベースに、MCP サーバーが必要とするキャラクターコンテキストを追加する。

        Args:
            batch_context: バッチ処理が ToolExecutor の挙動を切り替えるためのフラグ群。
                指定された場合は ``CHOTGOR_BATCH_CONTEXT`` 環境変数として JSON 文字列で渡し、
                MCP サーバー（別プロセス）が backend への HTTP リクエストに乗せて伝搬する。
                None / 空 dict なら設定しない（通常チャットと同じ挙動）。
            default_origin: inscribe_memory / post_working_memory_thread の保存時に
                付与する origin（"real" / "interlude"）。"real" 以外なら
                ``CHOTGOR_DEFAULT_ORIGIN`` env として MCP サーバーへ渡す。
                in-process プロバイダーと挙動差を出さないための経路。
        """
        env = _clean_env()
        if self.character_id:
            env["CHOTGOR_CHARACTER_ID"] = self.character_id
        if self.session_id:
            env["CHOTGOR_SESSION_ID"] = self.session_id
        if batch_context:
            env["CHOTGOR_BATCH_CONTEXT"] = json.dumps(batch_context, ensure_ascii=False)
        if default_origin and default_origin != "real":
            env["CHOTGOR_DEFAULT_ORIGIN"] = default_origin
        return env

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tool_executor,
    ) -> tuple[str, str]:
        """Claude CLI 内部の MCP ループに委譲する。

        tool_executor（Python 側）の execute は MCP サーバー経由で実行されるため、
        ここでは記憶系ツール（inscribe_memory 等）の結果反映には使われない。
        ただし switch_angle のみ例外で、ChatService が tool_executor.switch_request を
        読んで再ディスパッチを判定するため、raw stream-json から抽出して反映させる。

        ``tool_executor.batch_context``（例: ``{"force_insert_memory": True}``）は
        env 経由で MCP サーバーへ渡し、HTTP ペイロードに乗せて backend 側 ToolExecutor へ
        再構築する。in-process プロバイダー（Ollama 等）との挙動差を埋めるための経路。

        Returns:
            (text, thinking): thinking は思考ブロックが存在する場合その内容、なければ空文字列。
        """
        from backend.providers.base import LLMApiError

        batch_context = getattr(tool_executor, "batch_context", None) if tool_executor else None
        default_origin = getattr(tool_executor, "default_origin", "real") if tool_executor else "real"
        raw = await self._run_generate_raw(
            system_prompt, messages,
            batch_context=batch_context,
            default_origin=default_origin,
        )
        # _run_generate_raw はエラーを例外ではなく "[Error: ...]" 文字列で返す設計。
        # バッチ処理（ask_character_with_tools 等）が正しくフォールバックできるよう
        # ここで LLMApiError に変換して送出する。
        if raw.startswith("[Error:"):
            raise LLMApiError(raw)
        text = _parse_stream_json(raw)
        thinking = _extract_thinking_from_stream_json(raw)
        # MCP 経由で実行された switch_angle を tool_executor へ転写する。
        # ChatService._extract_switch_info が tool_executor.switch_request を見て
        # 再ディスパッチの可否を判定するため、ここで反映しないと switch_angle が無視される。
        switch_info = _extract_switch_angle_from_stream_json(raw)
        if switch_info is not None and tool_executor is not None:
            preset_name, self_instruction = switch_info
            tool_executor.execute(
                "switch_angle",
                {"preset_name": preset_name, "self_instruction": self_instruction},
            )
        return text, thinking

    async def _run_generate_raw(
        self,
        system_prompt: str,
        messages: list[dict],
        batch_context: dict | None = None,
        default_origin: str = "real",
    ) -> str:
        """Claude CLI を呼び出して raw stdout 文字列を返す内部メソッド。

        エラー時はエラーメッセージ文字列を返す（例外は送出しない）。

        Args:
            batch_context: ToolExecutor の挙動切り替えフラグ。指定すると CHOTGOR_BATCH_CONTEXT
                環境変数として MCP サーバーへ渡る（generate_with_tools 経由のみ使用）。
            default_origin: 記憶/スレッド保存時の origin ラベル。"real" 以外なら
                CHOTGOR_DEFAULT_ORIGIN として MCP サーバーへ伝搬される。
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
            result = await _run_claude(sys_file.name, msg_file.name, model=self.model, effort=self.thinking_level, env=self._make_env(batch_context=batch_context, default_origin=default_origin), allowed_tools=self.allowed_tools)

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
            raw_lines: list[str] = []  # 全NDJSONイベント行（ツール呼び出し含む、ログ用）
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
                    raw_lines.append(line)
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
                # ツール呼び出しを含む全NDJSONイベント行をログに記録する。
                # テキストのみだとtool_useブロックが欠落してログUIにバッジが表示されない。
                self._log_response("\n".join(raw_lines))
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
            raw_lines: list[str] = []  # 全NDJSONイベント行（ツール呼び出し含む、ログ用）
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
                    raw_lines.append(line)
                    try:
                        event = json.loads(line)
                        if event.get("type") == "assistant":
                            for block in event.get("message", {}).get("content", []):
                                btype = block.get("type")
                                if btype == "thinking" and block.get("thinking"):
                                    loop.call_soon_threadsafe(
                                        queue.put_nowait, ("thinking", block["thinking"])
                                    )
                                elif btype == "text" and block.get("text"):
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
                # ツール呼び出しを含む全NDJSONイベント行をログに記録する。
                # テキストのみだとtool_useブロックが欠落してログUIにバッジが表示されない。
                self._log_response("\n".join(raw_lines))
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
