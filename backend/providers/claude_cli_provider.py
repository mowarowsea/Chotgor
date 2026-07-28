"""Claude Code CLI プロバイダー。

`claude` CLI を asyncio.to_thread 経由のサブプロセスとして呼び出す（Windows 対応）。
"""

import asyncio
import contextvars
import json
import logging
import os
import shutil
import subprocess
import tempfile

from backend.lib.log_context import (
    current_log_dir_id,
    current_log_feature,
    current_log_target,
    current_message_id,
)
from backend.lib.stream_json import iter_stream_json_events
from backend.providers.base import BaseLLMProvider, safe_loop_call

logger = logging.getLogger(__name__)


def _find_claude() -> str:
    for name in ("claude", "claude.cmd", "claude.ps1"):
        path = shutil.which(name)
        if path:
            return path
    return "claude"


CLAUDE_BIN = _find_claude()


# 実行時 claude CLI を動かす作業ディレクトリ。
# Chotgor リポジトリの外（親ディレクトリ直下）の専用ディレクトリで CLI を起動することで、
# リポジトリの CLAUDE.md と自動生成 MEMORY.md を読み込ませず、呼び出しごとのトークン消費を
# 削減する（キャラクター応答の生成に CLAUDE.md / MEMORY.md は不要なため）。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_CLAUDE_CWD = os.path.join(os.path.dirname(_REPO_ROOT), "Chotgor_ClaudeCodeWorkDir")


def _ensure_claude_cwd() -> str | None:
    """実行時 claude CLI 用の作業ディレクトリを冪等に用意し、その絶対パスを返す。

    Chotgor リポジトリ外で CLI を起動すると、リポジトリの ``.claude/settings.json``
    （MCP ツール許可 permissions.allow）も読まれなくなる。非対話（``--print``）モードでは
    未許可ツールが拒否されてしまうため、リポジトリ側 settings.json をこの作業
    ディレクトリへ複製し、inscribe_memory などの MCP ツールが従来どおり許可される
    状態を保つ。リポジトリ側を source of truth として毎起動時に複製するため、
    許可の二重管理（追加し忘れ）が起きない。

    ディレクトリ準備に失敗した場合は None を返す。呼び出し側は cwd 指定なし
    （親プロセスのカレントディレクトリ継承）へフォールバックして従来どおり動作する。
    """
    try:
        claude_dir = os.path.join(_CLAUDE_CWD, ".claude")
        os.makedirs(claude_dir, exist_ok=True)
        repo_settings = os.path.join(_REPO_ROOT, ".claude", "settings.json")
        if os.path.isfile(repo_settings):
            shutil.copyfile(repo_settings, os.path.join(claude_dir, "settings.json"))
        return _CLAUDE_CWD
    except Exception:
        return None


# モジュールロード時に1回だけ作業ディレクトリを用意する（None ならフォールバック）。
_CLAUDE_CWD_READY = _ensure_claude_cwd()


def _build_tools_flag(allowed_tools: dict) -> str:
    """allowed_tools 設定から --tools フラグ文字列を組み立てる。

    現状、組み込みツール（WebSearch/WebFetch 等）は一切有効化しない方針のため
    常に空文字列を返す。外部情報の取得は Chotgor MCP の web_search ツール
    （Tavily 経由）に一本化されている。Google 系ツールは MCP 経由で
    --tools フラグでは制御されないため、ここでは何も加算しない。
    """
    return ""


def _build_cli_args(
    system_prompt: str,
    model: str = "",
    effort: str = "default",
    allowed_tools: dict | None = None,
    mcp_enabled: bool = True,
) -> list[str]:
    """claude CLI 呼び出しの共通フラグ列を組み立てる。

    model が空文字列の場合は --model フラグを付けない（CLIデフォルトを使用）。
    effort が "default" の場合は --effort フラグを付けない。
    allowed_tools が None または空の場合は --tools "" で全組み込みツールを無効化する。

    Args:
        mcp_enabled: False なら --strict-mcp-config を付け、Chotgor MCP サーバーを
            一切接続しない状態で CLI を起動する。chotgor MCP はユーザグローバルの
            ``~/.claude.json`` に登録されているため、何もしないと cwd に関わらず
            接続され、ツール無しのはずの問い合わせ（generate()）にもツール一覧が
            見えてしまう。その経路は character_id を env へ渡していないので、
            キャラクターがツールへ手を伸ばしても必ず
            ``[Error: CHOTGOR_CHARACTER_ID が設定されていません]`` になる
            （debug/2a46c2cb）。「使えない手を見せない」ために閉じる。
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
        "--thinking", "adaptive", # thinking blockが出ない問題対応。効かないかも。（ClaudeCodeCLI Issueに記載あり）
        "--thinking-display", "summarized", # thinking blockが出ない問題対応。効かないかも。（ClaudeCodeCLI Issueに記載あり）
        "--include-partial-messages", # 出力のストリーム化。効いてないっぽい。
    ]
    if not mcp_enabled:
        # --mcp-config を渡さずに --strict-mcp-config を付けると、参照する MCP 設定が
        # 「指定なし」に確定し、接続されるサーバーが 0 本になる。
        args.append("--strict-mcp-config")
    if model:
        args.extend(["--model", model])
    if effort and effort != "default":
        args.extend(["--effort", effort])
    return args


# MCP 接続レースガードの総起動回数（初回含む）。
# pending 起動は実測で約2割のため、3回で残存率は 1% 未満に落ちる。
_MCP_GUARD_MAX_ATTEMPTS = 3


def _mcp_pending_in_init(event: dict) -> bool:
    """stream-json の init イベントが「MCP サーバー未接続」を示しているか判定する。

    未接続のままリクエストが走ると、そのターンはツールが一切提供されず、
    キャラクターが「ツール使用の演技」（擬似構文のテキスト出力）に流れる
    事故が起きる（debug/cfd5bf43）。init 以外のイベントには常に False。

    --strict-mcp-config で MCP を意図的に閉じた起動（mcp_enabled=False）では
    mcp_servers が空リストになるため、判定は自然に False になりリトライも走らない。
    """
    if event.get("type") != "system" or event.get("subtype") != "init":
        return False
    servers = event.get("mcp_servers") or []
    return any(s.get("status") != "connected" for s in servers)


def _spawn_cli_mcp_guarded(
    args: list[str], stdin_bytes: bytes, env: dict
) -> tuple[subprocess.Popen, list[str]]:
    """CLI を起動し、init イベントが MCP 未接続を示したらプロセスを作り直す。

    stream-json の最初の1行（通常 init イベント）だけを先読みして判定する。
    最終試行では未接続でもそのまま続行する（ターン全損よりツール無しの方がまし）。

    Returns:
        (プロセス, 先読み済み行リスト)。先読み行には破棄した試行の init 行も
        含める（debug ログで再試行の痕跡を追えるようにするため）。呼び出し側は
        先読み行を処理してから残りの stdout を読み進めること（`_cli_output_lines`）。
    """
    consumed: list[str] = []
    proc: subprocess.Popen | None = None
    for attempt in range(1, _MCP_GUARD_MAX_ATTEMPTS + 1):
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=_CLAUDE_CWD_READY,
        )
        proc.stdin.write(stdin_bytes)
        proc.stdin.close()

        first = proc.stdout.readline().decode("utf-8", errors="replace").strip()
        if first:
            consumed.append(first)
        pending = False
        try:
            pending = _mcp_pending_in_init(json.loads(first)) if first else False
        except json.JSONDecodeError:
            pass

        if not pending:
            return proc, consumed
        if attempt < _MCP_GUARD_MAX_ATTEMPTS:
            logger.warning(
                "MCP サーバー未接続を init で検知 (attempt %d/%d) — CLI を再起動します",
                attempt, _MCP_GUARD_MAX_ATTEMPTS,
            )
            proc.kill()
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()
        else:
            logger.warning(
                "MCP サーバー未接続のまま続行します（リトライ上限 %d 回到達）。"
                "このターンはツール無しになる可能性があります", _MCP_GUARD_MAX_ATTEMPTS,
            )
    return proc, consumed


def _cli_output_lines(proc: subprocess.Popen, pre_lines: list[str]):
    """先読み済み行→残り stdout の順で、空行を除いた NDJSON 行を逐次 yield する。"""
    yield from pre_lines
    for line_bytes in iter(proc.stdout.readline, b""):
        line = line_bytes.decode("utf-8", errors="replace").strip()
        if line:
            yield line


# Claude サブプロセス起動前に除去すべき環境変数。
# CLAUDECODE          : ネストされたセッションエラーを防ぐ。
# ANTHROPIC_API_KEY   : 無効なキーによるOAuth迂回を防ぎ、OAuth認証を優先させる。
_CLAUDE_ENV_EXCLUDES = {"CLAUDECODE", "ANTHROPIC_API_KEY"}


def _clean_env() -> dict:
    """グローバル環境変数からClaudeネスト検出・誤認証キーを除き、コスト削減フラグを注入した dict を返す。"""
    env = {k: v for k, v in os.environ.items() if k not in _CLAUDE_ENV_EXCLUDES}
    # プロンプトキャッシュは有効のまま使う（docs/planned/prompt_cache_plan.md §7.0）。
    # サブスク（OAuth）運用ではキャッシュ読出がレート枠の節約に直結するため、
    # 旧実装にあった DISABLE_PROMPT_CACHING=1 の注入は撤去した。
    # 自動メモリ（~/.claude/projects/.../memory/MEMORY.md）の読み込みを無効化し、
    # システムプロンプトへの不要な注入を防いでトークン消費を削減する。
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
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


def _extract_usage_from_stream_json(raw: str) -> dict | None:
    """Claude CLI stream-json 出力からトークン使用量を抽出する。

    type == "result" イベントの usage（実行全体の合計）と total_cost_usd、
    および assistant イベントの message.model（実際に使われたモデルID）を集める。
    result イベントが存在しない（途中エラー等）場合は None を返す。

    Returns:
        {"model", "input_tokens", "output_tokens", "cache_read_input_tokens",
         "cache_creation_input_tokens", "total_cost_usd"} の辞書、または None。
    """
    model = ""
    usage: dict | None = None
    cost: float | None = None
    # CLI の生 stdout は1行1イベントだが、複数行またがり・非 JSON 行混在にも
    # 対応できるよう iter_stream_json_events（backend/lib/stream_json.py）で走査する。
    for event in iter_stream_json_events(raw):
        etype = event.get("type")
        if etype == "assistant" and not model:
            model = (event.get("message") or {}).get("model", "") or ""
        elif etype == "result":
            u = event.get("usage")
            if isinstance(u, dict):
                usage = u
            c = event.get("total_cost_usd")
            if isinstance(c, (int, float)):
                cost = float(c)
    if usage is None:
        return None
    return {
        "model": model,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
        "total_cost_usd": cost,
    }


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

    def _record_usage_from_raw(self, raw: str) -> None:
        """raw stream-json から使用量を抽出して llm_usage_events へ記録する。

        result イベントが無い（エラー中断等）場合は何もしない。
        記録失敗はチャット本流に影響しない（usage_recorder 側で握り潰す）。
        """
        from backend.lib.usage_recorder import record_usage

        usage = _extract_usage_from_stream_json(raw)
        if usage is None:
            return
        record_usage(
            provider=self.PROVIDER_ID,
            model=usage["model"] or self.model,
            preset_name=self.preset_name,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_input_tokens=usage["cache_read_input_tokens"],
            cache_creation_input_tokens=usage["cache_creation_input_tokens"],
            total_cost_usd=usage["total_cost_usd"],
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
                付与する origin（"real" / "usual" / "interlude" の3値）。"real" 以外なら
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
        # ログ文脈（request_id 等）の env→HTTP リレー:
        # Claude CLI が叩く MCP サーバー（mcp_server.py）は別プロセスのため、
        # backend 側 ContextVar（current_message_id 等）は届かない。ツール実行イベント
        # （tool_call_events）を元リクエストに紐付けるため、ここで env に積み、
        # mcp_server.py → /api/mcp/tools/call → ContextVar 復元の経路で伝搬する
        # （batch_context / default_origin と同じリレー思想）。
        env["CHOTGOR_LOG_CONTEXT"] = json.dumps(
            {
                "request_id": current_message_id.get(),
                "dir_id": current_log_dir_id.get(),
                "feature": current_log_feature.get(),
                "target": current_log_target.get() or "",
            },
            ensure_ascii=False,
        )
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
        return text, thinking

    async def _run_generate_raw(
        self,
        system_prompt: str,
        messages: list[dict],
        batch_context: dict | None = None,
        default_origin: str = "real",
        mcp_enabled: bool = True,
    ) -> str:
        """Claude CLI を呼び出して raw stdout 文字列を返す内部メソッド。

        エラー時はエラーメッセージ文字列を返す（例外は送出しない）。

        Args:
            batch_context: ToolExecutor の挙動切り替えフラグ。指定すると CHOTGOR_BATCH_CONTEXT
                環境変数として MCP サーバーへ渡る（generate_with_tools 経由のみ使用）。
            default_origin: 記憶/スレッド保存時の origin ラベル（"real" / "usual" / "interlude"）。
                "real" 以外なら CHOTGOR_DEFAULT_ORIGIN として MCP サーバーへ伝搬される。
            mcp_enabled: False なら Chotgor MCP を接続せずに起動する（_build_cli_args 参照）。
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
            result = await _run_claude(sys_file.name, msg_file.name, model=self.model, effort=self.thinking_level, env=self._make_env(batch_context=batch_context, default_origin=default_origin), allowed_tools=self.allowed_tools, mcp_enabled=mcp_enabled)

            if result.returncode != 0:
                err_msg = result.stderr.decode("utf-8", errors="replace")
                out_msg = result.stdout.decode("utf-8", errors="replace")
                # generate_with_tools が `raw.startswith("[Error:")` で LLMApiError へ
                # 変換するため、プレフィクスを必ず `[Error:` に揃える。
                err = (
                    f"[Error: Claude Code exited with code {result.returncode}\n"
                    f"STDERR: {err_msg[:1000]}\nSTDOUT: {out_msg[:2000]}]"
                )
                self._log_error(err)
                return err

            raw_stdout = result.stdout.decode("utf-8", errors="replace")
            self._log_response(raw_stdout)
            self._record_usage_from_raw(raw_stdout)
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
        """Claude CLI を呼び出してテキスト応答を返す（ツール無しの単発問い合わせ）。

        generate() は「ツールを使わせない問い合わせ」の契約であり、呼び出し側
        （ask_character・ambience judge・計器判定・GM 予定生成・翻訳）はいずれも
        応答テキストのパース結果だけを使う。それらの経路は provider へ character_id を
        渡さないため MCP ツールを呼んでも必ず失敗する。使えない手を見せて空振らせない
        よう、MCP を閉じて起動する（mcp_enabled=False）。
        ツールが要る問い合わせは ask_character_with_tools → generate_with_tools を使う。
        """
        raw = await self._run_generate_raw(system_prompt, messages, mcp_enabled=False)
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
                proc, pre_lines = _spawn_cli_mcp_guarded(
                    _build_cli_args(system_prompt, self.model, self.thinking_level, self.allowed_tools),
                    conversation.encode("utf-8"),
                    env,
                )

                for line in _cli_output_lines(proc, pre_lines):
                    raw_lines.append(line)
                    try:
                        event = json.loads(line)
                        if event.get("type") == "assistant":
                            for block in event.get("message", {}).get("content", []):
                                if block.get("type") == "text" and block["text"]:
                                    safe_loop_call(loop, queue.put_nowait, block["text"])
                    except json.JSONDecodeError:
                        pass

                proc.wait()
                if proc.returncode != 0:
                    err = proc.stderr.read().decode("utf-8", errors="replace")
                    safe_loop_call(
                        loop, queue.put_nowait,
                        RuntimeError(f"Claude CLI exited with code {proc.returncode}: {err[:500]}")
                    )
            except FileNotFoundError:
                safe_loop_call(
                    loop, queue.put_nowait,
                    RuntimeError("claude CLI が見つかりません。Claude Code がインストール・認証済みか確認してください")
                )
            except Exception as e:
                safe_loop_call(loop, queue.put_nowait, RuntimeError(str(e)))
            finally:
                # ツール呼び出しを含む全NDJSONイベント行をログに記録する。
                # テキストのみだとtool_useブロックが欠落してログUIにバッジが表示されない。
                raw_joined = "\n".join(raw_lines)
                self._log_response(raw_joined)
                # contextvars は ctx.run でコピー済みのため、このスレッドからの記録でも
                # feature / target / request_id が正しく付く。
                self._record_usage_from_raw(raw_joined)
                safe_loop_call(loop, queue.put_nowait, None)

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
        プロセス起動失敗・非ゼロ終了などのエラーは ("error", msg) として yield する。
        呼び出し側は ("error", ...) を出力に積まず、状況に応じて UI 表示・蒸留スキップ
        などの分岐を行う。

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
                proc, pre_lines = _spawn_cli_mcp_guarded(
                    _build_cli_args(system_prompt, self.model, self.thinking_level, self.allowed_tools),
                    conversation.encode("utf-8"),
                    env,
                )

                for line in _cli_output_lines(proc, pre_lines):
                    raw_lines.append(line)
                    try:
                        event = json.loads(line)
                        if event.get("type") == "assistant":
                            for block in event.get("message", {}).get("content", []):
                                btype = block.get("type")
                                if btype == "thinking" and block.get("thinking"):
                                    safe_loop_call(
                                        loop, queue.put_nowait, ("thinking", block["thinking"])
                                    )
                                elif btype == "text" and block.get("text"):
                                    safe_loop_call(
                                        loop, queue.put_nowait, ("text", block["text"])
                                    )
                    except json.JSONDecodeError:
                        pass

                proc.wait()
                if proc.returncode != 0:
                    err = proc.stderr.read().decode("utf-8", errors="replace")
                    safe_loop_call(
                        loop, queue.put_nowait,
                        RuntimeError(f"Claude CLI exited with code {proc.returncode}: {err[:500]}")
                    )
            except FileNotFoundError:
                safe_loop_call(
                    loop, queue.put_nowait,
                    RuntimeError("claude CLI が見つかりません。Claude Code がインストール・認証済みか確認してください")
                )
            except Exception as e:
                safe_loop_call(loop, queue.put_nowait, RuntimeError(str(e)))
            finally:
                # ツール呼び出しを含む全NDJSONイベント行をログに記録する。
                # テキストのみだとtool_useブロックが欠落してログUIにバッジが表示されない。
                raw_joined = "\n".join(raw_lines)
                self._log_response(raw_joined)
                # contextvars は ctx.run でコピー済みのため、このスレッドからの記録でも
                # feature / target / request_id が正しく付く。
                self._record_usage_from_raw(raw_joined)
                safe_loop_call(loop, queue.put_nowait, None)

        ctx = contextvars.copy_context()
        threading.Thread(target=lambda: ctx.run(run), daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                # CLI 非ゼロ終了・claude 不在・想定外例外を「エラー」型 chunk として通知する。
                # synopsis 蒸留や履歴蓄積に混入させず、呼び出し側が個別に表示・記録する。
                yield ("error", f"[Claude CLI error: {item}]")
                break
            yield item


async def _run_claude(
    sys_path: str,
    msg_path: str,
    model: str = "",
    effort: str = "default",
    env: dict | None = None,
    allowed_tools: dict | None = None,
    mcp_enabled: bool = True,
) -> subprocess.CompletedProcess:
    """Run claude CLI in a thread (Windows asyncio SelectorEventLoop workaround).

    env が None の場合は _clean_env() をフォールバックとして使用する。
    mcp_enabled=False なら Chotgor MCP を接続せずに起動する（_build_cli_args 参照）。
    """
    if env is None:
        env = _clean_env()

    with open(sys_path, encoding="utf-8") as f:
        system_content = f.read()
    with open(msg_path, encoding="utf-8") as f:
        msg_content = f.read()

    def run():
        import threading

        proc, pre_lines = _spawn_cli_mcp_guarded(
            _build_cli_args(system_content, model, effort, allowed_tools, mcp_enabled),
            msg_content.encode("utf-8"),
            env,
        )
        # stderr は別スレッドで並行に読み切る（stdout 全読み中のパイプ詰まり防止）
        stderr_chunks: list[bytes] = []
        drainer = threading.Thread(
            target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True
        )
        drainer.start()
        rest = proc.stdout.read().decode("utf-8", errors="replace")
        proc.wait()
        drainer.join()
        stdout_text = "\n".join(pre_lines + ([rest] if rest else []))
        return subprocess.CompletedProcess(
            proc.args,
            proc.returncode,
            stdout_text.encode("utf-8"),
            b"".join(stderr_chunks),
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
