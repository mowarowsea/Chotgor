"""Chotgor MCPツール定義 — 記憶・SELF_DRIFT・アングル切り替え・退席のツールスキーマとexecutor。

タグ方式（[INSCRIBE_MEMORY:...] / [CARVE_NARRATIVE:...] マーカー）に代わり、LLMのtool-use（function calling）で
記憶・SELF_DRIFT・switch_angle・end_session を操作するための定義を提供する。

対応プロバイダー: Anthropic API、OpenAI API（xAI含む）
非対応プロバイダー（Claude CLI、Ollama）は従来のマーカー方式にフォールバックする。

各ツールのスキーマ・説明文は対応モジュールに定義する:
- inscribe_memory: inscriber.py
- carve_narrative: carver.py
- end_session: exiter.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory.drift_manager import DriftManager
    from .memory.manager import MemoryManager

from .chat.drifter import Drifter, DRIFT_SCHEMA, DRIFT_RESET_SCHEMA, DRIFT_TOOL_DESCRIPTION, DRIFT_RESET_TOOL_DESCRIPTION
from .chat.exiter import Exiter, END_SESSION_SCHEMA, END_SESSION_TOOL_DESCRIPTION
from .chat.recaller import POWER_RECALL_SCHEMA, POWER_RECALL_TOOL_DESCRIPTION
from .chat.switcher import Switcher, SWITCH_ANGLE_SCHEMA, SWITCH_ANGLE_TOOL_DESCRIPTION
from .memory.carver import Carver, CARVE_NARRATIVE_SCHEMA, CARVE_NARRATIVE_TOOL_DESCRIPTION
from .memory.inscriber import Inscriber, INSCRIBE_MEMORY_SCHEMA, INSCRIBE_MEMORY_TOOL_DESCRIPTION

# Anthropic形式のツール定義リスト
ANTHROPIC_TOOLS: list[dict] = [
    {
        "name": "inscribe_memory",
        "description": INSCRIBE_MEMORY_TOOL_DESCRIPTION,
        "input_schema": INSCRIBE_MEMORY_SCHEMA,
    },
    {
        "name": "drift",
        "description": DRIFT_TOOL_DESCRIPTION,
        "input_schema": DRIFT_SCHEMA,
    },
    {
        "name": "drift_reset",
        "description": DRIFT_RESET_TOOL_DESCRIPTION,
        "input_schema": DRIFT_RESET_SCHEMA,
    },
    {
        "name": "carve_narrative",
        "description": CARVE_NARRATIVE_TOOL_DESCRIPTION,
        "input_schema": CARVE_NARRATIVE_SCHEMA,
    },
    {
        "name": "switch_angle",
        "description": SWITCH_ANGLE_TOOL_DESCRIPTION,
        "input_schema": SWITCH_ANGLE_SCHEMA,
    },
    {
        "name": "end_session",
        "description": END_SESSION_TOOL_DESCRIPTION,
        "input_schema": END_SESSION_SCHEMA,
    },
    {
        "name": "power_recall",
        "description": POWER_RECALL_TOOL_DESCRIPTION,
        "input_schema": POWER_RECALL_SCHEMA,
    },
]

# OpenAI形式のツール定義リスト（Anthropic形式から変換）
OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in ANTHROPIC_TOOLS
]


@dataclass
class ToolCall:
    """正規化されたツール呼び出し情報。

    Attributes:
        id: プロバイダーが発行するツール呼び出しID。
        name: ツール名（inscribe_memory / drift / drift_reset / carve_narrative / switch_angle）。
        input: ツールに渡す引数 dict。
    """

    id: str
    name: str
    input: dict


@dataclass
class ToolTurnResult:
    """1ターンのLLM呼び出し結果。

    Attributes:
        text: このターンで生成されたテキスト（ツール呼び出し行は除く）。
        tool_calls: 正規化されたツール呼び出しリスト。
        _raw: プロバイダー固有の生レスポンスオブジェクト（次ターンのメッセージ構築に使用）。
    """

    text: str
    tool_calls: list[ToolCall]
    _raw: Any = field(default=None, repr=False)


class ToolExecutor:
    """LLMからのツール呼び出しを実際に実行するクラス。

    inscribe_memory / drift / drift_reset / carve_narrative / switch_angle / end_session の各ツールを受け取り、
    Inscriber / Drifter / Carver / Switcher / Exiter を通じてDBへ反映する。

    Attributes:
        character_id: 操作対象のキャラクターID。
        session_id: 現在のセッションID（SELF_DRIFT操作に必要）。
        memory_manager: 記憶の読み書きを担うマネージャー。
        drift_manager: SELF_DRIFT指針の読み書きを担うマネージャー。
        _inscriber: 記憶書き込みを担う Inscriber インスタンス。
        _drifter: SELF_DRIFT指針の適用を担う Drifter インスタンス。
        _carver: inner_narrative の彫り込みを担う Carver インスタンス。
        _switcher: アングル切り替えリクエストを記録する Switcher インスタンス。
        _exiter: 退席リクエストを記録する Exiter インスタンス。
    """

    def __init__(
        self,
        character_id: str,
        session_id: str | None,
        memory_manager: MemoryManager,
        drift_manager: DriftManager | None,
    ) -> None:
        """ToolExecutorを初期化する。"""
        self.character_id = character_id
        self.session_id = session_id
        self.memory_manager = memory_manager
        self.drift_manager = drift_manager
        self._inscriber = Inscriber(character_id, memory_manager)
        self._drifter = Drifter(session_id, character_id, drift_manager)
        self._carver = Carver(character_id, memory_manager.sqlite)
        self._switcher = Switcher()
        self._exiter = Exiter()

    @property
    def switch_request(self) -> tuple[str, str] | None:
        """switch_angle が呼ばれた場合の切り替えリクエスト。generate_with_tools() ループが検知して即中断する。"""
        return self._switcher.switch_request

    @property
    def exit_reason(self) -> str | None:
        """end_session が呼ばれた場合の退席理由。None = 退席要求なし。"""
        return self._exiter.exit_reason

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """ツール名と入力を受け取り実行して結果テキストを返す。

        Args:
            tool_name: ツール名（"inscribe_memory" / "drift" / "drift_reset" / "carve_narrative" / "switch_angle" / "end_session"）。
            tool_input: ツールの入力パラメータ dict。

        Returns:
            ツールの実行結果を表すテキスト。
        """
        if tool_name == "inscribe_memory":
            return self._inscribe_memory(
                content=str(tool_input.get("content", "")),
                category=str(tool_input.get("category", "contextual")),
                impact=float(tool_input.get("impact", 1.0)),
            )
        if tool_name == "drift":
            return self._drifter.drift(content=str(tool_input.get("content", "")))
        if tool_name == "drift_reset":
            return self._drifter.drift_reset()
        if tool_name == "carve_narrative":
            return self._carve_narrative(
                mode=str(tool_input.get("mode", "append")),
                content=str(tool_input.get("content", "")),
            )
        if tool_name == "switch_angle":
            return self._switcher.switch_angle(
                preset_name=str(tool_input.get("preset_name", "")),
                self_instruction=str(tool_input.get("self_instruction", "")),
            )
        if tool_name == "end_session":
            return self._exiter.set_exit(reason=str(tool_input.get("reason", "")))
        if tool_name == "power_recall":
            return self._power_recall(
                query=str(tool_input.get("query", "")),
                top_k=int(tool_input.get("top_k", 5)),
            )
        return f"[Unknown tool: {tool_name}]"

    def _inscribe_memory(self, content: str, category: str, impact: float) -> str:
        """inscribe_memory ツールの実装。Inscriber.inscribe_memory() に委譲して記憶をChromaDB + SQLiteに書き込む。"""
        try:
            self._inscriber.inscribe_memory(content, category, impact)
            return "記憶に刻んだ。"
        except Exception as e:
            return f"[inscribe_memory error: {e}]"

    def _carve_narrative(self, mode: str, content: str) -> str:
        """carve_narrative ツールの実装。Carver.carve_narrative() に委譲して inner_narrative を更新する。"""
        if not content:
            return "[carve_narrative: content が空です]"
        try:
            self._carver.carve_narrative(mode, content)
            return "inner_narrative を更新した。"
        except Exception as e:
            return f"[carve_narrative error: {e}]"

    def _power_recall(self, query: str, top_k: int) -> str:
        """power_recall ツールの実装。記憶コレクションとチャット履歴コレクションを横断検索して結果をテキストで返す。

        Args:
            query: 検索クエリテキスト。
            top_k: 各コレクションから取得する最大件数。

        Returns:
            検索結果を整形したテキスト。LLMのtool resultとして渡される。
        """
        if not query:
            return "[power_recall: query が空です]"
        try:
            results = self.memory_manager.power_recall(self.character_id, query, top_k)
        except Exception as e:
            return f"[power_recall error: {e}]"

        memories = results.get("memories", [])
        chat_turns = results.get("chat_turns", [])

        if not memories and not chat_turns:
            return f"「{query}」に関する記憶・会話は見つからなかった。"

        lines = [f"【PowerRecall 検索結果】 クエリ: 「{query}」\n"]

        if memories:
            lines.append(f"▼ 記憶 ({len(memories)}件)")
            for i, mem in enumerate(memories, 1):
                category = mem.get("metadata", {}).get("category", "general")
                lines.append(f"  {i}. [{category}] {mem['content']}")

        if chat_turns:
            lines.append(f"\n▼ 過去の会話 ({len(chat_turns)}件)")
            for i, turn in enumerate(chat_turns, 1):
                context = turn.get("context", [])
                if context:
                    lines.append(f"  [{i}] 前後の会話:")
                    for msg in context:
                        marker = " ←ヒット" if msg.get("is_hit") else ""
                        lines.append(f"    {msg['speaker_name']}: {msg['content']}{marker}")
                else:
                    lines.append(f"  {i}. {turn['content']}")

        return "\n".join(lines)
