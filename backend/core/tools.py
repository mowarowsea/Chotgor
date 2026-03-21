"""Chotgor MCPツール定義 — 記憶・SELF_DRIFT・アングル切り替えのツールスキーマとexecutor。

タグ方式（[INSCRIBE_MEMORY:...] / [CARVE_NARRATIVE:...] マーカー）に代わり、LLMのtool-use（function calling）で
記憶・SELF_DRIFT・switch_angle を操作するための定義を提供する。

対応プロバイダー: Anthropic API、OpenAI API（xAI含む）
非対応プロバイダー（Claude CLI、Ollama）は従来のマーカー方式にフォールバックする。

各ツールのスキーマ・説明文は対応モジュールに定義する:
- inscribe_memory: inscriber.py
- carve_narrative: carver.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory.drift_manager import DriftManager
    from .memory.manager import MemoryManager

from .memory.carver import Carver, CARVE_NARRATIVE_SCHEMA, CARVE_NARRATIVE_TOOL_DESCRIPTION
from .memory.inscriber import Inscriber, INSCRIBE_MEMORY_SCHEMA, INSCRIBE_MEMORY_TOOL_DESCRIPTION

# --- drift ツールのパラメータスキーマ ---

_DRIFT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "このチャット内でのみ有効な一時的な行動指針のテキスト。",
        },
    },
    "required": ["content"],
}

# --- drift_reset ツールのパラメータスキーマ ---
_DRIFT_RESET_SCHEMA: dict = {
    "type": "object",
    "properties": {},
    "required": [],
}

# --- switch_angle ツールのパラメータスキーマ ---
_SWITCH_ANGLE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "preset_name": {
            "type": "string",
            "description": "切り替え先のプリセット名。利用可能なプリセットはシステムプロンプトに記載されている。",
        },
        "self_instruction": {
            "type": "string",
            "description": "切り替え後のプリセットに渡す自己指針テキスト。どのように応答するかを一言で。",
        },
    },
    "required": ["preset_name", "self_instruction"],
}

# Anthropic形式のツール定義リスト
ANTHROPIC_TOOLS: list[dict] = [
    {
        "name": "inscribe_memory",
        "description": INSCRIBE_MEMORY_TOOL_DESCRIPTION,
        "input_schema": INSCRIBE_MEMORY_SCHEMA,
    },
    {
        "name": "drift",
        "description": (
            "このチャット内でのみ有効な一時的な行動指針を設定する。最大3件まで保持。"
        ),
        "input_schema": _DRIFT_SCHEMA,
    },
    {
        "name": "drift_reset",
        "description": "現在有効な全SELF_DRIFT指針をリセットする。",
        "input_schema": _DRIFT_RESET_SCHEMA,
    },
    {
        "name": "carve_narrative",
        "description": CARVE_NARRATIVE_TOOL_DESCRIPTION,
        "input_schema": CARVE_NARRATIVE_SCHEMA,
    },
    {
        "name": "switch_angle",
        "description": (
            "プリセット（モデル）を切り替える。キャラクターは変わらない。"
            "切り替え後のモデルがこの会話に改めて応答する。"
            "利用可能なプリセットと切り替えタイミングはシステムプロンプトに記載されている。"
        ),
        "input_schema": _SWITCH_ANGLE_SCHEMA,
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

    inscribe_memory / drift / drift_reset / carve_narrative / switch_angle の各ツールを受け取り、
    Inscriber / DriftManager を通じてDBへ反映する。

    Attributes:
        character_id: 操作対象のキャラクターID。
        session_id: 現在のセッションID（SELF_DRIFT操作に必要）。
        memory_manager: 記憶の読み書きを担うマネージャー。
        drift_manager: SELF_DRIFT指針の読み書きを担うマネージャー。
        _inscriber: 記憶書き込みを担う Inscriber インスタンス。
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
        self._carver = Carver(character_id, memory_manager.sqlite)
        # switch_angle が呼ばれた場合に (preset_name, self_instruction) を格納する。
        # generate_with_tools() ループがこれを検知して即中断する。
        self.switch_request: tuple[str, str] | None = None

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """ツール名と入力を受け取り実行して結果テキストを返す。

        Args:
            tool_name: ツール名（"inscribe_memory" / "drift" / "drift_reset" / "carve_narrative" / "switch_angle"）。
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
            return self._drift(content=str(tool_input.get("content", "")))
        if tool_name == "drift_reset":
            return self._drift_reset()
        if tool_name == "carve_narrative":
            return self._carve_narrative(
                mode=str(tool_input.get("mode", "append")),
                content=str(tool_input.get("content", "")),
            )
        if tool_name == "switch_angle":
            return self._switch_angle(
                preset_name=str(tool_input.get("preset_name", "")),
                self_instruction=str(tool_input.get("self_instruction", "")),
            )
        return f"[Unknown tool: {tool_name}]"

    def _inscribe_memory(self, content: str, category: str, impact: float) -> str:
        """inscribe_memory ツールの実装。Inscriber.inscribe_memory() に委譲して記憶をChromaDB + SQLiteに書き込む。"""
        try:
            self._inscriber.inscribe_memory(content, category, impact)
            return "記憶に刻んだ。"
        except Exception as e:
            return f"[inscribe_memory error: {e}]"

    def _drift(self, content: str) -> str:
        """drift ツールの実装。セッション内一時指針をDBに追加する。"""
        if not self.drift_manager or not self.session_id:
            return "SELF_DRIFT は利用できない。"
        try:
            self.drift_manager.add_drift(self.session_id, self.character_id, content)
            return "指針を設定した。"
        except Exception as e:
            return f"[drift error: {e}]"

    def _drift_reset(self) -> str:
        """drift_reset ツールの実装。セッション内の全SELF_DRIFT指針をリセットする。"""
        if not self.drift_manager or not self.session_id:
            return "SELF_DRIFT は利用できない。"
        try:
            self.drift_manager.reset_drifts(self.session_id, self.character_id)
            return "指針をリセットした。"
        except Exception as e:
            return f"[drift_reset error: {e}]"

    def _carve_narrative(self, mode: str, content: str) -> str:
        """carve_narrative ツールの実装。Carver.carve_narrative() に委譲して inner_narrative を更新する。"""
        if not content:
            return "[carve_narrative: content が空です]"
        try:
            self._carver.carve_narrative(mode, content)
            return "inner_narrative を更新した。"
        except Exception as e:
            return f"[carve_narrative error: {e}]"

    def _switch_angle(self, preset_name: str, self_instruction: str) -> str:
        """switch_angle ツールの実装。アングル切り替えリクエストを記録する。

        実際の切り替え処理（再ディスパッチ）は service.py が担う。
        ここでは switch_request にリクエストを格納するだけ。
        generate_with_tools() ループがこれを検知して即中断する。
        """
        self.switch_request = (preset_name, self_instruction)
        return f"アングルを {preset_name} に切り替えます。"
