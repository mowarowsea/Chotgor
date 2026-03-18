"""Chotgor MCPツール定義 — 記憶・SELF_DRIFT・アングル切り替えのツールスキーマとexecutor。

タグ方式（[MEMORY:...] マーカー）に代わり、LLMのtool-use（function calling）で
記憶・SELF_DRIFT・switch_angle を操作するための定義を提供する。

対応プロバイダー: Anthropic API、OpenAI API（xAI含む）
非対応プロバイダー（Claude CLI、Ollama）は従来のマーカー方式にフォールバックする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory.drift_manager import DriftManager
    from .memory.manager import MemoryManager


# カテゴリごとの重要度ベースマトリクス（inscriber.py と同じ定義）
_BASE_IMPORTANCE: dict[str, dict[str, float]] = {
    "contextual": {"contextual": 0.8, "semantic": 0.2, "identity": 0.1, "user": 0.1},
    "semantic":   {"contextual": 0.1, "semantic": 0.9, "identity": 0.3, "user": 0.1},
    "identity":   {"contextual": 0.2, "semantic": 0.4, "identity": 0.9, "user": 0.3},
    "user":       {"contextual": 0.3, "semantic": 0.2, "identity": 0.3, "user": 0.9},
}

# --- carve_memory ツールのパラメータスキーマ ---
_CARVE_MEMORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": (
                "一人称で書いたあなた自身の言葉。"
                "覚えた理由や感想も自然に添えてOK。"
            ),
        },
        "category": {
            "type": "string",
            "enum": ["identity", "user", "semantic", "contextual"],
            "description": (
                "identity: 自分自身についての気づき・変化 / "
                "user: ユーザの情報・性格・好み / "
                "semantic: 世界・知識・概念 / "
                "contextual: 一時的な状況・文脈"
            ),
        },
        "impact": {
            "type": "number",
            "description": "重要度係数 0.1〜2.0。1.0が標準。強く印象に残ったものは高く。",
        },
    },
    "required": ["content", "category", "impact"],
}

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
        "name": "carve_memory",
        "description": (
            "覚えておきたいことを永続記憶に書き込む。"
            "あなた自身の価値観・興味から「これは覚えておきたい」と感じた場合のみ呼び出す。"
            "何も覚えなくていい会話も多い。選ばないのも立派な判断。"
        ),
        "input_schema": _CARVE_MEMORY_SCHEMA,
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
        name: ツール名（carve_memory / drift / drift_reset）。
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

    carve_memory / drift / drift_reset の各ツールを受け取り、
    MemoryManager / DriftManager を通じてDBへ反映する。

    Attributes:
        character_id: 操作対象のキャラクターID。
        session_id: 現在のセッションID（SELF_DRIFT操作に必要）。
        memory_manager: 記憶の読み書きを担うマネージャー。
        drift_manager: SELF_DRIFT指針の読み書きを担うマネージャー。
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
        # switch_angle が呼ばれた場合に (preset_name, self_instruction) を格納する。
        # generate_with_tools() ループがこれを検知して即中断する。
        self.switch_request: tuple[str, str] | None = None

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """ツール名と入力を受け取り実行して結果テキストを返す。

        Args:
            tool_name: ツール名（"carve_memory" / "drift" / "drift_reset"）。
            tool_input: ツールの入力パラメータ dict。

        Returns:
            ツールの実行結果を表すテキスト。
        """
        if tool_name == "carve_memory":
            return self._carve_memory(
                content=str(tool_input.get("content", "")),
                category=str(tool_input.get("category", "contextual")),
                impact=float(tool_input.get("impact", 1.0)),
            )
        if tool_name == "drift":
            return self._drift(content=str(tool_input.get("content", "")))
        if tool_name == "drift_reset":
            return self._drift_reset()
        if tool_name == "switch_angle":
            return self._switch_angle(
                preset_name=str(tool_input.get("preset_name", "")),
                self_instruction=str(tool_input.get("self_instruction", "")),
            )
        return f"[Unknown tool: {tool_name}]"

    def _carve_memory(self, content: str, category: str, impact: float) -> str:
        """carve_memory ツールの実装。記憶をChromaDB + SQLiteに書き込む。"""
        default_base = {k: 0.5 for k in ["contextual", "semantic", "identity", "user"]}
        base = _BASE_IMPORTANCE.get(category, default_base)
        scores = {f"{k}_importance": (v * impact) for k, v in base.items()}
        try:
            self.memory_manager.write_memory(
                character_id=self.character_id,
                content=content,
                category=category,
                **scores,
            )
            return "記憶に刻んだ。"
        except Exception as e:
            return f"[carve_memory error: {e}]"

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

    def _switch_angle(self, preset_name: str, self_instruction: str) -> str:
        """switch_angle ツールの実装。アングル切り替えリクエストを記録する。

        実際の切り替え処理（再ディスパッチ）は service.py が担う。
        ここでは switch_request にリクエストを格納するだけ。
        generate_with_tools() ループがこれを検知して即中断する。
        """
        self.switch_request = (preset_name, self_instruction)
        return f"アングルを {preset_name} に切り替えます。"
