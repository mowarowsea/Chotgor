"""Debug logger for recording I/O between frontend and LLM."""

import json
import os
from datetime import datetime
from typing import Any

DEBUG_DIR = "debug"


def is_debug_enabled() -> bool:
    return os.getenv("CHOTGOR_DEBUG") == "1"


def _write_log(prefix: str, content: str) -> None:
    if not is_debug_enabled():
        return
        
    os.makedirs(DEBUG_DIR, exist_ok=True)
    # Format: yyyyMMddhhmmssffff
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:18]
    filename = f"{timestamp}_{prefix}.log"
    filepath = os.path.join(DEBUG_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def log_front_input(data: Any) -> None:
    """フロント(OpenWebUI)からの入力リクエストを記録"""
    content = json.dumps(data, ensure_ascii=False, indent=2)
    _write_log("01_FrontInput", content)


def log_llm_request(system_prompt: str, messages: list[dict]) -> None:
    """LLMへの出力リクエスト(システムプロンプト+履歴)を記録"""
    data = {
        "system_prompt": system_prompt,
        "messages": messages
    }
    content = json.dumps(data, ensure_ascii=False, indent=2)
    _write_log("02_LLMRequest", content)


def log_llm_response(text: str) -> None:
    """LLMからの生の入力(レスポンス)を記録"""
    _write_log("03_LLMResponse", text)


def log_front_output(text: str) -> None:
    """フロントへ連携する最終的な出力(クリーンなテキスト)を記録"""
    _write_log("04_FrontOutput", text)
