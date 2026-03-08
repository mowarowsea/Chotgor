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


def log_front_output(text: str) -> None:
    """フロントへ連携する最終的な出力(クリーンなテキスト)を記録"""
    _write_log("04_FrontOutput", text)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def log_provider_request(provider: str, params: Any) -> None:
    """プロバイダーAPIへ実際に送るリクエスト全パラメータを記録"""
    content = json.dumps(params, ensure_ascii=False, indent=2, default=_json_default)
    _write_log(f"02_Request_{provider}", content)


def log_provider_response(provider: str, data: Any) -> None:
    """プロバイダーAPIからの生レスポンスを記録"""
    if isinstance(data, str):
        content = data
    else:
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    _write_log(f"03_Response_{provider}", content)
