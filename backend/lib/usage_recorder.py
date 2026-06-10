"""LLM 使用量レコーダー — プロバイダーのレスポンスからトークン使用量を記録する。

claude_cli / google プロバイダーがレスポンス解析後に record_usage() を呼び、
llm_usage_events テーブルへ1 API 呼び出し = 1行で追加する。
feature / target / request_id は log_context の ContextVar から自動補完するため、
呼び出し側はトークン数とモデル情報だけ渡せばよい。

記録失敗はチャット本流を絶対に妨げない（握り潰して WARNING ログのみ）。
"""

import logging

from backend.lib.log_context import (
    current_log_feature,
    current_log_target,
    current_message_id,
)

_log = logging.getLogger(__name__)

# SQLiteStore インスタンス（main.py 起動時に set_store() で注入される）
_store = None


def set_store(sqlite) -> None:
    """SQLiteStore を注入する。main.py の起動時に呼ぶ。"""
    global _store
    _store = sqlite


def record_usage(
    provider: str,
    model: str = "",
    preset_name: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    total_cost_usd: float | None = None,
) -> None:
    """LLM 使用量イベントを1行記録する。

    Args:
        provider: プロバイダーID（"claude_cli" / "google" 等）。
        model: 使用モデルID（レスポンス由来があればそちらを渡す）。
        preset_name: 使用プリセット名。
        input_tokens: 入力トークン数。
        output_tokens: 出力トークン数。
        cache_read_input_tokens: キャッシュ読込トークン数（claude_cli のみ）。
        cache_creation_input_tokens: キャッシュ作成トークン数（claude_cli のみ）。
        total_cost_usd: 概算コストUSD（claude_cli の result イベント由来）。
    """
    if _store is None:
        return
    # 全部ゼロ（usage が取れなかった）なら記録しない
    if not input_tokens and not output_tokens:
        return
    try:
        _store.add_llm_usage_event(
            provider=provider,
            model=model or None,
            preset_name=preset_name or None,
            target=current_log_target.get() or None,
            feature=current_log_feature.get() or None,
            request_id=current_message_id.get() or None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            total_cost_usd=total_cost_usd,
        )
    except Exception as e:
        _log.warning("LLM使用量の記録に失敗: provider=%s error=%s", provider, e)
