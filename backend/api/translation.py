"""翻訳APIエンドポイント。

思考ブロック（ThinkingBlock / Thoughts）の翻訳リクエストを受け付け、
指定されたモデルプリセットを使ってテキストを翻訳する。
翻訳結果はDBに保存せず、レスポンスとして返すのみ。
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.providers.registry import create_provider

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/translate", tags=["translation"])


class TranslateRequest(BaseModel):
    """翻訳リクエストモデル。"""

    text: str
    """翻訳するテキスト。"""

    preset_id: str | None = None
    """使用するモデルプリセットID。省略時はグローバル設定の translation_preset_id を使用する。"""


class TranslateResponse(BaseModel):
    """翻訳レスポンスモデル。"""

    translation: str
    """翻訳されたテキスト。"""


_TRANSLATION_SYSTEM_PROMPT = """\
You are a professional translator. Translate the given text to Japanese.
Output only the translation. No explanations, preambles, or comments.
If a portion cannot be translated (code, proper nouns, etc.), leave it as-is."""


@router.post("", response_model=TranslateResponse)
async def translate_text(request: Request, body: TranslateRequest) -> TranslateResponse:
    """テキストを日本語に翻訳する。

    preset_id 省略時は global_settings["translation_preset_id"] を使用する。
    翻訳結果はDBに保存せず、レスポンスとして返すのみ。

    Args:
        request: FastAPIリクエストオブジェクト。
        body: 翻訳リクエスト（text, preset_id）。

    Raises:
        HTTPException 400: preset_id が未指定かつ設定にも未設定の場合。
        HTTPException 404: プリセットが見つからない場合。
        HTTPException 503: プロバイダー呼び出しに失敗した場合。
    """
    state = request.app.state
    sqlite = state.sqlite
    settings = sqlite.get_all_settings()

    # preset_id 解決: リクエスト指定 > グローバル設定
    preset_id = body.preset_id or settings.get("translation_preset_id") or ""
    if not preset_id:
        raise HTTPException(
            status_code=400,
            detail="翻訳モデルが設定されていません。Settings 画面で翻訳モデルを設定してください。",
        )

    preset = sqlite.get_model_preset(preset_id)
    if preset is None:
        _log.error("翻訳: プリセット未発見 preset_id=%s", preset_id)
        raise HTTPException(
            status_code=404,
            detail=f"モデルプリセット '{preset_id}' が見つかりません。",
        )

    try:
        provider = create_provider(
            preset.provider,
            preset.model_id or "",
            settings,
            thinking_level=preset.thinking_level or "default",
            preset_name=preset.name,
        )
    except Exception as e:
        _log.error("翻訳: プロバイダー生成失敗 preset=%s error=%s", preset_id, e)
        raise HTTPException(
            status_code=503,
            detail="翻訳サービスの初期化に失敗しました。",
        ) from e

    messages = [{"role": "user", "content": body.text}]
    try:
        result = await provider.generate(_TRANSLATION_SYSTEM_PROMPT, messages)
    except Exception as e:
        _log.error("翻訳: LLM呼び出し失敗 preset=%s error=%s", preset_id, e)
        raise HTTPException(
            status_code=503,
            detail="翻訳サービスへのリクエストに失敗しました。",
        ) from e

    return TranslateResponse(translation=result or "")
