"""設定 UI パッケージ。

旧 `backend/api/ui.py`（単一ファイル）をページ単位で分割したパッケージ。

- common.py     — テンプレート保持・共有フォーム/レスポンスヘルパー
- dashboard.py  — ダッシュボード（index・LLM 使用量集計）
- characters.py — キャラクター管理
- memories.py   — 保存記憶＆ワーキングメモリ閲覧
- presets.py    — LLM モデルプリセット管理
- scenarios.py  — シナリオテンプレート＆NPC 管理
- settings.py   — グローバル設定・embedding・再インデックス

main.py からは従来通り `router` を include し、テンプレートは
`set_templates()` で注入する（互換ファサード）。
"""

from fastapi import APIRouter

from backend.api.ui.common import (  # noqa: F401
    get_templates,
    set_templates,
)
from backend.api.ui import characters, dashboard, instruments, memories, presets, scenarios, settings

# 各ページルーターを集約した親ルーター（main.py が include する）
router = APIRouter()
router.include_router(dashboard.router)
router.include_router(characters.router)
router.include_router(memories.router)
router.include_router(presets.router)
router.include_router(scenarios.router)
router.include_router(settings.router)
router.include_router(instruments.router)
