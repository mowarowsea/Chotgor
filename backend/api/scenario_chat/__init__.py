"""シナリオチャット API パッケージ。

旧 `backend/api/scenario_chat.py`（単一ファイル）をリソース単位で分割。
- schemas.py   — Pydantic リクエストスキーマ
- scenarios.py — シナリオテンプレート＆NPC CRUD・プリセット一覧
- sessions.py  — プレイセッション CRUD・履歴・あらすじ
- stream.py    — SSE ストリーム実行

`/api/scenario_chat/...` 配下に、シナリオテンプレートとプレイインスタンスの
2 層を扱う CRUD と SSE ストリームを提供する。

エンドポイント:
    -- シナリオテンプレート（CRUD・NPC 編集） --
    POST   /api/scenario_chat/scenarios                                   作成
    GET    /api/scenario_chat/scenarios                                   一覧
    GET    /api/scenario_chat/scenarios/{sid}                             詳細
    PATCH  /api/scenario_chat/scenarios/{sid}                             更新
    DELETE /api/scenario_chat/scenarios/{sid}                             削除（紐づくセッションも）
    POST   /api/scenario_chat/scenarios/{sid}/npcs                        NPC 追加
    PATCH  /api/scenario_chat/scenarios/{sid}/npcs/{nid}                  NPC 編集
    DELETE /api/scenario_chat/scenarios/{sid}/npcs/{nid}                  NPC 削除

    -- プレイインスタンス（起動・履歴・SSE） --
    POST   /api/scenario_chat/sessions                                    シナリオから起動
    GET    /api/scenario_chat/sessions                                    一覧
    GET    /api/scenario_chat/sessions/{sid}                              詳細（シナリオ・NPC込み）
    PATCH  /api/scenario_chat/sessions/{sid}                              更新（title / status）
    DELETE /api/scenario_chat/sessions/{sid}                              削除（テンプレ非影響）
    POST   /api/scenario_chat/sessions/{sid}/end                          終了
    GET    /api/scenario_chat/sessions/{sid}/turns                        履歴
    POST   /api/scenario_chat/sessions/{sid}/stream                       SSE

    -- 補助 --
    GET    /api/scenario_chat/presets                                     GM プリセット選択用一覧
"""

from fastapi import APIRouter

from backend.api.scenario_chat import scenarios, sessions, stream

# 各リソースルーターを集約した親ルーター（main.py が include する）
router = APIRouter()
router.include_router(scenarios.router)
router.include_router(sessions.router)
router.include_router(stream.router)
