"""シナリオチャット API テスト群の共有ヘルパー。

実 SQLiteStore を組み込んだ FastAPI アプリの構築と、シナリオ/セッションの
シードデータ作成を提供する。test_scenario_api_*.py から共通利用する。
ファイル名先頭がアンダースコアのため pytest のテスト収集対象にはならない。
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import scenario_chat as scenario_chat_module


def _build_app(sqlite_store) -> FastAPI:
    """テスト用に scenario_chat ルータを乗せた最小 FastAPI app を構築する。"""
    app = FastAPI()
    app.include_router(scenario_chat_module.router)
    app.state.sqlite = sqlite_store
    return app


def _seed_preset(sqlite_store, preset_id="preset-test", name="Test", provider="fake"):
    """テスト用に LLMModelPreset を 1 件作っておく。"""
    sqlite_store.create_model_preset(
        preset_id=preset_id,
        name=name,
        provider=provider,
        model_id="fake-model",
    )
    return preset_id


def _create_scenario(client, **overrides) -> dict:
    """POST /scenarios のラッパ。GM プリセットはセッション側で指定するので含まない。"""
    payload = {
        "title": "テストシナリオ",
        # 旧 user_alias は廃止。ユーザPCを pc_slots の先頭枠として渡す。
        "pc_slots": [{"slot_id": "user", "name": "プレイヤー", "description": ""}],
    }
    payload.update(overrides)
    res = client.post("/api/scenario_chat/scenarios", json=payload)
    assert res.status_code == 201, res.text
    return res.json()


def _start_session(
    client,
    scenario_id: str,
    title=None,
    gm_preset_id: str = "preset-test",
    synopsis_preset_id: str = None,
) -> dict:
    """POST /sessions のラッパ。gm_preset_id / synopsis_preset_id はセッション必須。

    synopsis_preset_id 省略時は gm_preset_id と同値を渡す（従来挙動と等価）。
    """
    if synopsis_preset_id is None:
        synopsis_preset_id = gm_preset_id
    payload = {
        "scenario_id": scenario_id,
        "gm_preset_id": gm_preset_id,
        "synopsis_preset_id": synopsis_preset_id,
    }
    if title:
        payload["title"] = title
    res = client.post("/api/scenario_chat/sessions", json=payload)
    assert res.status_code == 201, res.text
    return res.json()

