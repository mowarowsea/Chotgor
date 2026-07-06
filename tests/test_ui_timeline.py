"""Tests for backend.api.ui.timeline — タイムライン閲覧ページ（めぐり Phase 7）。

ユーザダイヤル（覗き窓）の適用面を検証する:
    - user_ui 投影がキャラの timeline_dial を反映して描画されること
    - ダイヤル切替 POST が characters.timeline_dial を更新すること
    - ダイヤル段階で見えるものが実際に減ること（生活の秘匿・最終形）
実テンプレート（timeline.html）を使い、Jinja の構文崩れも同時に検出する。
"""

import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from backend.api.ui import common as ui_common
from backend.api.ui.timeline import router

_TEMPLATES_DIR = str(Path(__file__).parent.parent / "backend" / "templates")


@pytest.fixture
def timeline_client(sqlite_store, monkeypatch):
    """timeline ルーターと実テンプレートを組み込んだテストクライアントを返すフィクスチャ。"""
    monkeypatch.setattr(ui_common, "templates", Jinja2Templates(directory=_TEMPLATES_DIR))
    app = FastAPI()
    app.include_router(router)
    app.state.sqlite = sqlite_store
    return TestClient(app)


def _seed_character_with_events(sqlite_store):
    """1on1会話・うつつシーン・記憶が混在するキャラクターを作るヘルパ。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name="はるテスト")
    # 1on1 会話
    sid = str(uuid.uuid4())
    sqlite_store.create_chat_session(session_id=sid, model_id="はるテスト@d")
    sqlite_store.create_chat_message(
        message_id=str(uuid.uuid4()), session_id=sid, role="user",
        content="チャットの中身テキスト",
    )
    # うつつシーン
    scen_id = str(uuid.uuid4())
    sqlite_store.create_scenario(
        scenario_id=scen_id, title="世界", owner_character_id=char_id,
    )
    usess = str(uuid.uuid4())
    sqlite_store.create_scenario_session(
        session_id=usess, scenario_id=scen_id, title="うつつ",
        gm_preset_id="p", synopsis_preset_id="p", engine_type="usual_days",
    )
    sqlite_store.create_scenario_turn(
        turn_id=str(uuid.uuid4()), session_id=usess, turn_index=0,
        speaker_type="npc", speaker_name="店主", content="うつつの中身テキスト",
    )
    # 記憶
    sqlite_store.create_inscribed_memory(
        memory_id=str(uuid.uuid4()), character_id=char_id,
        content="記憶の中身テキスト",
    )
    return char_id


class TestTimelinePage:
    """GET /ui/timeline の描画とダイヤル適用を検証するテストクラス。"""

    def test_renders_without_characters(self, timeline_client):
        """キャラクターゼロでも 200 で描画されること。"""
        resp = timeline_client.get("/ui/timeline")
        assert resp.status_code == 200
        assert "タイムライン" in resp.text

    def test_dial_0_shows_everything(self, timeline_client, sqlite_store):
        """ダイヤル0（全開）はチャット・うつつ・記憶の中身がすべて見える。"""
        char_id = _seed_character_with_events(sqlite_store)
        resp = timeline_client.get(f"/ui/timeline?character_id={char_id}")
        assert resp.status_code == 200
        assert "チャットの中身テキスト" in resp.text
        assert "うつつの中身テキスト" in resp.text
        assert "記憶の中身テキスト" in resp.text

    def test_dial_1_hides_usual_content(self, timeline_client, sqlite_store):
        """ダイヤル1（生活の秘匿）はうつつの中身が封筒止めになる。"""
        char_id = _seed_character_with_events(sqlite_store)
        sqlite_store.update_character(char_id, timeline_dial=1)
        resp = timeline_client.get(f"/ui/timeline?character_id={char_id}")
        assert "うつつの中身テキスト" not in resp.text
        assert "封筒のみ" in resp.text          # envelope 表示
        assert "チャットの中身テキスト" in resp.text  # real の会話は見える

    def test_dial_3_chat_only(self, timeline_client, sqlite_store):
        """ダイヤル3（最終形）はチャット応答以外すべて消える。"""
        char_id = _seed_character_with_events(sqlite_store)
        sqlite_store.update_character(char_id, timeline_dial=3)
        resp = timeline_client.get(f"/ui/timeline?character_id={char_id}")
        assert "チャットの中身テキスト" in resp.text
        assert "うつつの中身テキスト" not in resp.text
        assert "記憶の中身テキスト" not in resp.text
        assert "記憶を刻んだ" not in resp.text  # 存在ごと hidden

    def test_set_dial_persists(self, timeline_client, sqlite_store):
        """ダイヤル切替 POST が timeline_dial を更新しリダイレクトする。"""
        char_id = _seed_character_with_events(sqlite_store)
        resp = timeline_client.post(
            f"/ui/timeline/{char_id}/dial", data={"dial": "2"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert sqlite_store.get_character(char_id).timeline_dial == 2

    def test_set_dial_clamps_invalid(self, timeline_client, sqlite_store):
        """範囲外・不正なダイヤル値はクランプ/フォールバックされる。"""
        char_id = _seed_character_with_events(sqlite_store)
        timeline_client.post(
            f"/ui/timeline/{char_id}/dial", data={"dial": "9"},
            follow_redirects=False,
        )
        assert sqlite_store.get_character(char_id).timeline_dial == 3
        timeline_client.post(
            f"/ui/timeline/{char_id}/dial", data={"dial": "abc"},
            follow_redirects=False,
        )
        assert sqlite_store.get_character(char_id).timeline_dial == 0
