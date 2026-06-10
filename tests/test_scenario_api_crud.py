"""シナリオチャット API — テンプレート＆NPC＆セッション CRUD の統合テスト。

実 SQLiteStore を tmp fixture から差し込み、HTTP 経由で
シナリオ/NPC/プレイセッションの CRUD エンドポイントを検証する。
"""

import json

import pytest

from tests._scenario_api_helpers import (
    _build_app,
    _create_scenario,
    _seed_preset,
    _start_session,
)
from fastapi.testclient import TestClient

# ─── シナリオテンプレート CRUD ──────────────────────────────────────────────


class TestScenarioCRUD:
    """シナリオテンプレートの CRUD エンドポイントを検証する。

    Scenario = 何度でも遊べる「設定の塊」テンプレート。
    """

    def test_create_minimum(self, sqlite_store):
        """必須項目だけで作成し、レスポンスにフィールドが含まれること。

        gm_preset_id はテンプレートには含まれない（セッション単位の設定に移行）。
        """
        client = TestClient(_build_app(sqlite_store))
        body = _create_scenario(client)
        assert body["id"]
        assert body["title"] == "テストシナリオ"
        # 旧 user_alias は廃止。ユーザPCは pc_slots の先頭枠として保持される。
        assert body["pc_slots"][0]["name"] == "プレイヤー"
        assert "gm_preset_id" not in body

    def test_create_full(self, sqlite_store):
        """全フィールド指定で作成できること。"""
        client = TestClient(_build_app(sqlite_store))
        body = _create_scenario(
            client,
            scenario="魔導書探求。古城。詩的に語る。",
            history_max_turns=50,
        )
        assert body["scenario"] == "魔導書探求。古城。詩的に語る。"
        assert body["history_max_turns"] == 50

    def test_create_missing_required(self, sqlite_store):
        """必須フィールド未指定で 422。"""
        client = TestClient(_build_app(sqlite_store))
        # title 抜けで 422
        res = client.post(
            "/api/scenario_chat/scenarios",
            json={"scenario": "本文のみ"},
        )
        assert res.status_code == 422

    def test_list_empty(self, sqlite_store):
        """作成前は空配列。"""
        client = TestClient(_build_app(sqlite_store))
        res = client.get("/api/scenario_chat/scenarios")
        assert res.status_code == 200
        assert res.json() == []

    def test_list_has_entries(self, sqlite_store):
        """複数作成後に一覧で取得できる。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        _create_scenario(client, title="A")
        _create_scenario(client, title="B")
        titles = sorted(s["title"] for s in client.get("/api/scenario_chat/scenarios").json())
        assert titles == ["A", "B"]

    def test_get_includes_npcs(self, sqlite_store):
        """詳細 API は npcs 配列を含む。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        res = client.get(f"/api/scenario_chat/scenarios/{sid}")
        assert res.status_code == 200
        body = res.json()
        assert body["id"] == sid
        assert body["npcs"] == []

    def test_get_not_found(self, sqlite_store):
        """存在しないシナリオで 404。"""
        client = TestClient(_build_app(sqlite_store))
        res = client.get("/api/scenario_chat/scenarios/nope")
        assert res.status_code == 404

    def test_update(self, sqlite_store):
        """PATCH で複数フィールドを部分更新できる。"""
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        res = client.patch(
            f"/api/scenario_chat/scenarios/{sid}",
            json={"title": "更新後", "scenario": "新あらすじ"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["title"] == "更新後"
        assert body["scenario"] == "新あらすじ"

    def test_update_not_found(self, sqlite_store):
        client = TestClient(_build_app(sqlite_store))
        res = client.patch("/api/scenario_chat/scenarios/nope", json={"title": "x"})
        assert res.status_code == 404

    def test_delete(self, sqlite_store):
        """シナリオ削除すると詳細が 404 になる。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        res = client.delete(f"/api/scenario_chat/scenarios/{sid}")
        assert res.status_code == 200
        assert res.json() == {"deleted": True}
        assert client.get(f"/api/scenario_chat/scenarios/{sid}").status_code == 404

    def test_delete_cascades_sessions(self, sqlite_store):
        """シナリオ削除で紐づくセッションも消える。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        client.delete(f"/api/scenario_chat/scenarios/{sid}")
        assert client.get(f"/api/scenario_chat/sessions/{sess_id}").status_code == 404

    def test_update_keeps_sessions(self, sqlite_store):
        """シナリオ「更新」ではセッションが残る（仕様の核）。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        client.patch(
            f"/api/scenario_chat/scenarios/{sid}",
            json={"title": "編集後"},
        )
        assert client.get(f"/api/scenario_chat/sessions/{sess_id}").status_code == 200


# ─── NPC CRUD ────────────────────────────────────────────────────────────────


class TestNpcCRUD:
    """シナリオに紐づく NPC の追加・編集・削除エンドポイントを検証する。"""

    def test_add_npc(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        res = client.post(
            f"/api/scenario_chat/scenarios/{sid}/npcs",
            json={"name": "レイカ", "description": "魔法使い"},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["name"] == "レイカ"
        assert body["scenario_id"] == sid

    def test_add_npc_scenario_not_found(self, sqlite_store):
        client = TestClient(_build_app(sqlite_store))
        res = client.post(
            "/api/scenario_chat/scenarios/nope/npcs",
            json={"name": "X"},
        )
        assert res.status_code == 404

    def test_add_npc_duplicate(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        client.post(
            f"/api/scenario_chat/scenarios/{sid}/npcs", json={"name": "ダブり"}
        )
        res = client.post(
            f"/api/scenario_chat/scenarios/{sid}/npcs", json={"name": "ダブり"}
        )
        assert res.status_code == 400

    def test_edit_npc(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        nid = client.post(
            f"/api/scenario_chat/scenarios/{sid}/npcs",
            json={"name": "レイカ"},
        ).json()["id"]
        res = client.patch(
            f"/api/scenario_chat/scenarios/{sid}/npcs/{nid}",
            json={"description": "新プロフ"},
        )
        assert res.status_code == 200
        assert res.json()["description"] == "新プロフ"

    def test_edit_npc_rename_to_existing(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        client.post(f"/api/scenario_chat/scenarios/{sid}/npcs", json={"name": "A"})
        nid = client.post(
            f"/api/scenario_chat/scenarios/{sid}/npcs", json={"name": "B"}
        ).json()["id"]
        res = client.patch(
            f"/api/scenario_chat/scenarios/{sid}/npcs/{nid}",
            json={"name": "A"},
        )
        assert res.status_code == 400

    def test_edit_npc_not_found(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        res = client.patch(
            f"/api/scenario_chat/scenarios/{sid}/npcs/nope",
            json={"name": "X"},
        )
        assert res.status_code == 404

    def test_delete_npc(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        nid = client.post(
            f"/api/scenario_chat/scenarios/{sid}/npcs", json={"name": "削除予定"}
        ).json()["id"]
        res = client.delete(f"/api/scenario_chat/scenarios/{sid}/npcs/{nid}")
        assert res.status_code == 200
        body = client.get(f"/api/scenario_chat/scenarios/{sid}").json()
        assert body["npcs"] == []

    def test_delete_npc_scenario_mismatch(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid_a = _create_scenario(client, title="A")["id"]
        sid_b = _create_scenario(client, title="B")["id"]
        nid = client.post(
            f"/api/scenario_chat/scenarios/{sid_a}/npcs", json={"name": "X"}
        ).json()["id"]
        res = client.delete(f"/api/scenario_chat/scenarios/{sid_b}/npcs/{nid}")
        assert res.status_code == 404


# ─── プレイセッション CRUD ──────────────────────────────────────────────────


class TestSessionCRUD:
    """シナリオから起動するプレイセッションのライフサイクルを検証する。"""

    def test_start_session(self, sqlite_store):
        """シナリオ ID + gm_preset_id を指定してセッションを起動できる。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        scenario = _create_scenario(client, title="テンプレA")
        sess = _start_session(client, scenario["id"])
        assert sess["id"]
        assert sess["scenario_id"] == scenario["id"]
        # title 省略時はテンプレ title をコピー
        assert sess["title"] == "テンプレA"
        assert sess["status"] == "active"
        # gm_preset_id がセッションに紐づく
        assert sess["gm_preset_id"] == "preset-test"

    def test_start_session_with_title(self, sqlite_store):
        """title 指定でセッション名をカスタムできる。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess = _start_session(client, sid, title="プレイ #2")
        assert sess["title"] == "プレイ #2"

    def test_start_session_missing_scenario(self, sqlite_store):
        """存在しないシナリオで起動は 400。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        res = client.post(
            "/api/scenario_chat/sessions",
            json={
                "scenario_id": "missing",
                "gm_preset_id": "preset-test",
                "synopsis_preset_id": "preset-test",
            },
        )
        assert res.status_code == 400

    def test_start_session_invalid_preset(self, sqlite_store):
        """未登録の gm_preset_id でセッション起動は 400。"""
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        res = client.post(
            "/api/scenario_chat/sessions",
            json={
                "scenario_id": sid,
                "gm_preset_id": "missing",
                "synopsis_preset_id": "missing",
            },
        )
        assert res.status_code == 400

    def test_start_session_missing_preset_field(self, sqlite_store):
        """gm_preset_id 自体を渡さないと 422（必須項目）。"""
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        res = client.post(
            "/api/scenario_chat/sessions",
            json={"scenario_id": sid},
        )
        assert res.status_code == 422

    def test_list_sessions(self, sqlite_store):
        """セッション一覧。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        _start_session(client, sid)
        _start_session(client, sid, title="2nd")
        res = client.get("/api/scenario_chat/sessions")
        assert len(res.json()) == 2

    def test_get_session_includes_scenario_and_npcs(self, sqlite_store):
        """セッション詳細はシナリオ情報と NPC リストを含む。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        client.post(
            f"/api/scenario_chat/scenarios/{sid}/npcs", json={"name": "レイカ"}
        )
        sess = _start_session(client, sid)
        res = client.get(f"/api/scenario_chat/sessions/{sess['id']}")
        assert res.status_code == 200
        body = res.json()
        assert body["scenario"] is not None
        assert body["scenario"]["id"] == sid
        assert len(body["npcs"]) == 1
        assert body["npcs"][0]["name"] == "レイカ"

    def test_get_session_not_found(self, sqlite_store):
        client = TestClient(_build_app(sqlite_store))
        res = client.get("/api/scenario_chat/sessions/nope")
        assert res.status_code == 404

    def test_update_session(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}",
            json={"title": "リネーム"},
        )
        assert res.status_code == 200
        assert res.json()["title"] == "リネーム"

    def test_update_session_gm_preset(self, sqlite_store):
        """PATCH /sessions/{id} で GM プリセットを差し替えできること（チャット中のモデル切替）。"""
        _seed_preset(sqlite_store, preset_id="preset-test")
        _seed_preset(sqlite_store, preset_id="preset-other", name="Other")
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid, gm_preset_id="preset-test")["id"]
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}",
            json={"gm_preset_id": "preset-other"},
        )
        assert res.status_code == 200
        assert res.json()["gm_preset_id"] == "preset-other"

    def test_update_session_invalid_preset(self, sqlite_store):
        """存在しないプリセットへ切替は 400。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}",
            json={"gm_preset_id": "missing"},
        )
        assert res.status_code == 400

    def test_delete_session(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        res = client.delete(f"/api/scenario_chat/sessions/{sess_id}")
        assert res.status_code == 200
        # テンプレは残る
        assert client.get(f"/api/scenario_chat/scenarios/{sid}").status_code == 200

    def test_end_session(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        res = client.post(f"/api/scenario_chat/sessions/{sess_id}/end")
        assert res.status_code == 200
        assert res.json()["status"] == "ended"


