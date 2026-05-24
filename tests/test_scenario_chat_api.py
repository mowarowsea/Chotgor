"""シナリオチャット API (backend.api.scenario_chat) の統合テスト。

実 SQLiteStore を tmp fixture から差し込み、LLM プロバイダ呼出のみモックして
HTTP 経由でエンドポイントの振る舞いを検証する。

エンドポイント:
    -- シナリオテンプレート --
    POST   /api/scenario_chat/scenarios                   作成
    GET    /api/scenario_chat/scenarios                   一覧
    GET    /api/scenario_chat/scenarios/{sid}             詳細（NPC込み）
    PATCH  /api/scenario_chat/scenarios/{sid}             更新
    DELETE /api/scenario_chat/scenarios/{sid}             削除（紐づくセッションも）
    POST   /api/scenario_chat/scenarios/{sid}/npcs        NPC 追加
    PATCH  /api/scenario_chat/scenarios/{sid}/npcs/{nid}  NPC 編集
    DELETE /api/scenario_chat/scenarios/{sid}/npcs/{nid}  NPC 削除

    -- プレイセッション --
    POST   /api/scenario_chat/sessions                    起動
    GET    /api/scenario_chat/sessions                    一覧
    GET    /api/scenario_chat/sessions/{sid}              詳細
    PATCH  /api/scenario_chat/sessions/{sid}              更新
    DELETE /api/scenario_chat/sessions/{sid}              削除
    POST   /api/scenario_chat/sessions/{sid}/end          終了
    GET    /api/scenario_chat/sessions/{sid}/turns        履歴
    POST   /api/scenario_chat/sessions/{sid}/stream       SSE
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import scenario_chat as scenario_chat_module


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


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
        "user_alias": "プレイヤー",
    }
    payload.update(overrides)
    res = client.post("/api/scenario_chat/scenarios", json=payload)
    assert res.status_code == 201, res.text
    return res.json()


def _start_session(
    client, scenario_id: str, title=None, gm_preset_id: str = "preset-test"
) -> dict:
    """POST /sessions のラッパ。gm_preset_id はセッション必須。"""
    payload = {"scenario_id": scenario_id, "gm_preset_id": gm_preset_id}
    if title:
        payload["title"] = title
    res = client.post("/api/scenario_chat/sessions", json=payload)
    assert res.status_code == 201, res.text
    return res.json()


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
        assert body["user_alias"] == "プレイヤー"
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
            json={"user_alias": "p"},
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
            json={"scenario_id": "missing", "gm_preset_id": "preset-test"},
        )
        assert res.status_code == 400

    def test_start_session_invalid_preset(self, sqlite_store):
        """未登録の gm_preset_id でセッション起動は 400。"""
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        res = client.post(
            "/api/scenario_chat/sessions",
            json={"scenario_id": sid, "gm_preset_id": "missing"},
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


# ─── ターン取得 ──────────────────────────────────────────────────────────────


class TestListTurns:
    """履歴一覧 API を検証する。"""

    def test_empty(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        res = client.get(f"/api/scenario_chat/sessions/{sess_id}/turns")
        assert res.status_code == 200
        assert res.json() == []

    def test_with_direct_inserts(self, sqlite_store):
        """SQLite に直接挿入したターンを API で取得できる。"""
        import uuid as _uuid
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        sqlite_store.create_scenario_turn(
            turn_id=str(_uuid.uuid4()),
            session_id=sess_id,
            turn_index=0,
            speaker_type="user",
            speaker_name="プレイヤー",
            content="導入",
        )
        sqlite_store.create_scenario_turn(
            turn_id=str(_uuid.uuid4()),
            session_id=sess_id,
            turn_index=1,
            speaker_type="narrator",
            speaker_name="Narrator",
            content="夜",
        )
        body = client.get(f"/api/scenario_chat/sessions/{sess_id}/turns").json()
        contents = [t["content"] for t in body]
        assert contents == ["導入", "夜"]

    def test_session_not_found(self, sqlite_store):
        client = TestClient(_build_app(sqlite_store))
        res = client.get("/api/scenario_chat/sessions/nope/turns")
        assert res.status_code == 404


# ─── ターン削除（編集・再生成の前処理） ─────────────────────────────────────


class TestDeleteTurnsFrom:
    """指定ターン以降を削除する API を検証する。

    ユーザ発話の編集・GM ターン再生成の前処理として呼ばれる。
    """

    def test_delete_from_pivot(self, sqlite_store):
        """指定ターン以降が削除されること。"""
        import uuid as _uuid
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        ids = []
        for i in range(4):
            tid = str(_uuid.uuid4())
            sqlite_store.create_scenario_turn(
                turn_id=tid,
                session_id=sess_id,
                turn_index=i,
                speaker_type="user" if i % 2 == 0 else "narrator",
                speaker_name="P" if i % 2 == 0 else "Narrator",
                content=str(i),
            )
            ids.append(tid)
        pivot = ids[2]
        res = client.delete(
            f"/api/scenario_chat/sessions/{sess_id}/turns/from/{pivot}"
        )
        assert res.status_code == 200
        body = client.get(f"/api/scenario_chat/sessions/{sess_id}/turns").json()
        assert [t["content"] for t in body] == ["0", "1"]

    def test_delete_unknown_turn(self, sqlite_store):
        """存在しない turn_id で 404。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        res = client.delete(
            f"/api/scenario_chat/sessions/{sess_id}/turns/from/nope"
        )
        assert res.status_code == 404

    def test_delete_session_not_found(self, sqlite_store):
        """存在しないセッションで 404。"""
        client = TestClient(_build_app(sqlite_store))
        res = client.delete(
            "/api/scenario_chat/sessions/nope/turns/from/whatever"
        )
        assert res.status_code == 404


# ─── ストリーミング ──────────────────────────────────────────────────────────


class TestStream:
    """SSE ストリーム API をエンジンモック化で検証する。"""

    def test_basic_flow(self, sqlite_store, monkeypatch):
        """エンジンモックが期待通りの SSE 行列を返すこと。"""
        from backend.services.scenario_chat import service as svc
        from backend.services.scenario_chat.parser import UtteranceDelta
        from backend.services.scenario_chat.engine import EngineResult, TurnRecord

        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]

        class FakeEngine:
            """シナリオから起動された session に対する SceneEngine スタブ。"""

            async def generate_stream(self, **kwargs):
                yield UtteranceDelta(
                    speaker_type="narrator",
                    speaker_id=None,
                    speaker_name="Narrator",
                    content_delta="雨",
                    is_speaker_change=True,
                    is_known=True,
                )
                yield UtteranceDelta(
                    speaker_type="narrator",
                    speaker_id=None,
                    speaker_name="Narrator",
                    content_delta="\n",
                    is_speaker_change=False,
                    is_known=True,
                )
                yield TurnRecord(
                    speaker_type="narrator",
                    speaker_id=None,
                    speaker_name="Narrator",
                    content="雨\n",
                    is_known=True,
                )
                yield EngineResult(raw_response="@Narrator: 雨\n")

        monkeypatch.setattr(svc, "_build_default_engine", lambda sqlite: FakeEngine())

        with client.stream(
            "POST",
            f"/api/scenario_chat/sessions/{sess_id}/stream",
            json={"content": "やぁ"},
        ) as resp:
            assert resp.status_code == 200
            events = []
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                events.append(json.loads(line[len("data: ") :]))

        types = [e["type"] for e in events]
        assert "user_saved" in types
        assert "speaker_start" in types
        assert "content_delta" in types
        assert "speaker_end" in types
        assert "turn_complete" in types
        assert types[-1] == "done"

    def test_session_not_found(self, sqlite_store):
        client = TestClient(_build_app(sqlite_store))
        res = client.post(
            "/api/scenario_chat/sessions/nope/stream",
            json={"content": "x"},
        )
        assert res.status_code == 404

    def test_ended_session(self, sqlite_store):
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        client.post(f"/api/scenario_chat/sessions/{sess_id}/end")
        res = client.post(
            f"/api/scenario_chat/sessions/{sess_id}/stream",
            json={"content": "x"},
        )
        assert res.status_code == 400


# ─── あらすじ API（記憶捏造対策） ─────────────────────────────────────────────


class TestSynopsisAPI:
    """`/sessions/{id}/synopsis` 系エンドポイントを検証する。

    記憶捏造対策として導入された「セッション単位のあらすじ」機構の
    HTTP インターフェース層。重要観点:
        - GET で初期状態（空文字列・last_turn_index=-1）が取れる
        - PATCH で auto / manual を独立に更新できる
        - PATCH で manual を更新しても auto が破壊されない（記憶保護の核心）
        - 逆も同様（auto 更新で manual が壊れない）
        - 存在しないセッションは 404
        - 空ボディの PATCH は現状を返すだけ
    """

    def test_get_initial_synopsis_is_empty(self, sqlite_store):
        """新規セッションは auto / manual ともに空、last_turn_index=-1。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        res = client.get(f"/api/scenario_chat/sessions/{sess_id}/synopsis")
        assert res.status_code == 200
        data = res.json()
        assert data == {"auto": "", "manual": "", "last_turn_index": -1}

    def test_get_nonexistent_session_returns_404(self, sqlite_store):
        """存在しないセッション ID は 404。"""
        client = TestClient(_build_app(sqlite_store))
        res = client.get("/api/scenario_chat/sessions/nope/synopsis")
        assert res.status_code == 404

    def test_patch_manual_only(self, sqlite_store):
        """manual だけ更新して auto に手をつけないこと。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        # 事前に auto を入れておく
        client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis",
            json={"auto": "既存の自動要約"},
        )
        # manual だけ更新（auto は触らない）
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis",
            json={"manual": "プレイヤーが書いた経緯"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["auto"] == "既存の自動要約"
        assert data["manual"] == "プレイヤーが書いた経緯"

    def test_patch_auto_only_preserves_manual(self, sqlite_store):
        """auto を更新しても manual が破壊されないこと。

        ユーザが捏造記述を見つけて auto を編集する状況を想定したテスト。
        """
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        # 先に manual を設定
        client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis",
            json={"manual": "守りたい記述"},
        )
        # auto を上書き（manual には触れない）
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis",
            json={"auto": "ユーザが編集した自動あらすじ"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["auto"] == "ユーザが編集した自動あらすじ"
        assert data["manual"] == "守りたい記述"

    def test_patch_empty_body_returns_current_state(self, sqlite_store):
        """空ボディの PATCH は現状を返すだけ。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis",
            json={"auto": "X", "manual": "Y"},
        )
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis",
            json={},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["auto"] == "X"
        assert data["manual"] == "Y"

    def test_patch_nonexistent_returns_404(self, sqlite_store):
        """存在しないセッションへの PATCH は 404。"""
        client = TestClient(_build_app(sqlite_store))
        res = client.patch(
            "/api/scenario_chat/sessions/nope/synopsis",
            json={"manual": "x"},
        )
        assert res.status_code == 404

    def test_patch_auto_empty_resets_last_turn_index(self, sqlite_store):
        """auto を空文字列にクリアしたら `last_turn_index` が -1 へ連動リセットされること。

        この連動が無いと「auto は空なのに last_turn_index は過去の値を保持」という
        不整合状態が残り、以降の自動蒸留が `new_dropped` 空判定で永久にスキップ
        される（旧ユーザ報告: 「あらすじ削除後も再生成されない」）。
        手動で auto を白紙化する意図は「最初から蒸留し直したい」なので、
        進捗ポインタも初期状態へ戻すのが正しい。
        """
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        # 蒸留が進んだ状態を直接 DB レイヤで作る
        sqlite_store.update_scenario_session_synopsis(
            sess_id, auto="既存あらすじ", manual="守りたいメモ", last_turn_index=42,
        )
        # auto だけ空文字へクリア
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis",
            json={"auto": ""},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["auto"] == ""
        assert data["last_turn_index"] == -1
        # manual は触らない（保護対象）
        assert data["manual"] == "守りたいメモ"

    def test_patch_auto_nonempty_keeps_last_turn_index(self, sqlite_store):
        """auto を非空テキストに編集した場合は `last_turn_index` を据え置くこと。

        ユーザが捏造記述だけを部分削除・修正する用途（蒸留結果を「補正」する）
        では、last_turn_index は蒸留済みターン境界を表すので残すのが正しい。
        リセットは「白紙化（auto を完全に空にする）」時のみの特例。
        """
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        sqlite_store.update_scenario_session_synopsis(
            sess_id, auto="既存あらすじ", last_turn_index=42,
        )
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis",
            json={"auto": "ユーザが修正したあらすじ"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["auto"] == "ユーザが修正したあらすじ"
        assert data["last_turn_index"] == 42

    def test_regenerate_without_dropped_returns_current(self, sqlite_store):
        """履歴上限内なら regenerate は何もせず現状を返す（dropped が空のため）。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        sess_id = _start_session(client, sid)["id"]
        # 履歴ゼロのまま regenerate（dropped は空）
        res = client.post(
            f"/api/scenario_chat/sessions/{sess_id}/synopsis/regenerate"
        )
        assert res.status_code == 200
        data = res.json()
        assert data == {"auto": "", "manual": "", "last_turn_index": -1}

    def test_regenerate_nonexistent_returns_404(self, sqlite_store):
        """存在しないセッションは 404。"""
        client = TestClient(_build_app(sqlite_store))
        res = client.post("/api/scenario_chat/sessions/nope/synopsis/regenerate")
        assert res.status_code == 404
