"""シナリオチャット API — ターン取得/削除・SSE ストリーム・あらすじ API の統合テスト。

実 SQLiteStore を tmp fixture から差し込み、LLM プロバイダ呼出のみモックして
HTTP 経由でエンドポイントの振る舞いを検証する。
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


# ─── 枝分かれ（レスポンスガチャ）・手動書き換え ─────────────────────────────


def _seed_generation(store, session_id, generation_id, branch_point_index, contents):
    """1 リクエスト分の応答群（＝1 つの枝）を直接 SQLite へ積むヘルパ。

    実運用では stream エンドポイントが ContextVar へ枝情報を積むため、
    テストでも同じ経路を通す。
    """
    import uuid as _uuid

    from backend.lib.log_context import (
        current_branch_point_index,
        current_generation_id,
    )

    gen_token = current_generation_id.set(generation_id)
    branch_token = current_branch_point_index.set(branch_point_index)
    try:
        saved = []
        for c in contents:
            saved.append(
                store.create_scenario_turn(
                    turn_id=str(_uuid.uuid4()),
                    session_id=session_id,
                    turn_index=store.get_next_scenario_turn_index(session_id),
                    speaker_type="narrator",
                    speaker_name="Narrator",
                    content=c,
                )
            )
        return saved
    finally:
        current_generation_id.reset(gen_token)
        current_branch_point_index.reset(branch_token)


class TestTurnVariantsApi:
    """枝の切替・手動書き換え API を検証する。"""

    def _session(self, sqlite_store):
        """プリセットとセッションを用意して (client, session_id) を返す。"""
        _seed_preset(sqlite_store)
        client = TestClient(_build_app(sqlite_store))
        sid = _create_scenario(client)["id"]
        return client, _start_session(client, sid)["id"]

    def test_turns_expose_variant_info(self, sqlite_store):
        """ターン一覧に枝ナビ用の index / count / siblings が乗ること。"""
        client, sess_id = self._session(sqlite_store)
        _seed_generation(sqlite_store, sess_id, "gen1", -1, ["A"])
        turn = client.get(f"/api/scenario_chat/sessions/{sess_id}/turns").json()[0]
        assert turn["generation_id"] == "gen1"
        assert turn["variant_index"] == 1
        assert turn["variant_count"] == 1
        assert turn["variant_siblings"] == ["gen1"]

    def test_delete_keeps_variants_by_default(self, sqlite_store):
        """既定の巻き戻しは非活性化で、枝として選び直せること。"""
        client, sess_id = self._session(sqlite_store)
        first = _seed_generation(sqlite_store, sess_id, "gen1", -1, ["A"])
        client.delete(
            f"/api/scenario_chat/sessions/{sess_id}/turns/from/{first[0].id}"
        )
        _seed_generation(sqlite_store, sess_id, "gen2", -1, ["B"])

        body = client.get(f"/api/scenario_chat/sessions/{sess_id}/turns").json()
        assert [t["content"] for t in body] == ["B"]
        assert body[0]["variant_count"] == 2  # gen1 が枝として残っている

    def test_delete_with_keep_variants_false_wipes_branches(self, sqlite_store):
        """keep_variants=false は非活性な枝も含めて物理削除すること。

        起点は gen1 の行（turn_index が最小）。そこから後ろを一掃するので、
        非活性の gen1 も、その後に生えた gen2 もまとめて消える。
        """
        client, sess_id = self._session(sqlite_store)
        first = _seed_generation(sqlite_store, sess_id, "gen1", -1, ["A"])
        client.delete(
            f"/api/scenario_chat/sessions/{sess_id}/turns/from/{first[0].id}"
        )
        _seed_generation(sqlite_store, sess_id, "gen2", -1, ["B"])

        res = client.delete(
            f"/api/scenario_chat/sessions/{sess_id}/turns/from/{first[0].id}"
            "?keep_variants=false"
        )
        assert res.status_code == 200
        assert client.get(f"/api/scenario_chat/sessions/{sess_id}/turns").json() == []
        assert sqlite_store.list_scenario_generation_variants(sess_id) == {}

    def test_activate_switches_branch(self, sqlite_store):
        """枝を切り替えると本線が差し替わり、切替後の一覧が返ること。"""
        client, sess_id = self._session(sqlite_store)
        first = _seed_generation(sqlite_store, sess_id, "gen1", -1, ["A"])
        client.delete(
            f"/api/scenario_chat/sessions/{sess_id}/turns/from/{first[0].id}"
        )
        _seed_generation(sqlite_store, sess_id, "gen2", -1, ["B"])

        res = client.post(
            f"/api/scenario_chat/sessions/{sess_id}/turns/activate",
            json={"generation_id": "gen1"},
        )
        assert res.status_code == 200
        assert [t["content"] for t in res.json()] == ["A"]

    def test_activate_unknown_generation(self, sqlite_store):
        """存在しない枝は 404。"""
        client, sess_id = self._session(sqlite_store)
        res = client.post(
            f"/api/scenario_chat/sessions/{sess_id}/turns/activate",
            json={"generation_id": "nope"},
        )
        assert res.status_code == 404

    def test_patch_turn_content(self, sqlite_store):
        """発話本文を上書きできること（枝は増えない）。"""
        client, sess_id = self._session(sqlite_store)
        turns = _seed_generation(sqlite_store, sess_id, "gen1", -1, ["A"])

        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/turns/{turns[0].id}",
            json={"content": "書き換え"},
        )
        assert res.status_code == 200
        assert res.json()["content"] == "書き換え"
        body = client.get(f"/api/scenario_chat/sessions/{sess_id}/turns").json()
        assert [t["content"] for t in body] == ["書き換え"]

    def test_patch_unknown_turn(self, sqlite_store):
        """存在しないターンは 404。"""
        client, sess_id = self._session(sqlite_store)
        res = client.patch(
            f"/api/scenario_chat/sessions/{sess_id}/turns/nope",
            json={"content": "x"},
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
        assert "turn_start" in types
        assert "chunk" in types
        assert "turn_end" in types
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
        """履歴上限内なら regenerate は蒸留せず、現状の synopsis と進捗を返す。

        レスポンスは {"synopsis": {...}, "progress": {...}} 形式。蒸留対象が
        無いので synopsis は初期状態、progress も新規ターン 0 を返す。
        """
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
        assert data["synopsis"] == {"auto": "", "manual": "", "last_turn_index": -1}
        # 進捗は新規ターン 0。上限は設定/デフォルトに依存するので存在のみ確認する。
        assert data["progress"]["turns"] == 0
        assert data["progress"]["chars"] == 0
        assert "max_turns" in data["progress"]
        assert "max_chars" in data["progress"]

    def test_regenerate_nonexistent_returns_404(self, sqlite_store):
        """存在しないセッションは 404。"""
        client = TestClient(_build_app(sqlite_store))
        res = client.post("/api/scenario_chat/sessions/nope/synopsis/regenerate")
        assert res.status_code == 404
