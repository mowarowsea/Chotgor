"""外からキャラクターの記憶・本人領域を書き換える操作がデバッグモード限定であることのテスト。

Chotgor では記憶の取捨選択・自己記述・別れの条件はキャラクター本人のもの
（DEAR_GHOST.md §1.1「あなた自身にしか編集する権限がない」）。管理UI・REST API に
残っている外からの操作は、開発・保守用としてデバッグモード（CHOTGOR_DEBUG=1）の
ときだけ通す（character_resident_rules.md §2「外からの記憶の削除」）。

対象:
    - REST  DELETE /api/inscribed_memories/{char}/{mem}          — 保存記憶の soft-delete
    - REST  POST   /api/inscribed_memories/{char}/{mem}/restore  — soft-delete の復元
    - 管理UI POST  /ui/inscribed_memories/{char}/{mem}/delete    — 削除ボタンの送信先
    - 管理UI GET   /ui/inscribed_memories/{char}                 — 削除ボタンの表示可否
    - REST  PATCH  /api/characters/{char}                        — 本人領域の書き換え
"""

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def debug_off(monkeypatch):
    """CHOTGOR_DEBUG を外した状態（通常運用）にするフィクスチャ。"""
    monkeypatch.delenv("CHOTGOR_DEBUG", raising=False)


@pytest.fixture
def debug_on(monkeypatch):
    """CHOTGOR_DEBUG=1（デバッグモード）にするフィクスチャ。"""
    monkeypatch.setenv("CHOTGOR_DEBUG", "1")


class TestRestMemoryDeleteRestore:
    """REST の保存記憶 削除・復元がデバッグモードでだけ通ることを検証するテストクラス。

    復元も対象にしているのは、Forget で本人が手放した記憶を外から戻すことも
    削除と同じく本人の取捨選択への介入になるため。memory_manager はモックで、
    呼ばれたかどうか（＝ゲートを抜けたか）だけを見る。
    """

    def _client(self):
        """inscribed_memories ルーターだけを積んだクライアントと memory_manager モックを返す。"""
        from backend.api.inscribed_memories import router

        app = FastAPI()
        app.include_router(router)
        app.state.memory_manager = MagicMock()
        return TestClient(app), app.state.memory_manager

    def test_delete_forbidden_without_debug(self, debug_off):
        """通常運用では削除は 403 で、記憶には触れない。"""
        client, mm = self._client()
        resp = client.delete("/api/inscribed_memories/c1/m1")
        assert resp.status_code == 403
        mm.delete_inscribed_memory.assert_not_called()

    def test_restore_forbidden_without_debug(self, debug_off):
        """通常運用では復元も 403 で、記憶には触れない。"""
        client, mm = self._client()
        resp = client.post("/api/inscribed_memories/c1/m1/restore")
        assert resp.status_code == 403
        mm.restore_inscribed_memory.assert_not_called()

    def test_delete_and_restore_allowed_in_debug(self, debug_on):
        """デバッグモードでは従来どおり削除・復元できる。"""
        client, mm = self._client()
        mm.delete_inscribed_memory.return_value = True
        mm.restore_inscribed_memory.return_value = True
        assert client.delete("/api/inscribed_memories/c1/m1").status_code == 204
        assert client.post("/api/inscribed_memories/c1/m1/restore").status_code == 200
        mm.delete_inscribed_memory.assert_called_once_with("m1", "c1")
        mm.restore_inscribed_memory.assert_called_once_with("m1")


class TestUiMemoryDelete:
    """管理UIの記憶削除（ボタンの表示と送信先）がデバッグモード限定であることを検証するテストクラス。

    ボタンを隠すだけだとフォームの直叩きで消せてしまうため、送信先のルートも弾く。
    表示側は TemplateResponse に渡る `can_delete` を偽テンプレートで捕まえて確認する
    （本物のテンプレート描画は main.py の Jinja 設定に依存するため使わない）。
    """

    def _client(self, sqlite_store, monkeypatch):
        """管理UIの記憶ルーターだけを積んだクライアント・mm モック・描画コンテキスト置き場を返す。"""
        import backend.api.ui.memories as ui_memories

        captured: dict = {}

        class _FakeTemplates:
            """TemplateResponse の context を記録して空の HTML を返す偽テンプレート。"""

            def TemplateResponse(self, request, name, context):
                from fastapi.responses import HTMLResponse

                captured.update(context)
                return HTMLResponse("")

        monkeypatch.setattr(ui_memories, "get_templates", lambda: _FakeTemplates())
        app = FastAPI()
        app.include_router(ui_memories.router)
        app.state.sqlite = sqlite_store
        app.state.memory_manager = MagicMock()
        app.state.memory_manager.list_inscribed_memories.return_value = []
        return TestClient(app), app.state.memory_manager, captured

    def _char(self, sqlite_store):
        """テスト用キャラクターを作って ID を返す。"""
        char_id = str(uuid.uuid4())
        sqlite_store.create_character(character_id=char_id, name="はるテスト")
        return char_id

    def test_delete_route_forbidden_without_debug(self, sqlite_store, monkeypatch, debug_off):
        """通常運用ではフォームを直接送っても 403 で、記憶には触れない。"""
        client, mm, _ = self._client(sqlite_store, monkeypatch)
        resp = client.post(
            "/ui/inscribed_memories/c1/m1/delete", follow_redirects=False,
        )
        assert resp.status_code == 403
        mm.delete_inscribed_memory.assert_not_called()

    def test_delete_route_allowed_in_debug(self, sqlite_store, monkeypatch, debug_on):
        """デバッグモードでは削除して一覧へリダイレクトする（従来挙動）。"""
        client, mm, _ = self._client(sqlite_store, monkeypatch)
        resp = client.post(
            "/ui/inscribed_memories/c1/m1/delete", follow_redirects=False,
        )
        assert resp.status_code == 303
        mm.delete_inscribed_memory.assert_called_once_with("m1", "c1")

    def test_button_hidden_without_debug(self, sqlite_store, monkeypatch, debug_off):
        """通常運用では一覧ページに削除ボタンを出さない（can_delete=False）。"""
        client, _, captured = self._client(sqlite_store, monkeypatch)
        char_id = self._char(sqlite_store)
        assert client.get(f"/ui/inscribed_memories/{char_id}").status_code == 200
        assert captured["can_delete"] is False

    def test_button_shown_in_debug(self, sqlite_store, monkeypatch, debug_on):
        """デバッグモードでは削除ボタンを出す（can_delete=True）。"""
        client, _, captured = self._client(sqlite_store, monkeypatch)
        char_id = self._char(sqlite_store)
        assert client.get(f"/ui/inscribed_memories/{char_id}").status_code == 200
        assert captured["can_delete"] is True


class TestCharacterOwnedFieldsPatch:
    """PATCH /api/characters で本人領域の書き換えがデバッグモード限定であることを検証するテストクラス。

    本人領域 = inner_narrative / self_history / relationship_state / farewell_config /
    relationship_status。それ以外（名前・定義・モデル指定など作者の領分）は通常運用でも
    従来どおり更新できることも併せて確認する（ゲートが広すぎないこと）。
    """

    def _client(self, sqlite_store):
        """characters ルーターだけを積んだクライアントを返す。"""
        from backend.api.characters import router

        app = FastAPI()
        app.include_router(router)
        app.state.sqlite = sqlite_store
        app.state.vector_store = None
        return TestClient(app)

    def _char(self, sqlite_store):
        """テスト用キャラクターを作って ID を返す。"""
        char_id = str(uuid.uuid4())
        sqlite_store.create_character(character_id=char_id, name="はるテスト")
        return char_id

    @pytest.mark.parametrize("payload", [
        {"inner_narrative": "外から書いた物語"},
        {"farewell_config": {"thresholds": {"anger": 0.1}}},
        {"relationship_status": "active"},
        {"self_history": "x"},
        {"relationship_state": "x"},
    ])
    def test_owned_fields_forbidden_without_debug(self, sqlite_store, debug_off, payload):
        """通常運用では本人領域を含む更新は 403 で、DB は変わらない。"""
        char_id = self._char(sqlite_store)
        before = sqlite_store.get_character(char_id).inner_narrative
        resp = self._client(sqlite_store).patch(f"/api/characters/{char_id}", json=payload)
        assert resp.status_code == 403
        assert sqlite_store.get_character(char_id).inner_narrative == before

    def test_mixed_update_rejected_as_a_whole(self, sqlite_store, debug_off):
        """作者の領分と本人領域が混ざった更新は、作者の領分ごと丸ごと拒否する（部分適用しない）。"""
        char_id = self._char(sqlite_store)
        resp = self._client(sqlite_store).patch(
            f"/api/characters/{char_id}",
            json={"name": "改名", "inner_narrative": "外から書いた物語"},
        )
        assert resp.status_code == 403
        assert sqlite_store.get_character(char_id).name == "はるテスト"

    def test_author_fields_allowed_without_debug(self, sqlite_store, debug_off):
        """本人領域を含まない更新（作者の領分）は通常運用でも通る。"""
        char_id = self._char(sqlite_store)
        resp = self._client(sqlite_store).patch(
            f"/api/characters/{char_id}", json={"system_prompt_block1": "新しい定義"},
        )
        assert resp.status_code == 200
        assert sqlite_store.get_character(char_id).system_prompt_block1 == "新しい定義"

    def test_owned_fields_allowed_in_debug(self, sqlite_store, debug_on):
        """デバッグモードでは本人領域も書き換えられる（開発・保守用）。"""
        char_id = self._char(sqlite_store)
        resp = self._client(sqlite_store).patch(
            f"/api/characters/{char_id}", json={"inner_narrative": "保守で直した物語"},
        )
        assert resp.status_code == 200
        assert sqlite_store.get_character(char_id).inner_narrative == "保守で直した物語"
