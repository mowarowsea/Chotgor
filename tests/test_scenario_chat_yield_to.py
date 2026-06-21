"""シナリオチャット ensemble_pc — 「ターンを譲る」UI の初動ルーティングを検証する。

`run_scenario_turn(auto_advance=True, yield_to=...)` の挙動を検証する。
yield_to はフロントの「ターンを譲る」チップ（他PC枠 / @ALL / @GM）から渡される
パラメータで、ユーザがメッセージを書かずに初動ルーティングを直接指定する用途。

検証する観点:
    - yield_to=<PC枠名> → 該当 PC が最初に呼ばれる（GM を経由しない）
    - yield_to="GM"    → GM が最初に呼ばれる（PC は呼ばれない、従来 auto_advance と同じ）
    - yield_to="ALL"   → ランダム PC ルート（PC が 1 名のときは確実にその PC）
    - yield_to=None    → 従来挙動（GM 行き）
    - yield_to=<存在しないPC枠名> → GM フォールバック
    - yield_to=<ユーザPC枠名> → ユーザPC枠は譲れないため GM フォールバック
    - auto_advance=True なので user turn は保存されない（履歴に痕跡が残らない）
"""

import asyncio

import backend.services.scenario_chat.pc_runner as pc_runner_mod
import backend.services.scenario_chat.service as svc

from tests._scenario_sqlite_helpers import _make_scenario


# ─── セッションビルダー / モック差し替え ──────────────────────────────────────


def _build_pc_mode_session(store, *, with_user_pc: bool = False):
    """ensemble_pc セッション（2 キャラ PC）を組み立てて返す。

    PC1=「はる」、PC2=「もわ」。両方とも AI キャラ担当。with_user_pc=True のときは
    末尾にユーザ枠（"ユーザPC"）を追加し、yield_to がユーザPCを弾くか検証する。

    Returns:
        (session_id, {"haru": cid_haru, "mowa": cid_mowa})
    """
    cid_haru = "char-haru"
    cid_mowa = "char-mowa"
    pid = "preset-test"
    store.create_character(cid_haru, "はる")
    store.create_character(cid_mowa, "もわ")
    store.create_model_preset(pid, "テスト用", "anthropic", "claude-x")

    pc_slots = [
        {"slot_id": "pc1", "name": "はる", "description": "PC1。"},
        {"slot_id": "pc2", "name": "もわ", "description": "PC2。"},
    ]
    pc_assignments = [
        {"slot_id": "pc1", "player_type": "character",
         "character_id": cid_haru, "preset_id": pid},
        {"slot_id": "pc2", "player_type": "character",
         "character_id": cid_mowa, "preset_id": pid},
    ]
    if with_user_pc:
        pc_slots.append({"slot_id": "user", "name": "ユーザPC", "description": ""})
        pc_assignments.append({"slot_id": "user", "player_type": "user"})

    scenario = _make_scenario(
        store, title="ensemble_pc テスト", pc_slots=pc_slots,
    )
    sid = "sess-pc"
    store.create_scenario_session(
        session_id=sid,
        scenario_id=scenario.id,
        title="プレイ #1",
        gm_preset_id=pid,
        synopsis_preset_id=pid,
        engine_type="ensemble_pc",
        pc_assignments=pc_assignments,
    )
    return sid, {"haru": cid_haru, "mowa": cid_mowa}


def _install_mocks(monkeypatch, *, gm_text: str = "Narrator: 静かな場面。"):
    """_run_gm_turn / stream_pc_response をモック差し替えして呼出順を記録する。

    fake_gm は Narrator ターンを 1 件保存し、ルーティングは @<PC> 等を含まない既定文を
    使うため後続は終了する（次話者「無し」で while ループ break）。
    fake_pc は pc_done を 1 回 yield する。やはりメンションは含めないので終了する。

    Returns:
        呼出ログのリスト（dict）。{"kind": "gm" | "pc", "pc_name": str | None} の形式で
        run_scenario_turn 内で実行された順に積まれる。
    """
    call_log: list[dict] = []

    async def fake_gm(**kwargs):
        call_log.append({"kind": "gm", "pc_name": None})
        svc._save_turn(
            sqlite=kwargs["sqlite"],
            session_id=kwargs["session_id"],
            speaker_type="narrator",
            speaker_name="Narrator",
            content=gm_text,
            raw_response=gm_text,
        )
        return
        yield  # async generator にするためのダミー

    async def fake_pc(**kwargs):
        pc = kwargs["pc"]
        call_log.append({"kind": "pc", "pc_name": pc.name})
        # 後続ループの break 条件を満たすため、メンション無しの本文で pc_done を返す。
        yield ("pc_done", {
            "character": pc.name,
            "character_id": pc.character_id,
            "full_text": f"（{pc.name}は無言で頷いた）",
            "anticipation": None,
        })

    monkeypatch.setattr(svc, "_run_gm_turn", fake_gm)
    monkeypatch.setattr(pc_runner_mod, "stream_pc_response", fake_pc)
    # あらすじ進捗計算は副作用で SQLite に触るが、ここではルーティングのみが関心事。
    monkeypatch.setattr(svc, "compute_synopsis_progress", lambda *a, **k: None)
    return call_log


def _run(sid, sqlite, **kwargs):
    """run_scenario_turn を async generator として最後まで消費するヘルパー。"""

    async def _go():
        events: list[tuple[str, dict]] = []
        async for ev in svc.run_scenario_turn(
            session_id=sid,
            sqlite=sqlite,
            settings={},
            chat_service=object(),
            **kwargs,
        ):
            events.append(ev)
        return events

    return asyncio.run(_go())


# ─── テストケース ───────────────────────────────────────────────────────────


class TestYieldTo:
    """ensemble_pc + auto_advance + yield_to による初動ルーティングを検証する。"""

    def test_yield_to_pc_dispatches_to_that_pc(self, sqlite_store, monkeypatch):
        """yield_to="はる" → 最初の呼出が PC「はる」になること。GM は経由しない。"""
        sid, _ = _build_pc_mode_session(sqlite_store)
        call_log = _install_mocks(monkeypatch)

        _run(sid, sqlite_store, user_message="", auto_advance=True, yield_to="はる")

        assert len(call_log) >= 1, "PC レスポンスが 1 回も走っていない"
        assert call_log[0] == {"kind": "pc", "pc_name": "はる"}

    def test_yield_to_other_pc(self, sqlite_store, monkeypatch):
        """yield_to="もわ" → PC2「もわ」が最初に呼ばれること。"""
        sid, _ = _build_pc_mode_session(sqlite_store)
        call_log = _install_mocks(monkeypatch)

        _run(sid, sqlite_store, user_message="", auto_advance=True, yield_to="もわ")

        assert call_log[0] == {"kind": "pc", "pc_name": "もわ"}

    def test_yield_to_gm_dispatches_to_gm(self, sqlite_store, monkeypatch):
        """yield_to="GM" → 最初の呼出が GM になること（PC ではない）。

        ensemble_pc には「GM 直後にメンションが無ければ @ALL フォールバック」の
        特例があるため、GM 応答後に PC が呼ばれることはあり得る（設計どおり）。
        ここでは「初動が GM」であることだけを検証する。
        """
        sid, _ = _build_pc_mode_session(sqlite_store)
        call_log = _install_mocks(monkeypatch)

        _run(sid, sqlite_store, user_message="", auto_advance=True, yield_to="GM")

        assert call_log[0]["kind"] == "gm"

    def test_yield_to_all_with_single_pc(self, sqlite_store, monkeypatch):
        """yield_to="ALL" → PC が 1 名のときは確実にその PC に振られること。"""
        cid = "char-only"
        pid = "preset-test"
        sqlite_store.create_character(cid, "ひとり")
        sqlite_store.create_model_preset(pid, "テスト用", "anthropic", "claude-x")
        scenario = _make_scenario(
            sqlite_store,
            title="1PC テスト",
            pc_slots=[{"slot_id": "pc1", "name": "ひとり", "description": ""}],
        )
        sid = "sess-1pc"
        sqlite_store.create_scenario_session(
            session_id=sid,
            scenario_id=scenario.id,
            title="プレイ #1",
            gm_preset_id=pid,
            synopsis_preset_id=pid,
            engine_type="ensemble_pc",
            pc_assignments=[{
                "slot_id": "pc1", "player_type": "character",
                "character_id": cid, "preset_id": pid,
            }],
        )
        call_log = _install_mocks(monkeypatch)

        _run(sid, sqlite_store, user_message="", auto_advance=True, yield_to="ALL")

        assert call_log[0] == {"kind": "pc", "pc_name": "ひとり"}

    def test_yield_to_none_falls_back_to_gm(self, sqlite_store, monkeypatch):
        """yield_to=None（従来 auto_advance） → 初動は GM。後方互換性確認。"""
        sid, _ = _build_pc_mode_session(sqlite_store)
        call_log = _install_mocks(monkeypatch)

        _run(sid, sqlite_store, user_message="", auto_advance=True, yield_to=None)

        assert call_log[0]["kind"] == "gm"

    def test_yield_to_unknown_falls_back_to_gm(self, sqlite_store, monkeypatch):
        """存在しない PC枠名は GM フォールバック（誤った値での暴発を防ぐ）。"""
        sid, _ = _build_pc_mode_session(sqlite_store)
        call_log = _install_mocks(monkeypatch)

        _run(sid, sqlite_store, user_message="", auto_advance=True, yield_to="存在しない")

        assert call_log[0]["kind"] == "gm"

    def test_yield_to_user_pc_falls_back_to_gm(self, sqlite_store, monkeypatch):
        """ユーザPC枠は「譲れない先」なので GM フォールバック。

        UI 側はユーザPCをチップ候補から除外するが、API を直接叩かれても安全側に倒す。
        """
        sid, _ = _build_pc_mode_session(sqlite_store, with_user_pc=True)
        call_log = _install_mocks(monkeypatch)

        _run(sid, sqlite_store, user_message="", auto_advance=True, yield_to="ユーザPC")

        assert call_log[0]["kind"] == "gm"

    def test_yield_to_does_not_save_user_turn(self, sqlite_store, monkeypatch):
        """auto_advance=True なので user turn は保存されず、痕跡が履歴に残らないこと。"""
        sid, _ = _build_pc_mode_session(sqlite_store)
        _install_mocks(monkeypatch)

        _run(sid, sqlite_store, user_message="", auto_advance=True, yield_to="はる")

        turns = sqlite_store.list_scenario_turns(sid)
        assert all(t.speaker_type != "user" for t in turns), \
            "auto_advance=True + yield_to で user turn が保存されてしまった"
