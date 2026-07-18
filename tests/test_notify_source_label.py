"""ntfy 通知の発話経路ラベル（source）のテスト。

通知本文だけでは「うつつのシーン内発話」と「うつつ派生ではる発の 1on1 発話」の
区別がつかなかった問題（2026-07-18）への対処を検証する。

- notify_character_spoke: source ラベルが通知本文末尾に埋め込まれること、
  無効化時・キャラ名空のときは送信されないこと。
- _save_turn（シナリオ/うつつ共通のターン保存経路）: セッションの engine_type を見て
  "usual_days" なら「うつつ」、それ以外（対面シナリオプレイ）なら「シナリオ」の
  ラベルで通知が飛ぶこと。PC 以外の話者（GM/NPC/user）では通知しないこと。

なお 1on1 側（同期返信・預かり配達・能動 push / reach_out）はすべて source="1on1" で
統一する裁定（ユーザ発への返信とキャラ発の話しかけは区別しない）。呼び出し元の
リテラルを網羅テストはせず、ラベル生成の仕組み側をここで固める。
"""

import uuid

import backend.lib.notify as notify_mod
from backend.services.scenario_chat.turns import _save_turn


class _ImmediateThread:
    """threading.Thread 互換の即時実行スタブ。

    notify_character_spoke はデーモンスレッドへ送信を逃がすため、テストでは
    start() で target を同期実行してタイミング依存（flaky）を排除する。
    """

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


class TestNotifyCharacterSpoke:
    """notify_character_spoke 単体のラベル埋め込みと抑止条件のテスト。"""

    def _capture(self, monkeypatch):
        """送信を同期化し、_send に渡った本文をリストで捕捉するヘルパ。"""
        sent = []
        monkeypatch.setattr(notify_mod, "NTFY_ENABLED", True)
        monkeypatch.setattr(notify_mod.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(notify_mod, "_send", lambda message: sent.append(message))
        return sent

    def test_source_label_is_embedded(self, monkeypatch):
        """source ラベルが通知本文の末尾（全角括弧）に入る。"""
        sent = self._capture(monkeypatch)
        notify_mod.notify_character_spoke("はる", source="うつつ")
        assert sent == ["Chotgor:はるが発話（うつつ）"]

    def test_1on1_label(self, monkeypatch):
        """1on1 経路のラベル。ユーザ発への返信もキャラ発の話しかけも同一表記。"""
        sent = self._capture(monkeypatch)
        notify_mod.notify_character_spoke("はる", source="1on1")
        assert sent == ["Chotgor:はるが発話（1on1）"]

    def test_disabled_sends_nothing(self, monkeypatch):
        """NTFY_ENABLED=False なら送信しない。"""
        sent = self._capture(monkeypatch)
        monkeypatch.setattr(notify_mod, "NTFY_ENABLED", False)
        notify_mod.notify_character_spoke("はる", source="1on1")
        assert sent == []

    def test_empty_name_sends_nothing(self, monkeypatch):
        """キャラ名が空なら送信しない（従来挙動の維持）。"""
        sent = self._capture(monkeypatch)
        notify_mod.notify_character_spoke("", source="1on1")
        assert sent == []


class TestSaveTurnNotifySource:
    """_save_turn の通知ラベル出し分けテスト。

    シナリオの対面プレイとうつつ（無人生活）は同じ _save_turn 経路を通るため、
    セッションの engine_type だけが両者を区別する唯一の手がかりになる。
    ここが壊れると「うつつのシーン内発話」と「対面プレイの発話」の通知が
    再び見分けられなくなるので、engine_type ごとのラベルを固定する。
    """

    def _make_session(self, sqlite_store, engine_type: str) -> str:
        """指定 engine_type のシナリオセッションを作って ID を返すヘルパ。"""
        scenario_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        sqlite_store.create_scenario(scenario_id=scenario_id, title="テスト用シナリオ")
        sqlite_store.create_scenario_session(
            session_id=session_id,
            scenario_id=scenario_id,
            title="テスト用セッション",
            gm_preset_id="preset-gm",
            synopsis_preset_id="preset-gm",
            engine_type=engine_type,
        )
        return session_id

    def _capture_notify(self, monkeypatch):
        """notify_character_spoke 呼び出しを (name, source) タプルで捕捉するヘルパ。

        turns.py は関数内 import（from backend.lib.notify import ...）なので、
        notify モジュール側の属性を差し替えれば呼び出し時に反映される。
        """
        calls = []
        monkeypatch.setattr(
            notify_mod, "notify_character_spoke",
            lambda name, *, source: calls.append((name, source)),
        )
        return calls

    def test_usual_days_session_labels_utsutsu(self, sqlite_store, monkeypatch):
        """engine_type="usual_days" の PC 発話は「うつつ」ラベルで通知される。"""
        calls = self._capture_notify(monkeypatch)
        session_id = self._make_session(sqlite_store, "usual_days")
        _save_turn(
            sqlite=sqlite_store,
            session_id=session_id,
            speaker_type="pc",
            speaker_name="はる",
            content="……ん、今日はいい天気",
        )
        assert calls == [("はる", "うつつ")]

    def test_ensemble_pc_session_labels_scenario(self, sqlite_store, monkeypatch):
        """engine_type="ensemble_pc"（対面プレイ）の PC 発話は「シナリオ」ラベル。"""
        calls = self._capture_notify(monkeypatch)
        session_id = self._make_session(sqlite_store, "ensemble_pc")
        _save_turn(
            sqlite=sqlite_store,
            session_id=session_id,
            speaker_type="pc",
            speaker_name="はる",
            content="よし、行こうか",
        )
        assert calls == [("はる", "シナリオ")]

    def test_non_pc_speaker_does_not_notify(self, sqlite_store, monkeypatch):
        """GM / user 等の非 PC 話者では通知しない（従来挙動の維持）。"""
        calls = self._capture_notify(monkeypatch)
        session_id = self._make_session(sqlite_store, "usual_days")
        _save_turn(
            sqlite=sqlite_store,
            session_id=session_id,
            speaker_type="gm",
            speaker_name="GM",
            content="場面: 夕暮れの部屋",
        )
        _save_turn(
            sqlite=sqlite_store,
            session_id=session_id,
            speaker_type="user",
            speaker_name="もわ",
            content="ただいま",
        )
        assert calls == []
