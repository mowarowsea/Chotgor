"""log_context.ensure_message_id() — ID 未採番防御網のユニットテスト。

エントリポイントで new_message_id() を呼び忘れたコードパスがあっても、
debug/--------/ へのファイル堆積と debug_log_entries.request_id="--------" 行の
発生を構造的に防ぐ lazy 採番の動作を検証する。

背景:
    新機能追加のたびに new_message_id() の呼び忘れが再発し（weekly_schedule /
    pressure_interview / intent_pickup / ask_visibility …）、フォールバック先の
    debug/--------/ に無関係な機能のログが数ヶ月分混在していた。
    ensure_message_id() は debug_logger の書き込み直前に発動する最後の防衛線であり、
    「呼び忘れてもフォールバック溜めに落ちない」ことをここで保証する。

検証対象:
    - ensure_message_id() 単体の採番・冪等・巻き添え防止・警告
    - ChotgorLogger 経由（ファイル出力・DB INSERT）で "--------" が使われないこと
"""

import pytest

from backend.lib.debug_logger import ChotgorLogger
from backend.lib.log_context import (
    UNSET_MESSAGE_ID,
    current_log_db_entry_id,
    current_log_dir_id,
    current_log_feature,
    current_log_session_id,
    current_log_target,
    current_message_id,
    ensure_message_id,
    new_message_id,
    next_log_index,
)


def _reset_context() -> None:
    """ログ用 ContextVar をすべて未設定デフォルトへ戻すヘルパ。

    pytest はテスト間で contextvars を隔離しないため、
    各テストの冒頭で明示的にリセットして独立性を保つ。
    """
    current_message_id.set(UNSET_MESSAGE_ID)
    current_log_dir_id.set(UNSET_MESSAGE_ID)
    current_log_feature.set("chat")
    current_log_target.set(None)
    current_log_session_id.set(None)
    current_log_db_entry_id.set(None)


# ─── ensure_message_id 単体 ───────────────────────────────────────────────────


class TestEnsureMessageId:
    """ensure_message_id() 単体の動作を検証するテストクラス。

    未採番時の lazy 採番・採番済み時の無変更（冪等）・DB 用 ContextVar の
    巻き添え防止・警告ログの発報を確認する。
    """

    def test_unset_generates_fresh_id(self):
        """未採番（デフォルト "--------"）の場合、8桁 hex を採番し
        current_message_id / current_log_dir_id の両方へセットすること。"""
        _reset_context()
        msg_id = ensure_message_id()
        assert msg_id != UNSET_MESSAGE_ID
        assert len(msg_id) == 8
        int(msg_id, 16)  # 16進数として解釈できること
        assert current_message_id.get() == msg_id
        assert current_log_dir_id.get() == msg_id

    def test_already_set_returns_existing(self):
        """new_message_id() 採番済みの場合、その ID をそのまま返し変更しないこと。"""
        _reset_context()
        original = new_message_id()
        assert ensure_message_id() == original
        assert current_message_id.get() == original
        assert current_log_dir_id.get() == original

    def test_idempotent_within_context(self):
        """同一コンテキストで2回呼んでも同じ ID が返ること（lazy 採番は1回だけ）。"""
        _reset_context()
        first = ensure_message_id()
        second = ensure_message_id()
        assert first == second

    def test_does_not_touch_db_context_vars(self):
        """lazy 採番が session_id / target 等の DB 用 ContextVar を巻き添えに
        リセットしないこと。

        new_message_id() と違い、呼び忘れコンテキストで先にセット済みの値を
        保全するのが ensure_message_id() の仕様。
        """
        _reset_context()
        current_log_session_id.set("sess_123")
        current_log_target.set("はる")
        ensure_message_id()
        assert current_log_session_id.get() == "sess_123"
        assert current_log_target.get() == "はる"

    def test_counter_reset_on_lazy_numbering(self):
        """lazy 採番時に通し番号カウンタがリセットされ、次の番号が 1 になること。"""
        _reset_context()
        next_log_index()  # 汚染された counter を模擬（=1 に進める）
        ensure_message_id()
        assert next_log_index() == 1

    def test_warning_logged_on_lazy_numbering(self, caplog):
        """lazy 採番時に呼び忘れ検出の警告が feature 名付きで出ること。

        警告は開発者が「どの機能のエントリポイントに new_message_id() を
        足すべきか」を特定する手がかりになる。
        """
        _reset_context()
        current_log_feature.set("weekly_schedule")
        with caplog.at_level("WARNING", logger="backend.lib.log_context"):
            ensure_message_id()
        assert any("weekly_schedule" in r.message for r in caplog.records)

    def test_no_warning_when_already_set(self, caplog):
        """採番済みの場合は警告が出ないこと（正常経路にノイズを混ぜない）。"""
        _reset_context()
        new_message_id()
        with caplog.at_level("WARNING", logger="backend.lib.log_context"):
            ensure_message_id()
        assert not caplog.records


# ─── ChotgorLogger 経由の防御網 ────────────────────────────────────────────────


@pytest.fixture
def debug_dir(tmp_path, monkeypatch):
    """CHOTGOR_DEBUG=1 でファイルログを有効化し、出力先を tmp_path へ隔離するフィクスチャ。

    ChotgorLogger.DEBUG_DIR はクラス属性なので monkeypatch がテスト後に自動で戻す。
    """
    monkeypatch.setenv("CHOTGOR_DEBUG", "1")
    debug_root = tmp_path / "debug"
    monkeypatch.setattr(ChotgorLogger, "DEBUG_DIR", str(debug_root))
    return debug_root


@pytest.fixture
def store(tmp_path):
    """テスト用 SQLiteStore を返すフィクスチャ。"""
    from backend.repositories.sqlite.store import SQLiteStore
    return SQLiteStore(str(tmp_path / "test.db"))


class TestDebugLoggerFallbackPrevention:
    """ChotgorLogger が "--------" フォールバックへ書き込まないことを検証するテストクラス。

    weekly_schedule 週次バッチのような「new_message_id() を呼ばずに
    log_provider_request / log_front_input から始まる」コードパスを模擬し、
    ファイルログ・DB 行の両方が lazy 採番 ID を使うことを確認する。
    """

    def test_write_log_does_not_use_fallback_dir(self, debug_dir):
        """未採番のままファイルログを書いても debug/--------/ が作られず、
        lazy 採番 ID のフォルダへ書かれること（今回の週次バッチの再現シナリオ）。"""
        _reset_context()
        logger = ChotgorLogger()
        logger.log_provider_request("TestPreset", {"messages": []})
        assert not (debug_dir / UNSET_MESSAGE_ID).exists()
        dirs = [p.name for p in debug_dir.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        assert dirs[0] == current_log_dir_id.get()

    def test_db_insert_does_not_use_fallback_request_id(self, store, monkeypatch):
        """CHOTGOR_DEBUG=0（ファイルログ無効）でも DB 行の request_id が
        "--------" にならないこと。

        ファイル書き込みの ensure を素通りする経路（デバッグ無効環境）でも
        _insert_main_entry 側の ensure が効くことを確認する。
        """
        monkeypatch.delenv("CHOTGOR_DEBUG", raising=False)
        _reset_context()
        logger = ChotgorLogger()
        logger.set_store(store)
        logger.log_front_input({"content": "こんにちは"})
        ids, total = store.get_debug_log_request_ids_paged()
        assert total == 1
        assert ids[0] != UNSET_MESSAGE_ID

    def test_sub_entry_does_not_use_fallback_request_id(self, store, monkeypatch):
        """ambience 等のサブ行 INSERT でも request_id が "--------" にならないこと。"""
        monkeypatch.delenv("CHOTGOR_DEBUG", raising=False)
        _reset_context()
        current_log_feature.set("ambience")
        logger = ChotgorLogger()
        logger.set_store(store)
        logger.log_provider_response("TestPreset", "さようなら")
        ids, total = store.get_debug_log_request_ids_paged()
        assert total == 1
        assert ids[0] != UNSET_MESSAGE_ID

    def test_normal_path_keeps_explicit_id(self, debug_dir, store):
        """new_message_id() を正しく呼ぶ既存経路では、採番済み ID がそのまま
        ファイルフォルダ・DB 行の両方に使われること（防御網が正常経路を邪魔しない）。"""
        _reset_context()
        msg_id = new_message_id()
        logger = ChotgorLogger()
        logger.set_store(store)
        logger.log_front_input({"content": "テスト"})
        assert (debug_dir / msg_id).exists()
        ids, _ = store.get_debug_log_request_ids_paged()
        assert ids == [msg_id]
