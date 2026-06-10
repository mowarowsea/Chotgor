"""Tests for backend.lib.usage_recorder — LLM 使用量レコーダー。"""

import pytest

from backend.lib import usage_recorder
from backend.lib.log_context import (
    current_log_feature,
    current_log_target,
    current_message_id,
)


class _FakeStore:
    """add_llm_usage_event の呼び出しを記録するテスト用スタブ。"""

    def __init__(self, raise_on_add: bool = False):
        self.calls: list[dict] = []
        self._raise = raise_on_add

    def add_llm_usage_event(self, **kwargs):
        """呼び出し引数を記録する。raise_on_add=True なら DB 障害を模して例外を投げる。"""
        if self._raise:
            raise RuntimeError("simulated db failure")
        self.calls.append(kwargs)


@pytest.fixture
def fake_store(monkeypatch):
    """usage_recorder のモジュールグローバル _store をスタブへ差し替える。"""
    store = _FakeStore()
    monkeypatch.setattr(usage_recorder, "_store", store)
    return store


class TestRecordUsage:
    """record_usage の挙動を検証する。

    レコーダーはチャット本流（プロバイダーのレスポンス処理）の途中で呼ばれるため、
    「記録できないときに本流を壊さない」ことが最重要の契約：
    - store 未注入（起動前・テスト環境）では何もしない
    - トークンが全くない（usage が取れなかった）呼び出しは記録しない
    - DB 例外は握り潰す（呼び出し元へ伝播しない）
    加えて、feature / target / request_id が log_context の ContextVar から
    自動補完されることを確認する。
    """

    def test_noop_when_store_not_injected(self, monkeypatch):
        """set_store 前（_store=None）は例外も記録も発生しないこと。"""
        monkeypatch.setattr(usage_recorder, "_store", None)

        usage_recorder.record_usage(provider="claude_cli", input_tokens=10, output_tokens=5)

    def test_skips_when_no_tokens(self, fake_store):
        """input/output とも 0（usage が取れなかった）の呼び出しは記録しないこと。"""
        usage_recorder.record_usage(provider="claude_cli")

        assert fake_store.calls == []

    def test_records_with_context_vars(self, fake_store):
        """トークン情報と log_context 由来の feature/target/request_id が揃って記録されること。"""
        t_feature = current_log_feature.set("chronicle")
        t_target = current_log_target.set("織羽")
        t_msg = current_message_id.set("abcd1234")
        try:
            usage_recorder.record_usage(
                provider="claude_cli",
                model="claude-sonnet-4-6",
                preset_name="default",
                input_tokens=1200,
                output_tokens=340,
                cache_read_input_tokens=800,
                cache_creation_input_tokens=50,
                total_cost_usd=0.0123,
            )
        finally:
            current_log_feature.reset(t_feature)
            current_log_target.reset(t_target)
            current_message_id.reset(t_msg)

        assert fake_store.calls == [{
            "provider": "claude_cli",
            "model": "claude-sonnet-4-6",
            "preset_name": "default",
            "target": "織羽",
            "feature": "chronicle",
            "request_id": "abcd1234",
            "input_tokens": 1200,
            "output_tokens": 340,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 50,
            "total_cost_usd": 0.0123,
        }]

    def test_empty_strings_become_none(self, fake_store):
        """model / preset_name の空文字は NULL（None）として記録されること。"""
        usage_recorder.record_usage(
            provider="google", input_tokens=10, output_tokens=5,
        )

        assert len(fake_store.calls) == 1
        call = fake_store.calls[0]
        assert call["model"] is None
        assert call["preset_name"] is None

    def test_store_exception_is_swallowed(self, monkeypatch, caplog):
        """DB 書き込み例外が呼び出し元（チャット本流）へ伝播しないこと。"""
        monkeypatch.setattr(usage_recorder, "_store", _FakeStore(raise_on_add=True))

        usage_recorder.record_usage(provider="claude_cli", input_tokens=10, output_tokens=5)

        assert any("LLM使用量の記録に失敗" in r.message for r in caplog.records)
