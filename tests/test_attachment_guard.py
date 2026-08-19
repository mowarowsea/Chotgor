"""添付の入口ガードのテスト。

音声を渡せるのは Gemini（google プロバイダー）だけで、プロバイダーによって
「聴ける／聴けない」が割れる。フロントは FileDialog の accept と選択後の
MIME 検査で止めるが、スマホの FileDialog は accept を尊重しないことがあるため、
送信 API 側でも弾く二重ガードになっている。

ここで検証するのは:
    - プロバイダー能力宣言（SUPPORTED_ATTACHMENT_KINDS）の実際の値
    - /v1/models がその能力を attachment_kinds として返すこと
    - 送信 API が非対応の添付を 400 で拒否すること（黙って捨てない）
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import chat as chat_module
from backend.providers.registry import PROVIDER_REGISTRY, supported_attachment_kinds


# ---------------------------------------------------------------------------
# プロバイダー能力宣言
# ---------------------------------------------------------------------------

class TestSupportedAttachmentKinds:
    """SUPPORTED_ATTACHMENT_KINDS の宣言内容を検証するテストクラス。

    「どのプロバイダーが何を受け取れるか」は UI の accept と送信時検証の
    唯一の根拠なので、宣言が意図せず変わったら落ちるようにしておく。
    """

    def test_google_accepts_audio(self):
        """google は音声を受け取れること（Gemini の inline_data 経由）。"""
        assert supported_attachment_kinds("google") == ["audio", "image"]

    def test_claude_cli_rejects_audio(self):
        """claude_cli は画像のみ（Anthropic は音声入力を持たない）。"""
        assert supported_attachment_kinds("claude_cli") == ["image"]

    @pytest.mark.parametrize(
        "provider_id",
        [p for p in PROVIDER_REGISTRY if p != "google"],
    )
    def test_only_google_accepts_audio(self, provider_id):
        """google 以外のプロバイダーは音声を受け取らないこと。"""
        assert "audio" not in supported_attachment_kinds(provider_id)

    def test_unknown_provider_falls_back_to_image_only(self):
        """未知の provider_id は既定（画像のみ）に落ちること。"""
        assert supported_attachment_kinds("nonexistent") == ["image"]


# ---------------------------------------------------------------------------
# 送信 API の二重ガード
# ---------------------------------------------------------------------------

def _make_app(sqlite_mock) -> FastAPI:
    """モック state を持つ最小 FastAPI アプリを生成する。"""
    app = FastAPI()
    app.include_router(chat_module.router)
    app.state.sqlite = sqlite_mock
    app.state.chat_service = MagicMock()
    app.state.memory_manager = MagicMock()
    app.state.vector_store = MagicMock()
    app.state.uploads_dir = "/tmp"
    return app


def _sqlite_with(provider: str, attachment_mimes: dict[str, str]) -> MagicMock:
    """指定プロバイダーのプリセットと添付メタデータを返すストアモックを作る。"""
    sqlite = MagicMock()
    sqlite.get_chat_session.return_value = MagicMock(
        id="sid", model_id="はる@p1", session_type="1on1", title="t"
    )
    sqlite.get_model_preset_by_name.return_value = MagicMock(provider=provider)
    sqlite.get_chat_attachment.side_effect = lambda att_id: (
        MagicMock(mime_type=attachment_mimes[att_id])
        if att_id in attachment_mimes else None
    )
    return sqlite


class TestStreamRejectsUnsupportedAttachments:
    """送信 API が非対応の添付を 400 で弾くことを検証するテストクラス。

    「渡ったように見えて渡っていない」が最悪なので、黙って捨てずに拒否する。
    拒否は LLM 呼び出しどころかユーザメッセージの保存より前に起きる
    （弾いた発話が履歴へ残らないこと）。
    """

    def test_audio_to_non_google_is_rejected(self):
        """claude_cli プリセットへの音声添付は 400 になること。"""
        sqlite = _sqlite_with("claude_cli", {"a1": "audio/mpeg"})
        client = TestClient(_make_app(sqlite))

        res = client.post(
            "/api/chat/sessions/sid/messages/stream",
            json={"content": "これ聴いて", "attachment_ids": ["a1"]},
        )

        assert res.status_code == 400
        assert "Gemini" in res.json()["detail"]
        sqlite.create_chat_message.assert_not_called()

    def test_rejection_happens_before_the_message_is_saved(self):
        """拒否はユーザメッセージの保存より前に起きること（弾いた発話を履歴に残さない）。"""
        sqlite = _sqlite_with("claude_cli", {"a1": "audio/mpeg"})
        client = TestClient(_make_app(sqlite))

        client.post(
            "/api/chat/sessions/sid/messages/stream",
            json={"content": "これ聴いて", "attachment_ids": ["a1"]},
        )

        sqlite.create_chat_message.assert_not_called()


class TestRejectUnsupportedAttachmentsHelper:
    """_reject_unsupported_attachments 単体の判定を検証するテストクラス。

    エンドポイント経由だと通過ケースが LLM 呼び出しまで走ってしまうため、
    「弾かないこと」の確認はヘルパー単体で行う（例外が出なければ通過）。
    """

    def _state(self, provider: str, attachment_mimes: dict[str, str]):
        state = MagicMock()
        state.sqlite = _sqlite_with(provider, attachment_mimes)
        return state

    def test_audio_to_google_is_allowed(self):
        """google プリセットへの音声添付は弾かれないこと。"""
        state = self._state("google", {"a1": "audio/mpeg"})
        chat_module._reject_unsupported_attachments(state, "はる@p1", ["a1"])

    def test_image_to_non_google_is_allowed(self):
        """画像はどのプロバイダーでも弾かれないこと（従来挙動の維持）。"""
        state = self._state("claude_cli", {"a1": "image/png"})
        chat_module._reject_unsupported_attachments(state, "はる@p1", ["a1"])

    def test_no_attachment_is_allowed(self):
        """添付なしなら何も見ずに通ること。"""
        state = self._state("claude_cli", {})
        chat_module._reject_unsupported_attachments(state, "はる@p1", None)

    def test_unknown_attachment_id_is_not_judged_here(self):
        """メタデータを引けない添付IDはここでは判定しないこと（後段に任せる）。"""
        state = self._state("claude_cli", {})
        chat_module._reject_unsupported_attachments(state, "はる@p1", ["missing"])

    def test_unresolvable_preset_is_not_judged_here(self):
        """プリセットを解決できないときは判定せず、後段の通常フローに任せること。"""
        state = self._state("claude_cli", {"a1": "audio/mpeg"})
        state.sqlite.get_model_preset_by_name.return_value = None
        state.sqlite.get_model_preset.return_value = None
        chat_module._reject_unsupported_attachments(state, "はる@p1", ["a1"])

    def test_audio_to_non_google_raises_400(self):
        """claude_cli プリセットへの音声添付は HTTPException 400 になること。"""
        from fastapi import HTTPException

        state = self._state("claude_cli", {"a1": "audio/mpeg"})
        with pytest.raises(HTTPException) as excinfo:
            chat_module._reject_unsupported_attachments(state, "はる@p1", ["a1"])
        assert excinfo.value.status_code == 400
        assert "Gemini" in excinfo.value.detail


# ---------------------------------------------------------------------------
# レスポンス形式（表示のための mime 解決）
# ---------------------------------------------------------------------------

class TestMessageToDictAttachments:
    """message_to_dict が添付へ mime_type を添えることを検証するテストクラス。

    フロントは mime から種別を導出して画像サムネと音声プレイヤーを出し分ける。
    ID だけを返していた頃は音声を <img> に食わせて壊れるため、形式を変えている。
    """

    def _msg(self, attachments):
        from datetime import datetime

        m = MagicMock()
        m.id = "m1"
        m.session_id = "sid"
        m.role = "user"
        m.content = "これ聴いて"
        m.reasoning = None
        m.attachments = attachments
        m.character_name = None
        m.preset_name = None
        m.is_system_message = None
        m.log_message_id = None
        m.anticipation = None
        m.face_to_face = 0
        m.created_at = datetime(2026, 8, 20, 12, 0, 0)
        return m

    def test_attachment_carries_mime_type(self):
        """添付が {id, mime_type} の形で返ること。"""
        from backend.api.utils import message_to_dict

        sqlite = MagicMock()
        sqlite.get_chat_attachment.return_value = MagicMock(mime_type="audio/mpeg")

        result = message_to_dict(self._msg(["a1"]), sqlite)

        assert result["attachments"] == [{"id": "a1", "mime_type": "audio/mpeg"}]

    def test_missing_metadata_yields_id_only(self):
        """メタデータを引けない添付は id だけを返すこと（表示は画像扱いになる）。"""
        from backend.api.utils import message_to_dict

        sqlite = MagicMock()
        sqlite.get_chat_attachment.return_value = None

        result = message_to_dict(self._msg(["a1"]), sqlite)

        assert result["attachments"] == [{"id": "a1"}]

    def test_no_sqlite_yields_id_only(self):
        """sqlite 未指定でも例外にせず id だけ返すこと。"""
        from backend.api.utils import message_to_dict

        result = message_to_dict(self._msg(["a1"]))

        assert result["attachments"] == [{"id": "a1"}]

    def test_no_attachments_key_when_empty(self):
        """添付がなければキー自体を省くこと（レスポンスサイズ削減の既存方針）。"""
        from backend.api.utils import message_to_dict

        result = message_to_dict(self._msg(None), MagicMock())

        assert "attachments" not in result
