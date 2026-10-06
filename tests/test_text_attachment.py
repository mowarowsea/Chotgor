"""テキスト添付の受け入れ判定とアップロード API のテスト。

テキスト添付は画像・音声と同じ添付の仕組みに乗るが、ブラウザの MIME が当てにならない
（Windows で .md が空 MIME、.csv が application/vnd.ms-excel、.ts が video/mp2t）ため、
アップロード時に拡張子から text/plain へ正規化して保存する。以後の種別判定は
保存済み MIME だけで完結する。設計は docs/planned/text_attachment_plan.md。

ここで検証するのは:
    - attachment_kind が text/* とテキスト系 application/* を "text" と判定すること
    - resolve_upload_mime の正規化規則（MIME 優先・拡張子は救済のみ）
    - decode_text の文字コード救済（UTF-8 BOM / CP932）
    - アップロード API が正規化 MIME とファイル名を保存し、1 MB 超を 400 で弾くこと
    - 配信 API がテキストに charset=utf-8 を付けること
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import chat_attachments as attachments_module
from backend.lib.attachments import (
    MAX_TEXT_ATTACHMENT_BYTES,
    attachment_kind,
    decode_text,
    resolve_upload_mime,
)


class TestAttachmentKindText:
    """attachment_kind() のテキスト判定を検証するテストクラス。

    "text" 判定はアップロード可否・プロバイダー適合・コンテンツパート構築の
    すべての根拠になるため、受け入れる MIME の範囲を固定しておく。
    """

    @pytest.mark.parametrize(
        "mime",
        ["text/plain", "text/markdown", "text/csv", "text/plain; charset=utf-8",
         "application/json", "application/x-yaml", "application/xml"],
    )
    def test_text_like_mimes_are_text(self, mime):
        """text/* とテキスト系 application/* は "text" になること。"""
        assert attachment_kind(mime) == "text"

    @pytest.mark.parametrize("mime", ["application/pdf", "application/octet-stream", "", None])
    def test_binary_or_unknown_is_rejected(self, mime):
        """バイナリ・不明な MIME は扱わないこと（PDF 等は将来枠）。"""
        assert attachment_kind(mime) is None

    def test_image_and_audio_are_unchanged(self):
        """既存の画像・音声判定を壊していないこと。"""
        assert attachment_kind("image/png") == "image"
        assert attachment_kind("audio/mpeg") == "audio"


class TestResolveUploadMime:
    """resolve_upload_mime() — アップロード時の MIME 正規化を検証するテストクラス。

    拡張子はあくまで「MIME から種別が出ないときの救済」。MIME で判定できるものを
    拡張子で上書きしてはならない（.svg を image/svg+xml のまま画像として扱う等）。
    """

    @pytest.mark.parametrize(
        "content_type,filename",
        [("", "note.md"), ("application/octet-stream", "main.py"),
         ("application/vnd.ms-excel", "data.csv"), ("video/mp2t", "app.ts")],
    )
    def test_unreliable_mime_is_normalized_by_extension(self, content_type, filename):
        """当てにならない MIME でも、テキスト系拡張子なら text/plain に正規化されること。"""
        assert resolve_upload_mime(content_type, filename) == "text/plain"

    def test_extension_is_case_insensitive(self):
        """拡張子の大文字小文字は区別しないこと。"""
        assert resolve_upload_mime("", "README.MD") == "text/plain"

    def test_recognized_mime_is_kept(self):
        """MIME から種別が出るものはそのまま保存すること（拡張子で上書きしない）。"""
        assert resolve_upload_mime("image/svg+xml", "icon.svg") == "image/svg+xml"
        assert resolve_upload_mime("text/markdown", "a.md") == "text/markdown"

    def test_unknown_extension_keeps_original_mime(self):
        """テキスト系でない拡張子は元の MIME のまま（＝後段で 400 になる）。"""
        assert resolve_upload_mime("application/pdf", "doc.pdf") == "application/pdf"

    def test_missing_content_type_and_filename(self):
        """MIME もファイル名も無ければ空文字（例外にしない）。"""
        assert resolve_upload_mime(None, None) == ""


class TestDecodeText:
    """decode_text() の文字コード救済を検証するテストクラス。

    Windows 由来のメモは Shift_JIS（CP932）のことがあるため、UTF-8 で読めなければ
    CP932 を試す。どちらでも読めなければ置換文字で読み、例外にはしない。
    """

    def test_utf8(self):
        """UTF-8 はそのまま読めること。"""
        assert decode_text("はる".encode("utf-8")) == "はる"

    def test_utf8_bom_is_stripped(self):
        """UTF-8 の BOM は落とすこと。"""
        assert decode_text("\ufeffはる".encode("utf-8")) == "はる"

    def test_cp932_fallback(self):
        """UTF-8 で読めない CP932 は CP932 として読むこと。"""
        assert decode_text("はる".encode("cp932")) == "はる"


def _make_app(tmp_path) -> tuple[FastAPI, MagicMock]:
    """モック sqlite と一時 uploads_dir を持つ最小アプリを作る。"""
    app = FastAPI()
    app.include_router(attachments_module.router)
    sqlite = MagicMock()
    sqlite.get_chat_session.return_value = MagicMock(id="sid")
    app.state.sqlite = sqlite
    app.state.uploads_dir = str(tmp_path)
    return app, sqlite


class TestUploadTextAttachment:
    """アップロード API（POST /sessions/{id}/attachments）のテキスト受け入れを検証するテストクラス。

    受け入れ可否の正はバックエンド。フロントの拡張子判定をすり抜けても、
    ここで正規化・上限・ファイル名保存が効くことを確認する。
    """

    def test_markdown_with_empty_mime_is_saved_as_text(self, tmp_path):
        """空 MIME の .md が text/plain として保存され、ファイル名も記録されること。"""
        app, sqlite = _make_app(tmp_path)
        client = TestClient(app)

        res = client.post(
            "/api/chat/sessions/sid/attachments",
            files={"files": ("note.md", "# 見出し".encode("utf-8"), "application/octet-stream")},
        )

        assert res.status_code == 201
        body = res.json()[0]
        assert body["kind"] == "text"
        assert body["mime_type"] == "text/plain"
        assert body["filename"] == "note.md"
        kwargs = sqlite.create_chat_attachment.call_args.kwargs
        assert kwargs["mime_type"] == "text/plain"
        assert kwargs["filename"] == "note.md"
        assert (tmp_path / body["id"]).read_bytes() == "# 見出し".encode("utf-8")

    def test_oversized_text_is_rejected(self, tmp_path):
        """1 MB を超えるテキストは 400 で弾き、保存もしないこと。"""
        app, sqlite = _make_app(tmp_path)
        client = TestClient(app)

        res = client.post(
            "/api/chat/sessions/sid/attachments",
            files={"files": ("big.txt", b"a" * (MAX_TEXT_ATTACHMENT_BYTES + 1), "text/plain")},
        )

        assert res.status_code == 400
        assert "大きすぎます" in res.json()["detail"]
        sqlite.create_chat_attachment.assert_not_called()
        assert list(tmp_path.iterdir()) == []

    def test_text_at_limit_is_accepted(self, tmp_path):
        """ちょうど上限のテキストは受け入れること（境界値）。"""
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        res = client.post(
            "/api/chat/sessions/sid/attachments",
            files={"files": ("edge.txt", b"a" * MAX_TEXT_ATTACHMENT_BYTES, "text/plain")},
        )

        assert res.status_code == 201

    def test_unknown_binary_is_still_rejected(self, tmp_path):
        """テキストでも画像・音声でもないファイルは従来どおり 400 になること。"""
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        res = client.post(
            "/api/chat/sessions/sid/attachments",
            files={"files": ("doc.pdf", b"%PDF", "application/pdf")},
        )

        assert res.status_code == 400


class TestServeTextAttachment:
    """配信 API（GET /attachments/{id}）のテキスト配信を検証するテストクラス。

    charset が無いとブラウザが日本語を化かすため、テキストには utf-8 を付ける。
    """

    def test_text_is_served_with_utf8_charset(self, tmp_path):
        """テキスト添付は charset=utf-8 付きで返ること。"""
        app, sqlite = _make_app(tmp_path)
        (tmp_path / "a1").write_bytes("はる".encode("utf-8"))
        sqlite.get_chat_attachment.return_value = MagicMock(mime_type="text/plain")
        client = TestClient(app)

        res = client.get("/api/chat/attachments/a1")

        assert res.status_code == 200
        assert "charset=utf-8" in res.headers["content-type"]
        assert res.content.decode("utf-8") == "はる"

    def test_image_content_type_is_untouched(self, tmp_path):
        """画像の Content-Type には charset を付けないこと。"""
        app, sqlite = _make_app(tmp_path)
        (tmp_path / "a1").write_bytes(b"PNG")
        sqlite.get_chat_attachment.return_value = MagicMock(mime_type="image/png")
        client = TestClient(app)

        res = client.get("/api/chat/attachments/a1")

        assert res.headers["content-type"] == "image/png"
