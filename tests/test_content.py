"""backend.services.chat.content モジュールのユニットテスト。

対象関数:
    apply_context_window() — chronicle済みメッセージ数を制限してコンテキストを圧縮する
    attachment_trace()     — 過去ターンの添付を痕跡テキストへ落とす
    build_1on1_history()   — 履歴を Message リストへ変換する（添付の実体は載せない）
    build_message_content() — 最新ターンの添付をコンテンツパートへ載せる

テスト方針:
    - ChatMessage の代わりに SimpleNamespace で chronicled_at を持つ軽量オブジェクトを使う
    - DB・LLM へのアクセスは発生しないため外部 mock は不要
    - 境界値（0件・全件chronicle済み・全件未chronicle・混在）を網羅する
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from backend.services.chat.content import (
    apply_context_window,
    attachment_trace,
    build_1on1_history,
    build_message_content,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────────────

def _msg(chronicled: bool, label: str = "") -> SimpleNamespace:
    """テスト用メッセージオブジェクトを生成する。

    Args:
        chronicled: True なら chronicled_at に datetime をセット、False なら None。
        label: 識別用のラベル（content 相当）。

    Returns:
        chronicled_at と content を持つ SimpleNamespace。
    """
    return SimpleNamespace(
        chronicled_at=datetime(2026, 1, 1) if chronicled else None,
        content=label,
    )


# ─── apply_context_window ────────────────────────────────────────────────────


class TestApplyContextWindow:
    """apply_context_window() の動作を検証するテストクラス。"""

    def test_empty_history_returns_empty(self):
        """空リストを渡した場合、空リストを返すこと。"""
        assert apply_context_window([]) == []

    def test_all_unchronicled_returns_all(self):
        """chronicle未実行のメッセージはすべて返されること（圧縮なし）。"""
        msgs = [_msg(False, f"m{i}") for i in range(20)]
        result = apply_context_window(msgs, max_chronicled=10)
        assert result == msgs

    def test_all_chronicled_returns_last_n(self):
        """chronicle済みのみの場合、末尾 max_chronicled 件だけ返されること。"""
        msgs = [_msg(True, f"m{i}") for i in range(15)]
        result = apply_context_window(msgs, max_chronicled=5)
        assert result == msgs[-5:]

    def test_mixed_keeps_all_unchronicled_and_last_n_chronicled(self):
        """chronicle済みと未chronicle混在時、未chronicle全件 + chronicle済み末尾N件を返すこと。

        時系列順も保持されることを確認する。
        """
        # chronicle済み: 10件（インデックス 0〜9）
        # 未chronicle: 5件（インデックス 10〜14）
        chronicled = [_msg(True, f"c{i}") for i in range(10)]
        unchronicled = [_msg(False, f"u{i}") for i in range(5)]
        history = chronicled + unchronicled

        result = apply_context_window(history, max_chronicled=3)

        # chronicle済みの末尾3件 + 未chronicle全5件 = 8件
        assert len(result) == 8
        # 時系列順: chronicle済み末尾3件が先に来る
        assert result[0].content == "c7"
        assert result[1].content == "c8"
        assert result[2].content == "c9"
        assert result[3].content == "u0"
        assert result[-1].content == "u4"

    def test_max_chronicled_zero_removes_all_chronicled(self):
        """max_chronicled=0 の場合、chronicle済みメッセージをすべて除去すること。"""
        msgs = [_msg(True, f"c{i}") for i in range(5)] + [_msg(False, "u0")]
        result = apply_context_window(msgs, max_chronicled=0)
        assert len(result) == 1
        assert result[0].content == "u0"

    def test_max_chronicled_larger_than_total_returns_all(self):
        """max_chronicled が chronicle済み件数を超える場合、chronicle済みを全件保持すること。"""
        msgs = [_msg(True, f"c{i}") for i in range(3)] + [_msg(False, "u0")]
        result = apply_context_window(msgs, max_chronicled=100)
        assert result == msgs

    def test_preserves_original_order(self):
        """chronicle済みと未chronicleが交互に並ぶ場合、元の時系列順が保持されること。

        例: [c0, u1, c2, u3] で max_chronicled=1 の場合
        → c0 は除外、u1 は保持、c2 は保持（末尾1件）、u3 は保持
        → [u1, c2, u3] の順
        """
        c0 = _msg(True, "c0")
        u1 = _msg(False, "u1")
        c2 = _msg(True, "c2")
        u3 = _msg(False, "u3")
        history = [c0, u1, c2, u3]

        result = apply_context_window(history, max_chronicled=1)

        assert len(result) == 3
        assert result[0].content == "u1"
        assert result[1].content == "c2"
        assert result[2].content == "u3"

    def test_no_chronicled_at_attribute_treated_as_unchronicled(self):
        """chronicled_at 属性がないオブジェクトは未chronicle扱い（全件保持）になること。

        getattr(..., None) のフォールバック動作を確認する。
        """
        msgs = [SimpleNamespace(content=f"m{i}") for i in range(5)]
        result = apply_context_window(msgs, max_chronicled=2)
        # chronicled_at がなければ全件保持
        assert result == msgs

    def test_exact_boundary_max_chronicled_equals_count(self):
        """chronicle済みの件数と max_chronicled が等しい場合、全件保持されること。"""
        msgs = [_msg(True, f"c{i}") for i in range(5)]
        result = apply_context_window(msgs, max_chronicled=5)
        assert result == msgs


# ─── 添付の寿命（build_1on1_history / attachment_trace） ──────────────────────


class _FakeAttachment:
    """ChatAttachment ORM の代わりに mime_type だけを持つスタブ。"""

    def __init__(self, mime_type: str):
        self.mime_type = mime_type


class _FakeSqlite:
    """get_chat_attachment だけを提供する最小のストアスタブ。

    id → mime_type の辞書を受け取り、未登録IDには None を返す
    （実装が「メタデータを引けない添付」をどう扱うかも検証できるようにする）。
    """

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def get_chat_attachment(self, attachment_id: str):
        mime = self._mapping.get(attachment_id)
        return _FakeAttachment(mime) if mime else None


def _hist_msg(role: str, content: str, attachments=None, is_system=None) -> SimpleNamespace:
    """build_1on1_history 用の履歴メッセージスタブを作る。"""
    return SimpleNamespace(
        role=role,
        content=content,
        attachments=attachments,
        is_system_message=is_system,
    )


class TestAttachmentTrace:
    """attachment_trace() — 過去ターンの添付を痕跡テキストへ落とす変換のテスト。

    実体（base64）を履歴から外しつつ「何かを渡した」事実は残す、という
    添付の寿命ルールの中核。種別ごとの文言と重複のまとめ方を検証する。
    """

    def test_no_attachments_returns_text_unchanged(self):
        """添付がなければ本文をそのまま返すこと。"""
        assert attachment_trace("こんにちは", [], _FakeSqlite({})) == "こんにちは"

    def test_sqlite_missing_returns_text_unchanged(self):
        """sqlite が無ければ種別を引けないため本文をそのまま返すこと。"""
        assert attachment_trace("こんにちは", ["a1"], None) == "こんにちは"

    def test_image_appends_image_trace(self):
        """画像添付は [画像を見せた] を本文の次行へ足すこと。"""
        sqlite = _FakeSqlite({"a1": "image/png"})
        assert attachment_trace("これ見て", ["a1"], sqlite) == "これ見て\n[画像を見せた]"

    def test_audio_appends_audio_trace(self):
        """音声添付は [音声を聴かせた] を本文の次行へ足すこと。"""
        sqlite = _FakeSqlite({"a1": "audio/mpeg"})
        assert attachment_trace("これ聴いて", ["a1"], sqlite) == "これ聴いて\n[音声を聴かせた]"

    def test_same_kind_is_collapsed_into_one_line(self):
        """同種の添付が複数あっても痕跡は1行にまとめること（枚数は残さない）。"""
        sqlite = _FakeSqlite({"a1": "image/png", "a2": "image/jpeg"})
        assert attachment_trace("2枚", ["a1", "a2"], sqlite) == "2枚\n[画像を見せた]"

    def test_mixed_kinds_produce_one_line_each(self):
        """種別が違えばそれぞれ1行ずつ、出現順に並ぶこと。"""
        sqlite = _FakeSqlite({"a1": "audio/mpeg", "a2": "image/png"})
        result = attachment_trace("両方", ["a1", "a2"], sqlite)
        assert result == "両方\n[音声を聴かせた]\n[画像を見せた]"

    def test_unknown_mime_falls_back_to_generic_trace(self):
        """種別を導出できない MIME は汎用の痕跡になること（旧レコード対策）。"""
        sqlite = _FakeSqlite({"a1": "application/pdf"})
        assert attachment_trace("なにか", ["a1"], sqlite) == "なにか\n[ファイルを渡した]"

    def test_missing_metadata_falls_back_to_generic_trace(self):
        """DB にメタデータが無い添付IDも汎用の痕跡へ落ちること（黙って消さない）。"""
        assert attachment_trace("なにか", ["missing"], _FakeSqlite({})) == "なにか\n[ファイルを渡した]"

    def test_empty_text_yields_trace_only(self):
        """本文が空なら痕跡だけを返すこと（先頭の空行を作らない）。"""
        sqlite = _FakeSqlite({"a1": "audio/mpeg"})
        assert attachment_trace("", ["a1"], sqlite) == "[音声を聴かせた]"


class TestBuild1on1HistoryAttachments:
    """build_1on1_history() が履歴から添付の実体を落とすことを検証するテストクラス。

    ここを通る添付はすべて過去ターンのもの（最新ターンは呼び出し側が
    build_message_content で別に組む）。したがって content は必ず文字列になり、
    base64 を含むコンテンツパートのリストにはならない。
    """

    def test_history_attachment_becomes_text_trace(self):
        """履歴のユーザ添付は痕跡テキストへ置換され、content が文字列になること。"""
        sqlite = _FakeSqlite({"a1": "audio/mpeg"})
        history = [_hist_msg("user", "これ聴いて", ["a1"])]
        messages = build_1on1_history(history, sqlite, "/tmp/uploads")
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "これ聴いて\n[音声を聴かせた]"

    def test_history_without_attachment_is_plain_text(self):
        """添付のないユーザ発話は本文がそのまま渡ること。"""
        history = [_hist_msg("user", "ふつうの発話", None)]
        messages = build_1on1_history(history, _FakeSqlite({}), "/tmp/uploads")
        assert messages[0].content == "ふつうの発話"

    def test_character_role_maps_to_assistant(self):
        """character ロールは API 仕様上の role="assistant" へ写ること。"""
        history = [_hist_msg("character", "うん")]
        messages = build_1on1_history(history, _FakeSqlite({}), "/tmp/uploads")
        assert messages[0].role == "assistant"
        assert messages[0].content == "うん"

    def test_system_message_is_skipped(self):
        """システムメッセージ（退席通知等）は履歴に載らないこと。"""
        history = [
            _hist_msg("character", "掲示", is_system=True),
            _hist_msg("user", "やあ"),
        ]
        messages = build_1on1_history(history, _FakeSqlite({}), "/tmp/uploads")
        assert len(messages) == 1
        assert messages[0].content == "やあ"

    def test_no_base64_leaks_into_history(self):
        """複数ターン分の添付があっても、履歴のどこにも実体（リスト形式）が残らないこと。"""
        sqlite = _FakeSqlite({"a1": "image/png", "a2": "audio/mpeg"})
        history = [
            _hist_msg("user", "1枚目", ["a1"]),
            _hist_msg("character", "見たよ"),
            _hist_msg("user", "次は曲", ["a2"]),
        ]
        messages = build_1on1_history(history, sqlite, "/tmp/uploads")
        assert all(isinstance(m.content, str) for m in messages)


# ─── 添付パートの出し分け（build_message_content） ────────────────────────────


class TestBuildMessageContent:
    """build_message_content() — 最新ターンの添付をコンテンツパートへ載せる変換のテスト。

    mime から導出した種別で形式を出し分ける。画像は OpenAI vision の image_url
    （data URL）、音声は OpenAI 準拠の input_audio。どちらもプロバイダー非依存の
    内部表現であり、各プロバイダーがここから自分の形式へ載せ替える。
    """

    def _uploads(self, tmp_path, files: dict[str, bytes]) -> str:
        """uploads_dir を模した一時ディレクトリへ添付実体を置く。"""
        for att_id, data in files.items():
            (tmp_path / att_id).write_bytes(data)
        return str(tmp_path)

    def test_no_attachments_returns_plain_text(self):
        """添付がなければ文字列をそのまま返すこと。"""
        assert build_message_content("やあ", [], _FakeSqlite({}), "/tmp") == "やあ"

    def test_missing_sqlite_returns_plain_text(self):
        """sqlite 未指定ならメタデータを引けないため文字列を返すこと。"""
        assert build_message_content("やあ", ["a1"], None, "/tmp") == "やあ"

    def test_image_becomes_image_url_part(self, tmp_path):
        """画像は data URL 形式の image_url パートになること。"""
        uploads = self._uploads(tmp_path, {"a1": b"ABC"})
        sqlite = _FakeSqlite({"a1": "image/png"})
        result = build_message_content("これ見て", ["a1"], sqlite, uploads)
        assert result[0] == {"type": "text", "text": "これ見て"}
        assert result[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,QUJD"},
        }

    def test_audio_becomes_input_audio_part(self, tmp_path):
        """音声は OpenAI 準拠の input_audio パートになり、format が mime から導出されること。"""
        uploads = self._uploads(tmp_path, {"a1": b"ABC"})
        sqlite = _FakeSqlite({"a1": "audio/mpeg"})
        result = build_message_content("これ聴いて", ["a1"], sqlite, uploads)
        assert result[1] == {
            "type": "input_audio",
            "input_audio": {"data": "QUJD", "format": "mp3"},
        }

    def test_mixed_attachments_keep_order(self, tmp_path):
        """画像と音声が混在しても、指定された順にパートが並ぶこと。"""
        uploads = self._uploads(tmp_path, {"a1": b"ABC", "a2": b"ABC"})
        sqlite = _FakeSqlite({"a1": "audio/wav", "a2": "image/jpeg"})
        result = build_message_content("両方", ["a1", "a2"], sqlite, uploads)
        assert [p["type"] for p in result] == ["text", "input_audio", "image_url"]
        assert result[1]["input_audio"]["format"] == "wav"

    def test_unknown_mime_is_dropped_and_falls_back_to_text(self, tmp_path):
        """扱えない MIME はパート化せず、結果が本文だけならテキストへ戻ること。

        アップロードAPIで弾いているので通常は届かないが、旧レコード等が
        混ざっても不正なパートを LLM へ流さない。
        """
        uploads = self._uploads(tmp_path, {"a1": b"ABC"})
        sqlite = _FakeSqlite({"a1": "application/pdf"})
        assert build_message_content("なにか", ["a1"], sqlite, uploads) == "なにか"

    def test_missing_file_is_skipped(self, tmp_path):
        """実体ファイルが消えている添付は飛ばすこと（例外を投げない）。"""
        sqlite = _FakeSqlite({"a1": "image/png"})
        assert build_message_content("消えた", ["a1"], sqlite, str(tmp_path)) == "消えた"
