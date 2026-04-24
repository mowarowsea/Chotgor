"""chat/indexer.py のユニットテスト。

get_participant_char_ids と index_message_sync の動作を検証する。
SQLiteStore は実インメモリDB、ChromaStore はモックを使用する。

テスト対象:
  - get_participant_char_ids: 1on1 / グループ / 存在しないキャラの各ケース
  - index_message_sync: ユーザ発言・キャラ発言・システムメッセージ・ファンアウトの各ケース
  - リトライ機構: ChromaDB失敗時のリトライ動作とエラーログ記録
"""

import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from backend.services.chat.indexer import (
    get_participant_char_ids,
    index_message_sync,
    _CHAT_INDEX_MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# ヘルパー: ChatSession / ChatMessage の簡易スタブ
# ---------------------------------------------------------------------------

def _make_1on1_session(char_name: str, preset_name: str = "default") -> MagicMock:
    """1on1セッションのスタブを作成する。

    Args:
        char_name: キャラクター名。
        preset_name: プリセット名。

    Returns:
        session_type="1on1"、model_id="{char_name}@{preset_name}" のMagicMock。
    """
    s = MagicMock()
    s.session_type = "1on1"
    s.model_id = f"{char_name}@{preset_name}"
    return s


def _make_group_session(participants: list[dict], director_char_name: str) -> MagicMock:
    """グループセッションのスタブを作成する。

    Args:
        participants: [{"char_name": str, "preset_id": str}] のリスト。
        director_char_name: 司会キャラクター名。

    Returns:
        session_type="group"、group_config が JSON化されたMagicMock。
    """
    s = MagicMock()
    s.session_type = "group"
    s.group_config = json.dumps({
        "participants": participants,
        "director_char_name": director_char_name,
    })
    return s


def _make_message(
    *,
    role: str = "user",
    content: str = "こんにちは",
    character_name: str | None = None,
    is_system_message: int | None = None,
    session_id: str = "session-001",
    message_id: str | None = None,
) -> MagicMock:
    """ChatMessage ORM のスタブを作成する。

    Args:
        role: "user" または "character"。
        content: 発言テキスト。
        character_name: キャラクター名（role="character" 時に使用）。
        is_system_message: 1 のときシステムメッセージとして扱われる。
        session_id: セッションID。
        message_id: メッセージID（省略時はUUID自動生成）。

    Returns:
        各フィールドが設定された ChatMessage スタブ。
    """
    m = MagicMock()
    m.id = message_id or str(uuid.uuid4())
    m.session_id = session_id
    m.role = role
    m.content = content
    m.character_name = character_name
    m.is_system_message = is_system_message
    m.created_at = datetime(2026, 3, 23, 12, 0, 0)
    return m


# ---------------------------------------------------------------------------
# get_participant_char_ids のテスト
# ---------------------------------------------------------------------------

class TestGetParticipantCharIds:
    """get_participant_char_ids のユニットテスト。"""

    def test_1on1_resolves_character_id(self, sqlite_store):
        """1on1セッションでキャラ名からIDが正しく解決されることを確認する。

        model_id の "@" より前の部分がキャラ名として解釈され、
        SQLite から対応する character_id が返されることを検証する。
        """
        char_id = str(uuid.uuid4())
        sqlite_store.create_character(char_id, "Alice")
        session = _make_1on1_session("Alice")

        result = get_participant_char_ids(session, sqlite_store)

        assert result == [char_id]

    def test_1on1_unknown_char_returns_empty(self, sqlite_store):
        """1on1セッションで存在しないキャラ名の場合、空リストを返すことを確認する。

        DB に登録されていないキャラ名でも例外が発生せず、
        空リストで安全に終了することを検証する。
        """
        session = _make_1on1_session("存在しないキャラ")

        result = get_participant_char_ids(session, sqlite_store)

        assert result == []

    def test_group_resolves_all_participant_ids(self, sqlite_store):
        """グループセッションで全参加キャラのIDが解決されることを確認する。

        participants に含まれる全キャラと director を含む一意のIDリストが
        返されることを検証する。director が participants と重複する場合は1件のみ。
        """
        id_alice = str(uuid.uuid4())
        id_bob = str(uuid.uuid4())
        sqlite_store.create_character(id_alice, "Alice")
        sqlite_store.create_character(id_bob, "Bob")

        session = _make_group_session(
            participants=[
                {"char_name": "Alice", "preset_id": "p1"},
                {"char_name": "Bob",   "preset_id": "p2"},
            ],
            director_char_name="Alice",  # 重複
        )

        result = get_participant_char_ids(session, sqlite_store)

        # Alice は participants と director で重複するが1件のみ
        assert sorted(result) == sorted([id_alice, id_bob])

    def test_group_excludes_unknown_chars(self, sqlite_store):
        """グループセッションで存在しないキャラは除外されることを確認する。

        participants に存在しないキャラ名が含まれていても例外が発生せず、
        解決できたキャラのIDのみが返されることを検証する。
        """
        id_alice = str(uuid.uuid4())
        sqlite_store.create_character(id_alice, "Alice")

        session = _make_group_session(
            participants=[
                {"char_name": "Alice", "preset_id": "p1"},
                {"char_name": "NotExist", "preset_id": "p2"},
            ],
            director_char_name="Alice",
        )

        result = get_participant_char_ids(session, sqlite_store)

        assert result == [id_alice]

    def test_group_invalid_group_config_returns_empty(self):
        """group_config が不正な JSON の場合、空リストを返すことを確認する。

        JSON パース失敗時も例外を投げず、空リストで安全に終了することを検証する。
        """
        s = MagicMock()
        s.session_type = "group"
        s.group_config = "not-json"
        sqlite = MagicMock()

        result = get_participant_char_ids(s, sqlite)

        assert result == []


# ---------------------------------------------------------------------------
# index_message_sync のテスト
# ---------------------------------------------------------------------------

class TestIndexMessageSync:
    """index_message_sync のユニットテスト。"""

    def test_user_message_indexed_with_user_name(self):
        """ユーザ発言が "{user_name}: {content}" 形式でインデックス登録されることを確認する。

        role="user" のとき speaker_name として user_name 引数が使われ、
        chroma.add_chat_turn が正しいドキュメントで呼ばれることを検証する。
        """
        chroma = MagicMock()
        char_id = "char-001"
        msg = _make_message(role="user", content="こんにちは", session_id="s-1")

        index_message_sync(msg, [char_id], chroma, user_name="テストユーザ")

        chroma.add_chat_turn.assert_called_once()
        kwargs = chroma.add_chat_turn.call_args.kwargs
        assert kwargs["message_id"] == msg.id
        assert kwargs["content"] == "テストユーザ: こんにちは"
        assert kwargs["character_id"] == char_id
        assert kwargs["metadata"]["role"] == "user"
        assert kwargs["metadata"]["speaker_name"] == "テストユーザ"
        assert kwargs["metadata"]["session_id"] == "s-1"

    def test_character_message_indexed_with_character_name(self):
        """キャラ発言が "{character_name}: {content}" 形式でインデックス登録されることを確認する。

        role="character" のとき character_name 属性が speaker_name として使われ、
        chroma.add_chat_turn が正しいドキュメントで呼ばれることを検証する。
        """
        chroma = MagicMock()
        char_id = "char-001"
        msg = _make_message(role="character", content="はい！", character_name="Alice")

        index_message_sync(msg, [char_id], chroma)

        kwargs = chroma.add_chat_turn.call_args.kwargs
        assert kwargs["content"] == "Alice: はい！"
        assert kwargs["metadata"]["speaker_name"] == "Alice"
        assert kwargs["metadata"]["role"] == "character"

    def test_character_message_fallback_speaker_name(self):
        """character_name が None の場合、"キャラクター" がフォールバック使用されることを確認する。

        character_name 未設定時（旧データ等）でも "キャラクター" として
        インデックス登録が行われることを検証する。
        """
        chroma = MagicMock()
        msg = _make_message(role="character", content="ふむ", character_name=None)

        index_message_sync(msg, ["char-001"], chroma)

        kwargs = chroma.add_chat_turn.call_args.kwargs
        assert kwargs["content"] == "キャラクター: ふむ"

    def test_system_message_is_skipped(self):
        """is_system_message=1 のメッセージはインデックス登録されないことを確認する。

        退席通知などのシステムメッセージは検索対象外とするため、
        chroma.add_chat_turn が一切呼ばれないことを検証する。
        """
        chroma = MagicMock()
        msg = _make_message(role="character", content="Aliceは退席しました。", is_system_message=1)

        index_message_sync(msg, ["char-001"], chroma)

        chroma.add_chat_turn.assert_not_called()

    def test_fanout_writes_to_all_character_collections(self):
        """複数キャラが参加するセッションで全員のコレクションに書き込まれることを確認する。

        character_ids に3件渡したとき、chroma.add_chat_turn が3回呼ばれ、
        それぞれ異なる character_id で呼ばれることを検証する（ファンアウト）。
        """
        chroma = MagicMock()
        char_ids = ["char-A", "char-B", "char-C"]
        msg = _make_message(role="user", content="みんなこんにちは")

        index_message_sync(msg, char_ids, chroma, user_name="ユーザ")

        assert chroma.add_chat_turn.call_count == 3
        called_char_ids = [
            c.kwargs["character_id"] for c in chroma.add_chat_turn.call_args_list
        ]
        assert sorted(called_char_ids) == sorted(char_ids)

    def test_chroma_error_does_not_propagate(self):
        """chroma.add_chat_turn が例外を投げてもエラーが伝播しないことを確認する。

        ChromaDB の障害がチャット本体に影響しないよう、
        リトライを経た後も例外を外部に漏らさず正常終了することを検証する。
        time.sleep をパッチしてリトライ待機時間をスキップする。
        """
        chroma = MagicMock()
        chroma.add_chat_turn.side_effect = Exception("ChromaDB接続エラー")
        msg = _make_message(role="user", content="テスト")

        # リトライのsleepをパッチしてテストを高速化する
        with patch("backend.services.chat.indexer.time.sleep"):
            # 例外が発生しないこと
            index_message_sync(msg, ["char-001"], chroma)

    def test_empty_character_ids_does_nothing(self):
        """character_ids が空のとき、chroma.add_chat_turn が呼ばれないことを確認する。

        参加キャラが解決できなかったセッションでも安全に処理を終了することを検証する。
        """
        chroma = MagicMock()
        msg = _make_message(role="user", content="テスト")

        index_message_sync(msg, [], chroma)

        chroma.add_chat_turn.assert_not_called()

    def test_metadata_contains_created_at(self):
        """メタデータに created_at が ISO文字列として含まれることを確認する。

        ChromaDB のメタデータに保存される created_at が
        "YYYY-MM-DDTHH:MM:SS" 形式の文字列であることを検証する。
        """
        chroma = MagicMock()
        msg = _make_message(role="user", content="テスト")

        index_message_sync(msg, ["char-001"], chroma)

        metadata = chroma.add_chat_turn.call_args.kwargs["metadata"]
        assert "created_at" in metadata
        # ISO形式の文字列であること
        datetime.fromisoformat(metadata["created_at"])


# ---------------------------------------------------------------------------
# リトライ機構のテスト
# ---------------------------------------------------------------------------

class TestIndexMessageSyncRetry:
    """index_message_sync のリトライ機構ユニットテスト。

    ChromaDB書き込み失敗時に30秒ごとに最大5回リトライし、
    最終失敗時にエラーログを記録することを検証する。
    time.sleep はパッチして実テスト時間を短縮する。
    """

    def test_retries_on_failure_then_succeeds(self):
        """ChromaDB失敗後にリトライして最終的に成功するケースを確認する。

        2回目の試行で成功した場合、add_chat_turn が計2回呼ばれ
        例外が外部に漏れないことを検証する。
        """
        chroma = MagicMock()
        chroma.add_chat_turn.side_effect = [Exception("初回失敗"), None]
        msg = _make_message(role="user", content="テスト")

        with patch("backend.services.chat.indexer.time.sleep") as mock_sleep:
            index_message_sync(msg, ["char-001"], chroma)

        assert chroma.add_chat_turn.call_count == 2
        mock_sleep.assert_called_once()

    def test_logs_error_on_final_failure(self):
        """ChromaDB書き込みが最大リトライ回数すべて失敗した場合にエラーログが記録されることを確認する。

        _CHAT_INDEX_MAX_RETRIES 回すべて失敗したとき logger.error が呼ばれ、
        例外は外部に漏れず、チャット本体（SQLite）への影響がないことを検証する。
        """
        chroma = MagicMock()
        chroma.add_chat_turn.side_effect = Exception("永続的なChromaDB障害")
        msg = _make_message(role="user", content="テスト")

        with patch("backend.services.chat.indexer.time.sleep"):
            with patch("backend.services.chat.indexer.logger") as mock_logger:
                index_message_sync(msg, ["char-001"], chroma)

        assert mock_logger.error.called
        error_args = mock_logger.error.call_args[0]
        assert "最終失敗" in error_args[0]
        assert chroma.add_chat_turn.call_count == _CHAT_INDEX_MAX_RETRIES

    def test_no_retry_on_success(self):
        """初回成功時にリトライが行われないことを確認する。

        add_chat_turn が1回で成功した場合、time.sleep が呼ばれず
        呼び出し回数も1回であることを検証する。
        """
        chroma = MagicMock()
        msg = _make_message(role="user", content="テスト")

        with patch("backend.services.chat.indexer.time.sleep") as mock_sleep:
            index_message_sync(msg, ["char-001"], chroma)

        mock_sleep.assert_not_called()
        assert chroma.add_chat_turn.call_count == 1

    def test_each_char_id_retried_independently(self):
        """複数キャラが存在する場合、各キャラIDに対して独立してリトライが行われることを確認する。

        char-001 が失敗し続けても char-002 のインデックスが影響を受けないことを検証する。
        char-001 は最大試行後にエラーログ、char-002 は1回で成功することを確認する。
        """
        chroma = MagicMock()
        # char-001 は常に失敗、char-002 は成功
        def side_effect(**kwargs):
            if kwargs.get("character_id") == "char-001":
                raise Exception("char-001のChromaDB障害")
        chroma.add_chat_turn.side_effect = side_effect
        msg = _make_message(role="user", content="テスト")

        with patch("backend.services.chat.indexer.time.sleep"):
            with patch("backend.services.chat.indexer.logger") as mock_logger:
                index_message_sync(msg, ["char-001", "char-002"], chroma)

        # char-001 は最大試行 → エラーログ
        assert mock_logger.error.called
        # char-002 は成功（合計呼び出し回数 = char-001分 + char-002の1回）
        expected_calls = _CHAT_INDEX_MAX_RETRIES + 1
        assert chroma.add_chat_turn.call_count == expected_calls
