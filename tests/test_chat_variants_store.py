"""1on1 レスポンスガチャ（chat_message_variants）のストアテスト。

docs/planned/chat_turn_variants_plan.md。実 SQLite（conftest の一時 DB）で、
退避・復元・枝番号・封筒・後始末を検証する。

created_at の並びを確実にするため、メッセージ作成の間に短い sleep を挟む
（Windows の時刻分解能では連続 INSERT が同時刻になりうる）。
"""

import time
import uuid

from backend.repositories.sqlite.models import ChatMessageVariant


def _setup(sqlite_store):
    """キャラ・セッション・「u1 → c1」を作って (char_id, sid) を返す。"""
    char_id = str(uuid.uuid4())
    sqlite_store.create_character(character_id=char_id, name="はるテスト")
    sid = str(uuid.uuid4())
    sqlite_store.create_chat_session(session_id=sid, model_id="はるテスト@default")
    _msg(sqlite_store, sid, "u1", "user", "こんにちは")
    _msg(sqlite_store, sid, "c1", "character", "応答その1", reasoning="考えたこと1")
    return char_id, sid


def _msg(sqlite_store, sid, mid, role, content, **kwargs):
    time.sleep(0.02)
    return sqlite_store.create_chat_message(
        message_id=mid, session_id=sid, role=role, content=content, **kwargs
    )


def _ids(sqlite_store, sid):
    return [m.id for m in sqlite_store.list_chat_messages(sid)]


def _regenerate(sqlite_store, sid, new_id, content):
    """引き直し確定の流れ（旧応答を退避 → 新応答を保存）を再現する。"""
    archived = sqlite_store.archive_chat_messages_as_variant(sid, "u1")
    _msg(sqlite_store, sid, new_id, "character", content)
    return archived


class TestArchive:
    """引き直し確定時の退避（archive_chat_messages_as_variant）。

    旧応答は chat_messages から消えて退避テーブルへ移る。chat_messages を読む
    既存経路（クロニクル・うつつ・計器など）から見えなくなることが、
    別テーブル方式を採った理由そのもの。
    """

    def test_moves_tail_out_of_main_line(self, sqlite_store):
        """最新ターンの応答が本線から消え、中身ごと退避テーブルに残ること。"""
        _cid, sid = _setup(sqlite_store)

        archived = sqlite_store.archive_chat_messages_as_variant(sid, "u1")

        assert archived == ["c1"]
        assert _ids(sqlite_store, sid) == ["u1"]
        with sqlite_store.get_session() as s:
            v = s.get(ChatMessageVariant, "c1")
            assert v.content == "応答その1"
            assert v.reasoning == "考えたこと1"
            assert v.anchor_message_id == "u1"

    def test_refuses_when_anchor_is_not_latest_user_message(self, sqlite_store):
        """起点の後ろにユーザ発話がある（過去ターン）なら退避せず None を返すこと。

        呼び出し側はこれを見て従来の物理削除へ切り替える。
        """
        _cid, sid = _setup(sqlite_store)
        _msg(sqlite_store, sid, "u2", "user", "次の話")

        assert sqlite_store.archive_chat_messages_as_variant(sid, "u1") is None
        assert _ids(sqlite_store, sid) == ["u1", "c1", "u2"]

    def test_retracts_envelope(self, sqlite_store):
        """退避した応答の封筒は retracted になること（削除時と同じ扱い）。"""
        char_id, sid = _setup(sqlite_store)

        sqlite_store.archive_chat_messages_as_variant(sid, "u1")

        visible = sqlite_store.list_timeline_events(char_id)
        assert [e.source_id for e in visible] == ["u1"]


class TestVariantInfo:
    """枝ナビ情報（get_chat_variant_info）。

    並びは各グループ先頭行の created_at 順で、往復しても番号が変わらないこと。
    """

    def test_none_without_siblings(self, sqlite_store):
        """引き直していないターンには枝ナビ情報がないこと。"""
        _cid, sid = _setup(sqlite_store)
        assert sqlite_store.get_chat_variant_info(sid, "u1") is None

    def test_counts_archived_and_active(self, sqlite_store):
        """2回引き直すと 3/3（最新が本線）になること。"""
        _cid, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")
        _regenerate(sqlite_store, sid, "c3", "応答その3")

        assert sqlite_store.get_chat_variant_info(sid, "u1") == {"index": 3, "count": 3}

    def test_none_after_next_user_message(self, sqlite_store):
        """次のユーザ発話が積まれたら、前のターンの枝ナビは消えること（最新ターン限定）。"""
        _cid, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")
        _msg(sqlite_store, sid, "u2", "user", "次の話")

        assert sqlite_store.get_chat_variant_info(sid, "u1") is None


class TestActivate:
    """枝の切替（activate_chat_variant）。

    本線の応答を退避し、選んだ応答を同じ ID のまま戻す。戻した応答には新しい
    封筒を積み、retract の取り消しはしない（めぐりの不可逆性）。
    """

    def test_restores_selected_variant(self, sqlite_store):
        """1番目を選ぶと元の応答が同じ ID・中身で本線に戻り、今の応答は退避されること。"""
        _cid, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")

        archived_ids, restored = sqlite_store.activate_chat_variant(sid, "u1", 1)

        assert archived_ids == ["c2"]
        assert [m.id for m in restored] == ["c1"]
        msgs = sqlite_store.list_chat_messages(sid)
        assert [m.id for m in msgs] == ["u1", "c1"]
        assert msgs[1].content == "応答その1"
        assert msgs[1].reasoning == "考えたこと1"
        assert sqlite_store.get_chat_variant_info(sid, "u1") == {"index": 1, "count": 2}

    def test_round_trip_keeps_numbering(self, sqlite_store):
        """1 → 2 と往復しても、枝の番号と中身の対応が変わらないこと。"""
        _cid, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")

        sqlite_store.activate_chat_variant(sid, "u1", 1)
        sqlite_store.activate_chat_variant(sid, "u1", 2)

        msgs = sqlite_store.list_chat_messages(sid)
        assert [m.content for m in msgs] == ["こんにちは", "応答その2"]
        assert sqlite_store.get_chat_variant_info(sid, "u1") == {"index": 2, "count": 2}

    def test_active_variant_is_noop(self, sqlite_store):
        """本線にいる枝を指定したら何も動かないこと。"""
        _cid, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")

        archived_ids, current = sqlite_store.activate_chat_variant(sid, "u1", 2)

        assert archived_ids == []
        assert [m.id for m in current] == ["c2"]

    def test_out_of_range_returns_none(self, sqlite_store):
        """範囲外の枝番号は None を返すこと。"""
        _cid, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")

        assert sqlite_store.activate_chat_variant(sid, "u1", 3) is None
        assert sqlite_store.activate_chat_variant(sid, "u1", 0) is None

    def test_restored_message_gets_new_envelope(self, sqlite_store):
        """戻した応答には新しい封筒が積まれ、古い retracted 封筒はそのまま残ること。"""
        char_id, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")

        sqlite_store.activate_chat_variant(sid, "u1", 1)

        visible = sqlite_store.list_timeline_events(char_id)
        assert [e.source_id for e in visible] == ["u1", "c1"]
        c1_all = [
            e for e in sqlite_store.list_timeline_events(char_id, include_retracted=True)
            if e.source_id == "c1"
        ]
        assert len(c1_all) == 2
        assert sum(e.retracted_at is not None for e in c1_all) == 1


class TestCleanup:
    """退避応答の後始末。

    ユーザ発話の編集・削除とセッション削除で、その発話を起点とする退避応答も消える。
    発話が変わる／なくなる以上、それに対して引いた応答は無効なため。
    """

    def _count(self, sqlite_store):
        with sqlite_store.get_session() as s:
            return s.query(ChatMessageVariant).count()

    def test_delete_from_drops_variants_of_removed_anchor(self, sqlite_store):
        """起点ユーザ発話ごと巻き戻すと、その退避応答も消えること。"""
        _cid, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")

        sqlite_store.delete_chat_messages_from(sid, "u1")

        assert self._count(sqlite_store) == 0

    def test_delete_session_drops_variants(self, sqlite_store):
        """セッション削除で退避応答も消えること。"""
        _cid, sid = _setup(sqlite_store)
        _regenerate(sqlite_store, sid, "c2", "応答その2")

        sqlite_store.delete_chat_session(sid)

        assert self._count(sqlite_store) == 0
