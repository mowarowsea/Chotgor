"""キャラクター削除のカスケード（SQLite 側）を検証するテスト。

`delete_character_cascade` が、対象キャラクターに紐づく
- inscribed_memories
- working_memory_threads ＋ working_memory_posts
- session_drifts
- 1on1 chat_sessions ＋ chat_messages ＋ chat_images
を全て削除し、かつ
- 別キャラクターのデータは一切巻き込まない
- debug_log_entries は保持する（このテストでは対象外として明示）
ことを保証する。SQLite には FK の ON DELETE CASCADE が無いため、
このカスケードはアプリ側で明示削除されることが前提となる。
"""

import uuid

from backend.repositories.sqlite.store import (
    Character,
    InscribedMemory,
    WorkingMemoryThread,
    WorkingMemoryPost,
    SessionDrift,
    ChatSession,
    ChatMessage,
    ChatImage,
)


def _seed_character(store, name: str) -> str:
    """1キャラクターと、紐づく全種類のデータを1セット作成し character_id を返す。

    inscribed_memory / working_memory(thread+post) / 1on1 chat(session+message+image+drift)
    をひととおり作る。
    """
    char_id = str(uuid.uuid4())
    store.create_character(character_id=char_id, name=name, system_prompt_block1="def")

    # 保存記憶
    store.create_inscribed_memory(
        memory_id=str(uuid.uuid4()),
        character_id=char_id,
        content=f"{name} の記憶",
        memory_category="contextual",
    )

    # ワーキングメモリ（スレッド＋ポスト）
    thread_id = str(uuid.uuid4())
    store.add_working_memory_thread(thread_id=thread_id, character_id=char_id, type="topic")
    store.add_working_memory_post(post_id=str(uuid.uuid4()), thread_id=thread_id, content="post")

    # 1on1 チャット（model_id = "{name}@default"）
    session_id = str(uuid.uuid4())
    store.create_chat_session(session_id=session_id, model_id=f"{name}@default")
    msg_id = str(uuid.uuid4())
    store.create_chat_message(message_id=msg_id, session_id=session_id, role="user", content="hi")
    store.create_chat_image(
        image_id=str(uuid.uuid4()), session_id=session_id, mime_type="image/png", message_id=msg_id
    )
    store.add_session_drift(session_id=session_id, character_id=char_id, content="drift")

    return char_id


def _counts(store, char_id: str, char_name: str) -> dict:
    """指定キャラクターに紐づく各テーブルの行数を数えて返す。"""
    with store.get_session() as s:
        session_ids = [
            row[0]
            for row in s.query(ChatSession.id).filter(
                ChatSession.model_id.like(f"{char_name}@%")
            ).all()
        ]
        thread_ids = [
            row[0]
            for row in s.query(WorkingMemoryThread.id).filter(
                WorkingMemoryThread.character_id == char_id
            ).all()
        ]
        return {
            "character": s.query(Character).filter(Character.id == char_id).count(),
            "memories": s.query(InscribedMemory).filter(
                InscribedMemory.character_id == char_id
            ).count(),
            "threads": s.query(WorkingMemoryThread).filter(
                WorkingMemoryThread.character_id == char_id
            ).count(),
            "posts": (
                s.query(WorkingMemoryPost).filter(
                    WorkingMemoryPost.thread_id.in_(thread_ids)
                ).count()
                if thread_ids
                else 0
            ),
            "drifts": s.query(SessionDrift).filter(
                SessionDrift.character_id == char_id
            ).count(),
            "sessions": s.query(ChatSession).filter(
                ChatSession.model_id.like(f"{char_name}@%")
            ).count(),
            "messages": (
                s.query(ChatMessage).filter(
                    ChatMessage.session_id.in_(session_ids)
                ).count()
                if session_ids
                else 0
            ),
            "images": (
                s.query(ChatImage).filter(
                    ChatImage.session_id.in_(session_ids)
                ).count()
                if session_ids
                else 0
            ),
        }


def test_cascade_deletes_all_related_rows(sqlite_store):
    """対象キャラに紐づく全テーブルの行がカスケード削除されることを検証する。

    削除前は各テーブルに1件以上存在し、削除後は全て 0 件になることを確認する。
    """
    char_id = _seed_character(sqlite_store, "Alice")

    before = _counts(sqlite_store, char_id, "Alice")
    # 全種類のデータが種まきされていることを前提確認
    assert all(v >= 1 for v in before.values()), before

    result = sqlite_store.delete_character_cascade(char_id)
    assert result is True

    after = _counts(sqlite_store, char_id, "Alice")
    assert all(v == 0 for v in after.values()), after


def test_cascade_does_not_touch_other_character(sqlite_store):
    """別キャラクターのデータが巻き込まれないことを検証する。

    Alice を削除しても Bob のキャラクター・記憶・WM・チャット・ドリフトは
    すべて残っていなければならない。
    """
    alice_id = _seed_character(sqlite_store, "Alice")
    bob_id = _seed_character(sqlite_store, "Bob")

    sqlite_store.delete_character_cascade(alice_id)

    bob_after = _counts(sqlite_store, bob_id, "Bob")
    assert all(v >= 1 for v in bob_after.values()), bob_after


def test_cascade_returns_false_for_missing_character(sqlite_store):
    """存在しないキャラクターIDに対しては False を返し、例外を投げないことを検証する。"""
    assert sqlite_store.delete_character_cascade("does-not-exist") is False
