"""ghost_model / chronicle テスト群の共有フィクスチャ・ヘルパー。

InscribedMemoryManager / WorkingMemoryManager のフィクスチャと、
キャラクター＋メッセージのシードヘルパーを提供する。
test_ghost_model_*.py / test_chronicle_*.py から import して使用する
（フィクスチャは import するだけで pytest が認識する）。
sqlite_store は conftest.py で定義済み。
ファイル名先頭がアンダースコアのため pytest のテスト収集対象にはならない。
"""

import uuid

import pytest


@pytest.fixture
def memory_manager(sqlite_store):
    """テスト用 InscribedMemoryManager（LanceDB はモック）。"""
    from unittest.mock import MagicMock
    from backend.services.memory.manager import InscribedMemoryManager
    vector_store = MagicMock()
    vector_store.add_inscribed_memory = MagicMock()
    vector_store.delete_inscribed_memory = MagicMock()
    vector_store.search = MagicMock(return_value=[])
    return InscribedMemoryManager(sqlite_store, vector_store)


@pytest.fixture
def working_memory_manager(sqlite_store):
    """テスト用 WorkingMemoryManager。

    スレッド CRUD は実 SQLite（インメモリ一時DB）で動かし、embedding 層の
    LanceStore のみ MagicMock に置き換える。chronicle が棚卸し結果を
    実際にスレッドへ反映する挙動をそのまま検証できる。
    """
    from unittest.mock import MagicMock
    from backend.services.memory.working_memory_manager import WorkingMemoryManager
    return WorkingMemoryManager(sqlite_store, MagicMock())



def _setup_char_with_messages(sqlite_store, char_name: str = "Alice", n_messages: int = 2):
    """テスト用キャラクター・セッション・メッセージを一括作成するヘルパー。

    Returns:
        (char_id, session_id, message_ids, preset_id) のタプル。
    """
    preset_id = str(uuid.uuid4())
    char_id = str(uuid.uuid4())
    sqlite_store.create_model_preset(preset_id, "TestPreset", "google", "gemini-2.0-flash")
    sqlite_store.create_character(char_id, char_name, ghost_model=preset_id)

    session_id = str(uuid.uuid4())
    sqlite_store.create_chat_session(session_id=session_id, model_id=f"{char_name}@default")

    message_ids = []
    for i in range(n_messages):
        mid = str(uuid.uuid4())
        role = "user" if i % 2 == 0 else "character"
        sqlite_store.create_chat_message(
            message_id=mid,
            session_id=session_id,
            role=role,
            content=f"メッセージ {i}",
            character_name=char_name if role == "character" else None,
        )
        message_ids.append(mid)

    return char_id, session_id, message_ids, preset_id


# 棚卸し・蒸留とも「変更なし」を表す chronicle 応答。
_NO_UPDATE_RESPONSE = (
    '{"thread_updates": [], "new_threads": [], "merges": [],'
    ' "inscribe": [], "carve": null, "farewell_config": null}'
)


