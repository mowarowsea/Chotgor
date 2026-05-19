"""
グループチャットターン制御 (services/group_chat/service.py の run_group_turn) のユニットテスト。

テスト対象:
  - 司会モデル未設定時に director_error イベントを送出すること
  - 司会LLMエラー（decide_next_speakers が None）時に director_error を送出すること
  - forced_speaker 指定時に司会を介さず指定キャラクターを発言させ、その後司会が再開すること

司会判定 (decide_next_speakers) と各キャラクターのストリーミング応答
(_stream_character_response) はすべてモック化し、ネットワーク呼び出しを発生させない。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.group_chat import service as svc


def _make_sqlite() -> MagicMock:
    """会話履歴・セッション状態を返す最小モック SQLiteStore を生成するヘルパー。

    退席キャラクターなし・空の会話履歴を返す。
    """
    sqlite = MagicMock()
    sqlite.list_chat_messages.return_value = []
    session = MagicMock()
    session.exited_chars = None
    sqlite.get_chat_session.return_value = session
    return sqlite


def _group_config() -> dict:
    """テスト用グループ設定（参加者2名）を返すヘルパー。"""
    return {
        "participants": [
            {"char_name": "はる", "preset_id": "p1"},
            {"char_name": "冴木", "preset_id": "p2"},
        ],
        "max_auto_turns": 3,
    }


async def _collect(gen) -> list[tuple]:
    """非同期ジェネレーターが yield する全イベントをリストに集約するヘルパー。"""
    return [event async for event in gen]


class TestRunGroupTurnDirectorError:
    """司会エラー時のフォールバック挙動を検証するテストスイート。

    司会モデル未設定・司会LLM障害のいずれも、ユーザーへ明示的に伝わる
    director_error イベントとして送出されることを確認する。
    """

    @pytest.mark.asyncio
    async def test_emits_director_error_when_director_unconfigured(self):
        """司会モデルが設定にもgroup_configにも無いとき director_error を送出すること。"""
        sqlite = _make_sqlite()
        gen = svc.run_group_turn(
            session_id="sid",
            group_config=_group_config(),
            sqlite=sqlite,
            settings={},  # group_director_preset_id 未設定
            chat_service=MagicMock(),
            message_to_dict=lambda m: m,
        )
        events = await _collect(gen)

        assert events[-1][0] == "director_error"
        assert "司会モデル" in events[-1][1]["message"]

    @pytest.mark.asyncio
    async def test_emits_director_error_when_decide_returns_none(self):
        """司会判定が None（LLM障害）を返したとき director_error を送出すること。"""
        sqlite = _make_sqlite()
        with patch.object(svc, "decide_next_speakers", new=AsyncMock(return_value=None)):
            gen = svc.run_group_turn(
                session_id="sid",
                group_config=_group_config(),
                sqlite=sqlite,
                settings={"group_director_preset_id": "preset-director"},
                chat_service=MagicMock(),
                message_to_dict=lambda m: m,
            )
            events = await _collect(gen)

        assert events[-1][0] == "director_error"


class TestRunGroupTurnForcedSpeaker:
    """forced_speaker（ユーザーによる手動指名）の挙動を検証するテストスイート。"""

    @pytest.mark.asyncio
    async def test_forced_speaker_skips_director_then_director_resumes(self):
        """forced_speaker 指定時、最初のターンは司会を介さず指定キャラが発言し、
        その後のターンは通常通り司会が引き継ぐこと。

        例: ユーザが「冴木」を手動指名 → 冴木が発言 → 司会が次発言者を判断。
        司会は2ターン目で [] を返してユーザーターンへ戻すものとする。
        """
        sqlite = _make_sqlite()

        async def fake_stream(char_name, **kwargs):
            """指定キャラクターの応答ストリームを模したダミージェネレーター。"""
            yield ("character_done", {"character": char_name, "message": MagicMock()})

        # 司会は手動指名ターンの後（2ターン目）に呼ばれ、[] を返す
        decide_mock = AsyncMock(return_value=[])

        with patch.object(svc, "_stream_character_response", new=fake_stream), \
             patch.object(svc, "decide_next_speakers", new=decide_mock):
            gen = svc.run_group_turn(
                session_id="sid",
                group_config=_group_config(),
                sqlite=sqlite,
                settings={"group_director_preset_id": "preset-director"},
                chat_service=MagicMock(),
                message_to_dict=lambda m: m,
                forced_speaker="冴木",
            )
            events = await _collect(gen)

        # 1ターン目は手動指名された「冴木」が司会を介さず発言する
        speaker_events = [e for e in events if e[0] == "speaker_decided"]
        assert speaker_events[0][1]["speakers"] == ["冴木"]
        # 「冴木」の応答完了イベントが含まれる
        assert any(e[0] == "character_done" for e in events)
        # 手動指名ターンの後、司会が1度呼ばれて [] を返しユーザーターンへ戻る
        decide_mock.assert_awaited_once()
        assert events[-1][0] == "user_turn"
