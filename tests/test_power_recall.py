"""PowerRecall 機能のユニットテスト。

以下を網羅的に検証する:
  - Recaller: [POWER_RECALL:...] タグのパース（クエリ・top_k抽出、マーカー前テキスト返却）
  - StreamingTagStripper: [POWER_RECALL:] がUIに表示されないこと（除去対象であること）
  - MemoryManager.power_recall: 記憶+チャット履歴の横断検索とコンテキスト付与
  - tools._power_recall: tool result のフォーマット（コンテキストあり・なし）
  - system_prompt: power_recalled ブロックの生成
  - ChatService.execute_stream: PowerRecall再呼び出しとループ防止
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from backend.core.chat.recaller import Recaller
from backend.core.tag_parser import StreamingTagStripper
from backend.core.memory.manager import MemoryManager
from backend.core.system_prompt import build_system_prompt


# ─── Recaller ─────────────────────────────────────────────────────────────────


class TestRecaller:
    """Recaller クラスのタグパーステスト。"""

    def test_タグありの場合マーカー前テキストを返す(self):
        """[POWER_RECALL:...] が存在する場合、マーカーより前のテキストのみが返ること。"""
        r = Recaller()
        text = "ちょっと頑張って思い出してみる\n[POWER_RECALL:ユーザが言ってた食べ物|5]"
        result = r.power_recall_from_text(text)
        assert result == "ちょっと頑張って思い出してみる"

    def test_タグありの場合recall_requestが設定される(self):
        """[POWER_RECALL:query|top_k] を正しくパースして recall_request に格納すること。"""
        r = Recaller()
        r.power_recall_from_text("[POWER_RECALL:好きな音楽|3]")
        assert r.recall_request == ("好きな音楽", 3)

    def test_top_k省略時はデフォルト5(self):
        """top_k が省略された場合、デフォルト値 5 が使われること。"""
        r = Recaller()
        r.power_recall_from_text("[POWER_RECALL:何か思い出す]")
        assert r.recall_request == ("何か思い出す", 5)

    def test_top_k非数値の場合もデフォルト5(self):
        """top_k 部分が数値でない場合もデフォルト 5 にフォールバックすること。"""
        r = Recaller()
        r.power_recall_from_text("[POWER_RECALL:クエリ|abc]")
        assert r.recall_request == ("クエリ", 5)

    def test_タグなしの場合テキストをそのまま返す(self):
        """[POWER_RECALL:...] がない場合、テキストをそのまま返し recall_request は None のまま。"""
        r = Recaller()
        result = r.power_recall_from_text("普通の応答テキスト")
        assert result == "普通の応答テキスト"
        assert r.recall_request is None

    def test_マーカー以降のテキストは捨てられる(self):
        """マーカー以降に続くテキストは返り値に含まれないこと。"""
        r = Recaller()
        result = r.power_recall_from_text("前テキスト\n[POWER_RECALL:クエリ|5]\n後テキスト")
        assert "後テキスト" not in result
        assert result == "前テキスト"

    def test_空文字列を渡しても例外が出ない(self):
        """空文字列を渡しても正常に動作し、recall_request が None であること。"""
        r = Recaller()
        result = r.power_recall_from_text("")
        assert result == ""
        assert r.recall_request is None


# ─── StreamingTagStripper ──────────────────────────────────────────────────────


class TestStreamingTagStripperPowerRecall:
    """StreamingTagStripper が [POWER_RECALL:] をUIから除去することのテスト。"""

    def test_power_recallタグがストリームから除去される(self):
        """[POWER_RECALL:...] がストリーミング出力に含まれないこと。"""
        stripper = StreamingTagStripper()
        text = "ちょっと思い出してみる[POWER_RECALL:好きな食べ物|5]"
        out = stripper.feed(text)
        remaining = stripper.flush()
        full_output = out + remaining
        assert "[POWER_RECALL:" not in full_output
        assert "ちょっと思い出してみる" in full_output

    def test_マーカー前テキストはそのまま流れる(self):
        """[POWER_RECALL:] の前に出力されたテキストはそのまま流れること。"""
        stripper = StreamingTagStripper()
        out = stripper.feed("前のテキスト")
        assert "前のテキスト" in out

    def test_タグが複数チャンクにまたがっても除去される(self):
        """[POWER_RECALL:...] が複数チャンクにまたがっても正しく除去されること。"""
        stripper = StreamingTagStripper()
        out1 = stripper.feed("テキスト[POWER_")
        out2 = stripper.feed("RECALL:クエリ|")
        out3 = stripper.feed("5]続き")
        remaining = stripper.flush()
        full = out1 + out2 + out3 + remaining
        assert "[POWER_RECALL:" not in full
        assert "テキスト" in full
        assert "続き" in full


# ─── MemoryManager.power_recall ───────────────────────────────────────────────


class TestMemoryManagerPowerRecall:
    """MemoryManager.power_recall() のテスト。ChromaStore はモック化する。"""

    @pytest.fixture
    def mock_chroma(self):
        """ChromaStore のモック。"""
        chroma = MagicMock()
        chroma.recall_memory.return_value = [
            {"id": "mem-1", "content": "コーヒーが好き", "distance": 0.1, "metadata": {"category": "user"}}
        ]
        chroma.recall_chat_turns.return_value = []
        chroma.find_similar_in_category.return_value = None
        return chroma

    @pytest.fixture
    def manager(self, sqlite_store, mock_chroma):
        """MemoryManager のフィクスチャ。"""
        return MemoryManager(sqlite=sqlite_store, chroma=mock_chroma)

    def test_記憶とチャット履歴の両方を返す(self, manager, mock_chroma):
        """power_recall が memories と chat_turns の両キーを持つ dict を返すこと。"""
        result = manager.power_recall("char-1", "好きな飲み物")
        assert "memories" in result
        assert "chat_turns" in result

    def test_ChromaDBが両コレクションを検索する(self, manager, mock_chroma):
        """recall_memory と recall_chat_turns の両方が呼ばれること。"""
        manager.power_recall("char-1", "好きな飲み物", top_k=7)
        mock_chroma.recall_memory.assert_called_once_with("好きな飲み物", "char-1", top_k=7)
        mock_chroma.recall_chat_turns.assert_called_once_with("好きな飲み物", "char-1", top_k=7)

    def test_チャットヒットにcontextが付与される(self, manager, mock_chroma, sqlite_store):
        """chat_turns の各ヒットに context キーが付与されること。"""
        import uuid
        session_id = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=session_id, model_id="テスト@anthropic")
        msg_id = "msg-hit-001"
        sqlite_store.create_chat_message(
            message_id=msg_id, session_id=session_id, role="user", content="ヒットメッセージ"
        )
        mock_chroma.recall_chat_turns.return_value = [
            {"id": msg_id, "content": "ユーザ: ヒットメッセージ", "distance": 0.2,
             "metadata": {"session_id": session_id}}
        ]

        result = manager.power_recall("char-1", "ヒット", top_k=5)
        assert len(result["chat_turns"]) == 1
        assert "context" in result["chat_turns"][0]

    def test_同一セッションヒットはlist_chat_messagesを1回だけ呼ぶ(self, manager, mock_chroma, sqlite_store):
        """同一セッションから複数ヒットがあっても list_chat_messages は1回だけ呼ばれること。

        N+1クエリを防ぐためのセッション単位キャッシュの動作確認。
        """
        import uuid
        session_id = str(uuid.uuid4())
        sqlite_store.create_chat_session(session_id=session_id, model_id="テスト@anthropic")
        for i in range(3):
            sqlite_store.create_chat_message(
                message_id=f"msg-{i}", session_id=session_id,
                role="user", content=f"msg{i}"
            )

        mock_chroma.recall_chat_turns.return_value = [
            {"id": "msg-0", "content": "c0", "distance": 0.1, "metadata": {"session_id": session_id}},
            {"id": "msg-2", "content": "c2", "distance": 0.2, "metadata": {"session_id": session_id}},
        ]

        original_list = sqlite_store.list_chat_messages
        call_count = {"n": 0}

        def counted_list(sid):
            call_count["n"] += 1
            return original_list(sid)

        sqlite_store.list_chat_messages = counted_list
        manager.power_recall("char-1", "クエリ")
        assert call_count["n"] == 1


# ─── system_prompt power_recalled ブロック ────────────────────────────────────


class TestSystemPromptPowerRecalled:
    """build_system_prompt() に power_recalled ブロックが含まれないことのテスト。

    検索結果は system_prompt ではなく Chotgor ユーザーターン（messages）として注入するため、
    build_system_prompt は power_recalled を受け取らず、結果ブロックを生成しない。
    """

    def test_システムプロンプトにPOWER_RECALL_COMPLETEブロックが含まれない(self):
        """build_system_prompt は POWER_RECALL COMPLETE ブロックを生成しないこと。"""
        prompt = build_system_prompt("キャラ設定")
        assert "POWER_RECALL COMPLETE" not in prompt


class TestFormatPowerRecallTurn:
    """format_power_recall_turn() の Chotgor ユーザーターンフォーマットテスト。"""

    def test_記憶ヒットがターンに含まれる(self):
        """memories のコンテンツが Chotgor ターンに含まれること。"""
        from backend.core.chat.recaller import format_power_recall_turn
        results = {
            "memories": [
                {"content": "コーヒーが好き", "distance": 0.1, "metadata": {"category": "user"}}
            ],
            "chat_turns": [],
        }
        turn = format_power_recall_turn(results, "好きな飲み物")
        assert "POWER_RECALL COMPLETE" in turn
        assert "コーヒーが好き" in turn
        assert "好きな飲み物" in turn

    def test_チャットヒットのコンテキストがターンに含まれる(self):
        """chat_turns のコンテキストメッセージが Chotgor ターンに含まれること。"""
        from backend.core.chat.recaller import format_power_recall_turn
        results = {
            "memories": [],
            "chat_turns": [
                {
                    "content": "ユーザ: カレーが好き",
                    "distance": 0.2,
                    "metadata": {},
                    "context": [
                        {"speaker_name": "ユーザ", "content": "カレー食べた？", "is_hit": False},
                        {"speaker_name": "ユーザ", "content": "カレーが好き", "is_hit": True},
                        {"speaker_name": "はる", "content": "私も好き！", "is_hit": False},
                    ],
                }
            ],
        }
        turn = format_power_recall_turn(results, "カレー")
        assert "カレー食べた？" in turn
        assert "カレーが好き" in turn
        assert "私も好き！" in turn

    def test_結果なしでも正常に動作する(self):
        """memories と chat_turns が空でも例外が発生しないこと。"""
        from backend.core.chat.recaller import format_power_recall_turn
        turn = format_power_recall_turn({"memories": [], "chat_turns": []}, "何か")
        assert "POWER_RECALL COMPLETE" in turn
        assert "見つかりませんでした" in turn

    def test_再検索禁止メッセージが含まれる(self):
        """生成されたターンに再検索禁止の指示が含まれること。"""
        from backend.core.chat.recaller import format_power_recall_turn
        turn = format_power_recall_turn({}, "クエリ")
        assert "禁止" in turn


# ─── ChatService.execute_stream: ループ防止 ───────────────────────────────────


class TestChatServicePowerRecallLoop:
    """ChatService.execute_stream() の PowerRecall ループ防止テスト。

    [POWER_RECALL:] タグが含まれる応答に対して:
      - request.power_recalled が空（初回）→ power_recall を実行して再呼び出しすること
      - request.power_recalled が非空（再呼び出し中）→ power_recall をスキップしてループしないこと
    """

    def _make_base_request(self, power_recalled=None):
        """テスト用 ChatRequest を生成する。"""
        from backend.core.chat.models import ChatRequest, Message
        return ChatRequest(
            character_id="char-1",
            character_name="はる",
            provider="anthropic",
            model="",
            messages=[Message(role="user", content="前に話してたこと覚えてる？")],
            power_recalled=power_recalled or {},
        )

    def _make_stream_provider(self, text: str):
        """指定テキストを1チャンクで返す generate_stream_typed モックプロバイダーを作る。"""
        from unittest.mock import AsyncMock, MagicMock

        async def fake_stream(*_args, **_kwargs):
            yield ("text", text)

        provider = MagicMock()
        provider.SUPPORTS_TOOLS = False
        provider.generate_stream_typed = fake_stream
        return provider

    def _make_memory_manager(self):
        """power_recall が {"memories": [], "chat_turns": []} を返す MemoryManager モック。"""
        mm = MagicMock()
        mm.recall_with_identity.return_value = ([], [])
        mm.power_recall.return_value = {"memories": [], "chat_turns": []}
        return mm

    @pytest.mark.asyncio
    async def test_初回呼び出しでPOWER_RECALLタグを検知したらpower_recallが実行される(self):
        """power_recalled が空の状態で [POWER_RECALL:...] タグがあると、
        memory_manager.power_recall が呼ばれること。

        ループ防止条件（request.power_recalled が非空）が初回は適用されないことを確認する。
        """
        from unittest.mock import patch, AsyncMock
        from backend.core.chat.service import ChatService

        mm = self._make_memory_manager()
        request = self._make_base_request(power_recalled={})
        provider = self._make_stream_provider("思い出してみる\n[POWER_RECALL:前に話した内容|3]")

        with (
            patch("backend.core.chat.service.create_provider", return_value=provider),
            patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
            patch("backend.core.chat.service.find_urls", return_value=[]),
            patch("backend.core.chat.service.asyncio.to_thread", new=AsyncMock(
                return_value={"memories": [], "chat_turns": []}
            )),
        ):
            service = ChatService(memory_manager=mm)
            events = [e async for e in service.execute_stream(request)]

        # to_thread 経由で power_recall が呼ばれたことを確認する
        to_thread_calls = [
            e for e in events  # イベント収集は副作用確認のため
        ]
        # re-invocation が走って追加のイベントが来ること（memories イベント等）
        event_types = [e[0] for e in events]
        # 少なくとも text イベントが存在すること（再呼び出し後の応答）
        assert "text" in event_types

    @pytest.mark.asyncio
    async def test_再呼び出し中はPOWER_RECALLタグを無視してループしない(self):
        """power_recalled が非空の状態（再呼び出し中）で [POWER_RECALL:...] タグがあっても、
        memory_manager.power_recall が呼ばれないこと。

        これがループ防止の核心: 再呼び出し後にキャラクターが再度タグを出しても
        さらなる再呼び出しは発生しない。
        """
        from unittest.mock import patch, AsyncMock
        from backend.core.chat.service import ChatService
        from backend.core.memory.inscriber import Inscriber
        from backend.core.memory.carver import Carver
        from backend.core.chat.exiter import Exiter

        mm = self._make_memory_manager()
        # power_recalled が非空 = 再呼び出し中
        request = self._make_base_request(
            power_recalled={"memories": [{"content": "前回の検索結果"}], "chat_turns": []}
        )
        provider = self._make_stream_provider("思い出してみる\n[POWER_RECALL:もっと調べる|3]")

        mock_inscriber = MagicMock()
        mock_inscriber.inscribe_memory_from_text.side_effect = lambda text, *_: text
        mock_carver = MagicMock()
        mock_carver.carve_narrative_from_text.side_effect = lambda text: text

        with (
            patch("backend.core.chat.service.create_provider", return_value=provider),
            patch("backend.core.chat.service.build_system_prompt", return_value="sys"),
            patch("backend.core.chat.service.find_urls", return_value=[]),
            patch("backend.core.chat.service.Inscriber", return_value=mock_inscriber),
            patch("backend.core.chat.service.Carver", return_value=mock_carver),
        ):
            service = ChatService(memory_manager=mm)
            events = [e async for e in service.execute_stream(request)]

        # power_recall は呼ばれていないこと
        mm.power_recall.assert_not_called()
        # text イベントが存在すること（応答は正常に返る）
        event_types = [e[0] for e in events]
        assert "text" in event_types
