"""Chronicle 棚卸しの「繰り返し話題への気づき誘導」のテスト。

同じ話題を何度も持ち出してしまう失敗（もう Close 済みの結論があるのに、別スレッドで
「まだ言えていない」と書き足し続ける）に本人が気づけるよう、棚卸しプロンプトへ
参考情報を載せる仕組み（docs/planned/wm_repeat_awareness_plan.md 施策A・B）。

**強制ルールではない**ことが設計の要。システムは close を実行せず、材料を出すだけで、
閉じるかどうかの判断はキャラクター本人に委ねる。したがってテストの保証範囲も
「材料が正しく提示されること」までとする（close されるかは毎回 LLM 応答に依存し、
決定論的なテストにならない）。

検証する観点:
    - _short_date: ISO 文字列 → MM-DD 表記、壊れた値のフォールバック
    - _format_similarity_hints: ペアあり / なしの出力
    - 統合: run_chronicle が Open×Close の類似ペアを棚卸しプロンプトへ載せること
    - 統合: 類似ペアが無いときも該当なしの文言が入り、プロンプトが壊れないこと
    - 統合: embedding 検索が落ちても棚卸し全体は成功すること（記憶系の縮退耐性）
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import (
    _format_similarity_hints,
    _short_date,
    run_chronicle,
)

from tests._ghost_model_helpers import (  # noqa: F401
    _NO_UPDATE_RESPONSE,
    _setup_char_with_messages,
    working_memory_manager,
)


# ---------------------------------------------------------------------------
# _short_date
# ---------------------------------------------------------------------------

class TestShortDate:
    """Close 日の短縮表記の検証。

    「いつ決着した話か」が分かれば十分な参考情報なので年は落とす。値が壊れていても
    棚卸し全体を止めないよう、空文字へ倒す（呼び出し側がラベルごと省く）。
    """

    def test_iso_datetime_becomes_month_day(self):
        """ISO 8601 の日時から MM-DD だけを取り出すこと。"""
        assert _short_date("2026-08-13T21:04:05") == "08-13"

    def test_date_only_is_accepted(self):
        """日付のみの ISO 文字列も受け付けること。"""
        assert _short_date("2026-08-13") == "08-13"

    def test_broken_value_falls_back_to_empty(self):
        """None・空・非日付文字列は空文字に倒すこと。"""
        assert _short_date(None) == ""
        assert _short_date("") == ""
        assert _short_date("いつか") == ""


# ---------------------------------------------------------------------------
# _format_similarity_hints
# ---------------------------------------------------------------------------

def _thread_dict(thread_id: str, summary: str, updated_at: str | None = None) -> dict:
    """整形対象のスレッド dict（_format_similarity_hints が触るキーだけ）。"""
    return {"id": thread_id, "summary": summary, "updated_at": updated_at}


class TestFormatSimilarityHints:
    """重複疑いペアの整形の検証。

    出力は「Open 側の見出し」＋「⇔ Close 側の見出し」の 2 行 1 組。本人が中身を
    確かめに行けるよう短縮 ID を添える（応答 JSON でもこの短縮 ID を使える）。
    機械判定のスコア（relevance）は本人の判断を数値に引きずらせないため出さない。
    """

    def test_pair_is_rendered_with_short_ids_and_close_date(self):
        """Open / Close 双方の短縮 ID・summary と、Close 日が出ること。"""
        open_thread = _thread_dict(
            "48e9a044-1111-2222-3333-444455556666", "グラフRAG検証、コスト・粒度で断念方向へ",
        )
        closed_thread = _thread_dict(
            "a8d87fb6-aaaa-bbbb-cccc-ddddeeeeffff",
            "✅日食なつこ実験、WM閉じ忘れ確認まで完了",
            updated_at="2026-08-13T21:04:05",
        )

        text = _format_similarity_hints([(open_thread, closed_thread)])

        assert text == (
            "[48e9a044](Open) グラフRAG検証、コスト・粒度で断念方向へ\n"
            "  ⇔ [a8d87fb6](Close済み・08-13) ✅日食なつこ実験、WM閉じ忘れ確認まで完了"
        )

    def test_relevance_score_is_not_exposed(self):
        """機械判定のスコアは出力に含めないこと。"""
        open_thread = _thread_dict("11111111-x", "Open の話題")
        closed_thread = dict(_thread_dict("22222222-x", "Close の話題"), relevance=0.845)

        text = _format_similarity_hints([(open_thread, closed_thread)])

        assert "0.84" not in text
        assert "relevance" not in text

    def test_missing_close_date_omits_the_label(self):
        """Close 日が取れないときは日付ラベルごと省き、括弧内を壊さないこと。"""
        text = _format_similarity_hints([
            (_thread_dict("11111111-x", "Open の話題"), _thread_dict("22222222-x", "Close の話題")),
        ])

        assert "(Close済み)" in text
        assert "Close済み・" not in text

    def test_multiple_pairs_are_listed_in_order(self):
        """複数ペアは渡された順に 2 行ずつ並ぶこと。"""
        pairs = [
            (_thread_dict("aaaaaaaa-x", "Open A"), _thread_dict("11111111-x", "Close A")),
            (_thread_dict("bbbbbbbb-x", "Open B"), _thread_dict("22222222-x", "Close B")),
        ]

        lines = _format_similarity_hints(pairs).split("\n")

        assert len(lines) == 4
        assert lines[0].startswith("[aaaaaaaa](Open)")
        assert lines[2].startswith("[bbbbbbbb](Open)")

    def test_no_pairs_returns_empty_label(self):
        """ペアが無いときは該当なしの文言を返すこと（空文字でプロンプトを崩さない）。"""
        assert _format_similarity_hints([]) == "（類似の疑いがある組み合わせはありません）"


# ---------------------------------------------------------------------------
# run_chronicle への組み込み
# ---------------------------------------------------------------------------

class TestRunChronicleSimilarityHints:
    """run_chronicle が類似ペアを棚卸しプロンプトへ載せることの検証。

    Open な task/topic ごとに Close 済みスレッドを検索し、閾値を超えたものだけを
    参考情報として提示する。close は実行しない（本テストでも is_open が変わらない
    ことを確認する）。LLM 呼び出しとベクトル検索はモックする。
    """

    def _setup_threads(self, sqlite_store, working_memory_manager):
        """Open な task 1本 と Close 済み topic 1本を持つキャラを用意する。

        Returns:
            (char_id, open_thread_id, closed_thread_id) のタプル。
        """
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=2)
        open_thread = working_memory_manager.create_thread(
            character_id=char_id, type="task",
            summary="グラフRAG検証、コスト・粒度で断念方向へ",
            content="日食なつこの実験結果、まだちゃんと共有できてない",
        )
        closed_thread = working_memory_manager.create_thread(
            character_id=char_id, type="topic",
            summary="✅日食なつこ実験、WM閉じ忘れ確認まで完了",
            content="結論も経緯も両方伝わった上での放置だった",
        )
        working_memory_manager.set_open(closed_thread["id"], False)
        return char_id, open_thread["id"], closed_thread["id"]

    async def _run_and_capture(self, char_id, sqlite_store, working_memory_manager):
        """run_chronicle を1回まわし、(結果, 棚卸しプロンプト本文) を返す。"""
        captured: list[str] = []

        async def fake_generate(sys_prompt, messages):
            captured.append(messages[0]["content"])
            return _NO_UPDATE_RESPONSE

        mock_provider = AsyncMock()
        mock_provider.generate = fake_generate

        with patch("backend.services.character_query.create_provider", return_value=mock_provider):
            result = await run_chronicle(
                character_id=char_id, sqlite=sqlite_store,
                working_memory_manager=working_memory_manager,
            )
        return result, captured[0]

    @pytest.mark.asyncio
    async def test_similar_pair_appears_in_prompt(self, sqlite_store, working_memory_manager):
        """類似する Close 済みスレッドが見つかったら、ペア行がプロンプトに載ること。"""
        char_id, open_id, closed_id = self._setup_threads(sqlite_store, working_memory_manager)
        # 明確な重複の実測水準（relevance 0.845 ≒ distance 0.31）を返す検索にする
        working_memory_manager.vector_store.recall_working_memory_threads.return_value = [
            {"id": closed_id, "distance": 0.31},
        ]

        result, prompt = await self._run_and_capture(
            char_id, sqlite_store, working_memory_manager,
        )

        assert result["status"] == "success"
        assert "## 類似の疑いがある組み合わせ（機械判定・参考情報）" in prompt
        assert f"[{open_id[:8]}](Open) グラフRAG検証、コスト・粒度で断念方向へ" in prompt
        today = datetime.now().strftime("%m-%d")
        assert (
            f"  ⇔ [{closed_id[:8]}](Close済み・{today}) ✅日食なつこ実験、WM閉じ忘れ確認まで完了"
            in prompt
        )

    @pytest.mark.asyncio
    async def test_close_is_not_executed_by_the_system(self, sqlite_store, working_memory_manager):
        """材料を出すだけで、システム側が勝手に Open スレッドを閉じないこと。"""
        char_id, open_id, closed_id = self._setup_threads(sqlite_store, working_memory_manager)
        working_memory_manager.vector_store.recall_working_memory_threads.return_value = [
            {"id": closed_id, "distance": 0.31},
        ]

        await self._run_and_capture(char_id, sqlite_store, working_memory_manager)

        assert sqlite_store.get_working_memory_thread(open_id).is_open

    @pytest.mark.asyncio
    async def test_weak_similarity_is_not_offered(self, sqlite_store, working_memory_manager):
        """無関係水準の類似度しか無ければ、材料として提示しないこと。"""
        char_id, _, closed_id = self._setup_threads(sqlite_store, working_memory_manager)
        # 完全無関係ペアの実測水準（relevance 0.776 ≒ distance 0.45）
        working_memory_manager.vector_store.recall_working_memory_threads.return_value = [
            {"id": closed_id, "distance": 0.45},
        ]

        _, prompt = await self._run_and_capture(char_id, sqlite_store, working_memory_manager)

        assert "（類似の疑いがある組み合わせはありません）" in prompt

    @pytest.mark.asyncio
    async def test_embedding_failure_does_not_break_chronicle(
        self, sqlite_store, working_memory_manager,
    ):
        """embedding 検索が落ちても棚卸しは成功し、該当なしとして続行すること。

        infinity 停止時に記憶系は縮退するが、棚卸しそのものは止めない
        （気づき誘導はあくまで補助であり、棚卸しの前提条件ではない）。
        """
        char_id, _, _ = self._setup_threads(sqlite_store, working_memory_manager)
        working_memory_manager.vector_store.recall_working_memory_threads.side_effect = (
            RuntimeError("infinity unreachable")
        )

        result, prompt = await self._run_and_capture(
            char_id, sqlite_store, working_memory_manager,
        )

        assert result["status"] == "success"
        assert "（類似の疑いがある組み合わせはありません）" in prompt
