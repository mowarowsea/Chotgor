"""Retrace（時制の辿り直し）バッチのテスト。

ワーキングメモリに「今週」「昨日」のような相対表現が日付なしで残ると、書いた日から
離れるほど指す先がずれ、やがて「今のこと」として読まれ続ける（8/10〜8/14 の休みを
翌週以降も休みだと思い込む事故が実際に起きた）。書き方のルールは Chronicle と
post_working_memory_thread のツール説明へ入れたが、それは以後に書くぶんにしか効かない。
すでに書かれてしまったぶんは本人に直してもらう必要がある — それがこのバッチ。

Chotgor の前提として、記憶の書き換えは開発者ではなくキャラクター本人が行う。
したがってシステムは「日付つきの履歴を提示する」ところまでを担い、どう直すか・
そもそも直すかは本人の判断に委ねる。テストの保証範囲もそこまでとする。

検証する観点:
    - 材料提示: ポストが書かれた日付つきでプロンプトへ載ること
    - 逃げ道: 思い出せないものを捏造させない旨がプロンプトに含まれること
    - 反映: 本人が返した thread_updates がスレッドへ反映されること
    - 役割の限定: 新規作成・統合・長期記憶への昇格は受け付けないこと
    - 早期終了: Open スレッドが無ければ LLM を呼ばないこと
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.retrace_job import run_retrace
from backend.services.memory.format import short_date

from tests._ghost_model_helpers import (  # noqa: F401
    _setup_char_with_messages,
    working_memory_manager,
)


async def _run_and_capture(char_id, sqlite_store, wm, response: str):
    """run_retrace を1回まわし、(結果, 本人へ渡したプロンプト本文) を返す。"""
    captured: list[str] = []

    async def fake_generate(sys_prompt, messages):
        captured.append(messages[0]["content"])
        return response

    mock_provider = AsyncMock()
    mock_provider.generate = fake_generate
    with patch("backend.services.character_query.create_provider", return_value=mock_provider):
        result = await run_retrace(
            character_id=char_id, sqlite=sqlite_store, working_memory_manager=wm,
        )
    return result, (captured[0] if captured else "")


class TestRetracePrompt:
    """本人へ渡す設問の検証。

    「この『今週』はいつの週だったのか」を辿る手がかりは、各ポストが書かれた日付
    しかない。通常の棚卸しは最新ポストしか出さないため、このバッチでは履歴を
    日付つきで開く。
    """

    @pytest.mark.asyncio
    async def test_posts_are_listed_with_the_date_they_were_written(
        self, sqlite_store, working_memory_manager,
    ):
        """各ポストが「書かれた日付つき」で並ぶこと。"""
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=1)
        thread = working_memory_manager.create_thread(
            character_id=char_id, type="task", summary="今週はお盆休み",
            content="今週はずっと休みだから寝坊してる",
        )
        working_memory_manager.add_post(thread["id"], "休み明けが憂鬱")

        _, prompt = await _run_and_capture(
            char_id, sqlite_store, working_memory_manager, '{"thread_updates": []}',
        )

        today = short_date(datetime.now())
        assert f"  - [{today}] 今週はずっと休みだから寝坊してる" in prompt
        assert f"  - [{today}] 休み明けが憂鬱" in prompt

    @pytest.mark.asyncio
    async def test_prompt_forbids_inventing_dates(self, sqlite_store, working_memory_manager):
        """思い出せないものを捏造させないこと。

        辻褄合わせで日付をでっち上げるくらいなら「思い出せない」と書いてあるほうが
        よい、という逃げ道を必ず残す（記憶は正しさを取り繕う対象ではない）。
        """
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=1)
        working_memory_manager.create_thread(
            character_id=char_id, type="task", summary="今週はお盆休み", content="休み中",
        )

        _, prompt = await _run_and_capture(
            char_id, sqlite_store, working_memory_manager, '{"thread_updates": []}',
        )

        assert "無理に日付を決めないでください" in prompt
        assert "いつのことか思い出せない" in prompt


class TestRetraceApply:
    """本人の返答を反映する側の検証。"""

    @pytest.mark.asyncio
    async def test_rewritten_summary_is_saved(self, sqlite_store, working_memory_manager):
        """相対表現を日付へ直した summary が保存されること。"""
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=1)
        thread = working_memory_manager.create_thread(
            character_id=char_id, type="task", summary="今週はお盆休み", content="休み中",
        )
        response = (
            '{"thread_updates": [{"id": "' + thread["id"] + '",'
            ' "summary": "8/10(月)〜8/14(金)はお盆休み", "new_post": "あれはお盆の週のことだった"}]}'
        )

        result, _ = await _run_and_capture(
            char_id, sqlite_store, working_memory_manager, response,
        )

        assert result["status"] == "success"
        assert sqlite_store.get_working_memory_thread(thread["id"]).summary == (
            "8/10(月)〜8/14(金)はお盆休み"
        )

    @pytest.mark.asyncio
    async def test_new_threads_and_inscribe_are_ignored(
        self, sqlite_store, working_memory_manager,
    ):
        """書き直し以外（新規スレッド・昇格）は受け付けないこと。

        このバッチの役目は過去の書き直しであって、新しい記憶を作ることではない。
        本人が勢いで書いてきても、ここでは捨てる（通常の棚卸しで扱えばよい）。
        """
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=1)
        working_memory_manager.create_thread(
            character_id=char_id, type="task", summary="今週はお盆休み", content="休み中",
        )
        before = len(working_memory_manager.list_threads_by_type(char_id, is_open=True))
        response = (
            '{"thread_updates": [],'
            ' "new_threads": [{"type": "task", "summary": "新しい話", "post": "本文",'
            ' "importance": 0.5, "origin": "real"}],'
            ' "inscribe": [{"content": "刻みたい", "category": "contextual", "impact": 1.0}]}'
        )

        result, _ = await _run_and_capture(
            char_id, sqlite_store, working_memory_manager, response,
        )

        assert result["status"] == "success"
        assert len(working_memory_manager.list_threads_by_type(char_id, is_open=True)) == before

    @pytest.mark.asyncio
    async def test_no_open_threads_skips_the_llm_call(self, sqlite_store, working_memory_manager):
        """直す対象が無ければ本人を起こさないこと（無駄な問い合わせを投げない）。"""
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=1)

        result, prompt = await _run_and_capture(
            char_id, sqlite_store, working_memory_manager, '{"thread_updates": []}',
        )

        assert result["status"] == "success"
        assert prompt == ""
