"""ワーキングメモリ・意図の「時制アンカリング」のテスト。

記憶の内容は残るのに「それがいつ・誰と・どこでのことだったか」が落ちる問題への
対策一式を検証する。実害として観測されたのは次の2つ:

  1. 前日話した内容は覚えているのに「それが昨日のことだった」「ユーザの家で
     話した」が抜け、いつの話か本人にも辿れない。
  2. 8/10〜8/14 の一週間の休みを日付なしで書き留めたため、8/17週・8/24週も
     休みだと認識し続けた（書いた時点の「今週」と読む時点の「今週」のズレ）。

対策は「材料を見せる」と「書き方を決める」の二段構え:
  - 材料: WM スレッド一覧・意図一覧・Chronicle 棚卸しプロンプトへ絶対日付を出す。
          対面中の場所判定ラベルを本人へ返す。現在時刻へ曜日を添える。
  - 書き方: 読む時点で意味がずれる相対表現を日付で書くルールをプロンプトへ入れる。

検証する観点:
    - short_date: 当年/年またぎ/曜日抑制/壊れた値
    - _build_place_block: 対面時のみ・ラベルがあるときのみ場所を出す
    - _format_thread_index / _format_thread_with_post: 一覧と固定注入の日付表示
    - _target_date_label: Chronicle が扱う対象日の決定
    - 意図表示: 日常注入と拾い上げ設問に「いつからの〜したい」かが出る
    - 統合: 棚卸しプロンプトに対象日・絶対化ルール・スレッド日付が載ること
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.batch.chronicle_job import _target_date_label, run_chronicle
from backend.lib.time_awareness import japanese_weekday
from backend.services.chat.request_builder import (
    _build_motive_block,
    _build_place_block,
    _format_thread_index,
    _format_thread_with_post,
)
from backend.services.intents.pickup import build_pickup_question
from backend.services.memory.format import short_date

from tests._ghost_model_helpers import (  # noqa: F401
    _NO_UPDATE_RESPONSE,
    _setup_char_with_messages,
    working_memory_manager,
)


def _this_year(month: int, day: int, hour: int = 10) -> datetime:
    """当年の日付を作る。年またぎで表記が変わる仕様のため固定年を書かない。"""
    return datetime.now().replace(
        month=month, day=day, hour=hour, minute=0, second=0, microsecond=0
    )


# ---------------------------------------------------------------------------
# short_date
# ---------------------------------------------------------------------------

class TestShortDate:
    """記憶表示用の短い絶対日付の検証。

    プロンプトへ出す日付はトークンを食うので最小限に絞る。同じ年なら年を落とし、
    年をまたぐものだけ年を出す。曜日を添えるのは「今週」「先週」のような週単位の
    言い回しを本人が絶対日付へ結び直せるようにするため（お盆休みの事故対策）。
    値が壊れていても表示全体を止めないよう、空文字へ倒して呼び出し側にラベルごと
    省かせる。
    """

    def test_same_year_drops_the_year_and_keeps_weekday(self):
        """当年の日付は MM-DD(曜) になること。"""
        dt = _this_year(8, 14)
        assert short_date(dt.isoformat(timespec="seconds")) == f"08-14({japanese_weekday(dt)})"

    def test_other_year_keeps_the_year(self):
        """年をまたぐ日付は YYYY-MM-DD(曜) になること（何年前の話か分かるように）。"""
        dt = _this_year(8, 14).replace(year=datetime.now().year - 1)
        assert short_date(dt.isoformat(timespec="seconds")) == (
            f"{dt.year}-08-14({japanese_weekday(dt)})"
        )

    def test_weekday_can_be_suppressed(self):
        """with_weekday=False なら曜日を落とすこと（曜日が要らない参考情報用）。"""
        dt = _this_year(8, 14)
        assert short_date(dt.isoformat(timespec="seconds"), with_weekday=False) == "08-14"

    def test_datetime_is_accepted_directly(self):
        """ISO 文字列だけでなく datetime も受け付けること。"""
        dt = _this_year(8, 14)
        assert short_date(dt) == f"08-14({japanese_weekday(dt)})"

    def test_broken_value_falls_back_to_empty(self):
        """None・空・非日付文字列は空文字へ倒すこと。"""
        assert short_date(None) == ""
        assert short_date("") == ""
        assert short_date("いつか") == ""


# ---------------------------------------------------------------------------
# 対面中の場所
# ---------------------------------------------------------------------------

class TestPlaceBlock:
    """対面中の「いまいる場所」ブロックの検証。

    なりゆき（ambience）の judge は対面中に毎ターン場所を判定して
    chat_sessions.current_bg_label へ書いていたが、その値を読むのはフロントの
    背景画像切り替えだけで、本人には渡っていなかった。本人が「どこで話しているか」
    を知らなければ記憶にも場所が残らない、というのがこのブロックの動機。

    残置値の事故を避けるため、対面でないときは値があっても出さない
    （current_bg_label は対面を抜けても消されず残る仕様）。
    """

    def test_face_to_face_with_label_shows_the_place(self):
        """対面中かつラベルがあれば、場所ブロックを出すこと。"""
        text = _build_place_block(True, "もわの家 リビング")
        assert "## 現在の文脈（場所）" in text
        assert "もわの家 リビング" in text

    def test_not_face_to_face_shows_nothing(self):
        """対面でなければ、ラベルが残っていても出さないこと（残置値の誤表示防止）。"""
        assert _build_place_block(False, "もわの家 リビング") == ""

    def test_missing_label_shows_nothing(self):
        """ラベル未判定・空白のみなら、ブロックごと出さないこと。"""
        assert _build_place_block(True, None) == ""
        assert _build_place_block(True, "   ") == ""


# ---------------------------------------------------------------------------
# WM スレッドの表示
# ---------------------------------------------------------------------------

class TestThreadDisplayDates:
    """システムプロンプトへ注入されるスレッド行の日付表示の検証。

    従来この2つの整形関数は created_at / 最新ポストの日時を持っているのに表示で
    捨てており、本人の日常視界には時間情報が一切なかった。掘れば
    read_working_memory_thread で日時が見えるが、日常的には見えない状態だった。
    """

    def test_open_thread_shows_when_it_started(self):
        """Open スレッドの一覧行に「いつからの話か」が出ること。"""
        dt = _this_year(8, 10)
        line = _format_thread_index({
            "id": "a1b2c3d4-x", "type": "task", "summary": "お盆休みの過ごし方",
            "atmosphere_tag": "のんびり", "importance": 0.7, "is_open": True,
            "created_at": dt.isoformat(timespec="seconds"),
        })
        assert f"08-10({japanese_weekday(dt)})〜" in line

    def test_closed_thread_line_stays_minimal(self):
        """Close 済みは見出しのみのまま（件数が増え続けるため情報を足さない）。"""
        line = _format_thread_index({
            "id": "a1b2c3d4-x", "type": "task", "summary": "決着済みの話",
            "atmosphere_tag": "すっきり", "importance": 0.7, "is_open": False,
            "created_at": _this_year(8, 10).isoformat(timespec="seconds"),
        })
        assert line == "[a1b2c3d4] (task) 決着済みの話"

    def test_latest_post_shows_when_it_was_written(self):
        """固定注入の最新ポストに、書かれた日付が付くこと。

        ポスト本文には「今週」「昨日」のような書いた時点基準の言い回しが混じる。
        いつ書かれたかが読めないと、現時点の話として誤読される。
        """
        dt = _this_year(8, 11)
        text = _format_thread_with_post({
            "id": "a1b2c3d4-x", "type": "body", "summary": "体の調子",
            "atmosphere_tag": "だるい", "latest_post": "今週はずっと休みだから寝坊してる",
            "latest_post_at": dt.isoformat(timespec="seconds"),
        })
        assert f"→ [08-11({japanese_weekday(dt)})] 今週はずっと休み" in text

    def test_post_without_timestamp_renders_without_prefix(self):
        """日時が取れないポストは、日付ラベルごと省いて出すこと（表示を壊さない）。"""
        text = _format_thread_with_post({
            "id": "a1b2c3d4-x", "type": "body", "summary": "体の調子",
            "latest_post": "なんとなくだるい", "latest_post_at": None,
        })
        assert text.endswith("→ なんとなくだるい")

    def test_thread_dict_carries_latest_post_timestamp(self, sqlite_store, working_memory_manager):
        """WorkingMemoryManager が最新ポストの日時を dict に載せること（表示の前提）。"""
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=1)
        thread = working_memory_manager.create_thread(
            character_id=char_id, type="task", summary="お盆休み", content="8/10から休み",
        )
        threads = working_memory_manager.list_threads_by_type(char_id, is_open=True)
        target = next(t for t in threads if t["id"] == thread["id"])
        assert target["latest_post_at"] is not None
        assert short_date(target["latest_post_at"]) != ""


# ---------------------------------------------------------------------------
# Chronicle の対象日
# ---------------------------------------------------------------------------

class TestTargetDateLabel:
    """棚卸しが「いつの出来事を扱っているか」のラベル決定の検証。

    Chronicle は前日ぶんを未明（既定 03:00）に処理するため、プロンプトで「今日」と
    書くと本人の認識が実際の1日ずれる。ずれたまま WM へ「今日」「昨日」と書かれると、
    後から何日のことか復元できなくなる。
    """

    def test_explicit_target_date_is_used(self):
        """target_date 指定時は、その日付をラベルにすること。"""
        dt = _this_year(8, 29)
        label = _target_date_label(dt.strftime("%Y-%m-%d"))
        assert label == f"08-29({japanese_weekday(dt)})"

    def test_single_day_of_records_becomes_that_day(self):
        """未処理ぶん経路で、レコードが1日に収まればその日をラベルにすること。"""
        dt = _this_year(8, 29)
        label = _target_date_label(None, [SimpleNamespace(created_at=dt)])
        assert label == f"08-29({japanese_weekday(dt)})"

    def test_multiple_days_become_a_range(self):
        """複数日にまたがる未処理ぶんは、範囲表記にすること。"""
        first, last = _this_year(8, 27), _this_year(8, 29)
        label = _target_date_label(
            None,
            [SimpleNamespace(created_at=first)],
            [SimpleNamespace(created_at=last)],
        )
        assert label == (
            f"08-27({japanese_weekday(first)})〜08-29({japanese_weekday(last)})"
        )

    def test_no_records_falls_back_to_today(self):
        """会話ゼロでもスレッド整理は走るため、当日へ倒してラベルを空にしないこと。"""
        assert _target_date_label(None, [], []) == short_date(datetime.now())


# ---------------------------------------------------------------------------
# 意図の日付表示
# ---------------------------------------------------------------------------

def _intent(intent_id: str, description: str, days_ago: int, target: str | None = None):
    """意図 ORM の代わり（表示側が触る属性だけ持つ）。"""
    return SimpleNamespace(
        id=intent_id, description=description, target=target,
        created_at=datetime.now() - timedelta(days=days_ago),
    )


class TestIntentDates:
    """意図の「いつからの〜したい」かの表示の検証。

    意図圧は経過日数で重くなる設計なのに、本人には抱えている長さが見えていなかった。
    とくに拾い上げの終端遷移（14日超）は「しばらく経っている」としか伝えておらず、
    どれだけ抱えていたか分からないまま「果たした／手放す／もどかしい」を選ばせていた。
    """

    def test_daily_block_shows_since_when(self):
        """日常注入の意図行に「◯◯から」が出ること。"""
        since = datetime.now() - timedelta(days=18)
        text = _build_motive_block(None, [{
            "description": "あの話をちゃんと伝えたい", "target": "user",
            "created_at": since.isoformat(timespec="seconds"),
        }])
        assert "相手: user" in text
        assert f"{short_date(since)}から" in text

    def test_daily_block_without_timestamp_keeps_target_only(self):
        """日時が無い意図は、相手だけを括弧に残して壊れないこと。"""
        text = _build_motive_block(None, [{"description": "散歩したい", "target": "self"}])
        assert "- 散歩したい" in text
        assert "から）" not in text

    def test_pickup_question_dates_the_stale_intents(self):
        """終端遷移の候補行に、いつから抱えているかが出ること。"""
        stale = _intent("i-1", "あの話をちゃんと伝えたい", days_ago=18)
        text = build_pickup_question([], [stale])
        assert f"（{short_date(stale.created_at)}から）" in text

    def test_pickup_question_dates_the_active_intents(self):
        """既存 active の一覧にも日付が出ること（束ね判断の材料になる）。"""
        active = _intent("i-2", "映画を観たい", days_ago=3, target="self")
        text = build_pickup_question([active], [])
        assert f"{short_date(active.created_at)}から" in text


# ---------------------------------------------------------------------------
# 棚卸しプロンプトへの統合
# ---------------------------------------------------------------------------

class TestChroniclePromptTimeAnchoring:
    """run_chronicle が組む棚卸しプロンプトの検証。

    本人が「いつの棚卸しか」を知り、スレッドに付いた日付を手がかりに相対表現を
    日付へ直せる状態になっていることまでを保証する。実際に直すかどうかは本人の
    判断であり、決定論的なテストにならないため保証範囲に含めない。
    """

    async def _capture_prompt(self, char_id, sqlite_store, working_memory_manager):
        """run_chronicle を1回まわし、棚卸しプロンプト本文を返す。"""
        captured: list[str] = []

        async def fake_generate(sys_prompt, messages):
            captured.append(messages[0]["content"])
            return _NO_UPDATE_RESPONSE

        mock_provider = AsyncMock()
        mock_provider.generate = fake_generate
        with patch("backend.services.character_query.create_provider", return_value=mock_provider):
            await run_chronicle(
                character_id=char_id, sqlite=sqlite_store,
                working_memory_manager=working_memory_manager,
            )
        return captured[0]

    @pytest.mark.asyncio
    async def test_prompt_states_which_day_it_covers(self, sqlite_store, working_memory_manager):
        """棚卸しが扱う対象日を明示すること（「今日」と書かせない）。"""
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=2)

        prompt = await self._capture_prompt(char_id, sqlite_store, working_memory_manager)

        assert "この棚卸しが扱うのは" in prompt
        assert short_date(datetime.now()) in prompt

    @pytest.mark.asyncio
    async def test_prompt_carries_the_absolute_date_rule(self, sqlite_store, working_memory_manager):
        """相対表現を日付で書くルールが載ること。"""
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=2)

        prompt = await self._capture_prompt(char_id, sqlite_store, working_memory_manager)

        assert "「いつ・誰と・どこで」を日付で留めること" in prompt
        assert "読む時点で意味がずれる言い方" in prompt

    @pytest.mark.asyncio
    async def test_thread_list_carries_dates(self, sqlite_store, working_memory_manager):
        """スレッド一覧に開始日と最新ポストの日付が添えられること。

        「この『今週』はいつの週だったのか」を本人が辿るための唯一の手がかりになる。
        """
        char_id, _, _, _ = _setup_char_with_messages(sqlite_store, "はる", n_messages=2)
        working_memory_manager.create_thread(
            character_id=char_id, type="task", summary="お盆休みの過ごし方",
            content="今週はずっと休みだから寝坊してる",
        )

        prompt = await self._capture_prompt(char_id, sqlite_store, working_memory_manager)

        today = short_date(datetime.now())
        assert f"  開始: {today}" in prompt
        assert f"  最新ポスト: [{today}] 今週はずっと休み" in prompt
