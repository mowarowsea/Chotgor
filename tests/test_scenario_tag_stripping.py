"""シナリオ／うつつ — 表示用本文からのツールタグ除去（保存経路側の保険）を検証する。

タグ除去は 2 段構えになっている:

    1. engine の StreamingTagStripper が **話者分割より前** に剥がす（主）
       → tests/test_scenario_chat_engine.py::TestTagStrippingBeforeSpeakerSplit
    2. 保存直前に extract_anticipation / extract_scene_close を **全話者ブロック** へ掛ける（保険）
       → 本ファイル

このファイルが受け持つのは 2 の層。engine を FakeEngine に差し替えて TurnRecord を
直接流し込むため、1 の stripper を意図的にバイパスした状態＝「マーカーが素通りしてきた
とき、保存の手前で止められるか」を見ている。

保険を **全ブロック** に掛ける理由:
    旧実装は SCENE_CLOSE の除去を「headless かつ最終 GM ターン 1 件」だけに掛けていた。
    しかし 1 レスポンスが複数の話者ブロックへ割れるとき、マーカーがどのブロックへ落ちるかは
    GM の書き方次第で読めない。最終ターン以外に落ちた分はそのまま画面へ出ていた。

検証する観点:
    - 先頭・中間ブロックのマーカーも除去されること（最終ブロック限定になっていない）
    - 表記揺れ（[ scene close ] / 小文字）も除去されること
    - 除去しても raw_response は汚れず、予想（anticipation）は最終ターンへ載ること
    - 通常モード（headless でない）でも除去が効くこと
"""

import asyncio

import pytest

import backend.services.scenario_chat.service as svc
from backend.services.scenario_chat.engine import EngineResult, TurnRecord

from tests._scenario_sqlite_helpers import _make_scenario, _make_session


# ─── フェイクエンジン ────────────────────────────────────────────────────────


class FakeEngine:
    """`generate_stream` で固定のアイテム列を返すエンジンスタブ。

    `_run_gm_turn` は engine の yield するオブジェクトの型で分岐するだけなので、
    TurnRecord / EngineResult を直接並べれば GM 1 レスポンス分を再現できる。
    stripper を通らない経路になるため、保存直前の保険だけを切り出して検証できる。
    """

    def __init__(self, items: list):
        self.items = items

    async def generate_stream(self, **kwargs):
        """コンストラクタで受け取ったアイテムをそのまま順に yield する。"""
        for item in self.items:
            yield item


def _narrator_record(content: str) -> TurnRecord:
    """Narrator の TurnRecord を作るショートカット。"""
    return TurnRecord(
        speaker_type="narrator",
        speaker_id=None,
        speaker_name="Narrator",
        content=content,
        is_known=True,
    )


def _run_gm(store, session_id: str, items: list) -> None:
    """`_run_gm_turn` を最後まで消費する（SSE イベントは捨てる）。"""

    async def _go():
        async for _ev, _meta in svc._run_gm_turn(
            engine=FakeEngine(items),
            scenario=None,
            npcs=[],
            history=[],
            user_message="",
            settings={},
            gm_preset_id="preset-test",
            auto_advance=False,
            synopsis_auto="",
            synopsis_manual="",
            previous_anticipation="",
            pc_summary="",
            dice_pool="",
            suppress_names=set(),
            user_speaker_name="プレイヤー",
            sqlite=store,
            session_id=session_id,
            saved_turn_ids=[],
        ):
            pass

    asyncio.run(_go())


def _prepare_session(store) -> str:
    """シナリオ + セッションを作り、セッション ID を返す。"""
    scenario = _make_scenario(store, title="タグ除去検証")
    session = _make_session(store, scenario.id)
    return session.id


# ─── 保存直前のマーカー除去 ──────────────────────────────────────────────────


class TestMarkerStrippedFromEverySpeakerBlock:
    """マーカーがどの話者ブロックに落ちても表示用 content から消えることを検証する。

    「最終ターンだけを後追いで UPDATE する」旧方式が取りこぼしていたケースを
    そのまま並べている。1 レスポンスが 3 ブロックへ割れ、マーカーは先頭・中間に
    落ちる — GM がレスポンス途中で幕を引こうとしたときに実際に起きる形。
    """

    def test_marker_in_first_block_is_removed(self, sqlite_store):
        """先頭ブロックの [SCENE_CLOSE] が除去されること（最終ブロック限定でない）。"""
        sid = _prepare_session(sqlite_store)
        items = [
            _narrator_record("一日が終わった。[SCENE_CLOSE]"),
            _narrator_record("……と思ったが、まだ続きがあった。"),
            EngineResult(raw_response="raw with [SCENE_CLOSE]"),
        ]
        _run_gm(sqlite_store, sid, items)

        turns = sqlite_store.list_scenario_turns(sid)
        assert len(turns) == 2
        assert "SCENE_CLOSE" not in (turns[0].content or "")
        assert "一日が終わった。" in turns[0].content
        # raw_response は生のまま（停止判定 _has_scene_close の材料）
        assert "[SCENE_CLOSE]" in (turns[0].raw_response or "")

    def test_lenient_marker_spelling_is_removed(self, sqlite_store):
        """表記揺れ（[ scene close ] / 小文字）も除去されること。

        揺れた表記は stripper（完全一致プレフィックス方式）をすり抜けるため、
        この保険が最後の砦になる。
        """
        sid = _prepare_session(sqlite_store)
        items = [
            _narrator_record("夜が更けた。[ scene close ]"),
            _narrator_record("窓の外は静かだった。[Scene_Close]"),
            EngineResult(raw_response="raw"),
        ]
        _run_gm(sqlite_store, sid, items)

        turns = sqlite_store.list_scenario_turns(sid)
        assert len(turns) == 2
        assert all("scene" not in (t.content or "").lower() for t in turns)
        assert "夜が更けた。" in turns[0].content
        assert "窓の外は静かだった。" in turns[1].content

    def test_anticipate_debris_in_any_block_is_removed(self, sqlite_store):
        """ブロック内で閉じている予想タグは、最終ブロックでなくても除去されること。"""
        sid = _prepare_session(sqlite_store)
        items = [
            _narrator_record("朝。[ANTICIPATE_RESPONSE:出勤するだろう。]"),
            _narrator_record("昼になった。"),
            EngineResult(raw_response="raw"),
        ]
        _run_gm(sqlite_store, sid, items)

        turns = sqlite_store.list_scenario_turns(sid)
        assert "ANTICIPATE_RESPONSE" not in (turns[0].content or "")
        assert "朝。" in turns[0].content


class TestAnticipationStillTakenFromRaw:
    """本文からタグを消しても、予想の採用（raw 基準）が壊れないことを検証する。

    予想は「最後の話者ブロック」の anticipation カラムへ入り、次レスポンスの GM
    プロンプトへ注入される。表示の掃除と意味の抽出はここで初めて合流するので、
    片方を直したせいで他方が死んでいないかを見る。
    """

    def test_anticipation_saved_on_last_turn_and_absent_from_content(self, sqlite_store):
        """raw の予想が最終ターンへ保存され、どの content にも残らないこと。"""
        sid = _prepare_session(sqlite_store)
        raw = (
            "@Narrator: 朝。\n"
            "@Narrator: 昼。\n"
            "[ANTICIPATE_RESPONSE:\n@Narrator:\n（次は昼食に出るだろう。）]"
        )
        items = [
            _narrator_record("朝。"),
            _narrator_record("昼。"),
            EngineResult(raw_response=raw),
        ]
        _run_gm(sqlite_store, sid, items)

        turns = sqlite_store.list_scenario_turns(sid)
        assert len(turns) == 2
        assert turns[0].anticipation in (None, "")
        assert "昼食に出るだろう" in (turns[1].anticipation or "")
        assert all("ANTICIPATE_RESPONSE" not in (t.content or "") for t in turns)


@pytest.mark.parametrize("marker", ["[SCENE_CLOSE]", "[ANTICIPATE_RESPONSE:予想。]"])
def test_normal_mode_also_strips_markers(sqlite_store, marker):
    """通常モード（headless でない普通のシナリオ）でも除去が効くこと。

    旧実装の SCENE_CLOSE 除去は `sc.is_headless` の内側にあり、対面プレイでは
    そもそも走らなかった。除去を保存経路へ移したことで、モードに関係なく効く。
    """
    sid = _prepare_session(sqlite_store)
    items = [
        _narrator_record(f"扉が閉まった。{marker}"),
        EngineResult(raw_response="raw"),
    ]
    _run_gm(sqlite_store, sid, items)

    turns = sqlite_store.list_scenario_turns(sid)
    assert len(turns) == 1
    assert "SCENE_CLOSE" not in (turns[0].content or "")
    assert "ANTICIPATE_RESPONSE" not in (turns[0].content or "")
    assert "扉が閉まった。" in turns[0].content
