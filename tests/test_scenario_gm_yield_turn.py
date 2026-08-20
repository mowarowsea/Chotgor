"""GM がターンを明け渡したときのルーティング（ScenarioRouter）のテスト。

GM が行頭 `@<PC名>:` を書くのは「ここは PC が喋る番」という判断であり、その後ろに
GM が続けた描写は「まだ起きていない出来事」になる。パーサ／エンジン側の破棄と
ストリーム打ち切りは test_scenario_chat_parser / test_scenario_chat_engine が担当し、
本ファイルは **明け渡し先へ正しくルーティングされるか** だけを検証する。

検証する観点:
    - `yielded_to_name` が PC枠名／PCキャラ本名のどちらでも当該 PC へ解決されること
    - raw の末尾に GM 系メンション（@Narrator / NPC名）があっても明け渡しが優先されること
      ── 旧実装ではここで末尾解析が GM 系を拾い、指名した PC ではなく @ALL ランダム
         抽選へ落ちる誤ルーティングが起きていた（本ファイル最大の回帰対象）
    - フラグが 1 度で消費され、次のターンは通常の末尾メンション解析へ戻ること
    - 譲渡先がルーティング候補に居ない場合（うつつの不在ユーザPC）は @ALL へ倒すこと
    - 通常モードのユーザPC 指名はループを終了してユーザ入力待ちにすること
    - SCENE_CLOSE 抑止フラグと同時に立っても明け渡しが優先されること
"""

import asyncio
from unittest.mock import MagicMock

from backend.services.chat_flow.scene_loop import LoopState, SpeakerInfo, TurnResult
from backend.services.scenario_chat.loop_strategies import (
    ScenarioLoopState,
    ScenarioRouter,
)
from backend.services.scenario_chat.mention import PcAssignment


def _pc_char(name: str, character_name: str | None = None) -> PcAssignment:
    """AI キャラが演じる PC枠。`@<枠名>` でも `@<キャラ本名>` でも解決される。"""
    return PcAssignment(
        slot_id=f"slot-{name}",
        name=name,
        player_type="character",
        character_id=f"cid-{name}",
        character_name=character_name or name,
    )


def _pc_user(name: str) -> PcAssignment:
    """ユーザが演じる PC枠。"""
    return PcAssignment(slot_id=f"slot-{name}", name=name, player_type="user")


def _make_state(**overrides) -> ScenarioLoopState:
    """Router のルーティング判定に必要な最小 ScenarioLoopState を組む。

    Router は依存系（sqlite / engine / chat_service）に触らないため MagicMock で埋める。
    """
    defaults = dict(
        sqlite=MagicMock(),
        settings={},
        engine=MagicMock(),
        chat_service=MagicMock(),
        session_id="session-1",
        session=MagicMock(),
        scenario=MagicMock(),
        npcs=[],
        npc_names={"レイカ"},
        pcs=[],
        routing_pcs=[],
        pc_summary_text="",
        user_speaker_name="もわ",
        suppress_names=set(),
        gm_preset_id="",
        current_synopsis={},
        auto_advance=True,
        is_headless=False,
        is_pc_mode=True,
        max_responses=10,
    )
    defaults.update(overrides)
    return ScenarioLoopState(**defaults)


def _route_after_gm(sc: ScenarioLoopState, raw: str) -> tuple[str, str | None]:
    """GM ターン直後のルーティング判定を 1 回実行する。"""
    return ScenarioRouter()._next_after_gm(sc, TurnResult(text="", raw=raw))


def _next_speaker(sc: ScenarioLoopState, raw: str) -> SpeakerInfo | None:
    """GM ターン直後の next_speaker を 1 回実行する（iteration=1 の経路）。"""
    state = LoopState(
        iteration=1,
        last_speaker=SpeakerInfo(kind="gm", name="GM"),
        last_result=TurnResult(text="", raw=raw),
        context={"scenario_state": sc},
    )
    return asyncio.run(ScenarioRouter().next_speaker(state))


class TestYieldToPcRouting:
    """明け渡し先の解決（枠名・キャラ本名）と、末尾メンション解析に対する優先度。

    GM が `@<PC名>:` を書いた事実は `yielded_to_name` として Executor から Router へ
    渡される。Router はこれを raw の末尾メンション解析より先に消費する。
    """

    def test_枠名での指名が当該PCへ解決される(self):
        alice = _pc_char("アリス")
        sc = _make_state(routing_pcs=[alice], yielded_to_name="アリス")
        assert _route_after_gm(sc, "@アリス: ") == ("pc", "アリス")

    def test_キャラ本名での指名も枠名へ正規化される(self):
        """GM が配役名でなく中の人の本名を書いた場合も同じ PC へ渡す。"""
        alice = _pc_char("アリス", character_name="はる")
        sc = _make_state(routing_pcs=[alice], yielded_to_name="はる")
        assert _route_after_gm(sc, "@はる: ") == ("pc", "アリス")

    def test_指名後のGM系メンションより明け渡しが優先される(self):
        """raw 末尾が `@Narrator:` でも、指名した PC へ渡すこと。

        旧実装は raw を後ろから走査して最後のメンションを採用していたため、
        GM が指名の後ろに地の文を続けると @ALL ランダム抽選へ落ちていた。
        """
        alice = _pc_char("アリス")
        bob = _pc_char("ボブ")
        sc = _make_state(routing_pcs=[alice, bob], yielded_to_name="アリス")
        raw = "@アリス: どうする?\n@Narrator: 沈黙が落ちた。\n"
        assert _route_after_gm(sc, raw) == ("pc", "アリス")

    def test_指名後のNPCメンションより明け渡しが優先される(self):
        alice = _pc_char("アリス")
        bob = _pc_char("ボブ")
        sc = _make_state(routing_pcs=[alice, bob], yielded_to_name="アリス")
        raw = "@アリス: どうする?\n@レイカ: 早くしろ。\n"
        assert _route_after_gm(sc, raw) == ("pc", "アリス")

    def test_next_speakerがPCのSpeakerInfoを返す(self):
        """Router 全体を通しても当該 PC の SpeakerInfo になること。"""
        alice = _pc_char("アリス")
        sc = _make_state(routing_pcs=[alice], yielded_to_name="アリス")
        speaker = _next_speaker(sc, "@アリス: ")
        assert speaker is not None
        assert speaker.kind == "pc"
        assert speaker.name == "アリス"
        assert speaker.id == "cid-アリス"
        assert speaker.metadata["pc"] is alice


class TestYieldFlagConsumption:
    """明け渡しフラグが 1 度で消費され、次ターンへ持ち越されないこと。

    持ち越すと「GM が一度 PC を指名したら以降ずっとその PC へ回る」状態になり、
    通常の末尾メンション解析（GM の意図するその都度のルーティング）が死ぬ。
    """

    def test_1度で消費される(self):
        alice = _pc_char("アリス")
        sc = _make_state(routing_pcs=[alice], yielded_to_name="アリス")
        _route_after_gm(sc, "@アリス: ")
        assert sc.yielded_to_name is None

    def test_消費後は末尾メンション解析へ戻る(self):
        """2 度目の GM ターンは raw の末尾メンションで判定されること。"""
        alice = _pc_char("アリス")
        bob = _pc_char("ボブ")
        sc = _make_state(routing_pcs=[alice, bob], yielded_to_name="アリス")
        assert _route_after_gm(sc, "@アリス: ") == ("pc", "アリス")
        # 次ターンは明け渡しなし。raw 末尾の指名（ボブ）が採用される。
        assert _route_after_gm(sc, "@Narrator: 扉が開く。@ボブ、君は?\n") == ("pc", "ボブ")


class TestYieldTargetUnavailable:
    """譲渡先がルーティング候補に居ない／ユーザ枠のときの倒し方。

    うつつ（headless）ではユーザPC を `routing_pcs` から外しているため、GM が万一
    `@<ユーザ>:` を書いても譲れない。GM の出力は既に打ち切っているので、場を止めず
    @ALL へ倒す（不在の相手を待たない）。通常モードのユーザPC 指名は逆に、ユーザへ
    ターンが渡る＝ループを終了して入力待ちにする。
    """

    def test_うつつの不在ユーザ指名はALLへ倒れる(self):
        haru = _pc_char("はる")
        sc = _make_state(
            is_headless=True,
            routing_pcs=[haru],  # ユーザPC は除外済み
            yielded_to_name="もわ",
            user_speaker_name="もわ",
        )
        assert _route_after_gm(sc, "@もわ: ") == ("all", None)

    def test_ルーティング候補が空ならnone(self):
        """PC が 1 人も居ないシナリオでは @ALL へも倒せない。"""
        sc = _make_state(routing_pcs=[], yielded_to_name="もわ")
        assert _route_after_gm(sc, "@もわ: ") == ("none", None)

    def test_通常モードのユーザPC指名はループを終える(self):
        """ユーザ枠へ渡す＝ユーザ入力待ち。next_speaker が None を返すこと。"""
        user_pc = _pc_user("もわ")
        alice = _pc_char("アリス")
        sc = _make_state(
            is_headless=False,
            routing_pcs=[user_pc, alice],
            yielded_to_name="もわ",
            user_speaker_name="もわ",
        )
        assert _route_after_gm(sc, "@もわ: ") == ("none", None)

        sc.yielded_to_name = "もわ"
        assert _next_speaker(sc, "@もわ: ") is None


class TestYieldVersusSceneCloseSuppression:
    """SCENE_CLOSE 抑止フラグと同時に立ったときの優先順位。

    抑止の目的は「主人公が未発話なのに GM がシーンを閉じる」のを防ぐこと。GM が
    SCENE_CLOSE と PC 指名を同時に書いた場合、指名へ渡せばまさに主人公が発話するので、
    @ALL ランダム抽選へ倒すより明け渡しを優先するほうが目的に適う。
    """

    def test_明け渡しが抑止フラグより優先される(self):
        alice = _pc_char("アリス")
        bob = _pc_char("ボブ")
        sc = _make_state(
            is_headless=True,
            routing_pcs=[alice, bob],
            yielded_to_name="アリス",
            scene_close_suppressed=True,
        )
        assert _route_after_gm(sc, "@アリス: ") == ("pc", "アリス")

    def test_明け渡しが無ければ抑止フラグが効く(self):
        alice = _pc_char("アリス")
        sc = _make_state(
            is_headless=True,
            routing_pcs=[alice],
            scene_close_suppressed=True,
        )
        assert _route_after_gm(sc, "@Narrator: 幕が下りる。\n") == ("all", None)
        assert sc.scene_close_suppressed is False
