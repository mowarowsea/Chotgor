"""シナリオ用 SceneLoop Strategy。

シナリオ／うつつの「GM↔PC↔NPC を順次回す外側ループ」を chat_flow.SceneLoop の
TurnRouter / TurnExecutor に切り出したもの。``run_scenario_turn`` から呼び出され、
ループ制御は SceneLoop に委ね、Router/Executor がメンション解析・SCENE_CLOSE 抑止・
GM/PC ターン実行といったシナリオ固有の責務を担う。

設計メモ:
    - Router/Executor が共有する可変状態は ``ScenarioLoopState`` にまとめ、
      ``LoopState.context["scenario_state"]`` で渡す。これにより SceneLoop 側は
      シナリオ固有の概念を一切持たない。
    - Executor 内で SQLite への保存・ログ feature 切替・SCENE_CLOSE マーカー除去等の
      副作用も完結させる。Router は副作用を持たず、「次の話者は誰か」だけを判断する。
    - 循環 import を避けるため、scenario_chat.service の関数・定数は関数内で
      遅延 import する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from backend.lib.log_context import current_log_feature
from backend.services.chat_flow.scene_loop import (
    LoopState,
    SpeakerInfo,
    TurnResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ScenarioLoopState:
    """SceneLoop へ渡すシナリオ進行コンテキスト。

    ``LoopState.context["scenario_state"]`` で Router / Executor が共有する。
    保持する値は概ね 2 系統:
        - 不変パラメータ（セッション・シナリオ・PC 枠・OOC など、シーン中変わらない）
        - 進行可変（saved_turn_ids / fired_responses / pc_responses /
          last_speaker_name など、ターンごとに更新）
    """

    # 依存（不変）
    sqlite: Any
    settings: dict
    engine: Any
    chat_service: Any

    # シナリオ／セッション情報（不変）
    session_id: str
    session: Any
    scenario: Any
    npcs: list
    npc_names: set
    pcs: list
    routing_pcs: list
    pc_summary_text: str
    user_speaker_name: str
    suppress_names: set
    gm_preset_id: str
    current_synopsis: dict

    # モード判定（不変）
    auto_advance: bool
    is_headless: bool
    is_pc_mode: bool
    max_responses: int

    # うつつ固有（不変）
    usual_time_context: str = ""
    absent_user_block: str = ""
    extra_first_gm_ooc: str = ""

    # GM プロンプト補助（不変）
    previous_anticipation: str = ""
    user_message: str = ""

    # 初動ルーティング（不変）
    initial_next_kind: str = "gm"
    initial_next_target: str | None = None

    # 進行可変
    saved_turn_ids: list = field(default_factory=list)
    fired_responses: int = 0
    pc_responses: int = 0
    last_speaker_name: str | None = None
    # SCENE_CLOSE を抑止したターンの直後は、次のルーティングを強制 @ALL にフォールバックする。
    # （Executor 側で True にセットされ、Router 側で消費する）
    scene_close_suppressed: bool = False


class ScenarioRouter:
    """シナリオ用 TurnRouter。SCENE_CLOSE 検出とメンション解析で次の話者を決める。"""

    async def stop_condition(self, state: LoopState) -> tuple[bool, str]:
        """シナリオ固有の停止条件を判断する。

        - ``fired_responses`` が ``max_responses`` に達したら停止
        - headless かつ最終 GM ターンの生応答が ``[SCENE_CLOSE]`` を含み、
          かつ主人公 PC が一度でも発話していれば停止（早すぎる SCENE_CLOSE は Executor 側で抑止）
        """
        sc: ScenarioLoopState = state.context["scenario_state"]

        if sc.fired_responses >= sc.max_responses:
            return True, "max_responses"

        if (
            sc.is_headless
            and state.last_speaker is not None
            and state.last_speaker.kind == "gm"
            and state.last_result is not None
        ):
            from backend.services.scenario_chat.service import _has_scene_close

            if _has_scene_close(state.last_result.raw) and sc.pc_responses > 0:
                return True, "scene_close"

        return False, ""

    async def next_speaker(self, state: LoopState) -> SpeakerInfo | None:
        """次の話者を決定する。

        - 初回（iteration=0）は ``initial_next_kind`` / ``initial_next_target`` を採用
        - 2 回目以降は前ターン話者の kind に応じて _next_after_gm / _next_after_pc を呼ぶ
        - ``"all"`` を解決して PC 名に確定
        - PC がユーザ枠の場合は ``None`` を返してループを終了（ユーザ入力待ちへ）
        """
        sc: ScenarioLoopState = state.context["scenario_state"]
        from backend.services.scenario_chat.mention import pick_at_all_target

        if state.iteration == 0:
            kind = sc.initial_next_kind
            target = sc.initial_next_target
        else:
            last_speaker = state.last_speaker
            last_result = state.last_result
            if last_speaker is None:
                return None
            if last_speaker.kind == "gm":
                kind, target = self._next_after_gm(sc, last_result)
            elif last_speaker.kind == "pc":
                kind, target = self._next_after_pc(sc, last_result)
            else:
                kind, target = "none", None

        # @ALL を PC に解決する（直前話者を除いたランダム選択）。
        if kind == "all":
            pc = pick_at_all_target(sc.routing_pcs, last_speaker_name=sc.last_speaker_name)
            if pc is None or pc.is_user:
                return None
            kind = "pc"
            target = pc.name

        if kind == "gm":
            return SpeakerInfo(kind="gm", name="GM")
        if kind == "pc":
            pc = next((p for p in sc.routing_pcs if p.name == target), None)
            if pc is None or pc.is_user:
                return None
            return SpeakerInfo(
                kind="pc",
                name=pc.name,
                id=pc.character_id,
                metadata={"pc": pc},
            )
        return None

    def _next_after_gm(
        self, sc: ScenarioLoopState, last_result
    ) -> tuple[str, str | None]:
        """GM 直後の次の話者ルーティング。

        - PC モードでない／GM が無応答 → ``"none"``
        - GM の最終 raw からメンション解析。``pc``/``all`` 以外なら @ALL フォールバック
          （GM が NPC を呼び合うだけのループを防止）
        - SCENE_CLOSE 抑止フラグが立っていたら強制 @ALL（抑止フラグを消費する）
        """
        from backend.services.scenario_chat.mention import find_last_routing_mention

        raw = last_result.raw if last_result else ""

        if sc.scene_close_suppressed:
            sc.scene_close_suppressed = False
            if sc.routing_pcs:
                return "all", None
            return "none", None

        if not sc.is_pc_mode or not raw:
            return "none", None

        kind, target = find_last_routing_mention(raw, sc.routing_pcs, sc.npc_names)
        if kind not in {"pc", "all"}:
            if sc.routing_pcs:
                return "all", None
            return "none", None
        return kind, target

    def _next_after_pc(
        self, sc: ScenarioLoopState, last_result
    ) -> tuple[str, str | None]:
        """PC 直後の次の話者ルーティング。

        - PC 末尾のメンション解析。``"none"`` の場合、headless なら GM に継続、
          通常モードならユーザ待ちで終了する。
        """
        from backend.services.scenario_chat.mention import find_last_routing_mention

        text = last_result.text if last_result else ""
        kind, target = find_last_routing_mention(text, sc.routing_pcs, sc.npc_names)
        if kind == "none":
            if sc.is_headless:
                return "gm", None
        return kind, target


class ScenarioTurnExecutor:
    """シナリオ用 TurnExecutor。speaker.kind に応じて GM / PC ターンを実行する。"""

    async def execute(
        self,
        speaker: SpeakerInfo,
        state: LoopState,
    ) -> AsyncIterator[tuple[str, Any]]:
        """speaker.kind を見て GM / PC のいずれかを実行する。"""
        sc: ScenarioLoopState = state.context["scenario_state"]

        if speaker.kind == "gm":
            async for ev in self._run_gm(state, sc):
                yield ev
        elif speaker.kind == "pc":
            async for ev in self._run_pc(speaker, state, sc):
                yield ev
        else:
            yield (
                "turn_result",
                TurnResult(text="", error=f"unknown speaker kind: {speaker.kind}"),
            )

    async def _run_gm(
        self,
        state: LoopState,
        sc: ScenarioLoopState,
    ) -> AsyncIterator[tuple[str, Any]]:
        """GM レスポンス 1 回分を実行する（旧 run_scenario_turn の "next_kind == gm" ブロック相当）。

        SCENE_CLOSE 検出時の表示用 content マーカー除去、早すぎる SCENE_CLOSE 抑止、
        feature ログの "usual_days" 復帰、ターン副産物のカウントもここで完結させる。
        """
        from backend.services.scenario_chat.engine import generate_dice_pool
        from backend.services.scenario_chat.service import (
            _build_usual_gm_appendix,
            _has_scene_close,
            _run_gm_turn,
            extract_scene_close,
        )

        # GM ターン中にプロバイダエラーを検知したら保持する（末尾でシーンを閉じる判定に使う）。
        gm_error: str | None = None

        # PC ターンが feature を "usual_days_pc" に書き換えているため、GM ターン側で
        # "usual_days" へ戻す（ログの取り違い防止）。
        if sc.is_headless:
            current_log_feature.set("usual_days")

        # うつつ向け OOC 追記（常設フレーミング・偶発イベント・ソフト収束ヒント）
        gm_ooc = ""
        if sc.is_headless:
            gm_ooc = _build_usual_gm_appendix(
                sc.scenario,
                sc.fired_responses,
                sc.max_responses,
                is_first_gm=(sc.fired_responses == 0),
                absent_user_block=sc.absent_user_block,
            )
            if sc.fired_responses == 0 and sc.extra_first_gm_ooc.strip():
                gm_ooc = (sc.extra_first_gm_ooc.strip() + "\n" + gm_ooc).strip()

        async for ev, _meta in _run_gm_turn(
            engine=sc.engine,
            scenario=sc.scenario,
            npcs=sc.npcs,
            history=sc.sqlite.list_scenario_turns(sc.session_id),
            user_message=sc.user_message if sc.fired_responses == 0 else "",
            settings=sc.settings,
            gm_preset_id=sc.gm_preset_id,
            auto_advance=sc.auto_advance if sc.fired_responses == 0 else True,
            synopsis_auto=sc.current_synopsis.get("auto", ""),
            synopsis_manual=sc.current_synopsis.get("manual", ""),
            previous_anticipation=sc.previous_anticipation,
            pc_summary=sc.pc_summary_text,
            dice_pool=(
                generate_dice_pool(getattr(sc.scenario, "dice_pool_spec", None))
                if (sc.is_pc_mode and not sc.is_headless)
                else ""
            ),
            suppress_names=sc.suppress_names,
            user_speaker_name=sc.user_speaker_name,
            sqlite=sc.sqlite,
            session_id=sc.session_id,
            saved_turn_ids=sc.saved_turn_ids,
            time_context=sc.usual_time_context,
            gm_ooc_appendix=gm_ooc,
        ):
            # プロバイダエラー（503 等）は _run_gm_turn が ("error", {...}) を 1 度だけ
            # 流して scenario_turn を保存せず return する。ここで捕捉しておかないと、
            # 後段で古い GM ターンの raw を拾って誤ルーティングしたままループが継続し、
            # max_responses まで暴走する（PC ターンと非対称な穴だった）。
            if isinstance(ev, tuple) and ev and ev[0] == "error":
                payload = ev[1] if len(ev) > 1 and isinstance(ev[1], dict) else {}
                gm_error = payload.get("message") or "GM provider error"
            yield ev

        # GM がエラー回答だった場合は、そのシーンをここで閉じる（後続ターンへ進めない）。
        # error 付き TurnResult を返すと SceneLoop が done_reason="executor_error" で
        # ループを脱出する。古い raw でのメンション誤ルーティングもこれで止まる。
        if gm_error is not None:
            yield ("turn_result", TurnResult(text="", error=gm_error))
            return

        # GM の最終 raw_response を直近保存ターン（=最後の話者ブロック）から取り直す。
        latest = sc.sqlite.list_scenario_turns(sc.session_id)
        gm_last_raw = ""
        gm_last_name = None
        gm_last_turn = None
        for t in reversed(latest):
            if getattr(t, "speaker_type", "") in {"narrator", "npc"}:
                gm_last_raw = getattr(t, "raw_response", "") or ""
                gm_last_name = getattr(t, "speaker_name", "") or sc.last_speaker_name
                gm_last_turn = t
                break

        # SCENE_CLOSE 検出時の表示用 content マーカー除去 & 早すぎる SCENE_CLOSE 抑止判定。
        if sc.is_headless and _has_scene_close(gm_last_raw):
            if gm_last_turn is not None:
                cleaned, _ = extract_scene_close(getattr(gm_last_turn, "content", "") or "")
                sc.sqlite.update_scenario_turn(gm_last_turn.id, content=cleaned)
            if sc.pc_responses == 0 and sc.routing_pcs:
                # 早すぎる: Router 側で @ALL フォールバックさせるためフラグを立てる。
                # stop_condition 側は pc_responses==0 のとき停止しないので、自然に継続する。
                sc.scene_close_suppressed = True
                logger.info(
                    "うつつ: GM の早すぎる SCENE_CLOSE を抑止（主人公が未発話）"
                    " session=%s fired=%d",
                    sc.session_id,
                    sc.fired_responses + 1,
                )
            else:
                logger.info(
                    "うつつ: GM が SCENE_CLOSE を宣言 session=%s fired=%d",
                    sc.session_id,
                    sc.fired_responses + 1,
                )

        if gm_last_name:
            sc.last_speaker_name = gm_last_name
        sc.fired_responses += 1

        # raw のみ TurnResult に格納（メンション解析と SCENE_CLOSE 判定で使う）。
        yield ("turn_result", TurnResult(text="", raw=gm_last_raw))

    async def _run_pc(
        self,
        speaker: SpeakerInfo,
        state: LoopState,
        sc: ScenarioLoopState,
    ) -> AsyncIterator[tuple[str, Any]]:
        """PC レスポンス 1 回分を実行する（旧 run_scenario_turn の "next_kind == pc" ブロック相当）。"""
        from backend.services.scenario_chat.pc_runner import stream_pc_response
        from backend.services.scenario_chat.serializers import scenario_turn_to_dict
        from backend.services.scenario_chat.service import _resolve_pc_preset_id, _save_turn

        pc = speaker.metadata["pc"]

        if sc.chat_service is None:
            logger.warning(
                "PC レスポンス実行不可: chat_service が None session=%s pc=%s",
                sc.session_id,
                pc.name,
            )
            yield ("turn_result", TurnResult(text="", error="chat_service unavailable"))
            return

        preset_id = _resolve_pc_preset_id(pc, sc.sqlite)
        if not preset_id:
            logger.warning(
                "PC 用プリセット未解決 session=%s slot=%s character_id=%s",
                sc.session_id,
                pc.slot_id,
                pc.character_id,
            )
            yield ("turn_result", TurnResult(text="", error="preset unresolved"))
            return

        latest_history = sc.sqlite.list_scenario_turns(sc.session_id)

        yield ("turn_start", {"character": pc.name, "character_id": pc.character_id})

        full_text = ""
        anticipation_text: str | None = None
        try:
            async for ev_type, payload in stream_pc_response(
                pc=pc,
                scenario_title=sc.scenario.title,
                user_alias=sc.user_speaker_name,
                history=latest_history,
                preset_id=preset_id,
                sqlite=sc.sqlite,
                settings=sc.settings,
                chat_service=sc.chat_service,
                scenario_session_id=sc.session_id,
                default_origin="usual" if sc.is_headless else "interlude",
            ):
                if ev_type == "pc_done":
                    full_text = payload["full_text"]
                    anticipation_text = payload.get("anticipation")
                yield (ev_type, payload)
        except Exception as e:
            logger.exception(
                "PC レスポンス実行エラー session=%s pc=%s",
                sc.session_id,
                pc.name,
            )
            yield (
                "error",
                {
                    "character": pc.name,
                    "character_id": pc.character_id,
                    "message": str(e),
                },
            )
            yield ("turn_result", TurnResult(text="", error=str(e)))
            return

        if full_text.strip():
            saved = _save_turn(
                sqlite=sc.sqlite,
                session_id=sc.session_id,
                speaker_type="pc",
                speaker_name=pc.name,
                content=full_text,
                speaker_id=pc.character_id,
                attach_log_request_id=True,
                anticipation=anticipation_text,
            )
            sc.saved_turn_ids.append(saved.id)
            yield ("turn_end", {"turn": scenario_turn_to_dict(saved)})

        sc.last_speaker_name = pc.name
        sc.fired_responses += 1
        # キャラ PC が実際に発話した → 以降の GM SCENE_CLOSE は正規に受理される。
        sc.pc_responses += 1

        yield ("turn_result", TurnResult(text=full_text, raw=full_text))


__all__ = ["ScenarioLoopState", "ScenarioRouter", "ScenarioTurnExecutor"]
