"""シナリオチャットサービス — API 層から呼ばれるファサード。

1 ユーザターン分のレスポンス連鎖をストリーム実行する非同期ジェネレータ `run_scenario_turn` を提供する。

責務:
    - プレイヤー発話を scenario_turns に保存
    - EnsembleEngine を駆動して話者単位の SSE イベントを生成
    - 各発話を scenario_turns に保存
    - raw_response はそのレスポンス内の話者ブロック群（=ターン群）に共通で紐付ける

非同期ジェネレータで SSE 用イベントを yield することで、API 側の StreamingResponse 実装を一貫させる。

周辺責務は分割済み（本モジュールが後方互換で再エクスポートする）:
    - serializers.py    — ORM → dict 変換・ユーザ話者名解決
    - auto_synopsis.py  — あらすじ自動更新トリガー・進捗計算
    - turns.py          — ターン保存ヘルパ・intro 展開
    - scene_close.py    — [SCENE_CLOSE] マーカーの検出・除去
    - usual_days.py     — うつつ（無人生活シーン）のセッション管理・演出素材・シーン駆動
"""

import logging
from typing import Any, AsyncGenerator

from backend.character_actions.anticipator import extract_anticipation
from backend.lib.log_context import current_log_feature, current_log_session_id, new_message_id
from backend.lib.time_awareness import format_time_context
from backend.lib.tool_event_recorder import record_tool_event
from backend.services.scenario_chat.engine import (
    EngineResult,
    EnsembleEngine,
    SceneEngine,
    TurnRecord,
)
from backend.services.scenario_chat.parser import UtteranceDelta

# 後方互換の再エクスポート: API 層・テストは従来通り service 経由で import できる
from backend.services.scenario_chat.auto_synopsis import (  # noqa: F401
    SYNOPSIS_AUTO_TRIGGER_RATIO,
    compute_synopsis_progress,
    maybe_update_auto_synopsis,
)
from backend.services.scenario_chat.serializers import (  # noqa: F401
    resolve_user_speaker_name,
    scenario_npc_to_dict,
    scenario_session_to_dict,
    scenario_to_dict,
    scenario_turn_to_dict,
)
from backend.services.scenario_chat.turns import (  # noqa: F401
    _save_turn,
    parse_intro_to_turns,
    seed_intro_turns,
)
from backend.services.scenario_chat.scene_close import (  # noqa: F401
    _has_scene_close,
    extract_scene_close,
)
from backend.services.scenario_chat.usual_days import (  # noqa: F401
    _DEFAULT_USUAL_MAX_RESPONSES,
    _build_absent_user_block,
    _build_real_contact_block,
    _build_usual_gm_appendix,
    _build_usual_pc_assignments,
    _usual_event_categories,
    _usual_event_probability,
    ensure_usual_session,
    roll_usual_event,
    run_usual_days_scene,
    sync_usual_session_presets,
    usual_elapsed_note,
)

logger = logging.getLogger(__name__)


def _build_default_engine(sqlite) -> SceneEngine:
    """SQLite を preset_loader として束ねる既定 EnsembleEngine を作る。"""
    def loader(preset_id: str):
        return sqlite.get_model_preset(preset_id)
    return EnsembleEngine(preset_loader=loader)




def _latest_scenario_anticipation(history: list) -> str:
    """シナリオ履歴から直近の非空 anticipation（GM の前回予想）を返す。無ければ空文字列。

    Args:
        history: ScenarioTurn ORM の時系列昇順リスト。

    Returns:
        直近ターンに保存された予想文字列。無ければ空文字列。
    """
    for turn in reversed(history):
        anticipation = getattr(turn, "anticipation", None)
        if anticipation:
            return anticipation
    return ""


# 1 ユーザターンあたりの最大レスポンス数（GM + PC の LLM 呼出合計）。
# 無限連鎖防止用ガード。
# 用語: 「レスポンス」= LLM 1 呼出 (= 1 raw_response)。
#       「ターン」= @話者: ブロック単位 (scenario_turns 1 行)。両者は別軸。
_MAX_RESPONSES_PER_USER_TURN = 10


async def run_scenario_turn(
    session_id: str,
    user_message: str,
    sqlite,
    settings: dict,
    engine: SceneEngine | None = None,
    auto_advance: bool = False,
    chat_service=None,
    headless: bool = False,
    extra_first_gm_ooc: str = "",
    yield_to: str | None = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """ユーザ発話を受け取り、シナリオ 1 ユーザターン分の SSE イベントを順次 yield する。

    用語: 「レスポンス」= LLM 1 呼出 (= 1 raw_response)。GM 1 回・PC 1 回をそれぞれ 1 レスポンスと数える。
          「ターン」= @話者: ブロック単位 (scenario_turns 1 行)。GM の 1 レスポンス内に複数ターンが入りうる。

    プレイセッション（scenario_sessions）から元シナリオ（scenarios）を lookup し、
    そのシナリオの NPC・PC枠・GM プリセット等で 1 ユーザターン分のレスポンス連鎖を進行させる。

    動作モード:
        - engine_type == "ensemble":
            - 従来通り GM 1 レスポンスのみを実行してユーザ入力待ちへ戻す。
        - engine_type == "ensemble_pc":
            - メンション主導の話者ループで動作する:
              1. ユーザ発話末尾のメンションを解析して次話者を決定。
              2. メンション無し / @GM / @Narrator / @NPC → GM レスポンス。
              3. @<PC枠名> / @<キャラ本名> → 該当 PC レスポンス（直接ディスパッチ）。
              4. @ALL → 直前話者を除いた PC から random.choice。
              5. 各話者の発話末尾を再度メンション解析して次話者を決める。
              6. 次話者が「ユーザ枠」または「メンション無し」になればループ終了。
              7. 上限 `_MAX_RESPONSES_PER_USER_TURN` で打ち切る（無限連鎖防止）。

    PC レスポンス実行のため `chat_service` を渡すこと（None だと PC 行きの分岐は
    スキップされて GM のみ進む）。

    auto_advance=True の場合:
        - user_message は無視される（呼び出し側は空文字を渡してよい）
        - user turn は保存されない
        - SSE の "user_saved" イベントも発火しない
        - GM プロンプト末尾に「プレイヤーは無言、場面を進めて」という OOC 指示が入る

    yield_to は ensemble_pc の「ターンを譲る」UI 向け。auto_advance=True と組み合わせて使い、
    初動ルーティングをユーザが直接指定する。値の意味:
        - PC枠名（pc_slot.name）: そのPCに直接振る（@<PC>相当）。
        - "ALL": ランダムPC（@ALL相当）。
        - "GM" / None / 解決不能な値: 従来どおりの GM 行き。
    ensemble モード（GMのみ）や auto_advance=False のときは無視される。

    headless=True（うつつ / Usual Days 無人ループ）の場合:
        - engine_type=="usual_days" でも自動的に headless 扱いになる。
        - ユーザ枠ゼロを許容し、ユーザの介入なしに GM↔PC を連鎖させ続ける。
        - PC 発話末尾にメンションが無くても break せず GM レスポンスへ継続する
          （無人でも場面が止まらないようにする）。
        - 停止条件は GM 出力の `[SCENE_CLOSE]` 検出（主）か、
          `usual_config.max_responses_per_scene`（既定 _DEFAULT_USUAL_MAX_RESPONSES）への到達（保険）。
          旧キー名 max_turns_per_scene は後方互換のため読み出し時にフォールバック。
        - 記憶/スレッドは origin="usual" で保存される（stream_pc_response 経由）。
        - 通常 auto_advance=True と併用して呼ぶ（無人なのでユーザ発話保存をしない）。
    """
    current_log_feature.set("scenario_chat")
    session = sqlite.get_scenario_session(session_id)
    if not session:
        yield ("error", {"message": f"セッション '{session_id}' が見つかりません"})
        return
    if session.status != "active":
        yield ("error", {"message": "セッションは終了しています"})
        return
    scenario = sqlite.get_scenario(session.scenario_id)
    if not scenario:
        yield ("error", {"message": "元シナリオが見つかりません（孤児セッション）"})
        return

    npcs = sqlite.list_scenario_npcs(scenario.id)
    npc_names = {n.name for n in npcs if getattr(n, "name", None)}
    engine_type = getattr(session, "engine_type", "ensemble") or "ensemble"
    # うつつ無人ループ判定。明示フラグか engine_type=="usual_days" のいずれかで有効化する。
    is_headless = headless or engine_type == "usual_days"
    # usual_days は GM + PC ディスパッチ機構を ensemble_pc と共有するため PC モード扱いにする
    # （無人ループ制御だけが service 側の分岐で異なる）。
    is_pc_mode = engine_type in ("ensemble_pc", "usual_days")

    # うつつの GM レスポンスは PC レスポンス（pc_runner）と同様、独立した MAIN ログ行として扱う。
    # new_message_id() で log_dir_id を fresh にしないと、既定値 "--------" の旧ログ溜めへ
    # GM の debug ログが書かれ、数ヶ月前の旧エラー（旧 OpenRouter 402 等）と混在して
    # /ui/logs が誤検出する。feature ラベルは /ui/logs で識別できるよう "usual_days" にする。
    if is_headless:
        new_message_id()
        current_log_feature.set("usual_days")
        # new_message_id() で session_id も None にリセットされるので再セット（pc_runner と同様、
        # debug_log_entries.session_id が NULL だとシナリオ別フィルタが効かなくなる）。
        current_log_session_id.set(session_id)

    # 1 シーンあたりの上限レスポンス数。headless は usual_config.max_responses_per_scene を優先し、
    # 無指定なら旧キー max_turns_per_scene → さらに無指定なら _DEFAULT_USUAL_MAX_RESPONSES。
    # 通常モードは従来どおり _MAX_RESPONSES_PER_USER_TURN。
    # （ここでの「レスポンス」= GM/PC それぞれ 1 LLM 呼出 = 1 raw_response。
    #  「ターン」= @話者: ブロック単位 (scenario_turns 1 行) とは別軸。）
    if is_headless:
        _usual_cfg = getattr(scenario, "usual_config", None) or {}
        max_responses = int(
            _usual_cfg.get("max_responses_per_scene")
            or _usual_cfg.get("max_turns_per_scene")
            or _DEFAULT_USUAL_MAX_RESPONSES
        )
    else:
        max_responses = _MAX_RESPONSES_PER_USER_TURN

    # うつつの時間文脈（日付・曜日・時間帯・季節）はシーン中ほぼ不変なので 1 度だけ算出する。
    usual_time_context = format_time_context() if is_headless else ""

    # うつつの「不在のユーザ」ブロック（NPC 言及制御）。シーン中は不変なので 1 度だけ算出して
    # ループ内の _build_usual_gm_appendix へ素通しで渡す。owner キャラの
    # user_visibility_note（本人が周囲への伝達範囲を自分の言葉で書いたもの）が source of truth。
    # 空欄なら完全秘匿（NPC は触れない）。owner_character_id が無い ensemble シナリオでは空文字列。
    absent_user_block = ""
    if is_headless:
        owner_id = getattr(scenario, "owner_character_id", None)
        if owner_id:
            owner_char = sqlite.get_character(owner_id)
            if owner_char is not None:
                absent_user_block = _build_absent_user_block(
                    character_name=getattr(owner_char, "name", "") or "",
                    user_label=getattr(owner_char, "user_label", "") or "",
                    user_position=getattr(owner_char, "user_position", "") or "",
                    visibility_note=getattr(owner_char, "user_visibility_note", "") or "",
                )
                # 現実の接触の封筒（めぐり / タイムライン投影）を GM へ注入する。
                # observer="world_frame" は chat.*（real）を envelope 止めで見る —
                # 「いつ・どのくらい接触したか」の外形だけを渡し、中身は渡さない
                # （性質4: 因果的一貫性の穴埋め。docs/planned/aliveness_plan.md §2.4）。
                real_contact_block = _build_real_contact_block(sqlite, owner_char)
                if real_contact_block:
                    absent_user_block = (
                        f"{absent_user_block}\n\n{real_contact_block}"
                        if absent_user_block else real_contact_block
                    )

    from backend.services.scenario_chat.mention import (
        format_pc_summary,
        normalize_pc_assignments,
        normalize_pc_slots,
    )

    # PC枠・配役は engine_type に依存せず常に正規化する（ユーザPCも 1 枠として扱うため）。
    # 旧 user_alias 廃止後、ユーザの @タグ名は user 割当スロットの name から解決する。
    pc_slots = normalize_pc_slots(getattr(scenario, "pc_slots", None))
    pcs = normalize_pc_assignments(
        getattr(session, "pc_assignments", None), pc_slots, sqlite,
    )
    # うつつ（headless）はユーザにレスポンス順を回さない（ユーザは不在の人物）。ルーティングの
    # 候補からユーザPCを除外する。
    routing_pcs = [p for p in pcs if not p.is_user] if is_headless else pcs

    # PCロスター。通常モードは全PCを「全員代弁するな」と均一提示する。
    # うつつ(headless)では不在のユーザPCを名簿から除外する ── ユーザを「配役可能なPC」として
    # 並べると、GM が「いまは不在だがいずれ登場するキャスト」と誤読し、ユーザの言動を捏造する
    # 温床になる（観測済み: GM が `@<ユーザ>:` 形式でユーザのメッセージ全文を捏造した）。
    # ユーザの扱いは _build_absent_user_block の主語ベース3段ルールへ一本化する。
    # 一方 suppress_names には不在ユーザを残す（下記）── 万一 GM が `@<ユーザ>:` を書いても
    # parser 段の最終バックストップで受ける。routing_pcs は通常モードでは pcs と一致するため、
    # 通常モードの pc_summary は従来どおり全PCを含む。
    pc_summary_text = format_pc_summary(routing_pcs)

    # ユーザの @タグ名: player_type="user" のスロット name。無ければフォールバック。
    # うつつ（headless）では無言進行 OOC で使う総称を既定の「プレイヤー」のままにする
    # （ユーザPC名を user_alias に出すと _USUAL_GM_STANDING の「プレイヤー＝この人物自身」と
    #  齟齬が出るため。不在ユーザの言動制御は absent_user_block / suppress_names 側で扱う）。
    user_pc = next((p for p in pcs if p.is_user), None)
    user_speaker_name = "プレイヤー" if is_headless else (user_pc.name if user_pc else "プレイヤー")

    # GM が代弁してはならない名前: 全 PC枠名 + 全 AI キャラ本名（不在ユーザPCも含む）。
    # うつつでは不在ユーザを pc_summary（名簿）からは外すが、ここには残す ── プロンプトで
    # 生成を止めるのが本丸だが、万一 GM が `@<ユーザ>:` を書いた時に parser 段で破棄するための
    # 最終バックストップとして機能させる（pcs を回すので不在ユーザPCも自然に含まれる）。
    suppress_names: set[str] = {user_speaker_name}
    for pc in pcs:
        suppress_names.add(pc.name)
        if pc.is_character and pc.character_name:
            suppress_names.add(pc.character_name)

    previous_anticipation = _latest_scenario_anticipation(sqlite.list_scenario_turns(session_id))
    gm_preset_id = getattr(session, "gm_preset_id", "") or ""
    current_synopsis = sqlite.get_scenario_session_synopsis(session_id) or {
        "auto": "",
        "manual": "",
        "last_turn_index": -1,
    }

    if not auto_advance:
        user_turn = _save_turn(
            sqlite=sqlite,
            session_id=session_id,
            speaker_type="user",
            speaker_name=user_speaker_name,
            content=user_message,
        )
        yield ("user_saved", {"turn": scenario_turn_to_dict(user_turn)})

    if engine is None:
        engine = _build_default_engine(sqlite)

    saved_turn_ids: list[str] = []

    # 初動ルーティング決定
    next_kind: str = "gm"
    next_target: str | None = None
    if is_pc_mode and not auto_advance:
        next_kind, next_target = find_last_routing_mention(
            user_message, routing_pcs, npc_names,
        )
        if next_kind == "none":
            next_kind = "gm"
    elif is_pc_mode and auto_advance and yield_to:
        # 「ターンを譲る」UI（auto_advance + yield_to）。メッセージ本文の代わりに
        # フロントが直接指定した宛先で初動ルーティングを決める。
        if yield_to == "ALL":
            if routing_pcs:
                next_kind = "all"
                next_target = None
        elif yield_to != "GM":
            # PC枠名指定。PC が見つからなければ・ユーザPCなら GM フォールバック。
            target_pc = next((p for p in routing_pcs if p.name == yield_to), None)
            if target_pc is not None and not target_pc.is_user:
                next_kind = "pc"
                next_target = target_pc.name

    # ループ制御は SceneLoop に委ねる。シナリオ固有の判断（メンション解析・SCENE_CLOSE 抑止・
    # GM↔PC 実行）は ScenarioRouter / ScenarioTurnExecutor が ScenarioLoopState を介して担う。
    from backend.services.chat_flow.scene_loop import LoopState, SceneLoop
    from backend.services.scenario_chat.loop_strategies import (
        ScenarioLoopState,
        ScenarioRouter,
        ScenarioTurnExecutor,
    )

    scenario_state = ScenarioLoopState(
        sqlite=sqlite,
        settings=settings,
        engine=engine,
        chat_service=chat_service,
        session_id=session_id,
        session=session,
        scenario=scenario,
        npcs=npcs,
        npc_names=npc_names,
        pcs=pcs,
        routing_pcs=routing_pcs,
        pc_summary_text=pc_summary_text,
        user_speaker_name=user_speaker_name,
        suppress_names=suppress_names,
        gm_preset_id=gm_preset_id,
        current_synopsis=current_synopsis,
        auto_advance=auto_advance,
        is_headless=is_headless,
        is_pc_mode=is_pc_mode,
        max_responses=max_responses,
        usual_time_context=usual_time_context,
        absent_user_block=absent_user_block,
        extra_first_gm_ooc=extra_first_gm_ooc,
        previous_anticipation=previous_anticipation,
        user_message=user_message,
        initial_next_kind=next_kind,
        initial_next_target=next_target,
        saved_turn_ids=saved_turn_ids,
        last_speaker_name=user_speaker_name if not auto_advance else None,
    )

    loop = SceneLoop(
        router=ScenarioRouter(),
        executor=ScenarioTurnExecutor(),
        max_iterations=max_responses,
    )
    loop_state = LoopState(context={"scenario_state": scenario_state})

    try:
        async for ev in loop.run(initial_state=loop_state):
            # SceneLoop 終端通知は API 層に渡す必要がない（turn_complete を別途 yield する）。
            if ev[0] == "loop_complete":
                continue
            yield ev
    except Exception as e:
        logger.exception("シナリオレスポンス実行エラー session=%s", session_id)
        yield ("error", {"message": str(e)})
        return

    if scenario_state.fired_responses >= max_responses:
        logger.info(
            "シナリオレスポンス上限到達 session=%s fired=%d cap=%d headless=%s",
            session_id, scenario_state.fired_responses, max_responses, is_headless,
        )

    sqlite.update_scenario_session(session_id, status=session.status)

    # turn_ids は保存された話者ブロック ID（=ターン）、fired_responses は LLM 呼出回数（GM + PC）。
    # 後者は run_usual_days_scene の集計ログで使う。
    yield ("turn_complete", {
        "turn_ids": saved_turn_ids,
        "fired_responses": scenario_state.fired_responses,
    })

    progress = compute_synopsis_progress(sqlite, settings, scenario, session_id)
    if progress is not None:
        yield ("synopsis_progress", progress)






async def _run_gm_turn(
    engine,
    scenario,
    npcs: list,
    history: list,
    user_message: str,
    settings: dict,
    gm_preset_id: str,
    auto_advance: bool,
    synopsis_auto: str,
    synopsis_manual: str,
    previous_anticipation: str,
    pc_summary: str,
    dice_pool: str,
    suppress_names: set[str] | None,
    user_speaker_name: str,
    sqlite,
    session_id: str,
    saved_turn_ids: list[str],
    time_context: str = "",
    gm_ooc_appendix: str = "",
) -> AsyncGenerator[tuple[Any, Any], None]:
    """GM 1 レスポンス分を engine 経由で実行し、SSE イベントを yield しつつ
    scenario_turns へ保存する内部ヘルパ。

    yield 値はメンション主導ループ側で扱いやすいよう (event_tuple, None) の
    2-tuple にしている（将来 metadata を後付けする余地）。

    time_context / gm_ooc_appendix はうつつ（Usual Days）専用の GM プロンプト追記
    （時間文脈・偶発イベント指示・ソフト収束ヒント）。通常モードでは空文字列。
    """
    raw_response = ""
    turn_records_pending: list[TurnRecord] = []
    provider_error: str | None = None

    async for item in engine.generate_stream(
        scenario=scenario,
        npcs=npcs,
        history=history,
        user_message=user_message,
        settings=settings,
        gm_preset_id=gm_preset_id,
        auto_advance=auto_advance,
        synopsis_auto=synopsis_auto,
        synopsis_manual=synopsis_manual,
        previous_anticipation=previous_anticipation,
        pc_summary=pc_summary,
        dice_pool=dice_pool,
        suppress_names=suppress_names,
        user_speaker_name=user_speaker_name,
        time_context=time_context,
        gm_ooc_appendix=gm_ooc_appendix,
    ):
        if isinstance(item, UtteranceDelta):
            if item.is_speaker_change:
                yield ((
                    "turn_start",
                    {
                        "speaker_type": item.speaker_type,
                        "speaker_id": item.speaker_id,
                        "speaker_name": item.speaker_name,
                        "is_known": item.is_known,
                    },
                ), None)
            yield (("chunk", {"text": item.content_delta}), None)
        elif isinstance(item, TurnRecord):
            turn_records_pending.append(item)
        elif isinstance(item, EngineResult):
            raw_response = item.raw_response
            provider_error = item.provider_error
            # 計器 Tier 1（fabrication_backstop）: GM がユーザ（user_alias）を代弁した
            # ブロックを parser が破棄した = バックストップの発火。正常条件は発火 0 回
            # であり、発火した事実そのものが「幻想の穴」の証拠なので即時アラームにする。
            user_fabrications = [
                w for w in item.parser_warnings if w.startswith("user_alias")
            ]
            if user_fabrications:
                from backend.lib.instrument_recorder import fire_alarm
                fire_alarm(
                    "fabrication_backstop",
                    details={
                        "session_id": session_id,
                        "warnings": user_fabrications,
                        "all_parser_warnings": item.parser_warnings,
                    },
                )

    if provider_error is not None:
        # プロバイダ由来エラー: scenario_turns への保存とあらすじ蒸留対象化を回避する。
        # turn_records_pending は engine 側で flush せず空のまま渡されるので、ここでは
        # 保存をスキップし、UI に通知だけ流して終わる。次の user 発話時に
        # 同一の last_turn_index を維持したまま再試行できる。
        logger.warning(
            "GM プロバイダエラーで scenario turn 保存をスキップ session=%s 内容=%s",
            session_id, provider_error[:300],
        )
        yield (("error", {"message": provider_error}), None)
        return

    _, turn_anticipation = extract_anticipation(raw_response)
    # GM の予想はここで採用が確定する（最終ターン=最後の話者ブロックの anticipation カラムへ保存され、
    # 次レスポンスの GM プロンプトに注入される）ため、この地点で実行イベントとして記録する。
    if turn_anticipation:
        record_tool_event(
            "anticipate_response", {"content": turn_anticipation}, source="anticipation",
        )
    last_index = len(turn_records_pending) - 1
    for i, rec in enumerate(turn_records_pending):
        rec_content, _ = extract_anticipation(rec.content)
        saved = _save_turn(
            sqlite=sqlite,
            session_id=session_id,
            speaker_type=rec.speaker_type,
            speaker_name=rec.speaker_name,
            content=rec_content,
            speaker_id=rec.speaker_id,
            raw_response=raw_response,
            attach_log_request_id=True,
            anticipation=turn_anticipation if i == last_index else None,
        )
        saved_turn_ids.append(saved.id)
        yield (("turn_end", {"turn": scenario_turn_to_dict(saved)}), None)


def _resolve_pc_preset_id(pc, sqlite) -> str:
    """PC キャラに使う LLMModelPreset の ID を返す。

    解決順:
        1. pc.preset_id（セッション側 pc_assignments で指定された preset）
        2. キャラの enabled_providers の任意の1エントリ

    Args:
        pc: PcAssignment（player_type="character"）。
        sqlite: SQLiteStore。

    Returns:
        プリセット ID。解決不能なら空文字列。
    """
    if getattr(pc, "preset_id", None):
        preset_id = pc.preset_id
        if sqlite.get_model_preset(preset_id):
            return preset_id
    char = sqlite.get_character(pc.character_id)
    if not char:
        return ""
    enabled = getattr(char, "enabled_providers", None) or {}
    for preset_id in enabled.keys():
        if sqlite.get_model_preset(preset_id):
            return preset_id
    return ""


