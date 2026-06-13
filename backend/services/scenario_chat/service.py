"""シナリオチャットサービス — API 層から呼ばれるファサード。

1 ターンのストリーム実行を司る非同期ジェネレータ `run_scenario_turn` を提供する。

責務:
    - プレイヤー発話を scenario_turns に保存
    - EnsembleEngine を駆動して話者単位の SSE イベントを生成
    - 各発話を scenario_turns に保存
    - raw_response はそのターン内の発話レコードに共通で紐付ける

group_chat.run_group_turn と同じ思想（非同期ジェネレータで SSE 用イベントを yield）
にすることで、API 側の StreamingResponse 実装を一貫させる。

周辺責務は分割済み（本モジュールが後方互換で再エクスポートする）:
    - serializers.py    — ORM → dict 変換・ユーザ話者名解決
    - auto_synopsis.py  — あらすじ自動更新トリガー・進捗計算
    - turns.py          — ターン保存ヘルパ・intro 展開
"""

import logging
import random
from typing import Any, AsyncGenerator

from backend.character_actions.anticipator import extract_anticipation
from backend.lib.log_context import current_log_feature
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


# 1 ユーザターンあたりの最大話者ターン数（GM + PC の合計）。
# 無限連鎖防止用ガード。
_MAX_TURNS_PER_USER_TURN = 10

# うつつ（Usual Days）無人ループの 1 シーンあたり既定上限ターン数。
# usual_config.max_turns_per_scene が無指定のときのフォールバック（ハード上限の保険）。
_DEFAULT_USUAL_MAX_TURNS = 8

# GM がシーンの幕引きを宣言するマーカー。うつつ無人ループの主たる停止条件。
# Phase 3 で anticipator と同じ抽出機構（本文からの除去・OOC ソフト収束）に発展させるが、
# Phase 2 では GM 生出力にこのマーカーが含まれるかどうかの検出で停止を担う。
_SCENE_CLOSE_MARKER = "[SCENE_CLOSE]"


def _has_scene_close(text: str | None) -> bool:
    """GM の生出力にシーン幕引きマーカー（[SCENE_CLOSE]）が含まれるか判定する。

    うつつ無人ループの主たる停止判断（判断主体は GM）。大文字小文字は区別しない。
    """
    if not text:
        return False
    return _SCENE_CLOSE_MARKER.lower() in text.lower()


# シーン終盤、残りこのターン数以下になったら GM へソフト収束ヒント（OOC）を出す。
_USUAL_SOFT_CLOSE_REMAINING = 2

# ソフト収束ヒント本文。GM に「そろそろ畳んで [SCENE_CLOSE] してよい」と促す。
_USUAL_SOFT_CLOSE_HINT = (
    f"[OOC] そろそろこのシーンを自然に畳む頃合い。区切りがついたら、"
    f"地の文の最後に {_SCENE_CLOSE_MARKER} と書いてシーンを締めてよい"
    f"（無理に引き延ばさないこと）。"
)


def roll_usual_event(
    usual_config: dict | None,
    now=None,
    rng: random.Random | None = None,
) -> str:
    """うつつの偶発イベントを抽選し、発生時は GM 向けのカテゴリ提示文を返す（非発生は空文字列）。

    混合方式: 発生可否は ``event_probability`` で機械抽選して確実に制御し、
    中身はカテゴリ名だけを GM へ渡して即興に委ねる（世界＝GM が外的フレームを与える思想）。

    ``event_categories`` は柔軟に受け付ける:
        - list: そのままカテゴリ候補。
        - dict: 時間帯/曜日/季節などのバケツ。全 value（list）を平坦化して候補にする。

    Args:
        usual_config: scenarios.usual_config の dict。
        now: 基準時刻（将来のバケツ選択用。現状は未使用だが API を揃える）。
        rng: 乱数源。省略時は random.Random()（engine.generate_dice_pool と同じ思想の乱数）。

    Returns:
        発生時は ``[OOC] …「<カテゴリ>」…`` のヒント文。非発生・候補なしは空文字列。
    """
    cfg = usual_config or {}
    try:
        prob = float(cfg.get("event_probability") or 0.0)
    except (TypeError, ValueError):
        prob = 0.0
    raw_cats = cfg.get("event_categories") or []
    # dict なら全 value を平坦化、list ならそのまま。要素は文字列化して空を除く。
    flat: list[str] = []
    if isinstance(raw_cats, dict):
        for v in raw_cats.values():
            if isinstance(v, (list, tuple)):
                flat.extend(str(x).strip() for x in v if str(x).strip())
            elif str(v).strip():
                flat.append(str(v).strip())
    elif isinstance(raw_cats, (list, tuple)):
        flat = [str(x).strip() for x in raw_cats if str(x).strip()]
    if prob <= 0.0 or not flat:
        return ""
    if rng is None:
        rng = random.Random()
    if rng.random() >= prob:
        return ""
    category = rng.choice(flat)
    return (
        f"[OOC] 今日のこのシーンでは「{category}」にまつわる偶発的な出来事が起きてよい。"
        f"具体的な中身はあなた（GM）が即興で決めること（キャラの内面・選択には踏み込まない）。"
    )


def _build_usual_gm_appendix(
    scenario,
    fired_turns: int,
    max_turns: int,
    is_first_gm: bool,
    rng: random.Random | None = None,
) -> str:
    """うつつ GM ターンの OOC 追記（偶発イベント＋ソフト収束）を組み立てる。

    - 偶発イベント: シーンの最初の GM ターン（is_first_gm）でのみ抽選する
      （1 シーンに 1 度だけ種をまく）。
    - ソフト収束: 残りターン数が _USUAL_SOFT_CLOSE_REMAINING 以下になったら、
      GM に [SCENE_CLOSE] での自然な幕引きを促すヒントを添える（停止判断は GM）。

    Returns:
        OOC 追記文字列（複数行）。何も無ければ空文字列。
    """
    parts: list[str] = []
    if is_first_gm:
        event_hint = roll_usual_event(getattr(scenario, "usual_config", None), rng=rng)
        if event_hint:
            parts.append(event_hint)
    if max_turns - fired_turns <= _USUAL_SOFT_CLOSE_REMAINING:
        parts.append(_USUAL_SOFT_CLOSE_HINT)
    return "\n".join(parts)


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
) -> AsyncGenerator[tuple[str, Any], None]:
    """ユーザ発話を受け取り、シナリオ 1 ターン分の SSE イベントを順次 yield する。

    プレイセッション（scenario_sessions）から元シナリオ（scenarios）を lookup し、
    そのシナリオの NPC・PC枠・GM プリセット等で 1 ターンを進行させる。

    動作モード:
        - engine_type == "ensemble":
            - 従来通り GM 1 ターンのみを実行してユーザターンへ戻す。
        - engine_type == "ensemble_pc":
            - メンション主導の話者ループで動作する:
              1. ユーザ発話末尾のメンションを解析して次話者を決定。
              2. メンション無し / @GM / @Narrator / @NPC → GM ターン。
              3. @<PC枠名> / @<キャラ本名> → 該当 PC ターン（直接ディスパッチ）。
              4. @ALL → 直前話者を除いた PC から random.choice。
              5. 各話者の発話末尾を再度メンション解析して次話者を決める。
              6. 次話者が「ユーザ枠」または「メンション無し」になればループ終了。
              7. 上限 `_MAX_TURNS_PER_USER_TURN` で打ち切る（無限連鎖防止）。

    PC ターン実行のため `chat_service` を渡すこと（None だと PC 行きの分岐は
    スキップされて GM のみ進む）。

    auto_advance=True の場合:
        - user_message は無視される（呼び出し側は空文字を渡してよい）
        - user turn は保存されない
        - SSE の "user_saved" イベントも発火しない
        - GM プロンプト末尾に「プレイヤーは無言、場面を進めて」という OOC 指示が入る

    headless=True（うつつ / Usual Days 無人ループ）の場合:
        - engine_type=="usual_days" でも自動的に headless 扱いになる。
        - ユーザ枠ゼロを許容し、ユーザの介入なしに GM↔PC を連鎖させ続ける。
        - PC 発話末尾にメンションが無くても break せず GM ターンへ継続する
          （無人でも場面が止まらないようにする）。
        - 停止条件は GM 出力の `[SCENE_CLOSE]` 検出（主）か、
          `usual_config.max_turns_per_scene`（既定 _DEFAULT_USUAL_MAX_TURNS）への到達（保険）。
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

    # 1 シーンあたりの上限ターン数。headless は usual_config.max_turns_per_scene を優先し、
    # 無指定なら _DEFAULT_USUAL_MAX_TURNS。通常モードは従来どおり _MAX_TURNS_PER_USER_TURN。
    if is_headless:
        _usual_cfg = getattr(scenario, "usual_config", None) or {}
        max_turns = int(_usual_cfg.get("max_turns_per_scene") or _DEFAULT_USUAL_MAX_TURNS)
    else:
        max_turns = _MAX_TURNS_PER_USER_TURN

    # うつつの時間文脈（日付・曜日・時間帯・季節）はシーン中ほぼ不変なので 1 度だけ算出する。
    usual_time_context = format_time_context() if is_headless else ""

    from backend.services.scenario_chat.mention import (
        format_pc_summary,
        normalize_pc_assignments,
        normalize_pc_slots,
        find_last_routing_mention,
        pick_at_all_target,
    )
    from backend.services.scenario_chat.engine import generate_dice_pool

    # PC枠・配役は engine_type に依存せず常に正規化する（ユーザPCも 1 枠として扱うため）。
    # 旧 user_alias 廃止後、ユーザの @タグ名は user 割当スロットの name から解決する。
    pc_slots = normalize_pc_slots(getattr(scenario, "pc_slots", None))
    pcs = normalize_pc_assignments(
        getattr(session, "pc_assignments", None), pc_slots, sqlite,
    )
    # PCロスター（ユーザPC含む全PC）。GM へ「全PCを代弁するな」と均一に提示する。
    pc_summary_text = format_pc_summary(pcs)

    # ユーザの @タグ名: player_type="user" のスロット name。無ければフォールバック。
    user_pc = next((p for p in pcs if p.is_user), None)
    user_speaker_name = user_pc.name if user_pc else "プレイヤー"

    # GM が代弁してはならない名前: 全 PC枠名 + 全 AI キャラ本名（ユーザPCも含む）。
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
            user_message, pcs, npc_names,
        )
        if next_kind == "none":
            next_kind = "gm"

    last_speaker_name: str | None = user_speaker_name if not auto_advance else None
    fired_turns = 0

    try:
        while fired_turns < max_turns:
            if next_kind == "gm":
                # GM ターン実行
                gm_last_raw, gm_last_name = "", None
                # うつつ: 時間文脈（外的フレーム）と OOC（偶発イベント・ソフト収束）を GM へ注入。
                gm_ooc = ""
                if is_headless:
                    gm_ooc = _build_usual_gm_appendix(
                        scenario, fired_turns, max_turns, is_first_gm=(fired_turns == 0),
                    )
                    # シーン冒頭のみ、スケジューラからの経過時間メモ（「前回から N 時間後」等）を先頭に添える。
                    if fired_turns == 0 and extra_first_gm_ooc.strip():
                        gm_ooc = (extra_first_gm_ooc.strip() + "\n" + gm_ooc).strip()
                async for ev, _meta in _run_gm_turn(
                    engine=engine,
                    scenario=scenario,
                    npcs=npcs,
                    history=sqlite.list_scenario_turns(session_id),
                    user_message=user_message if fired_turns == 0 else "",
                    settings=settings,
                    gm_preset_id=gm_preset_id,
                    auto_advance=auto_advance if fired_turns == 0 else True,
                    synopsis_auto=current_synopsis.get("auto", ""),
                    synopsis_manual=current_synopsis.get("manual", ""),
                    previous_anticipation=previous_anticipation,
                    pc_summary=pc_summary_text,
                    dice_pool=(
                        generate_dice_pool(getattr(scenario, "dice_pool_spec", None))
                        if is_pc_mode else ""
                    ),
                    suppress_names=suppress_names,
                    user_speaker_name=user_speaker_name,
                    sqlite=sqlite,
                    session_id=session_id,
                    saved_turn_ids=saved_turn_ids,
                    time_context=usual_time_context,
                    gm_ooc_appendix=gm_ooc,
                ):
                    yield ev

                # GM の最終 raw_response を直近保存ターンから取り直す
                latest = sqlite.list_scenario_turns(session_id)
                for t in reversed(latest):
                    if getattr(t, "speaker_type", "") in {"narrator", "npc"}:
                        gm_last_raw = getattr(t, "raw_response", "") or ""
                        gm_last_name = getattr(t, "speaker_name", "") or last_speaker_name
                        break
                if gm_last_name:
                    last_speaker_name = gm_last_name
                fired_turns += 1

                # うつつ無人ループ: GM がシーン幕引き（[SCENE_CLOSE]）を宣言したら
                # そこでシーン終了。停止の主たる判断主体は GM（キャラではない）。
                if is_headless and _has_scene_close(gm_last_raw):
                    logger.info(
                        "うつつ: GM が SCENE_CLOSE を宣言 session=%s fired=%d",
                        session_id, fired_turns,
                    )
                    break

                if is_pc_mode and gm_last_raw:
                    next_kind, next_target = find_last_routing_mention(
                        gm_last_raw, pcs, npc_names,
                    )
                    # GM 直後の特例: 明示的に @<PC> / @ALL のいずれかが
                    # 指定されていない限り、必ず @ALL にフォールバックする。
                    # find_last_routing_mention は GM ラベル / NPC名 / Narrator も
                    # kind="gm" として返すが、GM が NPC を呼び合うだけのループを
                    # 防ぐため、ここでは pc/all 以外を ALL に格上げする。
                    # PC が 1 人もいなければ ALL は無意味なので従来どおり終了。
                    if next_kind not in {"pc", "all"}:
                        if pcs:
                            next_kind = "all"
                            next_target = None
                            continue
                        break
                    continue
                # 通常モード（ensemble）または PC モードでもメンション無し → 終了
                break

            if next_kind == "all":
                pc = pick_at_all_target(pcs, last_speaker_name=last_speaker_name)
                if pc is None or pc.is_user:
                    break
                next_kind = "pc"
                next_target = pc.name
                # フォールスルー: 同じイテレーションで PC 実行へ

            if next_kind == "pc":
                pc = next((p for p in pcs if p.name == next_target), None)
                if pc is None:
                    break
                if pc.is_user:
                    # ユーザ枠が指名された → ユーザターンへ
                    break
                if chat_service is None:
                    logger.warning(
                        "PC ターン実行不可: chat_service が None session=%s pc=%s",
                        session_id, pc.name,
                    )
                    break

                preset_id = _resolve_pc_preset_id(pc, sqlite)
                if not preset_id:
                    logger.warning(
                        "PC 用プリセット未解決 session=%s slot=%s character_id=%s",
                        session_id, pc.slot_id, pc.character_id,
                    )
                    break

                latest_history = sqlite.list_scenario_turns(session_id)

                yield ("pc_start", {
                    "character": pc.name,
                    "character_id": pc.character_id,
                })
                full_text = ""
                anticipation_text: str | None = None
                try:
                    from backend.services.scenario_chat.pc_runner import stream_pc_response
                    async for ev_type, payload in stream_pc_response(
                        pc=pc,
                        scenario_title=scenario.title,
                        user_alias=user_speaker_name,
                        history=latest_history,
                        preset_id=preset_id,
                        sqlite=sqlite,
                        settings=settings,
                        chat_service=chat_service,
                        scenario_session_id=session_id,
                        default_origin="usual" if is_headless else "interlude",
                    ):
                        if ev_type == "pc_done":
                            full_text = payload["full_text"]
                            anticipation_text = payload.get("anticipation")
                        yield (ev_type, payload)
                except Exception as e:
                    logger.exception(
                        "PC ターン実行エラー session=%s pc=%s",
                        session_id, pc.name,
                    )
                    yield ("pc_error", {
                        "character": pc.name,
                        "character_id": pc.character_id,
                        "message": str(e),
                    })
                    break

                if full_text.strip():
                    saved = _save_turn(
                        sqlite=sqlite,
                        session_id=session_id,
                        speaker_type="pc",
                        speaker_name=pc.name,
                        content=full_text,
                        speaker_id=pc.character_id,
                        attach_log_request_id=True,
                        anticipation=anticipation_text,
                    )
                    saved_turn_ids.append(saved.id)
                    yield ("speaker_end", {"turn": scenario_turn_to_dict(saved)})

                last_speaker_name = pc.name
                fired_turns += 1

                # PC 発話末尾のメンションで次話者を決める
                next_kind, next_target = find_last_routing_mention(
                    full_text, pcs, npc_names,
                )
                if next_kind == "none":
                    # うつつ無人ループ: メンションが無くてもユーザに戻さず GM ターンへ継続し、
                    # 場面が止まらないようにする（通常モードはユーザターンへ戻して終了）。
                    if is_headless:
                        next_kind = "gm"
                        next_target = None
                        continue
                    break
                continue

            # 未知の kind: 防御的に break
            break
    except Exception as e:
        logger.exception("シナリオターン実行エラー session=%s", session_id)
        yield ("error", {"message": str(e)})
        return

    if fired_turns >= max_turns:
        logger.info(
            "シナリオターン上限到達 session=%s fired=%d cap=%d headless=%s",
            session_id, fired_turns, max_turns, is_headless,
        )

    sqlite.update_scenario_session(session_id, status=session.status)

    yield ("turn_complete", {"turn_ids": saved_turn_ids})

    progress = compute_synopsis_progress(sqlite, settings, scenario, session_id)
    if progress is not None:
        yield ("synopsis_progress", progress)


async def run_usual_days_scene(
    session_id: str,
    sqlite,
    settings: dict,
    chat_service,
    engine: SceneEngine | None = None,
    extra_first_gm_ooc: str = "",
) -> dict:
    """うつつ（Usual Days）の 1 シーンを無人で回し、結果サマリを返す薄いトリガー。

    `run_scenario_turn(headless=True, auto_advance=True)` を内部で駆動して SSE イベントを
    すべて drain する（うつつは SSE 配信を伴わないため、戻り値の dict だけ使う）。
    Phase 4 のスケジューラと、デバッグ用の手動 1 シーン実行の双方から呼ぶ共通入口。

    Args:
        session_id: うつつセッション（engine_type="usual_days"）の ID。
        sqlite: SQLiteStore。
        settings: グローバル設定辞書。
        chat_service: PC ターン実行に必須の ChatService。
        engine: GM エンジン。None なら既定エンジンを使う。
        extra_first_gm_ooc: シーン冒頭の GM へ添える経過時間メモ等（「前回から N 時間後」）。

    Returns:
        {"saved_turn_ids": [...], "fired_turns": int, "scene_closed": bool,
         "error": str | None} の集計 dict。
    """
    saved_turn_ids: list[str] = []
    scene_closed = False
    error: str | None = None
    async for ev_type, payload in run_scenario_turn(
        session_id=session_id,
        user_message="",
        sqlite=sqlite,
        settings=settings,
        engine=engine,
        auto_advance=True,
        chat_service=chat_service,
        headless=True,
        extra_first_gm_ooc=extra_first_gm_ooc,
    ):
        if ev_type == "turn_complete":
            saved_turn_ids = list(payload.get("turn_ids", []))
        elif ev_type == "error":
            error = str(payload.get("message", ""))
    # シーンが GM の [SCENE_CLOSE] で閉じたかは、最終 GM ターンの生出力から判定する。
    for turn in reversed(sqlite.list_scenario_turns(session_id)):
        if getattr(turn, "speaker_type", "") in {"narrator", "npc"}:
            scene_closed = _has_scene_close(getattr(turn, "raw_response", "") or "")
            break
    return {
        "saved_turn_ids": saved_turn_ids,
        "fired_turns": len(saved_turn_ids),
        "scene_closed": scene_closed,
        "error": error,
    }


def ensure_usual_session(sqlite, scenario):
    """うつつ世界の永続セッション（engine_type="usual_days"）を find-or-create して返す。

    1 キャラ 1 世界・セッション永続1本の前提（plan §2）。既存の active な usual_days
    セッションがあればそれを返し、無ければ usual_config の GM/PC プリセットと owner キャラで
    新規起動する。起動に必要な情報（GM プリセット・owner・PC枠）が欠ければ None を返す。

    Args:
        sqlite: SQLiteStore。
        scenario: owner_character_id / usual_config / pc_slots を持つうつつシナリオ。

    Returns:
        ScenarioSession（既存または新規）。起動不能なら None。
    """
    import uuid

    from backend.services.scenario_chat.mention import normalize_pc_slots

    # 既存の active な usual_days セッションを優先（永続1本）。
    for s in sqlite.list_scenario_sessions_by_scenario(scenario.id):
        if getattr(s, "engine_type", "") == "usual_days" and getattr(s, "status", "") == "active":
            return s

    cfg = getattr(scenario, "usual_config", None) or {}
    gm_pid = (cfg.get("gm_preset_id") or "").strip()
    pc_pid = (cfg.get("pc_preset_id") or gm_pid).strip()
    owner_id = getattr(scenario, "owner_character_id", None)
    pc_slots = normalize_pc_slots(getattr(scenario, "pc_slots", None))
    if not gm_pid or not owner_id or not pc_slots:
        logger.warning(
            "うつつ: セッション起動に必要な情報が不足 owner=%s gm_preset=%s slots=%d",
            owner_id, gm_pid, len(pc_slots),
        )
        return None

    slot_id = pc_slots[0].slot_id
    session_id = str(uuid.uuid4())
    return sqlite.create_scenario_session(
        session_id=session_id,
        scenario_id=scenario.id,
        title=f"{getattr(scenario, 'title', 'うつつ')}（うつつ）",
        gm_preset_id=gm_pid,
        synopsis_preset_id=gm_pid,
        engine_type="usual_days",
        pc_assignments=[{
            "slot_id": slot_id,
            "player_type": "character",
            "character_id": owner_id,
            "preset_id": pc_pid,
        }],
    )


def usual_elapsed_note(sqlite, session_id: str, now=None) -> str:
    """前回シーン（最新ターン）からの経過時間を GM 向けの一文にして返す。

    うつつは間欠的に進むため、GM へ「前回からどれだけ時間が空いたか」を伝える。
    履歴が無い／時刻が取れない場合は空文字列。

    Args:
        sqlite: SQLiteStore。
        session_id: うつつセッション ID。
        now: 基準時刻。省略時は datetime.now()。

    Returns:
        "[OOC] 前回の場面から約N時間が経過した。…" の一文。算出不能なら空文字列。
    """
    from datetime import datetime as _dt

    from backend.lib.utils import format_time_delta

    if now is None:
        now = _dt.now()
    turns = sqlite.list_scenario_turns(session_id)
    last_at = None
    for turn in reversed(turns):
        created = getattr(turn, "created_at", None)
        if created is not None:
            last_at = created
            break
    if last_at is None:
        return ""
    try:
        delta_str = format_time_delta(now - last_at)
    except Exception:
        return ""
    if not delta_str:
        return ""
    return (
        f"[OOC] 前回の場面から{delta_str}が経過した。"
        f"その間の出来事は地の文で自然に補ってよい（時間の飛びを意識すること）。"
    )


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
    """GM 1 ターン分を engine 経由で実行し、SSE イベントを yield しつつ
    scenario_turns へ保存する内部ヘルパ。

    yield 値はメンション主導ループ側で扱いやすいよう (event_tuple, None) の
    2-tuple にしている（将来 metadata を後付けする余地）。

    time_context / gm_ooc_appendix はうつつ（Usual Days）専用の GM プロンプト追記
    （時間文脈・偶発イベント指示・ソフト収束ヒント）。通常モードでは空文字列。
    """
    raw_response = ""
    turn_records_pending: list[TurnRecord] = []

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
                    "speaker_start",
                    {
                        "speaker_type": item.speaker_type,
                        "speaker_id": item.speaker_id,
                        "speaker_name": item.speaker_name,
                        "is_known": item.is_known,
                    },
                ), None)
            yield (("content_delta", {"text": item.content_delta}), None)
        elif isinstance(item, TurnRecord):
            turn_records_pending.append(item)
        elif isinstance(item, EngineResult):
            raw_response = item.raw_response

    _, turn_anticipation = extract_anticipation(raw_response)
    # GM の予想はここで採用が確定する（最終ターンの anticipation カラムへ保存され、
    # 次ターンの GM プロンプトに注入される）ため、この地点で実行イベントとして記録する。
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
        yield (("speaker_end", {"turn": scenario_turn_to_dict(saved)}), None)


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


