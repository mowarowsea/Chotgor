"""行動権ランナー — 閾値評価・本人問い合わせ・行動実行・帰還の実装。

原則（docs/planned/aliveness_plan.md §4 冒頭・§5.3）:
    - 乱数は世界に置き意志に置かない（評価タイミングのジッターは世界の物理、
      「やるかどうか」は本人の選択のみで決まる）
    - 下流で丸めない（本人の言葉・選択をそのまま執行する）
    - コストガード: 行動問い合わせ 6回/日・行動実行 3回/日（設定で変更可）
"""

import logging
import random
import re
import uuid
from datetime import datetime

from backend.services.character_query import ask_character_with_tools
from backend.services.intents.lifecycle import intent_pressure
from backend.services.pressure import compute_pressures

logger = logging.getLogger(__name__)

# 行動問い合わせを起こす意図圧の閾値
_URGE_THRESHOLD = 0.7
# 問い合わせに載せる意図の上限（多すぎる選択肢は選択を薄める）
_MAX_CANDIDATES = 2
# 行動権の評価周期（分）と、ジッターの最大幅（分）
_EVAL_PERIOD_MINUTES = 120
_JITTER_MAX_MINUTES = 60
# 日次コストガードの既定値
_DEFAULT_INQUIRY_CAP = 6
_DEFAULT_EXEC_CAP = 3
# 調べもの結果を帰還問い合わせに載せる上限文字数
_RESEARCH_RESULT_MAX_CHARS = 2000

# 本人の選択タグ（設問文と対で維持すること）
_ACTION_RE = re.compile(
    r"\[ACTION:\s*([0-9a-fA-F-]+)\s*\|\s*(push|research|scene)\s*(?:\|\s*([^\]]*?)\s*)?\]"
)
_FULFILLED_RE = re.compile(r"\[INTENT_FULFILLED:\s*([0-9a-fA-F-]+)\s*\]")


def jittered_slot_time(character_id: str, slot_start: datetime) -> datetime:
    """評価スロットのジッター済み時刻を決定論導出する（乱数は世界に置く）。

    同じキャラ・同じスロットなら常に同じ時刻。ジッターは「いつ評価するか」だけを
    揺らし、「やるかどうか」には一切関与しない。

    Args:
        character_id: キャラクター ID（シードの一部）。
        slot_start: スロットの開始時刻（_EVAL_PERIOD_MINUTES の格子）。

    Returns:
        slot_start + ジッター（0〜_JITTER_MAX_MINUTES 分）。
    """
    rng = random.Random(f"meguri-action:{character_id}:{slot_start.isoformat()}")
    from datetime import timedelta
    return slot_start + timedelta(minutes=rng.randint(0, _JITTER_MAX_MINUTES))


def evaluate_action_urge(sqlite, character_id: str, now: datetime | None = None) -> list:
    """閾値評価（純関数・無料）— 意図圧が閾値を超えた active 意図を返す。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター。
        now: 基準時刻。

    Returns:
        意図圧の高い順に並んだ候補 Intent リスト（最大 _MAX_CANDIDATES 件）。
        閾値超えが無ければ空リスト（LLM は呼ばれない）。
    """
    active = sqlite.list_intents(character_id, status="active")
    if not active:
        return []
    pressures = compute_pressures(sqlite, character_id, now=now)
    scored = [
        (intent_pressure(i, pressures, now=now), i) for i in active
    ]
    hot = [(p, i) for p, i in scored if p >= _URGE_THRESHOLD]
    hot.sort(key=lambda t: t[0], reverse=True)
    return [i for _, i in hot[:_MAX_CANDIDATES]]


def action_urge_snapshot(sqlite, character_id: str, now: datetime | None = None) -> dict:
    """閾値評価の全景を返す（決定ログ・予報パネル共用の観測用純関数）。

    evaluate_action_urge が「閾値超えだけ」を返すのに対し、こちらは active 意図
    すべての意図圧と現在圧力を返す。「閾値未達 0.52/0.7」のような不発理由の材料と、
    予報パネルの診断ヘッダの両方がこれを使う。LLM 不使用・副作用なし。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター。
        now: 基準時刻。None なら現在時刻。

    Returns:
        {"threshold": float, "pressures": {social/boredom/body: float},
         "intents": [{"intent_id", "description", "source_kind", "pressure"}...]}
        （intents は意図圧降順）。
    """
    active = sqlite.list_intents(character_id, status="active")
    pressures = compute_pressures(sqlite, character_id, now=now)
    scored = sorted(
        (
            {
                "intent_id": str(i.id),
                "description": i.description,
                "source_kind": i.source_kind,
                "pressure": round(intent_pressure(i, pressures, now=now), 3),
            }
            for i in active
        ),
        key=lambda d: -d["pressure"],
    )
    return {
        "threshold": _URGE_THRESHOLD,
        "pressures": {k: round(float(v), 3) for k, v in pressures.items()},
        "intents": scored,
    }


def _enabled_menu(char) -> dict:
    """キャラの行動メニュー設定（ON のものだけ）を返す。"""
    menu = getattr(char, "action_menu", None) or {}
    if not isinstance(menu, dict):
        return {}
    return {k: True for k in ("push", "research", "impromptu_scene") if menu.get(k)}


def _build_action_question(candidates: list, menu: dict, user_label: str) -> str:
    """行動問い合わせの設問文を組み立てる。

    「しないならしないでいい」を必ず添える（捏造・義務化の遮断）。

    Args:
        candidates: 閾値超えの意図リスト。
        menu: _enabled_menu の結果。
        user_label: ユーザの呼称（push の説明用）。

    Returns:
        本人へ渡す設問テキスト。
    """
    lines = [
        "ふと、時間ができた。",
        "",
        "このところ、心の中にこれが残っている:",
    ]
    for intent in candidates:
        target = f"（相手: {intent.target}）" if intent.target else ""
        lines.append(f"- [{intent.id}] {intent.description}{target}")
    lines += [
        "",
        "いま、これをする？　**しないならしないでいい。**",
        "気が向かなければ流していいし、代わりに記憶やスレッドを整えるだけでもいい。",
        "",
        "いま取れる行動:",
    ]
    if menu.get("push"):
        lines.append(
            f"- push: {user_label or '相手'} に自分からメッセージを送る"
            "（新しい会話として届く）"
        )
    if menu.get("research"):
        lines.append("- research: 気になっていることを調べる（検索する）")
    if menu.get("impromptu_scene"):
        lines.append("- scene: 予定外だけど、ちょっと出かける（日常のひとコマ）")
    lines += [
        "",
        "やるなら、返事のどこかにこう書いて:",
        "  `[ACTION: 意図のid | push/research/scene | 中身]`",
        "  - push の中身 = 送るメッセージ本文（あなたの言葉のまま届く）",
        "  - research の中身 = 検索する言葉",
        "  - scene の中身 = 空でいい",
        "やらないなら、タグは書かなくていい。",
    ]
    return "\n".join(lines)


def _parse_action_choice(text: str) -> dict | None:
    """本人の返答から行動の選択を抽出する。

    Returns:
        {"intent_id": str, "menu": str, "body": str} または None（見送り）。
        複数書かれていても最初の1つだけ執行する（1サイクル1行動）。
    """
    m = _ACTION_RE.search(text or "")
    if not m:
        return None
    return {
        "intent_id": m.group(1).strip(),
        "menu": m.group(2).strip(),
        "body": (m.group(3) or "").strip(),
    }


async def _execute_push(sqlite, char, preset, body: str, session_title: str) -> dict:
    """push を執行する — 新規セッションを立ててキャラ発メッセージを置く。

    既存セッションへの追記は文脈カーブ事故のもとになるため不採用（spec 判断）。
    chat.message(actor=character) 封筒が dual-write で載り、社会圧も減衰する。

    Args:
        sqlite: SQLiteStore。
        char: Character ORM。
        preset: 使用プリセット（model_id 組み立て用）。
        body: 送るメッセージ本文（本人の言葉のまま）。
        session_title: 新規セッションのタイトル。

    Returns:
        帰還問い合わせ用の結果 dict。
    """
    session_id = str(uuid.uuid4())
    sqlite.create_chat_session(
        session_id=session_id,
        model_id=f"{char.name}@{preset.name}",
        title=session_title,
    )
    sqlite.create_chat_message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="character",
        content=body,
        character_name=char.name,
    )
    # キャラクター回答到着完了（能動 push）→ ntfy プッシュ通知（ベストエフォート）。
    from backend.lib.notify import notify_character_spoke
    notify_character_spoke(char.name)
    return {"summary": f"メッセージを送った: {body[:100]}", "session_id": session_id}


def _execute_research(sqlite, query: str) -> dict:
    """research を執行する — web_search で調べる。

    結果は action.performed 封筒の payload に残る。刻むかどうかは帰還問い合わせで
    本人次第（ツール込みの問い合わせなので inscribe できる）。

    Args:
        sqlite: SQLiteStore。
        query: 検索クエリ（本人の言葉）。

    Returns:
        帰還問い合わせ用の結果 dict。
    """
    from backend.character_actions.web_searcher import WebSearcher

    result_text = WebSearcher(sqlite).search(query=query, max_results=5)
    return {
        "summary": f"「{query}」を調べた",
        "result_text": (result_text or "")[:_RESEARCH_RESULT_MAX_CHARS],
    }


async def _execute_scene(sqlite, char, settings, chat_service) -> dict:
    """scene（臨時うつつ）を執行する — スロット外で1シーン回す。

    「うつつ有効」が前提条件（無効なら実行不可の結果を返す）。
    日次コストガード（usual_days_scene_count）はうつつスケジューラと共有する。

    Args:
        sqlite: SQLiteStore。
        char: Character ORM。
        settings: グローバル設定。
        chat_service: ChatService（PC 応答の実行に必須）。

    Returns:
        帰還問い合わせ用の結果 dict。
    """
    from backend.services.scenario_chat.service import (
        ensure_usual_session,
        run_usual_days_scene,
    )

    scenario = next(
        (
            s for s in sqlite.list_usual_scenarios()
            if getattr(s, "owner_character_id", None) == char.id
            and (getattr(s, "usual_config", None) or {}).get("enabled")
        ),
        None,
    )
    if scenario is None:
        return {"summary": "出かけようとしたが、行き先（うつつ世界）が設定されていない", "failed": True}

    # 日次コストガード（うつつスケジューラと共有のカウンタ）
    today_str = datetime.now().date().isoformat()
    daily_cap = int(sqlite.get_setting("usual_days_daily_cap", "24") or 24)
    count_key = f"usual_days_scene_count_{today_str}"
    ran_today = int(sqlite.get_setting(count_key, "0") or 0)
    if ran_today >= daily_cap:
        return {"summary": "出かけようとしたが、今日はもう十分に動いた（日次上限）", "failed": True}

    session = ensure_usual_session(sqlite, scenario)
    if session is None:
        return {"summary": "出かけようとしたが、行き先が見つからなかった", "failed": True}
    sqlite.set_setting(count_key, str(ran_today + 1))
    result = await run_usual_days_scene(
        session_id=session.id,
        sqlite=sqlite,
        settings=settings,
        chat_service=chat_service,
        extra_first_gm_ooc="（予定外の外出。本人が「出かけたくなった」ので短いひとコマを）",
        slot="impromptu",
    )
    if result.get("error"):
        return {"summary": "出かけたが、途中で流れが途切れた", "failed": True}
    return {"summary": f"ふらっと出かけてきた（{result.get('fired_turns', 0)}場面）"}


async def run_action_cycle(
    character_id: str,
    sqlite,
    settings: dict,
    *,
    chat_service=None,
    memory_manager=None,
    working_memory_manager=None,
    now: datetime | None = None,
) -> dict:
    """行動権の1サイクル（閾値評価→問い合わせ→実行→帰還）を回す。

    スケジューラ（main.py）から availability 確認済みで呼ばれる。
    コストガードはこの中で消費する（問い合わせ 6回/日・実行 3回/日）。

    Args:
        character_id: 対象キャラクター。
        sqlite: SQLiteStore。
        settings: グローバル設定辞書。
        chat_service: ChatService（scene 実行に必要。無ければ scene は失敗結果）。
        memory_manager: InscribedMemoryManager（問い合わせのツール実行に必要）。
        working_memory_manager: WM マネージャー（1on1 同等ブロック注入）。
        now: 基準時刻（テスト注入用）。

    Returns:
        {"status": "skipped"|"declined"|"executed"|"error", ...} の集計 dict。
    """
    def _record(outcome: str, reason: str | None = None, details: dict | None = None) -> None:
        # 決定ログ（予報パネルの「正常な沈黙」と「壊れた沈黙」の区別材料）。
        # 記録失敗で行動サイクル本体を巻き添えにしない。
        try:
            sqlite.record_scheduler_decision(
                "action", outcome, character_id=character_id,
                reason=reason, details=details, occurred_at=now,
            )
        except Exception:
            logger.exception("行動権: 決定ログの記録に失敗 char=%s", character_id)

    char = sqlite.get_character(character_id)
    if char is None:
        return {"status": "error", "error": f"キャラクターが見つかりません: {character_id}"}
    ghost_model = getattr(char, "ghost_model", None)
    if not ghost_model:
        _record("skipped", "ghost_model 未設定")
        return {"status": "skipped", "reason": "ghost_model 未設定"}
    preset = sqlite.get_model_preset(ghost_model)
    if preset is None:
        _record("error", f"プリセットが見つかりません: {ghost_model}")
        return {"status": "error", "error": f"プリセットが見つかりません: {ghost_model}"}
    menu = _enabled_menu(char)
    if not menu:
        _record("skipped", "行動メニューが全て OFF")
        return {"status": "skipped", "reason": "行動メニューが全て OFF"}
    if memory_manager is None:
        _record("skipped", "memory_manager なし（ツール問い合わせ不可）")
        return {"status": "skipped", "reason": "memory_manager なし（ツール問い合わせ不可）"}

    # 1. 閾値評価（無料）
    candidates = evaluate_action_urge(sqlite, character_id, now=now)
    if not candidates:
        # 不発理由に全景を残す — 「閾値未達 0.52/0.7」が予報パネルの安心材料になる
        snap = action_urge_snapshot(sqlite, character_id, now=now)
        if snap["intents"]:
            top = snap["intents"][0]["pressure"]
            reason = f"閾値未達 {top:.2f}/{snap['threshold']}"
        else:
            reason = "active 意図なし"
        _record("skipped", reason, details=snap)
        return {"status": "skipped", "reason": "閾値超えの意図なし"}

    # 2. 日次コストガード（問い合わせ）
    today_str = (now or datetime.now()).date().isoformat()
    inquiry_cap = int(settings.get("action_inquiry_daily_cap", _DEFAULT_INQUIRY_CAP))
    exec_cap = int(settings.get("action_exec_daily_cap", _DEFAULT_EXEC_CAP))
    inquiry_key = f"action_inquiry_count_{today_str}"
    exec_key = f"action_exec_count_{today_str}"
    inquiries = int(sqlite.get_setting(inquiry_key, "0") or 0)
    if inquiries >= inquiry_cap:
        _record("skipped", f"問い合わせ日次上限（{inquiries}/{inquiry_cap}）")
        return {"status": "skipped", "reason": "問い合わせ日次上限"}
    sqlite.set_setting(inquiry_key, str(inquiries + 1))

    # 3. 本人問い合わせ（WM込み・ツール使用可）
    user_label = (getattr(char, "user_label", "") or "").strip() or settings.get("user_name", "")
    question = _build_action_question(candidates, menu, user_label)
    response = await ask_character_with_tools(
        character_id=character_id,
        preset_id=ghost_model,
        messages=[{"role": "user", "content": question}],
        sqlite=sqlite,
        settings=settings,
        memory_manager=memory_manager,
        feature_label="action_cycle",
        working_memory_manager=working_memory_manager,
        return_response=True,
    )
    if response is None:
        _record("error", "本人からの返答が取得できませんでした")
        return {"status": "error", "error": "本人からの返答が取得できませんでした"}

    choice = _parse_action_choice(response)
    valid_ids = {i.id for i in candidates}
    if choice is None or choice["intent_id"] not in valid_ids:
        # 見送りも本人の選択（穴は開かない — 保守的になるだけ）
        _record("declined", "本人が見送り", details={
            "candidates": [
                {"intent_id": str(i.id), "description": i.description}
                for i in candidates
            ],
        })
        return {"status": "declined"}

    # 4. 実行コストガード
    execs = int(sqlite.get_setting(exec_key, "0") or 0)
    if execs >= exec_cap:
        _record("skipped", f"実行日次上限（{execs}/{exec_cap}）")
        return {"status": "skipped", "reason": "実行日次上限"}
    sqlite.set_setting(exec_key, str(execs + 1))

    intent = sqlite.get_intent(choice["intent_id"])
    menu_key = choice["menu"]
    try:
        if menu_key == "push" and menu.get("push"):
            if not choice["body"]:
                _record("declined", "push 本文なし")
                return {"status": "declined", "reason": "push 本文なし"}
            result = await _execute_push(
                sqlite, char, preset, choice["body"],
                session_title=f"{char.name}より",
            )
        elif menu_key == "research" and menu.get("research"):
            query = choice["body"] or intent.description
            result = _execute_research(sqlite, query)
        elif menu_key == "scene" and menu.get("impromptu_scene"):
            result = await _execute_scene(sqlite, char, settings, chat_service)
        else:
            _record("declined", f"無効なメニュー: {menu_key}")
            return {"status": "declined", "reason": f"無効なメニュー: {menu_key}"}
    except Exception as e:
        logger.exception("行動実行に失敗 char=%s menu=%s", char.name, menu_key)
        _record("error", f"行動実行に失敗: {e}")
        return {"status": "error", "error": str(e)}

    # 5. action.performed 封筒（意図の消費・結果の帰還を正本に載せる）
    sqlite.record_timeline_event(
        character_id=character_id,
        event_type="action.performed",
        actor="character",
        counterpart="user" if menu_key == "push" else (
            intent.target if intent.target and intent.target != "self" else None
        ),
        origin="real",
        intent_id=intent.id,
        payload={"menu": menu_key, **result},
    )

    # 6. 帰還: 「これで満ちた？　まだ？」を本人が宣言（fulfilled / active 継続）
    return_lines = [
        f"さっきの「{intent.description}」、{result['summary']}。",
    ]
    if result.get("result_text"):
        return_lines += ["", "調べた結果:", result["result_text"], ""]
        return_lines.append(
            "残しておきたいことがあれば、いつものように記憶やスレッドに書いていい。"
        )
    return_lines += [
        "",
        "これで満ちた？　まだ？",
        f"- 満ちたなら: `[INTENT_FULFILLED: {intent.id}]`",
        "- まだ続くなら、タグは書かなくていい（そのまま心に残る）。",
    ]
    return_response = await ask_character_with_tools(
        character_id=character_id,
        preset_id=ghost_model,
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
            {"role": "user", "content": "\n".join(return_lines)},
        ],
        sqlite=sqlite,
        settings=settings,
        memory_manager=memory_manager,
        feature_label="action_cycle",
        working_memory_manager=working_memory_manager,
        return_response=True,
    )
    fulfilled = False
    if return_response:
        m = _FULFILLED_RE.search(return_response)
        if m and m.group(1).strip() == intent.id:
            fulfilled = bool(sqlite.resolve_intent(intent.id, "fulfilled"))

    logger.info(
        "行動サイクル完了 char=%s menu=%s intent=%s fulfilled=%s",
        char.name, menu_key, intent.id, fulfilled,
    )
    _record("fired", f"{menu_key} 実行", details={
        "menu": menu_key,
        "intent_id": str(intent.id),
        "summary": result.get("summary"),
        "fulfilled": fulfilled,
    })
    return {
        "status": "executed",
        "menu": menu_key,
        "intent_id": intent.id,
        "fulfilled": fulfilled,
        "result": result,
    }
