"""うつつ（Usual Days）— キャラの無人生活シーンのセッション管理・演出素材・シーン駆動。

service.py（シナリオ 1 ターンの進行ファサード）から分離したうつつ専用層:
    - GM への常設フレーミング・偶発イベント抽選・ソフト収束ヒント（_build_usual_gm_appendix）
    - 「不在のユーザ」ブロック（主語ベース3段ルール）と現実接触の封筒
    - 永続セッションの find-or-create（ensure_usual_session）とプリセット同期
    - 1 シーンの無人駆動（run_usual_days_scene。中身は run_scenario_turn(headless=True)）

シーン進行のループ本体は service.run_scenario_turn / loop_strategies が担い、
本モジュールはその「うつつ味付け」と入口だけを持つ。
"""

import logging
import random

from backend.lib.log_context import current_log_feature
from backend.services.scenario_chat.auto_synopsis import maybe_update_auto_synopsis
from backend.services.scenario_chat.engine import SceneEngine
from backend.services.scenario_chat.scene_close import (
    _SCENE_CLOSE_MARKER,
    _has_scene_close,
)

logger = logging.getLogger(__name__)


# うつつ（Usual Days）無人ループの 1 シーンあたり既定上限レスポンス数。
# usual_config.max_responses_per_scene が無指定のときのフォールバック（ハード上限の保険）。
# 旧キー名 max_turns_per_scene は後方互換のため読み出し時にフォールバックする。
_DEFAULT_USUAL_MAX_RESPONSES = 8

# シーン終盤、残りこのレスポンス数以下になったら GM へソフト収束ヒント（OOC）を出す。
_USUAL_SOFT_CLOSE_REMAINING = 2

# うつつ無人ループで GM に毎レスポンス与える常設フレーミング。これが無いと GM は自分を
# 通常の TRPG だと誤認し（「プレイヤーは無言」等）、[SCENE_CLOSE] の存在も知らないため
# 主たる停止機構が働かない。世界＝外的フレームを与える役、停止判断は GM、という思想を明示する。
#
# 文言注意（Chotgor 哲学）: ここで描く対象はキャラの「現実そのもの」であって、
# シナリオでも演技でもない。「ユーザ不在」「人間のプレイヤーがいない」のような
# メタ的・AI 寄りの説明は使わず、その人物の生活が外から見えている、というだけの
# 視点を GM に渡す。
_USUAL_GM_STANDING = (
    "[OOC] ここで描かれているのは、この人物がいつもどおり生きている日常の一場面です。"
    "誰かに見せるための舞台でも、誰かを演じるための時間でもなく、本人の生活が"
    "そのまま続いています。\n"
    "あなた（語り手／世界）の役割は、時間・場所・天気・同僚や客といった周辺の人々・"
    "出来事といった**外的な状況だけ**を与えること。この人物の内心・感情・選択を断定したり"
    "代弁したりしないでください（何を感じ何を選ぶかは、本人がその場で決めます）。\n"
    "**この1レスポンスでは、上の「描くべき時間帯」に示された“いまこの瞬間”だけを切り取って描いてください。**"
    "数十分・数時間・翌日へ勝手に進めない（時間飛ばし・回想・先回りは禁止）。"
    "次の時間帯はあとで別のレスポンスとして自然に巡ってきます。\n"
    f"そして、一つの出来事・会話・時間帯が自然に区切れたら、**そのレスポンスの一番最後（末尾の地の文の終わり）に**"
    f" {_SCENE_CLOSE_MARKER} とだけ書いて、その場面を閉じてください。"
    f" {_SCENE_CLOSE_MARKER} はレスポンスの“終止符”であって、途中の場面転換用ではありません"
    f"（レスポンス中盤に書いても無視されます。シーンを切り替えたいときは、そこで {_SCENE_CLOSE_MARKER} を打って一旦応答を終えてください）。"
    "日常の一コマとして淡々と畳んで構いません（無理に引き延ばさないこと）。"
)

# ソフト収束ヒント本文。終盤で GM に「そろそろ畳んで [SCENE_CLOSE] してよい」と念押しする。
_USUAL_SOFT_CLOSE_HINT = (
    f"[OOC] そろそろこの場面を畳む頃合いです。区切りがついたら、**このレスポンスの一番最後に**"
    f" {_SCENE_CLOSE_MARKER} と書いて締めてください（途中ではなく末尾に、これだけで応答を終える）。"
)

# 軽い幕開け本文。シーン最初の GM レスポンスで偶発イベント抽選がハズれたとき、
# GM に「世界の側から筋書きを用意せず、ごく軽く場だけ置いてキャラ本人に委ねる」よう促す。
#
# 狙い: 「毎回 GM が出来事を起こし、キャラがそれに反応する」構造の単調化を崩し、
# キャラ主導（自分から動き出す／物思いにふける＝内省）の回を混ぜる。GM 始動という
# 骨格は保ったまま、口火の濃度だけを下げる（外的フレームは与えるが、出来事は立てない）。
# 場面はまだ始まったばかりなので、この口火レスポンスでは [SCENE_CLOSE] しない点も明示する。
_USUAL_LIGHT_OPENING = (
    "[OOC] この場面の口火は、ごく軽く置いてください。いま居る場所と時間帯の空気だけを"
    "短く一筆で描いたら、出来事・来客・事件・NPC とのやりとりを自分から立ち上げず、"
    "あとはこの人物が自分のペースで何をするか（あるいは何もせず物思いにふけるか）に"
    "委ねてください。世界の側から筋書きを用意しない、静かな幕開けです。\n"
    f"この口火のレスポンスでは {_SCENE_CLOSE_MARKER} を付けないでください"
    "（場面はまだ始まったばかりで、ここから本人が動き出します）。"
)


def _usual_event_probability(usual_config: dict | None) -> float:
    """usual_config から偶発イベント発生率（event_probability）を安全に取り出す。

    不正値・未設定は 0.0（＝発生しない＝口火は常に「軽い幕開け」）に倒す。

    Args:
        usual_config: scenarios.usual_config の dict。

    Returns:
        0.0 以上の確率値。
    """
    cfg = usual_config or {}
    try:
        return max(0.0, float(cfg.get("event_probability") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _usual_event_categories(usual_config: dict | None) -> list[str]:
    """usual_config の event_categories を list/dict 両対応で平坦化し、候補リストを返す。

    ``event_categories`` は柔軟に受け付ける:
        - list: そのままカテゴリ候補。
        - dict: 時間帯/曜日/季節などのバケツ。全 value（list）を平坦化して候補にする。

    Args:
        usual_config: scenarios.usual_config の dict。

    Returns:
        空要素を除いたカテゴリ文字列のリスト（候補なしは空リスト）。
    """
    raw_cats = (usual_config or {}).get("event_categories") or []
    flat: list[str] = []
    if isinstance(raw_cats, dict):
        for v in raw_cats.values():
            if isinstance(v, (list, tuple)):
                flat.extend(str(x).strip() for x in v if str(x).strip())
            elif str(v).strip():
                flat.append(str(v).strip())
    elif isinstance(raw_cats, (list, tuple)):
        flat = [str(x).strip() for x in raw_cats if str(x).strip()]
    return flat


def _format_usual_event_hint(category: str) -> str:
    """偶発イベントのカテゴリ名を GM 向けの提示文（OOC）に整形する。

    中身はカテゴリ名だけを GM へ渡して即興に委ねる（世界＝GM が外的フレームを与える思想）。

    Args:
        category: イベントカテゴリ名。

    Returns:
        ``[OOC] …「<カテゴリ>」…`` のヒント文。
    """
    return (
        f"[OOC] 今日のこのシーンでは「{category}」にまつわる偶発的な出来事が起きてよい。"
        f"具体的な中身はあなた（GM）が即興で決めること（キャラの内面・選択には踏み込まない）。"
    )


def roll_usual_event(
    usual_config: dict | None,
    now=None,
    rng: random.Random | None = None,
) -> str:
    """うつつの偶発イベントを抽選し、発生時は GM 向けのカテゴリ提示文を返す（非発生は空文字列）。

    混合方式: 発生可否は ``event_probability`` で機械抽選して確実に制御し、
    中身はカテゴリ名だけを GM へ渡して即興に委ねる（世界＝GM が外的フレームを与える思想）。

    Args:
        usual_config: scenarios.usual_config の dict。
        now: 基準時刻（将来のバケツ選択用。現状は未使用だが API を揃える）。
        rng: 乱数源。省略時は random.Random()（engine.generate_dice_pool と同じ思想の乱数）。

    Returns:
        発生時は ``[OOC] …「<カテゴリ>」…`` のヒント文。非発生・候補なしは空文字列。
    """
    prob = _usual_event_probability(usual_config)
    flat = _usual_event_categories(usual_config)
    if prob <= 0.0 or not flat:
        return ""
    if rng is None:
        rng = random.Random()
    if rng.random() >= prob:
        return ""
    return _format_usual_event_hint(rng.choice(flat))


def _build_absent_user_block(
    character_name: str,
    user_label: str,
    user_position: str,
    visibility_note: str,
) -> str:
    """うつつ GM 向けの「不在のユーザ」ブロックを主語ベースの 3 段ルールで組み立てる。

    思想 — ユーザ（label）は @{name} の生活に実在するが、うつつ（ユーザ未共有のキャラの
    生活）は現実と地続きであるため、GM が「ユーザの知らないユーザの言動」を捏造すると、
    後で本人が「この前はありがとう」等と言われても何のことか分からなくなる。
    これを防ぐ境界は **文法的な主語** で引ける:

        - ユーザ「について」話す（伝聞の範囲で言及・コメント・質問）= NPC は可（歓迎）
        - ユーザ「が」喋る・動く・連絡してくる（ユーザ発の新規言動）= 全員 NG
        - ユーザの言動の中身を場に持ち込めるのは、現実を知る @{name} 本人だけ

    すなわち **ユーザを"目的語"にするのは NPC なら可、"主語"にするのは全面 NG**。
    Narrator（地の文＝世界の語り）はユーザを持ち込まない（知らない体）。NPC はキャラ本人が
    周囲へ明かした範囲（user_visibility_note）でユーザを話題にしてよい。捏造が予想チャネル
    （[ANTICIPATE_RESPONSE]）経由でターンをまたいで自己成就するのも併せて禁じる。

    なお「Narrator はユーザを知らない」は厳密には実現不能（Narrator も NPC も中身は同一の
    GM-LLM で同じ system prompt を読む）。実体は「情報は手元にあるが、役割ごとの開示ルールで
    使い分ける」── 完全な情報遮断ではない。

    Args:
        character_name: 主人公キャラ名（PC枠と一致する想定）。
        user_label: ユーザの呼称（characters.user_label）。空ならブロック自体を生成しない。
        user_position: ユーザの位置づけ（characters.user_position）。NPC が別呼称で言及しても
            同一人物と分かるための手がかり。
        visibility_note: キャラ本人が周囲への伝達範囲を自分の言葉で書き下ろした文章
            （characters.user_visibility_note）。空なら完全秘匿モード（NPC も触れない）。

    Returns:
        GM へ渡す OOC ブロック文字列。user_label が空なら空文字列（ブロック非生成）。
    """
    label = (user_label or "").strip()
    if not label:
        return ""
    position = (user_position or "").strip()
    note = (visibility_note or "").strip()
    name = (character_name or "").strip() or "本人"

    lines = [
        f"# {label} の扱い（この世界における重要ルール）",
        f"{label} は @{name} の生活に実在する人物だが、この場面には居ない。",
    ]
    if position:
        lines.append(f"※ {label} ＝ {position}（NPC が別呼称で言及しても同一人物）。")

    # 原則: ユーザ「について」可 ／ ユーザ「が」不可。中身の供給源はキャラ本人のみ。
    lines += [
        "",
        f"▼ 原則 — {label}「について」触れるのは可、{label}「が」動く・話すのは不可。",
        f"- あなた（語り手／世界）と NPC は、{label} 本人を**登場させない・代弁しない・"
        f"{label} を主語にした言動を一切描かない**。連絡・訪問・依頼・通知・メッセージ・電話・"
        f"LINE・メールなど、{label} 発の出来事を起こさないこと（物理的な姿だけでなく遠隔接触も同様）。",
        f"- 特に @Narrator（地の文）は {label} をこの場面に持ち込まない（{label} 発の出来事を"
        f"地の文で起こさない）。",
        f"- {label} が何を言った・何をした・何を伝えてきたか、その**中身を場に持ち込めるのは "
        f"@{name} 本人だけ**。@{name} は {label} と現実で交流しており、自分から {label} の話を"
        f"持ち出すことがある。それは妨げない（むしろ自然）。",
    ]

    if note:
        # NPC は伝聞の範囲で「ユーザについて」話題化してよい（質問に限らず・歓迎）。
        lines += [
            "",
            f"▼ NPC は {label}「について」触れてよい（歓迎）。",
            f"@{name} 本人が周囲にどう伝えているかの範囲で、NPC は {label} を話題にし、尋ね、"
            f"コメントしてよい（質問だけに縛らない。会話の自然な呼び水として歓迎する）:",
            "",
            "  " + note.replace("\n", "\n  "),
            "",
            f"書かれている範囲のことだけが「周囲が知っている前提」。書かれていないことは周囲も知らない。"
            f"ただし会話の中心に {label} を据えたり、{label} の登場・連絡を促したりはしない"
            f"（あくまで今日の {name} の一日が主役）。",
        ]
    else:
        lines += [
            "",
            f"▼ @{name} は {label} のことを周囲に明かしていない。",
            f"NPC は {label} を話題に出さない（暗に匂わせもしない）。{label} に関する質問・噂・言及は"
            f"NPC 側から発生させない。",
        ]

    # 予想（ANTICIPATE）経由の自己成就を断つ。
    lines += [
        "",
        f"▼ 予想でも先取りしない — [ANTICIPATE_RESPONSE] に「{label} から連絡が来る」等、"
        f"{label} の言動を書いて次の場面で起こさないこと。{label} 発の展開を予想・仕込みしない。",
    ]
    return "\n".join(lines)


def _build_real_contact_block(sqlite, owner_char) -> str:
    """うつつ GM 向けの「現実の接触の封筒」ブロックを組み立てる（めぐり Phase 1）。

    タイムライン正本から直近7日の real イベント（chat.* / action.performed）を
    observer="world_frame" で投影し、封筒（存在・時刻・往復数）だけを GM に渡す。
    中身（何を話したか）は投影ポリシーが envelope 止めにするため決して載らない。

    Args:
        sqlite: SQLiteStore。
        owner_char: うつつ世界の所有者 Character。

    Returns:
        GM へ渡す OOC ブロック文字列。素材が無ければ空文字列。
    """
    from datetime import datetime, timedelta

    from backend.services.timeline import format_real_contact_block, project

    try:
        events = project(
            character_id=owner_char.id,
            observer="world_frame",
            sqlite=sqlite,
            since=datetime.now() - timedelta(days=7),
            origins=["real"],
            types=["chat.*", "action.performed"],
        )
        return format_real_contact_block(
            events,
            character_name=getattr(owner_char, "name", "") or "",
            user_label=getattr(owner_char, "user_label", "") or "",
        )
    except Exception:
        # 封筒注入は補助情報なので、失敗してもシーン進行は止めない
        logger.exception("現実接触封筒の構築に失敗 owner=%s", getattr(owner_char, "id", "?"))
        return ""


def _build_usual_gm_appendix(
    scenario,
    fired_responses: int,
    max_responses: int,
    is_first_gm: bool,
    rng: random.Random | None = None,
    absent_user_block: str = "",
) -> str:
    """うつつ GM レスポンスの OOC 追記（常設フレーミング＋偶発イベント＋ソフト収束）を組み立てる。

    - 常設フレーミング（_USUAL_GM_STANDING）: 毎レスポンス必ず添える。これが [SCENE_CLOSE] の
      存在と「無人の日常・GM は外的フレームのみ」をGMに伝える土台で、主たる停止機構の前提。
    - 不在のユーザブロック（absent_user_block）: 非空なら毎レスポンス末尾に添える。
      キャラ本人の周知設定（user_visibility_note）に基づく NPC 言及制御。
    - 口火モード抽選: シーン最初の GM レスポンス（is_first_gm）でのみ、
      event_probability で 1 回抽選してこのシーンの幕開け方を決める（1 シーンに 1 度だけ）。
        * ヒット → イベント駆動。従来どおり状況提示し、カテゴリ候補があれば偶発イベントの
          種を 1 つ撒く（候補なしでも素の状況提示は成立する）。
        * ハズレ → 軽い幕開け（_USUAL_LIGHT_OPENING）。GM は場だけ軽く置き、出来事を立てず
          キャラ本人に委ねる（キャラ主導・内省の回を混ぜ、単調な「GM起点→キャラ反応」を崩す）。
      2 レスポンス目以降は常に従来どおりの状況提示（軽い幕開けは初回口火限定）。
    - ソフト収束: 残りレスポンス数が _USUAL_SOFT_CLOSE_REMAINING 以下になったら、
      終盤の念押しを添える（停止判断はあくまで GM）。

    Returns:
        OOC 追記文字列（複数行）。常設フレーミングを含むため headless では常に非空。
    """
    parts: list[str] = [_USUAL_GM_STANDING]
    if absent_user_block:
        parts.append(absent_user_block)
    if is_first_gm:
        # 口火モードを event_probability で 1 回だけ抽選する（カテゴリ有無に依らず確率で決める）。
        usual_config = getattr(scenario, "usual_config", None)
        prob = _usual_event_probability(usual_config)
        roll = rng if rng is not None else random.Random()
        if prob > 0.0 and roll.random() < prob:
            # ヒット = イベント駆動。候補があれば偶発イベントの種を 1 つ撒く。
            categories = _usual_event_categories(usual_config)
            if categories:
                parts.append(_format_usual_event_hint(roll.choice(categories)))
        else:
            # ハズレ = 軽い幕開け。GM は場だけ置いてキャラに委ねる。
            parts.append(_USUAL_LIGHT_OPENING)
    if max_responses - fired_responses <= _USUAL_SOFT_CLOSE_REMAINING:
        parts.append(_USUAL_SOFT_CLOSE_HINT)
    return "\n\n".join(parts)


async def run_usual_days_scene(
    session_id: str,
    sqlite,
    settings: dict,
    chat_service,
    engine: SceneEngine | None = None,
    extra_first_gm_ooc: str = "",
    slot: str = "",
) -> dict:
    """うつつ（Usual Days）の 1 シーンを無人で回し、結果サマリを返す薄いトリガー。

    `run_scenario_turn(headless=True, auto_advance=True)` を内部で駆動して SSE イベントを
    すべて drain する（うつつは SSE 配信を伴わないため、戻り値の dict だけ使う）。
    スケジューラ（main.py）と、デバッグ用の手動 1 シーン実行の双方から呼ぶ共通入口。

    Args:
        session_id: うつつセッション（engine_type="usual_days"）の ID。
        sqlite: SQLiteStore。
        settings: グローバル設定辞書。
        chat_service: PC レスポンス実行に必須の ChatService。
        engine: GM エンジン。None なら既定エンジンを使う。
        extra_first_gm_ooc: シーン冒頭の GM へ添える経過時間メモ等（「前回から N 時間後」）。
        slot: 起動スロット時刻（"13:00" 等、スケジューラ起動時のみ）。scene.closed 封筒の payload に載る。

    Returns:
        {"saved_turn_ids": [...], "fired_responses": int, "fired_turns": int,
         "scene_closed": bool, "error": str | None} の集計 dict。
        fired_responses は LLM 呼出回数（GM + PC）、fired_turns は scenario_turns 行数（=話者ブロック数）。
        本人の reach_out でシーンが一時停止した場合は "paused_for_push": True が付く
        （蒸留・封筒・意図拾い上げはスキップされ、再開はスケジューラの領分）。
    """
    # service とは相互依存（service.run_scenario_turn がうつつ味付けに本モジュールを使う）の
    # ため、ループ本体の import はここで遅延させて循環を断つ。
    from backend.services.scenario_chat.service import run_scenario_turn

    saved_turn_ids: list[str] = []
    fired_responses = 0
    scene_closed = False
    error: str | None = None
    # うつつシーン進行中マーカー（めぐり Phase 5）: シーン中は availability ゲートが
    # unavailable("usual_scene") を返す。クラッシュ時も TTL で自然失効する。
    _pre_session = sqlite.get_scenario_session(session_id)
    _pre_scenario = sqlite.get_scenario(_pre_session.scenario_id) if _pre_session else None
    _gate_owner_id = getattr(_pre_scenario, "owner_character_id", None) if _pre_scenario else None
    if _gate_owner_id:
        from backend.services.gate import mark_usual_scene_running
        mark_usual_scene_running(sqlite, _gate_owner_id, True)
    try:
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
                fired_responses = int(payload.get("fired_responses", 0))
            elif ev_type == "error":
                # 統一エラーイベント: GM 由来は message のみ、PC 由来は character も付く。
                # 無人運転なので最初の 1 件を観測ログとして記録する。
                character = payload.get("character")
                if character:
                    # PC（キャラ）応答のエラー。ループは break され、それまでのターン（話者ブロック）は保存される。
                    if error is None:
                        error = f"PC応答エラー（{character}）: {payload.get('message', '')}"
                else:
                    # GM レスポンス等の致命的エラー（多くは例外送出）。シーンはここで打ち切られる。
                    error = str(payload.get("message", ""))
    finally:
        if _gate_owner_id:
            from backend.services.gate import mark_usual_scene_running
            mark_usual_scene_running(sqlite, _gate_owner_id, False)

    # reach_out ポーズ検知: 本人がシーン中に現実へメッセージを送った場合、ループは
    # 本人の発言終了で停止している（loop_strategies）。ポーズ要求キーが立っていれば
    # このシーンは「完走」ではなく「一時停止」— 蒸留・封筒・意図拾い上げは行わず、
    # スケジューラ（main.py）による再開（GM 継続）を待つ。
    paused_for_push = False
    if _gate_owner_id:
        from backend.character_actions.messenger import read_push_pause
        paused_for_push = read_push_pause(sqlite, _gate_owner_id) is not None
    if paused_for_push:
        logger.info(
            "うつつ: reach_out によりシーン一時停止 session=%s fired=%d",
            session_id, fired_responses,
        )
        return {
            "saved_turn_ids": saved_turn_ids,
            "fired_responses": fired_responses,
            "fired_turns": len(saved_turn_ids),
            "scene_closed": False,
            "paused_for_push": True,
            "error": error,
        }

    # シーンが GM の [SCENE_CLOSE] で閉じたかは、最終 GM ターン（話者ブロック）の生出力から判定する。
    for turn in reversed(sqlite.list_scenario_turns(session_id)):
        if getattr(turn, "speaker_type", "") in {"narrator", "npc"}:
            scene_closed = _has_scene_close(getattr(turn, "raw_response", "") or "")
            break

    # あらすじ蒸留（うつつ専用の発火点）。
    # 通常シナリオはフロントが進捗バーを見てユーザがプリセットを選び regenerate API を
    # 叩くが、うつつは無人ゆえ介入者がいない。そこで「1 シーン完走」を蒸留の
    # チェックポイントにし、ここで best-effort に蒸留する。force=False なので毎シーン
    # 必ず LLM を叩くわけではなく、前回蒸留以降の未蒸留ターンが history 上限の
    # SYNOPSIS_AUTO_TRIGGER_RATIO に達したときだけ実走する。これにより
    # スライディングウィンドウから古いターンが押し出される前にあらすじが先回りして
    # カバーし、GM が過去を忘れる事故を防ぐ。プリセットはセッションの
    # synopsis_preset_id（ensure_usual_session で GM プリセットと共通に記録済み）を使う。
    session = sqlite.get_scenario_session(session_id)
    scenario = sqlite.get_scenario(session.scenario_id) if session else None
    if session is not None and scenario is not None:
        # 蒸留の debug ログを "usual_days"（シーン進行）と取り違えないよう、
        # regenerate API と同じく feature を "synopsis" に切り替えてから呼ぶ。
        current_log_feature.set("synopsis")
        await maybe_update_auto_synopsis(
            sqlite=sqlite,
            settings=settings,
            scenario=scenario,
            history=sqlite.list_scenario_turns(session_id),
            session_id=session_id,
            synopsis_preset_id=(
                getattr(session, "synopsis_preset_id", "")
                or getattr(session, "gm_preset_id", "")
                or ""
            ),
            force=False,
        )

    # 計器 Tier 1（usual_scene_error）: 生活の中断は即時アラーム。
    # run_usual_days_scene が error なく完走することが正常条件。
    if error is not None:
        from backend.lib.instrument_recorder import fire_alarm
        fire_alarm("usual_scene_error", details={
            "session_id": session_id,
            "slot": slot,
            "error": error,
            "fired_responses": fired_responses,
        })

    # タイムライン封筒（scene.closed）: うつつシーンの完走を正本に載せる（payload 完結型）。
    # エラーで打ち切られたシーンは「完走」ではないため載せない
    # （エラー自体は計器 Tier 1 `usual_scene_error` の領分）。
    if error is None and scenario is not None and scenario.owner_character_id:
        sqlite.record_timeline_event(
            character_id=scenario.owner_character_id,
            event_type="scene.closed",
            actor="system",
            origin="usual",
            session_id=session_id,
            payload={
                "slot": slot,
                "turns": len(saved_turn_ids),
                "closed_by": "scene_close" if scene_closed else "turn_limit",
            },
        )
        # 意図の拾い上げ（めぐり Phase 4・うつつシーン完走後）: auto_synopsis と同じ
        # チェックポイントで「あとに残りそうな『〜したい』はある？」を本人に問う。
        # ターンが1つも保存されなかったシーンは体験ゼロなのでスキップする。
        # 拾い上げの失敗はシーン結果を壊さない。
        if saved_turn_ids:
            try:
                from backend.services.intents import run_intent_pickup
                current_log_feature.set("intent_pickup")
                await run_intent_pickup(
                    scenario.owner_character_id,
                    sqlite,
                    settings,
                    born_from="usual_scene",
                    memory_manager=getattr(chat_service, "memory_manager", None),
                    working_memory_manager=getattr(
                        chat_service, "working_memory_manager", None
                    ),
                )
            except Exception:
                logger.exception(
                    "意図の拾い上げに失敗 owner=%s", scenario.owner_character_id
                )

    return {
        "saved_turn_ids": saved_turn_ids,
        "fired_responses": fired_responses,
        "fired_turns": len(saved_turn_ids),
        "scene_closed": scene_closed,
        "error": error,
    }


def _build_usual_pc_assignments(pc_slots, owner_id: str, pc_pid: str) -> list[dict]:
    """うつつセッションの pc_assignments を組み立てる。

    主人公（owner キャラ）の character 割当に加え、ユーザPC枠（slot_id="user"）が
    定義されていれば「不在のユーザPC」割当（player_type="user"）を足す。ユーザPCは
    無人ループ中レスポンス順を取らない（run_scenario_turn の routing から除外される）が、
    GM の pc_summary には不在の PC として並び、GM の「PC を代弁しない」保護を効かせる。

    Args:
        pc_slots: normalize_pc_slots 済みの PC枠リスト。
        owner_id: 主人公キャラの ID。
        pc_pid: 主人公 PC を動かすプリセット ID。

    Returns:
        create_scenario_session に渡す pc_assignments のリスト。
    """
    char_slot = next((s for s in pc_slots if s.slot_id != "user"), pc_slots[0])
    assignments = [{
        "slot_id": char_slot.slot_id,
        "player_type": "character",
        "character_id": owner_id,
        "preset_id": pc_pid,
    }]
    if any(s.slot_id == "user" for s in pc_slots):
        assignments.append({"slot_id": "user", "player_type": "user"})
    return assignments


def sync_usual_session_presets(sqlite, scenario) -> int:
    """うつつシナリオの ``usual_config`` を、対応する active な ``usual_days`` セッションへ追従させる。

    Backend のうつつ設定 UI で ``gm_preset_id`` / ``pc_preset_id`` を更新した際、
    既に走っている usual_days セッションの ``session.gm_preset_id`` と
    ``pc_assignments[*].preset_id``（player_type="character" のもの）を最新値で
    上書きする。後勝ちルール — 最後に変更された側を真とする。

    PC 側プリセットの解決順は :func:`ensure_usual_session` と同一：
    ``usual_config.pc_preset_id`` → owner.ghost_model → ``usual_config.gm_preset_id``。

    Args:
        sqlite: SQLiteStore。
        scenario: 最新化済みの ``Scenario``（``owner_character_id`` / ``usual_config`` 必須）。

    Returns:
        反映を行ったセッションの数。同期不要（差分なし／対象なし）なら 0。
    """
    cfg = getattr(scenario, "usual_config", None) or {}
    gm_pid = (cfg.get("gm_preset_id") or "").strip()
    owner_id = getattr(scenario, "owner_character_id", None)
    if not gm_pid or not owner_id:
        return 0

    owner_char = sqlite.get_character(owner_id)
    owner_ghost = (getattr(owner_char, "ghost_model", None) or "") if owner_char else ""
    pc_pid = ((cfg.get("pc_preset_id") or "").strip() or owner_ghost.strip() or gm_pid)

    updated = 0
    for s in sqlite.list_scenario_sessions_by_scenario(scenario.id):
        if getattr(s, "engine_type", "") != "usual_days":
            continue
        if getattr(s, "status", "") != "active":
            continue

        updates: dict = {}
        if (getattr(s, "gm_preset_id", "") or "") != gm_pid:
            updates["gm_preset_id"] = gm_pid

        raw = list(getattr(s, "pc_assignments", None) or [])
        new_assignments: list[dict] = []
        changed_pc = False
        for entry in raw:
            if not isinstance(entry, dict):
                new_assignments.append(entry)
                continue
            if entry.get("player_type") == "character":
                if (entry.get("preset_id") or "") != pc_pid:
                    entry = {**entry, "preset_id": pc_pid}
                    changed_pc = True
            new_assignments.append(entry)
        if changed_pc:
            updates["pc_assignments"] = new_assignments

        if updates:
            sqlite.update_scenario_session(s.id, **updates)
            updated += 1

    return updated


def _attach_usual_user_assignment(sqlite, session, pc_slots) -> None:
    """既存うつつセッションに、後から定義されたユーザPC枠の割当を冪等に補う。

    ユーザPC（不在の人物）設定は本機能の途中から追加されたため、それ以前に作られた
    セッションには user 割当が無い。シナリオに user 枠が定義済みで、セッションの
    pc_assignments に user 割当が無ければ、ここで足して永続化する（再呼び出しでは no-op）。

    Args:
        sqlite: SQLiteStore。
        session: 既存の usual_days セッション（ORM オブジェクト）。
        pc_slots: normalize_pc_slots 済みの PC枠リスト。
    """
    if not any(s.slot_id == "user" for s in pc_slots):
        return
    raw = list(getattr(session, "pc_assignments", None) or [])
    if any(isinstance(e, dict) and e.get("player_type") == "user" for e in raw):
        return
    raw.append({"slot_id": "user", "player_type": "user"})
    sqlite.update_scenario_session(session.id, pc_assignments=raw)
    # 呼び出し側が直後に使う in-memory オブジェクトにも反映する。
    try:
        session.pc_assignments = raw
    except Exception:
        pass


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

    pc_slots = normalize_pc_slots(getattr(scenario, "pc_slots", None))

    # 既存の active な usual_days セッションを優先（永続1本）。
    # ユーザPC枠（不在の人物）が後から設定された場合は、既存セッションにも割当を補う。
    for s in sqlite.list_scenario_sessions_by_scenario(scenario.id):
        if getattr(s, "engine_type", "") == "usual_days" and getattr(s, "status", "") == "active":
            _attach_usual_user_assignment(sqlite, s, pc_slots)
            return s

    cfg = getattr(scenario, "usual_config", None) or {}
    gm_pid = (cfg.get("gm_preset_id") or "").strip()
    owner_id = getattr(scenario, "owner_character_id", None)
    # PC（主人公）を動かすプリセットの決定順:
    #   1. usual_config.pc_preset_id（明示指定があれば最優先）
    #   2. owner キャラの Ghost Model（このキャラの内省・自己処理に使う既定モデル）
    #   3. GM プリセット（いずれも無いときの最終フォールバック）
    # 「指定が無ければ、そのキャラ本来の Ghost Model でそのキャラを動かす」のが自然なため、
    # GM への素通しより Ghost Model を優先する。
    owner_char = sqlite.get_character(owner_id) if owner_id else None
    owner_ghost = (getattr(owner_char, "ghost_model", None) or "") if owner_char else ""
    pc_pid = ((cfg.get("pc_preset_id") or "").strip() or owner_ghost.strip() or gm_pid)
    if not gm_pid or not owner_id or not pc_slots:
        logger.warning(
            "うつつ: セッション起動に必要な情報が不足 owner=%s gm_preset=%s slots=%d",
            owner_id, gm_pid, len(pc_slots),
        )
        return None

    session_id = str(uuid.uuid4())
    return sqlite.create_scenario_session(
        session_id=session_id,
        scenario_id=scenario.id,
        title=getattr(scenario, 'title', 'うつつ'),
        gm_preset_id=gm_pid,
        synopsis_preset_id=gm_pid,
        engine_type="usual_days",
        pc_assignments=_build_usual_pc_assignments(pc_slots, owner_id, pc_pid),
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
