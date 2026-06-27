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
"""

import logging
import random
import re
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

# うつつ（Usual Days）無人ループの 1 シーンあたり既定上限レスポンス数。
# usual_config.max_responses_per_scene が無指定のときのフォールバック（ハード上限の保険）。
# 旧キー名 max_turns_per_scene は後方互換のため読み出し時にフォールバックする。
_DEFAULT_USUAL_MAX_RESPONSES = 8

# GM がシーンの幕引きを宣言するマーカー。うつつ無人ループの主たる停止条件。
# 検出は raw_response に対して行い、表示用 content からは extract_scene_close で除去する
# （anticipator と同じく「タグは機能、本文には残さない」方針）。
_SCENE_CLOSE_MARKER = "[SCENE_CLOSE]"

# [SCENE_CLOSE] を大小・全角半角の揺れも込みで拾う正規表現（検出と本文除去の共通源）。
_SCENE_CLOSE_RE = re.compile(r"\[\s*scene[_\s]?close\s*\]", re.IGNORECASE)


def _has_scene_close(text: str | None) -> bool:
    """GM の生出力にシーン幕引きマーカー（[SCENE_CLOSE]）が含まれるか判定する。

    うつつ無人ループの主たる停止判断（判断主体は GM）。大文字小文字・区切り揺れを許容する。
    """
    if not text:
        return False
    return bool(_SCENE_CLOSE_RE.search(text))


def extract_scene_close(text: str | None) -> tuple[str, bool]:
    """テキストから [SCENE_CLOSE] マーカーを除去し、(クリーン本文, 検出フラグ) を返す。

    anticipator.extract_anticipation と同じ思想で、マーカーは「機能」として扱い、
    表示・要約に残る本文からは取り除く。検出は揺れ（大小・全半角・空白）に寛容。

    Args:
        text: GM の発話テキスト。

    Returns:
        (marker を除いた本文, 1 つ以上検出されたか) のタプル。
    """
    if not text:
        return "", False
    found = bool(_SCENE_CLOSE_RE.search(text))
    if not found:
        return text, False
    cleaned = _SCENE_CLOSE_RE.sub("", text)
    # 除去で生じた行末の余白・空行を軽く整える。
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()
    return cleaned, True


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
    """うつつ GM 向けの「不在のユーザ」ブロックを組み立てる。

    キャラ本人が周囲（NPC）にユーザのことをどう伝えているか（characters.user_visibility_note）
    を素通しで載せ、NPC の自発的言及度を本人の流儀に委ねる。空欄なら完全秘匿として
    「NPC は触れない・話題にも出さない」を明示する（安全側デフォルト）。

    本ブロックは「ユーザを画面に出さない」という既存保護（pc_slots[user] の不在マーカー
    と prompt_builder の PC領分節）とは独立した追加レイヤー: 「周囲がユーザのことを
    どう扱うか」を本人の言葉でコントロールする。

    Args:
        character_name: 主人公キャラ名（PC枠 pc1 と一致する想定）。
        user_label: ユーザの呼称（characters.user_label）。空ならブロック自体を生成しない。
        user_position: ユーザの位置づけ（characters.user_position）。NPC が別名・別呼称で
            ユーザに言及してきても同一人物と分かるための手がかり。
        visibility_note: キャラ本人が周囲への伝達範囲を自分の言葉で書き下ろした文章
            （characters.user_visibility_note）。空なら完全秘匿モード。

    Returns:
        GM へ渡す OOC ブロック文字列。user_label が空なら空文字列（ブロック非生成）。
    """
    label = (user_label or "").strip()
    if not label:
        return ""
    position = (user_position or "").strip()
    note = (visibility_note or "").strip()
    name = (character_name or "").strip() or "本人"

    head_lines = [
        f"# 不在の {label} について（周囲の人物が知っていること）",
        f"今この場面に @{name} の生活上の人物「{label}」は居ない。",
    ]
    if position:
        head_lines.append(f"※ {label} ＝ {position}（NPC が別呼称で言及しても同一人物）。")

    if note:
        body_lines = [
            "",
            f"NPC が {label} のことに触れてよい範囲は、@{name} 本人が周囲にどう伝えているかに従う:",
            "",
            "  " + note.replace("\n", "\n  "),
            "",
            f"書かれている範囲のことだけが「周囲が知っている前提」。書かれていないことは周囲も知らない。",
            f"NPC が自発的に {label} に軽く触れるのは構わないが、会話の中心に据えたり、"
            f"{label} の登場を促したりしてはいけない。主役はあくまで今日の {name} の一日。",
        ]
    else:
        body_lines = [
            "",
            f"@{name} は {label} のことを周囲には明かしていない。",
            f"NPC は {label} を話題に出さない（暗に匂わせもしない）。"
            f" {label} に関する質問・噂・言及は NPC 側から発生させない。",
        ]
    return "\n".join(head_lines + body_lines)


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
    # PCロスター（不在ユーザPC含む全PC）。GM へ「全PCを代弁するな」と均一に提示する。
    pc_summary_text = format_pc_summary(pcs)

    # うつつ（headless）はユーザにレスポンス順を回さない（ユーザは不在の人物）。ルーティングの
    # 候補からユーザPCを除外する。pc_summary / suppress_names は全PC（不在ユーザ含む）のまま
    # GM に渡し、GM の「PC を代弁しない」保護を不在ユーザにも効かせる。
    routing_pcs = [p for p in pcs if not p.is_user] if is_headless else pcs

    # ユーザの @タグ名: player_type="user" のスロット name。無ければフォールバック。
    # うつつ（headless）では無言進行 OOC で使う総称を既定の「プレイヤー」のままにする
    # （ユーザPC名を user_alias に出すと _USUAL_GM_STANDING の「プレイヤー＝この人物自身」と
    #  齟齬が出るため。不在ユーザの呼称は pc_summary / suppress_names 側で扱う）。
    user_pc = next((p for p in pcs if p.is_user), None)
    user_speaker_name = "プレイヤー" if is_headless else (user_pc.name if user_pc else "プレイヤー")

    # GM が代弁してはならない名前: 全 PC枠名 + 全 AI キャラ本名（不在ユーザPCも含む）。
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
        chat_service: PC レスポンス実行に必須の ChatService。
        engine: GM エンジン。None なら既定エンジンを使う。
        extra_first_gm_ooc: シーン冒頭の GM へ添える経過時間メモ等（「前回から N 時間後」）。

    Returns:
        {"saved_turn_ids": [...], "fired_responses": int, "fired_turns": int,
         "scene_closed": bool, "error": str | None} の集計 dict。
        fired_responses は LLM 呼出回数（GM + PC）、fired_turns は scenario_turns 行数（=話者ブロック数）。
    """
    saved_turn_ids: list[str] = []
    fired_responses = 0
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


