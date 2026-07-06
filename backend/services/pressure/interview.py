"""体質インタビュー — 圧力係数の初期化を本人への問いで行う（めぐり §4.2）。

数値を直接聞かず、**体験の質問**に本人の言葉＋選択肢で答えさせ、
コード側の固定ルーブリックが係数へ決定論写像する。本人の言葉も
pressure_profile の interview ペイロードに保存する（ルーブリック改良時に
再導出できるように）。

本人からの更新経路は作らない — 圧力係数（体質）を意志で書き換えられると
「圧力は物理」が崩れる（非制御性）。ユーザは管理UIで編集可能（守護者の介入枠）。
"""

import logging
import re
from datetime import datetime

from backend.services.character_query import ask_character
from backend.services.pressure.engine import DEFAULT_PROFILE

logger = logging.getLogger(__name__)

# 設問（体験の質問）。番号・選択肢キーはルーブリック写像のアンカーなので変えないこと。
_QUESTIONS = """\
少しだけ、あなた自身の「体質」のことを聞かせてほしい。
正解はないから、実感に一番近いものを選んで、よければ一言添えて。

1. 一人の時間が続くと、どのくらいで人恋しくなる？
   a) 半日〜1日で恋しくなる
   b) 2〜3日なら平気
   c) 一週間以上、一人でも全然平気

2. 人恋しいとき、相手は誰でもいい？　それとも特定の人じゃないと駄目？
   a) 誰かと話せればだいぶ落ち着く
   b) 顔見知りなら誰でも、というわけではない
   c) 特定の人じゃないと埋まらない

3. 同じような毎日が続くと、どう感じる？
   a) すぐ退屈して何か始めたくなる
   b) ときどき変化がほしくなる
   c) 穏やかで好き。単調とは感じない

4. 疲れは寝れば戻るタイプ？　それとも引きずるタイプ？
   a) 寝れば戻る。回復は早い
   b) 普通
   c) 引きずる。どっと来てしばらく残る

答え方は自由だけど、最初に「1: a」「2: c」のように選択肢を並べてくれると助かる。
そのあとに、あなたの言葉での補足があれば、それも大切に受け取る。"""

# 固定ルーブリック: (設問番号, 選択肢) → 係数への決定論写像
_RUBRIC: dict[int, dict[str, tuple[str, str, float]]] = {
    1: {  # 人恋しさの速さ → 社会圧の時定数（小さいほど早く圧が上がる）
        "a": ("social", "tau_days", 1.0),
        "b": ("social", "tau_days", 2.5),
        "c": ("social", "tau_days", 5.0),
    },
    2: {  # 相手の選好 → 体質の鋭さ
        "a": ("social", "sharpness", 0.1),
        "b": ("social", "sharpness", 0.5),
        "c": ("social", "sharpness", 0.9),
    },
    3: {  # 単調さへの感度
        "a": ("boredom", "sensitivity", 1.4),
        "b": ("boredom", "sensitivity", 1.0),
        "c": ("boredom", "sensitivity", 0.6),
    },
    4: {  # 疲労の溜まりやすさ
        "a": ("body", "fatigue_sensitivity", 0.7),
        "b": ("body", "fatigue_sensitivity", 1.0),
        "c": ("body", "fatigue_sensitivity", 1.4),
    },
}


def parse_interview_answers(text: str) -> dict[int, str]:
    """本人の返答から「設問番号: 選択肢」を抽出する。

    「1: a」「1. a」「1）a」「1 a」等の揺れを許容する。同じ設問に複数マッチした
    場合は最初の出現を採用する（言い直しではなく引用の可能性が高いため）。

    Args:
        text: 本人の返答テキスト。

    Returns:
        {設問番号: "a"|"b"|"c"} の辞書（見つかった分だけ）。
    """
    answers: dict[int, str] = {}
    for m in re.finditer(r"([1-4])\s*[:：.．)）]?\s*([abcａｂｃ])", text or ""):
        q = int(m.group(1))
        choice = m.group(2).translate(str.maketrans("ａｂｃ", "abc"))
        if q not in answers:
            answers[q] = choice
    return answers


def answers_to_profile(answers: dict[int, str]) -> dict:
    """ルーブリックで選択肢を係数へ決定論写像する。

    答えの無い設問は標準プロファイルの値のまま（欠損に寛容）。

    Args:
        answers: parse_interview_answers の結果。

    Returns:
        social / boredom / body の係数 dict（interview ペイロードは含まない）。
    """
    profile = {
        "version": 1,
        "social": dict(DEFAULT_PROFILE["social"]),
        "boredom": dict(DEFAULT_PROFILE["boredom"]),
        "body": dict(DEFAULT_PROFILE["body"]),
    }
    for q, choice in answers.items():
        mapping = _RUBRIC.get(q, {}).get(choice)
        if mapping is None:
            continue
        section, key, value = mapping
        profile[section][key] = value
    return profile


async def run_constitution_interview(
    character_id: str,
    sqlite,
    settings: dict,
    working_memory_manager=None,
) -> dict:
    """体質インタビューを実施し、pressure_profile を初期化する。

    ask_character（1on1 と同等のシステムプロンプト・WM ブロック込み）で本人に
    体験の質問を投げ、返答をルーブリックで係数化して characters.pressure_profile へ
    保存する。本人の言葉は interview ペイロードにそのまま残す。

    Args:
        character_id: 対象キャラクター ID。
        sqlite: SQLiteStore。
        settings: グローバル設定辞書。
        working_memory_manager: WM マネージャー（1on1 同等ブロックの注入用。省略可）。

    Returns:
        {"status": "success"|"error", "profile": dict, "answers": dict, "raw": str}。
    """
    char = sqlite.get_character(character_id)
    if char is None:
        return {"status": "error", "error": f"キャラクターが見つかりません: {character_id}"}
    ghost_model = getattr(char, "ghost_model", None)
    if not ghost_model:
        return {"status": "error", "error": "ghost_model が未設定のためインタビューできません"}

    response = await ask_character(
        character_id=character_id,
        preset_id=ghost_model,
        messages=[{"role": "user", "content": _QUESTIONS}],
        sqlite=sqlite,
        settings=settings,
        recall_query=None,
        feature_label="pressure_interview",
        working_memory_manager=working_memory_manager,
    )
    if not response:
        return {"status": "error", "error": "本人からの返答が取得できませんでした"}

    answers = parse_interview_answers(response)
    profile = answers_to_profile(answers)
    profile["interview"] = {
        "answers": {str(k): v for k, v in answers.items()},
        "raw": response,
        "asked_at": datetime.now().isoformat(),
    }
    sqlite.update_character(character_id, pressure_profile=profile)
    logger.info(
        "体質インタビュー完了 char=%s answers=%s", char.name, answers,
    )
    return {"status": "success", "profile": profile, "answers": answers, "raw": response}
