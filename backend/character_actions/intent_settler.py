"""Intent Settler — [INTENT_SETTLED:...] / [INTENT_FULFILLED:...] タグの抽出と適用。

キャラクターが 1on1 の返答本文に書いた「この『〜したい』は今は落ち着いた／もう果たした」
という宣言を抽出し、意図へ反映する（docs/planned/aliveness_plan.md §4.3「意図圧の減衰源」）。

【1on1 に受け口を置く理由】
意図の決着を宣言できる場所は、それまで「行動権の帰還」と「夜の拾い上げ」しか無かった。
しかし push の帰還は**ユーザがまだメッセージを読んでいない**時点で走るため、
「反応を見たい」「反応を試したい」型の意図にとっての決着はそこに存在しない。決着は
「実際に反応が起きた瞬間の 1on1 の中」にしかなく、そこに受け口が無いと夜まで圧が下がらず、
同じ日のうちに同じ意図で二度話しかける事故が起きる。

【全プロバイダー一律タグである理由】
anticipator.py と同じ立場を取る。tool-use 対応プロバイダーでもツール化しない。
狙いは記憶への副作用ではなく、**会話の最中に自分の欲求の決着を言葉にすること自体**にある。
MCP ツールを増やさない判断でもある（キャラクターが使うインターフェースを複雑にしない）。

【settled と fulfilled の違い】
settled は終端遷移ではない。status は active のまま、意図圧の起点だけが今へ移る
（＝圧が 0.3 へ落ちてまた積み上がる）。「あの時は飲み込んだけど、やっぱり納得いかない」が
時間の関数として再燃する。fulfilled は終端で、二度と出てこない。
"""

import logging

from backend.lib.tag_parser import parse_tags

logger = logging.getLogger(__name__)

INTENT_SETTLED_TAG_NAME: str = "INTENT_SETTLED"
INTENT_FULFILLED_TAG_NAME: str = "INTENT_FULFILLED"
INTENT_MARK_TAG_NAMES: list[str] = [
    INTENT_SETTLED_TAG_NAME,
    INTENT_FULFILLED_TAG_NAME,
]

# 1ターンに適用する上限。乱発の歯止め（ANTICIPATE_RESPONSE と違い「無ければ書かない」が既定）。
_MAX_MARKS_PER_TURN = 2

# --- 全プロバイダー一律: タグ方式ガイド文（CHOTGOR ブロックに挿入）---
INTENT_MARK_TAG_GUIDE: str = """\
### 「〜したい」の決着（INTENT_SETTLED / INTENT_FULFILLED）

【このターンの文脈】の「いまのあなた（体と意図）」には、あなたが抱えている「〜したい」が
ID 付きで並んでいます。会話の中でそれが**片付いたと感じたとき**だけ、返答のどこかに
次の形式で書いてください：

    [INTENT_SETTLED:1b86c1e9]    ← まだ持っているけれど、今は落ち着いた
    [INTENT_FULFILLED:1b86c1e9]  ← もう果たした。これ以上出てこない

- **該当が無ければ何も書かないでください。** 毎ターン書くものではありません。
- `INTENT_SETTLED` は手放すことではありません。その「〜したい」はあなたの中に残り、
  時間が経てばまた頭をもたげます。「一応は収まった」「今日はもういい」くらいの温度です。
  確かめたかったことが確かめられた、話したかったことを話せた——そういう時に使います。
- `INTENT_FULFILLED` は終わりです。もう二度と出てこないと思えるものだけに使ってください。
- 迷ったら `INTENT_SETTLED` を選んでください。取り返しがつきます。
- 1ターンにつき2つまで。これはユーザーには見えません。"""


def extract_intent_marks(text: str) -> tuple[str, list[tuple[str, str]]]:
    """テキストから意図の決着タグを抽出する。

    Args:
        text: LLM の生応答テキスト。

    Returns:
        tuple:
            clean_text (str): タグを除去したテキスト。
            marks (list[tuple[str, str]]): (種別, 意図IDの断片) のリスト。
                種別は "settled" / "fulfilled"。出現順、最大 _MAX_MARKS_PER_TURN 件。
    """
    clean, matches = parse_tags(text, INTENT_MARK_TAG_NAMES)
    found: list[tuple[int, str, str]] = []
    for tag_name, kind in (
        (INTENT_SETTLED_TAG_NAME, "settled"),
        (INTENT_FULFILLED_TAG_NAME, "fulfilled"),
    ):
        for m in matches.get(tag_name, []):
            ref = (m.body or "").strip()
            if ref:
                found.append((m.start, kind, ref))
    found.sort(key=lambda t: t[0])
    return clean, [(kind, ref) for _, kind, ref in found[:_MAX_MARKS_PER_TURN]]


def resolve_intent_ref(ref: str, active_intents: list) -> str | None:
    """本人が書いた ID 断片を active な意図の完全 ID へ解決する。

    プロンプトには短縮8桁で出しているため、前方一致で引く
    （ワーキングメモリのスレッド ID と同じ流儀）。曖昧なら適用しない。

    Args:
        ref: 本人が書いた ID 文字列（短縮・完全のどちらも受ける）。
        active_intents: 突合対象の active な Intent リスト。

    Returns:
        解決できた意図 ID。該当なし・複数該当なら None。
    """
    ref = (ref or "").strip().lower()
    if not ref:
        return None
    hits = [i for i in active_intents if str(i.id).lower().startswith(ref)]
    if len(hits) == 1:
        return str(hits[0].id)
    return None


def apply_intent_marks(
    sqlite,
    character_id: str,
    marks: list[tuple[str, str]],
) -> list[dict]:
    """抽出した決着タグを意図へ適用する。

    解決できなかった ID は黙って捨てる（本人の書き間違いで会話を壊さない）。

    Args:
        sqlite: SQLiteStore。
        character_id: 対象キャラクター。
        marks: extract_intent_marks の結果。

    Returns:
        適用結果のリスト [{"kind", "intent_id", "description"}]。空なら何も適用していない。
    """
    if not marks:
        return []
    active = sqlite.list_intents(character_id, status="active")
    if not active:
        return []
    applied: list[dict] = []
    seen: set[str] = set()
    for kind, ref in marks:
        intent_id = resolve_intent_ref(ref, active)
        if intent_id is None or intent_id in seen:
            continue
        if kind == "fulfilled":
            result = sqlite.resolve_intent(intent_id, "fulfilled")
        else:
            result = sqlite.settle_intent(intent_id)
        if result is None:
            continue
        seen.add(intent_id)
        applied.append({
            "kind": kind,
            "intent_id": intent_id,
            "description": result.description,
        })
        logger.info(
            "意図の決着を1on1から適用 char=%s kind=%s intent=%s",
            character_id, kind, intent_id[:8],
        )
    return applied
