"""シナリオ PC モード用 メンション解析モジュール。

GM の発話本文中に現れる `@<name>` を抽出し、PC配役（pc_assignments）と
照合してそのターン中に発話すべき PC を順序付きで決定する。

挙動の要点:
    - 行頭 `@<name>:` の話者宣言と、地の文中の `@<name>` 言及の両方を抽出する。
      （`ScenarioChatParser` は行頭 `@:` のみ話者扱い／本文中 `@` は本文として通すため、
      呼びかけ宛先の検出はここで別途行う）
    - メンション名は **本名 → 配役名** の順でフォールバック解決する。
    - `@ALL` は直前話者以外からランダム1名（同じ PC が連続で呼ばれることを防ぐ）。
    - PC配役本名・PC配役名・NPC名・user_alias の三方衝突（実際は四方）は
      `detect_name_conflicts` で検知し、セッション起動時にエラーとして UI へ返す。

このモジュールは LLM 呼び出しを行わない純粋関数の集合。
ランダム選択は `random` モジュールを使う（ユニットテスト時は `random.seed` で固定可）。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# `@<name>` のスキャン用パターン。`@` の直後から、空白・コロン・`@`・改行・記号類
# に到達するまでを名前とみなす。NPC名は日本語/英数を許容するため広めに取り、
# 末尾の句読点や記号は呼び出し側で剥がす設計にしている。
_MENTION_RE = re.compile(r"@([^\s@:、。,.!\?！？\)\]」』]+)")


@dataclass
class PcAssignment:
    """PC配役の正規化レコード。

    Attributes:
        character_id: Chotgor の Character.id。実在チェックは呼び出し側で行う。
        role_name: シナリオ内での配役表示名。`@<role_name>` でメンションされる。
        character_name: Chotgor 上のキャラクター本名（最初に lookup して埋める）。
            `@<character_name>` でメンションされた場合の解決に使う。
    """

    character_id: str
    role_name: str
    character_name: str


def normalize_pc_assignments(
    pc_assignments_raw: list[dict] | None, sqlite,
) -> list[PcAssignment]:
    """生 pc_assignments JSON を PcAssignment のリストに正規化する。

    Args:
        pc_assignments_raw: ScenarioSession.pc_assignments の JSON
            （[{"character_id": "...", "role_name": "..."}]）。None / 空なら空リスト。
        sqlite: SQLiteStore（character_name の lookup に使用）。

    Returns:
        正規化された PcAssignment のリスト。character_id が SQLite で見つからない
        要素はスキップする（呼び出し側でバリデーション済みである前提だが念のため）。
    """
    if not pc_assignments_raw:
        return []
    out: list[PcAssignment] = []
    for entry in pc_assignments_raw:
        if not isinstance(entry, dict):
            continue
        cid = str(entry.get("character_id", "")).strip()
        role = str(entry.get("role_name", "")).strip()
        if not cid or not role:
            continue
        char = sqlite.get_character(cid)
        if char is None:
            continue
        out.append(PcAssignment(
            character_id=cid,
            role_name=role,
            character_name=char.name,
        ))
    return out


def extract_mentions(text: str) -> list[str]:
    """テキストから `@<name>` メンションを出現順に抽出する（重複は保持）。

    話者宣言行（行頭 `@<name>:`）も本文中言及（`@<name>` 単体）もどちらも拾う。
    `name` 末尾の句読点系記号はパターン側で除外しているが、念のため strip もかける。
    """
    if not text:
        return []
    return [m.strip().rstrip(":") for m in _MENTION_RE.findall(text) if m.strip()]


def resolve_pc(name: str, pcs: list[PcAssignment]) -> PcAssignment | None:
    """メンション名から PC を解決する。

    解決順:
        1. 本名（character_name）完全一致
        2. 配役名（role_name）完全一致

    Args:
        name: メンション名（`@` は含めない）。
        pcs: 正規化済み PC 配役リスト。

    Returns:
        ヒットした PcAssignment。どちらにも一致しなければ None。
    """
    if not name:
        return None
    for pc in pcs:
        if pc.character_name == name:
            return pc
    for pc in pcs:
        if pc.role_name == name:
            return pc
    return None


def resolve_pc_mentions_in_order(
    text: str, pcs: list[PcAssignment],
) -> list[PcAssignment]:
    """GM 出力テキストから PC メンションを出現順に解決して PC リストを返す。

    同一 PC が複数回メンションされても、ターン内で同一 PC は1回のみ発話させる
    （重複を出現順を保ったまま除く）。

    `@ALL` は専用シンボルとして扱われるためここでは展開しない。呼び出し側で
    `expand_at_all` を別途使うこと（順序保持の都合で2フェーズに分けている）。

    Args:
        text: GM 発話本文。
        pcs: 正規化済み PC 配役リスト。

    Returns:
        ターンで発話させる PC のリスト（出現順、重複なし）。
    """
    seen: set[str] = set()
    out: list[PcAssignment] = []
    for name in extract_mentions(text):
        pc = resolve_pc(name, pcs)
        if pc is None:
            continue
        if pc.character_id in seen:
            continue
        seen.add(pc.character_id)
        out.append(pc)
    return out


def pick_at_all_target(
    pcs: list[PcAssignment],
    last_pc_speaker_id: str | None = None,
    rng: random.Random | None = None,
) -> PcAssignment | None:
    """`@ALL` の応答対象を 1 名ランダム選択する。

    直前に発話した PC を選択候補から除外することで、@ALL が連続したときに
    同じ PC ばかりが応える事態を避ける。PC が 1 人しかいない場合は除外を緩和し、
    その 1 名を返す（除外で空集合になるのを防ぐ）。

    Args:
        pcs: 正規化済み PC 配役リスト。
        last_pc_speaker_id: 直前ターンで応答した PC の character_id。
            該当する PC は候補から除外する。
        rng: テスト用に注入できる random.Random インスタンス。

    Returns:
        選ばれた PC。PC が 1 人もいなければ None。
    """
    if not pcs:
        return None
    rng = rng or random
    if len(pcs) == 1:
        return pcs[0]
    if last_pc_speaker_id is None:
        return rng.choice(pcs)
    candidates = [p for p in pcs if p.character_id != last_pc_speaker_id]
    if not candidates:
        # 全員が last_speaker と同じ ID（理論上ありえないが防御）
        return rng.choice(pcs)
    return rng.choice(candidates)


def detect_name_conflicts(
    pcs: list[PcAssignment],
    npc_names: set[str],
    user_alias: str,
    narrator_name: str = "Narrator",
) -> list[str]:
    """PC本名・PC配役名・NPC名・user_alias・narrator_name の名前衝突を検知する。

    シナリオ起動時のバリデーション用。衝突があると `@<name>:` の話者解決が
    曖昧になり、GM が PC を NPC として書く／NPC を PC として書くといった
    事故の原因になる。

    Returns:
        衝突している名前のリスト（重複なし、見つかった順）。空なら問題なし。
    """
    seen: dict[str, int] = {}

    def bump(name: str) -> None:
        if not name:
            return
        seen[name] = seen.get(name, 0) + 1

    # user_alias と narrator は1個ずつカウント
    bump(user_alias)
    bump(narrator_name)
    for n in npc_names:
        bump(n)
    for pc in pcs:
        bump(pc.character_name)
        # 本名と配役名が同名の場合、ここで2カウントになるがそれは意図通り
        # （PC 本名 == 配役名 は事故源なので警告したい）
        if pc.role_name != pc.character_name:
            bump(pc.role_name)
        else:
            # 同名なら 1 件で済ませる代わりに、外向きにはこの名前を衝突として返す
            seen[pc.role_name] = max(seen.get(pc.role_name, 0), 2)

    return [name for name, count in seen.items() if count >= 2]


def format_pc_summary(pcs: list[PcAssignment], user_alias: str) -> str:
    """GM の system prompt に差し込む `{pc_summary}` ブロック本文を整形する。

    各 PC を「@<role_name>（本名: <char_name>） ← PC。別の AI が演じる。代弁禁止」の
    1 行で並べる。先頭に総括文を添え、GM が PC を別人格として扱う気持ちを強める。

    Args:
        pcs: 正規化済み PC 配役リスト。
        user_alias: ユーザの @タグ用呼称（本文の引き合いに出すために受け取る）。

    Returns:
        プロンプトに差し込み可能なテキスト。pcs が空なら空文字列。
    """
    if not pcs:
        return ""
    lines = [
        f"以下の PC は @{user_alias} と同じく「別の人格が演じるプレイヤーキャラクター」です。",
        "あなたは絶対に代弁しません。呼びかけは NPC の台詞や状況描写で行ってください。",
        "",
    ]
    for pc in pcs:
        if pc.character_name == pc.role_name:
            lines.append(f"@{pc.role_name} ← PC。別のAIが演じる。代弁禁止。")
        else:
            lines.append(
                f"@{pc.role_name}（本名: @{pc.character_name}）"
                " ← PC。別のAIが演じる。代弁禁止。"
            )
    return "\n".join(lines)
