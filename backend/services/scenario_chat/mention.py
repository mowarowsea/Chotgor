"""シナリオ PC モード用 メンション解析モジュール。

発話本文中に現れる `@<name>` を抽出し、PC枠（pc_slots）と PC配役（pc_assignments）
に照合して「次に発話すべき話者」を決定する。

設計（新形式）:
    - シナリオ側に `pc_slots = [{slot_id, name, description}]` を定義する（PC枠）。
    - セッション側に `pc_assignments = [{slot_id, player_type, character_id?, preset_id?}]`
      を割り当てる。player_type は "user"（ユーザが演じる）か "character"（Chotgor キャラが演じる）。
    - メンション解決時、@<name> の name は **PC枠 name → Chotgor キャラ本名 → NPC名/Narrator** の順で
      フォールバック解決する。
    - 行頭 `@<name>:` の話者宣言と地の文中の `@<name>` 言及の両方を抽出する。

ルーティング判定:
    - 発話本文の中で最後のメンションが「PC枠の name」 → そのスロットを担当する話者へ
        - player_type="user" のスロット → ユーザターン
        - player_type="character" のスロット → 該当 Chotgor キャラのターン
    - メンションが「@GM」「@Narrator」「NPC名」 → GM ターン
    - メンションが `@ALL` → 直前話者を除外したランダム1名（PC枠側からランダム抽選）
    - メンションが何も無い → GM ターン（地の文を進めるため）

このモジュールは LLM 呼び出しを行わない純粋関数の集合。
ランダム選択は `random` モジュールを使う（ユニットテスト時は `random.seed` で固定可）。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# `@<name>` のスキャン用パターン。`@` の直後から、空白・コロン・`@`・改行・記号類
# に到達するまでを名前とみなす。NPC名は日本語/英数を許容するため広めに取り、
# 末尾の句読点や記号は呼び出し側で剥がす設計にしている。
_MENTION_RE = re.compile(r"@([^\s@:、。,.!\?！？\)\]」』]+)")


@dataclass
class PcSlot:
    """PC枠（シナリオ側で定義）の正規化レコード。

    Attributes:
        slot_id: 枠の識別子（例: "pc1"）。UI 上でユーザが入力するか自動生成。
        name: 枠の表示名。`@<name>` でメンションされる。
        description: 人物像・口調・知っていることを全部詰めた自由テキスト。
            ensemble_pc 時、AI キャラ担当枠ならこれが PC の「配役メモ」として
            キャラのシステムプロンプトへ注入される。
    """

    slot_id: str
    name: str
    description: str = ""


@dataclass
class PcAssignment:
    """セッション側でのスロット割当てを表す正規化レコード。

    Attributes:
        slot_id: 紐づく PC枠の ID。
        name: PC枠の name（表示・解決用にキャッシュ）。
        description: PC枠の description（配役メモ）。
        player_type: "user" か "character"。
        character_id: player_type="character" のとき、Chotgor キャラの ID。
        character_name: player_type="character" のとき、Chotgor キャラの本名。
            `@<character_name>` でメンションされた場合のフォールバック解決に使う。
        preset_id: player_type="character" のとき、使う LLMModelPreset の ID。
            未指定なら呼び出し側で enabled_providers 先頭をフォールバックする。
    """

    slot_id: str
    name: str
    player_type: str
    description: str = ""
    character_id: str | None = None
    character_name: str | None = None
    preset_id: str | None = None

    @property
    def is_user(self) -> bool:
        """ユーザが演じる枠かどうか。"""
        return self.player_type == "user"

    @property
    def is_character(self) -> bool:
        """AI キャラが演じる枠かどうか。"""
        return self.player_type == "character"


def normalize_pc_slots(pc_slots_raw: list[dict] | None) -> list[PcSlot]:
    """生の pc_slots JSON を PcSlot のリストへ正規化する。

    Args:
        pc_slots_raw: Scenario.pc_slots の JSON。None / 空なら空リスト。

    Returns:
        正規化された PcSlot のリスト。slot_id・name 必須、欠落要素はスキップ。
    """
    if not pc_slots_raw:
        return []
    out: list[PcSlot] = []
    for entry in pc_slots_raw:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("slot_id", "")).strip()
        name = str(entry.get("name", "")).strip()
        if not sid or not name:
            continue
        out.append(PcSlot(
            slot_id=sid,
            name=name,
            description=str(entry.get("description", "") or "").strip(),
        ))
    return out


def normalize_pc_assignments(
    pc_assignments_raw: list[dict] | None,
    pc_slots: list[PcSlot],
    sqlite,
) -> list[PcAssignment]:
    """生 pc_assignments JSON を PcAssignment のリストに正規化する。

    Args:
        pc_assignments_raw: ScenarioSession.pc_assignments の JSON。
            形式: [{"slot_id":"pc1","player_type":"user"|"character",
                    "character_id":"...","preset_id":"..."}]。
        pc_slots: 親シナリオで定義済みの PC枠リスト（slot_id → name/description の参照元）。
        sqlite: SQLiteStore（character_name の lookup に使用）。

    Returns:
        正規化された PcAssignment のリスト。slot_id が pc_slots に無い／
        player_type 不正／character_id 不在の要素はスキップする。
    """
    if not pc_assignments_raw or not pc_slots:
        return []
    slot_by_id = {s.slot_id: s for s in pc_slots}
    out: list[PcAssignment] = []
    for entry in pc_assignments_raw:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("slot_id", "")).strip()
        ptype = str(entry.get("player_type", "")).strip()
        if not sid or ptype not in {"user", "character"}:
            continue
        slot = slot_by_id.get(sid)
        if slot is None:
            continue
        if ptype == "character":
            cid = str(entry.get("character_id", "")).strip()
            if not cid:
                continue
            char = sqlite.get_character(cid)
            if char is None:
                continue
            preset_id = str(entry.get("preset_id", "") or "").strip() or None
            out.append(PcAssignment(
                slot_id=sid,
                name=slot.name,
                description=slot.description,
                player_type="character",
                character_id=cid,
                character_name=char.name,
                preset_id=preset_id,
            ))
        else:
            out.append(PcAssignment(
                slot_id=sid,
                name=slot.name,
                description=slot.description,
                player_type="user",
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
    """メンション名から PC スロットを解決する。

    解決順:
        1. 枠 name（pc_slot.name）完全一致
        2. AI キャラの本名（character_name）完全一致

    Args:
        name: メンション名（`@` は含めない）。
        pcs: 正規化済み PC 配役リスト。

    Returns:
        ヒットした PcAssignment。どちらにも一致しなければ None。
    """
    if not name:
        return None
    for pc in pcs:
        if pc.name == name:
            return pc
    for pc in pcs:
        if pc.is_character and pc.character_name == name:
            return pc
    return None


def pick_at_all_target(
    pcs: list[PcAssignment],
    last_speaker_name: str | None = None,
    rng: random.Random | None = None,
) -> PcAssignment | None:
    """`@ALL` の応答対象を 1 名ランダム選択する。

    直前に発話した話者を選択候補から除外することで、@ALL が連続したときに
    同じ話者ばかりが応える事態を避ける。候補が空になる場合は除外を緩和する。

    Args:
        pcs: 正規化済み PC 配役リスト。
        last_speaker_name: 直前ターンの話者名。一致する PC は候補から除外する。
        rng: テスト用に注入できる random.Random インスタンス。

    Returns:
        選ばれた PC。PC が 1 人もいなければ None。
    """
    if not pcs:
        return None
    rng = rng or random
    if len(pcs) == 1:
        return pcs[0]
    if last_speaker_name is None:
        return rng.choice(pcs)
    candidates = [p for p in pcs if p.name != last_speaker_name]
    if not candidates:
        return rng.choice(pcs)
    return rng.choice(candidates)


def detect_name_conflicts(
    pcs: list[PcAssignment],
    npc_names: set[str],
    narrator_name: str = "Narrator",
) -> list[str]:
    """PC枠 name・PCキャラ本名・NPC名・Narrator の名前衝突を検知する。

    シナリオ起動時のバリデーション用。衝突があると `@<name>:` の話者解決が
    曖昧になり、GM が PC を NPC として書く／NPC を PC として書くといった
    事故の原因になる。

    Note: user_alias は新形式では「user player_type のスロット name」が
    その役を担うので、独立フィールドとしての衝突判定からは外す（pc.name に
    含まれるため自動的に判定される）。

    Returns:
        衝突している名前のリスト（重複なし、見つかった順）。空なら問題なし。
    """
    seen: dict[str, int] = {}

    def bump(name: str) -> None:
        if not name:
            return
        seen[name] = seen.get(name, 0) + 1

    bump(narrator_name)
    for n in npc_names:
        bump(n)
    for pc in pcs:
        bump(pc.name)
        if pc.is_character and pc.character_name and pc.character_name != pc.name:
            bump(pc.character_name)

    return [name for name, count in seen.items() if count >= 2]


def format_pc_summary(pcs: list[PcAssignment]) -> str:
    """GM の system prompt に差し込む `{pc_summary}` ブロック本文を整形する。

    全 PC を「@<name> ← PC。<description>」形式で**均一に**並べる。各 PC の「中の人」
    （人間プレイヤーか別の AI キャラか）は GM に一切開示しない — TRPG の卓では全員が
    等価なプレイヤーキャラクターであり、GM は誰が人間かを意識せず全 PC を等しく扱う。
    GM は PC を代弁してはならないため、強い禁止文言を付ける。

    Args:
        pcs: 正規化済み PC 配役リスト（ユーザPC・AIキャラPC を区別せず全件）。

    Returns:
        プロンプトに差し込み可能なテキスト。pcs が空なら空文字列。
    """
    if not pcs:
        return ""
    lines = [
        "以下の PC は「プレイヤーキャラクター」枠です。あなたは絶対に代弁しません。",
        "PC への呼びかけは NPC の台詞や状況描写で行ってください。",
        "各 PC が反応するかどうかは、それぞれを演じる本人に任せます。",
        "",
    ]
    for pc in pcs:
        # description は GM 側に共有して問題ないが、長文なので 1 行サマリだけ抜く。
        desc_brief = (pc.description or "").strip().replace("\n", " ")
        if len(desc_brief) > 80:
            desc_brief = desc_brief[:80] + "…"
        # 中の人（user/character）も AI キャラ本名も GM には出さない（区別させない）。
        if desc_brief:
            lines.append(f"@{pc.name} ← PC。{desc_brief}")
        else:
            lines.append(f"@{pc.name} ← PC。")
    return "\n".join(lines)


# ─── ルーティング判定 ───────────────────────────────────────────────────────


def find_last_routing_mention(
    text: str,
    pcs: list[PcAssignment],
    npc_names: set[str],
    narrator_name: str = "Narrator",
) -> tuple[str, str | None]:
    """発話末尾のルーティングメンションから「次の話者」分類を返す。

    話者切替を決める「最後の呼びかけ」を本文から拾う。GM 系メンション
    (`@GM`/`@Narrator`/NPC名) があれば GM 行きとし、PC スロット系メンションがあれば
    対象 PC 行きとする。`@ALL` は専用シンボル。

    Args:
        text: 1 発話分の本文。
        pcs: 正規化済み PC 配役リスト。
        npc_names: シナリオに定義された NPC 名集合。
        narrator_name: Narrator のタグ名。

    Returns:
        (kind, target) のタプル。
            kind ∈ {"pc", "all", "gm", "none"}
            target は kind="pc" のとき PC.name、kind="all" のとき None、
            kind="gm"/"none" のとき None。
        メンションが全く無ければ ("none", None)。
    """
    if not text:
        return ("none", None)
    names = extract_mentions(text)
    if not names:
        return ("none", None)
    # 後ろから順に「ルーティング判定に使えるメンション」を探す
    pc_names = {pc.name for pc in pcs}
    pc_real_names = {pc.character_name for pc in pcs if pc.is_character and pc.character_name}
    gm_aliases = {"GM", "gm", narrator_name}
    for name in reversed(names):
        if name.upper() == "ALL":
            return ("all", None)
        if name in pc_names:
            return ("pc", name)
        if name in pc_real_names:
            # 本名指名 → PC枠 name に正規化
            for pc in pcs:
                if pc.is_character and pc.character_name == name:
                    return ("pc", pc.name)
        if name in gm_aliases or name in npc_names:
            return ("gm", None)
    return ("none", None)
