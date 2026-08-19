"""シナリオ設定テキスト用「値タグ」の展開。

GM システムプロンプト専用の**ブロックタグ**（`{scenario}` / `{history_block}` /
`{npc_details}` など）とは分けて、シナリオ本文・intro・PC枠/NPC の description
といった**設定テキスト側**でも使える短い名前タグだけを切り出したモジュール。

二層に分ける理由:
    設定テキストでブロックタグまで許すと、`{scenario}` の自己再帰や
    `{history_block}` の多重展開が起きる。値タグは 1 段展開・固定長なので、
    どこに書かれても安全に解決できる。

サポートするタグ:
    ``{user_alias}``     ユーザが演じている PC 枠の名前
    ``{narrator_name}``  Narrator のタグ名
    ``{pc_name[1]}``     PC 枠の名前（1 始まり。pc_slots の並び順）
    ``{pc_name[pc1]}``   PC 枠の名前（slot_id 指定。並べ替え・枠追加に強い）
    ``{npc_name[1]}``    NPC の名前（1 始まり。NPC の作成順）

解決できないタグ（範囲外の番号・未定義の slot_id・名前未入力の枠）は
**書かれたまま残す**。黙って空文字にすると、タイポで名前が抜け落ちた文章が
そのまま LLM へ渡っても誰も気づけないため。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# `{tag}` / `{tag[key]}` の両形をまとめて拾う。key に `{}[]` を許さないのは、
# 入れ子や壊れた記法を「解決できないタグ」として素通しさせるため。
_VALUE_TAG_RE = re.compile(
    r"\{(user_alias|narrator_name|pc_name|npc_name)(?:\[([^\[\]{}]*)\])?\}"
)


@dataclass(frozen=True)
class TagContext:
    """値タグの解決に必要な名前だけを束ねたコンテキスト。

    Attributes:
        user_alias: ユーザが演じる PC 枠の名前。うつつ（headless）では
            GM にユーザ実名を出さない方針のため "プレイヤー" が入る。
        narrator_name: Narrator のタグ名。
        pc_names: PC 枠の名前（pc_slots の並び順）。名前未入力の枠も
            番号がずれないよう空文字のまま位置を保持する。
        pc_names_by_slot: slot_id（小文字化）→ 枠名。
        npc_names: NPC の名前（作成順）。
    """

    user_alias: str = "プレイヤー"
    narrator_name: str = "Narrator"
    pc_names: tuple[str, ...] = ()
    pc_names_by_slot: Mapping[str, str] = field(default_factory=dict)
    npc_names: tuple[str, ...] = ()


def _text_of(entry: Any, key: str) -> str:
    """dict でも ORM/dataclass でも同じように 1 フィールドを文字列で取り出す。

    pc_slots は生 JSON（dict）で渡ってくる経路と、正規化済み PcSlot /
    PcAssignment で渡ってくる経路の両方があるため両対応にしている。
    """
    if isinstance(entry, Mapping):
        value = entry.get(key, "")
    else:
        value = getattr(entry, key, "")
    return str(value or "").strip()


def build_tag_context(
    *,
    user_alias: str = "プレイヤー",
    narrator_name: str = "Narrator",
    pc_entries: Iterable[Any] | None = None,
    npcs: Iterable[Any] | None = None,
) -> TagContext:
    """値タグ解決用のコンテキストを組み立てる。

    Args:
        user_alias: `{user_alias}` に入る名前。
        narrator_name: `{narrator_name}` に入る名前。
        pc_entries: PC 枠。`Scenario.pc_slots` の生 JSON でも、正規化済みの
            PcSlot / PcAssignment でも受け付ける。番号は**シナリオ定義の並び順**を
            基準にする（セッションごとの配役や、うつつでの名簿除外で番号が
            動かないようにするため）。
        npcs: ScenarioNpc ORM 風オブジェクト（作成順）。
    """
    names: list[str] = []
    by_slot: dict[str, str] = {}
    for entry in pc_entries or []:
        name = _text_of(entry, "name")
        names.append(name)
        slot_id = _text_of(entry, "slot_id").lower()
        # 同じ slot_id が重複した場合は先勝ち（先に定義された枠を優先）。
        if slot_id and slot_id not in by_slot:
            by_slot[slot_id] = name
    npc_names = tuple(_text_of(n, "name") for n in npcs or [])
    return TagContext(
        user_alias=(user_alias or "").strip() or "プレイヤー",
        narrator_name=(narrator_name or "").strip() or "Narrator",
        pc_names=tuple(names),
        pc_names_by_slot=by_slot,
        npc_names=npc_names,
    )


def expand_value_tags(text: str, ctx: TagContext) -> str:
    """テキスト中の値タグを 1 段だけ展開する。

    展開結果に含まれる `{...}` は再展開しない（re.sub は置換後の文字列を
    走査し直さないため、この性質は無料で得られる）。

    Returns:
        展開後のテキスト。解決できなかったタグは書かれたまま残る。
    """
    if not text or "{" not in text:
        return text

    def _resolve(m: re.Match) -> str:
        tag, key = m.group(1), m.group(2)
        if tag == "user_alias":
            # 添字付き（`{user_alias[1]}` 等）は誤記なので解決しない。
            return ctx.user_alias if key is None else m.group(0)
        if tag == "narrator_name":
            return ctx.narrator_name if key is None else m.group(0)
        if key is None:
            # `{pc_name}` のような添字なしは、どの枠を指すか決められない。
            return m.group(0)
        names = ctx.pc_names if tag == "pc_name" else ctx.npc_names
        k = key.strip()
        if k.isdigit():
            idx = int(k)
            if 1 <= idx <= len(names) and names[idx - 1]:
                return names[idx - 1]
            return m.group(0)
        if tag == "pc_name":
            return ctx.pc_names_by_slot.get(k.lower()) or m.group(0)
        # NPC は slot_id を持たないので、番号以外の指定は解決しない。
        return m.group(0)

    return _VALUE_TAG_RE.sub(_resolve, text)
