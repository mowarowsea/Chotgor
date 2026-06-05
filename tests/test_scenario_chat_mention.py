"""シナリオ PC モード — メンション解析モジュールの単体テスト。

`backend/services/scenario_chat/mention.py` の純粋関数群を検証する。

検証観点:
    - `extract_mentions`: 行頭 `@<name>:` と地の文 `@<name>` を順序保ったまま抽出
       句読点・記号での末尾切り詰めが正しく動くこと
    - `resolve_pc`: 本名 → 配役名フォールバック解決
       両方ヒットしないときは None
    - `resolve_pc_mentions_in_order`: 同一 PC は最初の出現のみで重複除去、順序保持
    - `pick_at_all_target`: 直前話者除外、PC が 1 名なら除外緩和、決定性のため
       random.Random を注入する
    - `detect_name_conflicts`: NPC・PC本名・PC配役名・user_alias・narrator の四方衝突
       で全衝突名が返ること（重複なし）
    - `format_pc_summary`: 配役名と本名が同名のときの表示分岐

ダミー sqlite は `normalize_pc_assignments` だけが使うので、本ファイルでは
``resolve_pc`` 等の引数として PcAssignment を直接組み立てる経路でテストする。
"""

import random

import pytest

from backend.services.scenario_chat.mention import (
    PcAssignment,
    detect_name_conflicts,
    extract_mentions,
    format_pc_summary,
    pick_at_all_target,
    resolve_pc,
    resolve_pc_mentions_in_order,
)


def _pc(role: str, character_name: str | None = None, cid: str | None = None) -> PcAssignment:
    """テスト用の PcAssignment ファクトリ。

    role: 配役名（@<role>: でメンションされる名前）
    character_name: Chotgor の本名（@<本名>: でメンションされる名前。省略時は role と同じ）
    cid: character_id。省略時は role から生成。
    """
    return PcAssignment(
        character_id=cid or f"cid-{role}",
        role_name=role,
        character_name=character_name or role,
    )


class TestExtractMentions:
    """`extract_mentions` の境界を検証する。"""

    def test_行頭_at_name_colon_は配役名としてヒットする(self):
        """行頭の `@<name>:` は名前として抽出されること（`:` を含めない）。"""
        names = extract_mentions("@アリス: 何の用?\n")
        assert names == ["アリス"]

    def test_地の文中の_at_name_も抽出する(self):
        """地の文に書かれた `@<name>` も呼びかけとして抽出すること。"""
        names = extract_mentions("@Narrator: アリス、@アリス は道を見つめている。\n")
        # Narrator と アリス が拾われる（順序保持）
        assert "Narrator" in names
        assert "アリス" in names
        assert names.index("Narrator") < names.index("アリス")

    def test_句読点で名前が切れる(self):
        """`@<name>。` のような句読点は名前に含めない。"""
        # 句点・読点・閉じ括弧の前で切れる
        names = extract_mentions("@アリス、こちらへ。@ボブ。@キャロル)行こう\n")
        assert "アリス" in names
        assert "ボブ" in names
        assert "キャロル" in names

    def test_空文字列は空リストを返す(self):
        """入力が空ならエラーにならず空リスト。"""
        assert extract_mentions("") == []
        assert extract_mentions(None) == []  # type: ignore[arg-type]

    def test_at_all_も抽出する_大文字判定は呼び出し側(self):
        """`@ALL` も名前として抽出される（大文字判定はこの関数の責務外）。"""
        names = extract_mentions("@ALL こんにちは\n")
        assert "ALL" in names


class TestResolvePc:
    """`resolve_pc` の本名→配役名フォールバックを検証する。"""

    def test_本名で先にヒットする(self):
        """本名と配役名が別なら、本名一致が優先される。"""
        pcs = [_pc(role="アリス", character_name="はる")]
        hit = resolve_pc("はる", pcs)
        assert hit is not None
        assert hit.character_id == "cid-アリス"

    def test_配役名でフォールバック(self):
        """本名にヒットしなくても配役名で解決される。"""
        pcs = [_pc(role="アリス", character_name="はる")]
        hit = resolve_pc("アリス", pcs)
        assert hit is not None
        assert hit.character_id == "cid-アリス"

    def test_どちらにもヒットしないとNone(self):
        pcs = [_pc(role="アリス", character_name="はる")]
        assert resolve_pc("ボブ", pcs) is None

    def test_空名前はNone(self):
        pcs = [_pc(role="アリス")]
        assert resolve_pc("", pcs) is None


class TestResolvePcMentionsInOrder:
    """`resolve_pc_mentions_in_order` の順序保持と重複除去を検証する。"""

    def test_同一PCは初出のみ_順序保持(self):
        """同じ PC が複数回メンションされても、初出位置の順序のみ残る。"""
        pcs = [
            _pc(role="アリス", character_name="はる"),
            _pc(role="ボブ", character_name="なつ"),
        ]
        text = "@アリス: やあ\n@ボブ: ...\n@アリス: ねえ\n"
        result = resolve_pc_mentions_in_order(text, pcs)
        # アリス→ボブ の順で、アリスは1回のみ
        assert [r.role_name for r in result] == ["アリス", "ボブ"]

    def test_ALLはここでは展開しない(self):
        """`@ALL` は PC 名ではないため、この関数は無視する。"""
        pcs = [_pc(role="アリス")]
        text = "@ALL: 一斉に頼む\n"
        result = resolve_pc_mentions_in_order(text, pcs)
        assert result == []

    def test_未知メンションはスキップ(self):
        """PC配役にない名前は静かに無視。"""
        pcs = [_pc(role="アリス")]
        text = "@モブ通行人: 邪魔だ\n@アリス: すみません\n"
        result = resolve_pc_mentions_in_order(text, pcs)
        assert [r.role_name for r in result] == ["アリス"]


class TestPickAtAllTarget:
    """`pick_at_all_target` のランダム選択と直前話者除外を検証する。"""

    def test_PCゼロならNone(self):
        assert pick_at_all_target([], last_pc_speaker_id=None) is None

    def test_PCが1名のときはその1名_除外緩和(self):
        """PC が 1 人しかいないケースでは、たとえそれが直前話者でも返す（除外緩和）。"""
        pcs = [_pc(role="アリス", cid="cid-A")]
        assert pick_at_all_target(pcs, last_pc_speaker_id="cid-A") is pcs[0]
        assert pick_at_all_target(pcs, last_pc_speaker_id=None) is pcs[0]

    def test_直前話者を除外して残りからランダム選択(self):
        """直前話者が指定されていれば、それ以外から選ばれる（再現性のため seed 固定）。"""
        pcs = [
            _pc(role="アリス", cid="cid-A"),
            _pc(role="ボブ", cid="cid-B"),
            _pc(role="キャロル", cid="cid-C"),
        ]
        rng = random.Random(42)
        # cid-A を直前話者とすると、選ばれるのは B/C のいずれか
        for _ in range(20):
            chosen = pick_at_all_target(pcs, last_pc_speaker_id="cid-A", rng=rng)
            assert chosen is not None
            assert chosen.character_id != "cid-A"


class TestDetectNameConflicts:
    """`detect_name_conflicts` の四方衝突検知。"""

    def test_衝突なしなら空リスト(self):
        pcs = [_pc(role="アリス", character_name="はる")]
        npcs = {"レイカ", "ガイ"}
        assert detect_name_conflicts(pcs, npcs, user_alias="マスター") == []

    def test_PC配役名とNPC名衝突を検知(self):
        """PC配役の `アリス` と NPC の `アリス` が同名なら衝突として返る。"""
        pcs = [_pc(role="アリス", character_name="はる")]
        npcs = {"アリス", "ガイ"}
        conflicts = detect_name_conflicts(pcs, npcs, user_alias="マスター")
        assert "アリス" in conflicts

    def test_PC本名とuser_alias衝突を検知(self):
        pcs = [_pc(role="アリス", character_name="プレイヤー")]
        npcs = {"ガイ"}
        conflicts = detect_name_conflicts(pcs, npcs, user_alias="プレイヤー")
        assert "プレイヤー" in conflicts

    def test_PC本名と配役名が同名なら衝突として返る(self):
        """1 人の PC の中で本名 == 配役名 は事故源として警告したい。"""
        pcs = [_pc(role="ハル", character_name="ハル")]
        npcs: set[str] = set()
        conflicts = detect_name_conflicts(pcs, npcs, user_alias="マスター")
        assert "ハル" in conflicts

    def test_Narratorとの衝突も検知(self):
        pcs = [_pc(role="Narrator", character_name="はる")]
        conflicts = detect_name_conflicts(pcs, set(), user_alias="マスター")
        assert "Narrator" in conflicts


class TestFormatPcSummary:
    """`format_pc_summary` の表示分岐を検証する。"""

    def test_PCゼロなら空文字列(self):
        assert format_pc_summary([], user_alias="マスター") == ""

    def test_本名と配役名が違うときは両方表示する(self):
        pcs = [_pc(role="アリス", character_name="はる")]
        text = format_pc_summary(pcs, user_alias="マスター")
        assert "@アリス" in text
        assert "@はる" in text
        # 「代弁禁止」の文言が含まれる（GM への明示）
        assert "代弁禁止" in text

    def test_本名と配役名が同じなら本名表示は省略(self):
        pcs = [_pc(role="ハル", character_name="ハル")]
        text = format_pc_summary(pcs, user_alias="マスター")
        # `@ハル ← PC` の形で1回だけ言及（本名表記は出ない）
        assert text.count("@ハル") == 1
