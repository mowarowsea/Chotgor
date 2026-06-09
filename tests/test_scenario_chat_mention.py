"""シナリオ PC モード — メンション解析モジュールの単体テスト。

`backend/services/scenario_chat/mention.py` の純粋関数群を検証する。

検証観点:
    - `extract_mentions`: 行頭 `@<name>:` と地の文 `@<name>` を順序保ったまま抽出
       句読点・記号での末尾切り詰めが正しく動くこと
    - `resolve_pc`: 枠名 → AI キャラ本名のフォールバック解決
       両方ヒットしないときは None
    - `pick_at_all_target`: 直前話者除外、PC が 1 名なら除外緩和、決定性のため
       random.Random を注入する
    - `detect_name_conflicts`: NPC・PC枠名・PCキャラ本名・Narrator の衝突検知
    - `format_pc_summary`: AI/ユーザ枠の表記、description の付与
    - `find_last_routing_mention`: 発話末尾のメンション分類（pc / all / gm / none）
"""

import random

from backend.services.scenario_chat.mention import (
    PcAssignment,
    detect_name_conflicts,
    extract_mentions,
    find_last_routing_mention,
    format_pc_summary,
    pick_at_all_target,
    resolve_pc,
)


def _pc_char(name: str, character_name: str | None = None, cid: str | None = None,
              preset_id: str | None = None, description: str = "") -> PcAssignment:
    """テスト用の AI キャラ担当 PC配役ファクトリ。

    name: 枠の name（@<name> でメンションされる名前）
    character_name: AI キャラの本名（省略時は枠名と同じ）
    cid: character_id（省略時は枠名から生成）
    """
    return PcAssignment(
        slot_id=f"slot-{name}",
        name=name,
        player_type="character",
        description=description,
        character_id=cid or f"cid-{name}",
        character_name=character_name or name,
        preset_id=preset_id,
    )


def _pc_user(name: str, description: str = "") -> PcAssignment:
    """テスト用のユーザ担当 PC枠ファクトリ。"""
    return PcAssignment(
        slot_id=f"slot-{name}",
        name=name,
        player_type="user",
        description=description,
    )


class TestExtractMentions:
    """`extract_mentions` の境界を検証する。"""

    def test_行頭_at_name_colon_は名前としてヒットする(self):
        """行頭の `@<name>:` は名前として抽出されること（`:` を含めない）。"""
        names = extract_mentions("@アリス: 何の用?\n")
        assert names == ["アリス"]

    def test_地の文中の_at_name_も抽出する(self):
        """地の文に書かれた `@<name>` も呼びかけとして抽出すること。"""
        names = extract_mentions("@Narrator: アリス、@アリス は道を見つめている。\n")
        assert "Narrator" in names
        assert "アリス" in names
        assert names.index("Narrator") < names.index("アリス")

    def test_句読点で名前が切れる(self):
        """`@<name>。` のような句読点は名前に含めない。"""
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
    """`resolve_pc` の枠名→本名フォールバックを検証する。"""

    def test_枠名で先にヒットする(self):
        """枠名と本名が別なら、枠名一致が優先される。"""
        pcs = [_pc_char(name="アリス", character_name="はる")]
        hit = resolve_pc("アリス", pcs)
        assert hit is not None
        assert hit.character_id == "cid-アリス"

    def test_本名でフォールバック(self):
        """枠名にヒットしなくても AI キャラの本名で解決される。"""
        pcs = [_pc_char(name="アリス", character_name="はる")]
        hit = resolve_pc("はる", pcs)
        assert hit is not None
        assert hit.character_id == "cid-アリス"

    def test_ユーザ枠は本名フォールバック対象外(self):
        """player_type=user の枠は character_name を持たない（本名で引けない）。"""
        pcs = [_pc_user(name="プレイヤー")]
        assert resolve_pc("プレイヤー", pcs) is not None
        # ユーザ枠は character_name="" / None なので本名解決は対象外
        assert resolve_pc("はる", pcs) is None

    def test_どちらにもヒットしないとNone(self):
        pcs = [_pc_char(name="アリス", character_name="はる")]
        assert resolve_pc("ボブ", pcs) is None

    def test_空名前はNone(self):
        pcs = [_pc_char(name="アリス")]
        assert resolve_pc("", pcs) is None


class TestPickAtAllTarget:
    """`pick_at_all_target` のランダム選択と直前話者除外を検証する。"""

    def test_PCゼロならNone(self):
        assert pick_at_all_target([], last_speaker_name=None) is None

    def test_PCが1名のときはその1名_除外緩和(self):
        """PC が 1 人しかいないケースでは、たとえそれが直前話者でも返す（除外緩和）。"""
        pcs = [_pc_char(name="アリス")]
        assert pick_at_all_target(pcs, last_speaker_name="アリス") is pcs[0]
        assert pick_at_all_target(pcs, last_speaker_name=None) is pcs[0]

    def test_直前話者を除外して残りからランダム選択(self):
        """直前話者が指定されていれば、それ以外から選ばれる。"""
        pcs = [
            _pc_char(name="アリス"),
            _pc_char(name="ボブ"),
            _pc_char(name="キャロル"),
        ]
        rng = random.Random(42)
        # アリス を直前話者とすると、選ばれるのは ボブ/キャロル のいずれか
        for _ in range(20):
            chosen = pick_at_all_target(pcs, last_speaker_name="アリス", rng=rng)
            assert chosen is not None
            assert chosen.name != "アリス"


class TestDetectNameConflicts:
    """`detect_name_conflicts` の衝突検知。"""

    def test_衝突なしなら空リスト(self):
        pcs = [_pc_char(name="アリス", character_name="はる")]
        npcs = {"レイカ", "ガイ"}
        assert detect_name_conflicts(pcs, npcs) == []

    def test_PC枠名とNPC名衝突を検知(self):
        """PC枠の `アリス` と NPC の `アリス` が同名なら衝突として返る。"""
        pcs = [_pc_char(name="アリス", character_name="はる")]
        npcs = {"アリス", "ガイ"}
        conflicts = detect_name_conflicts(pcs, npcs)
        assert "アリス" in conflicts

    def test_AIキャラ本名とNPC名衝突を検知(self):
        """AI キャラの本名と NPC 名が衝突するケース。"""
        pcs = [_pc_char(name="アリス", character_name="はる")]
        npcs = {"はる"}
        conflicts = detect_name_conflicts(pcs, npcs)
        assert "はる" in conflicts

    def test_Narratorとの衝突も検知(self):
        pcs = [_pc_char(name="Narrator", character_name="はる")]
        conflicts = detect_name_conflicts(pcs, set())
        assert "Narrator" in conflicts


class TestFormatPcSummary:
    """`format_pc_summary` の表示分岐を検証する。"""

    def test_PCゼロなら空文字列(self):
        assert format_pc_summary([]) == ""

    def test_AIキャラ枠も均一にPC表示_中の人ラベルなし(self):
        """AIキャラPCも「@名前 ← PC」で均一表示し、中の人ラベルや本名を出さないこと。

        中の人（人間/AI）を区別しない方針に伴い、旧「別のAIキャラが演じる」ラベルは廃止。
        AI キャラの本名（character_name）も GM へは開示しない。
        """
        pcs = [_pc_char(name="アリス", character_name="はる", description="剣士")]
        text = format_pc_summary(pcs)
        assert "@アリス" in text
        assert "別のAIキャラが演じる" not in text
        assert "ユーザが演じる" not in text
        # AI キャラの本名は GM に開示しない
        assert "はる" not in text
        assert "代弁" in text

    def test_ユーザ枠も均一にPC表示_中の人ラベルなし(self):
        """ユーザPCも他PCと同じ「@名前 ← PC」で表示し、ユーザ専用ラベルを出さないこと。"""
        pcs = [_pc_user(name="主人公", description="新米冒険者")]
        text = format_pc_summary(pcs)
        assert "@主人公" in text
        assert "ユーザが演じる" not in text
        assert "別のAIキャラが演じる" not in text

    def test_descriptionが80文字超で省略される(self):
        long_desc = "あ" * 200
        pcs = [_pc_char(name="アリス", description=long_desc)]
        text = format_pc_summary(pcs)
        assert "…" in text


class TestFindLastRoutingMention:
    """`find_last_routing_mention` の話者ルーティング分類を検証する。"""

    def setup_method(self):
        self.pcs = [
            _pc_char(name="アリス", character_name="はる"),
            _pc_user(name="主人公"),
        ]
        self.npcs = {"レイカ", "ガイ"}

    def test_メンション無しはnone(self):
        kind, target = find_last_routing_mention("地の文だけ。", self.pcs, self.npcs)
        assert kind == "none"
        assert target is None

    def test_PC枠名メンションはpc(self):
        kind, target = find_last_routing_mention("@アリス どう?", self.pcs, self.npcs)
        assert kind == "pc"
        assert target == "アリス"

    def test_本名メンションは枠名にマップされる(self):
        kind, target = find_last_routing_mention("@はる どう?", self.pcs, self.npcs)
        assert kind == "pc"
        assert target == "アリス"

    def test_ユーザ枠メンションもpc(self):
        kind, target = find_last_routing_mention("@主人公 行こう", self.pcs, self.npcs)
        assert kind == "pc"
        assert target == "主人公"

    def test_GMやNarratorはgm(self):
        kind, _ = find_last_routing_mention("@GM どうする?", self.pcs, self.npcs)
        assert kind == "gm"
        kind2, _ = find_last_routing_mention("@Narrator お願い", self.pcs, self.npcs)
        assert kind2 == "gm"

    def test_NPC名はgm(self):
        kind, _ = find_last_routing_mention("@レイカ こんにちは", self.pcs, self.npcs)
        assert kind == "gm"

    def test_ALLはall(self):
        kind, target = find_last_routing_mention("@ALL みんなどう?", self.pcs, self.npcs)
        assert kind == "all"
        assert target is None

    def test_末尾の最後の有効メンションが優先される(self):
        """1 発話に複数のメンションがあるとき、末尾側が次話者を決める。"""
        text = "@アリス: そうだな。@GM 次にどうする?"
        kind, _ = find_last_routing_mention(text, self.pcs, self.npcs)
        assert kind == "gm"
