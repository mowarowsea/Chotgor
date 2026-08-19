"""シナリオ設定テキストの「値タグ」展開のテスト。

対象:
    backend.services.scenario_chat.template_tags
        （expand_value_tags / build_tag_context — 純粋関数）
    その適用経路 — GM システムプロンプト・intro・あらすじ蒸留プロンプト。

背景:
    タグはもともと GM の custom_system_prompt でしか使えなかった。
    シナリオ本文・intro・PC枠/NPC の description でも名前を差し込みたい、という
    要望を受けて「値タグ（名前もの・1 段展開）」と「ブロックタグ（GM プロンプト専用）」
    の二層に分けた。ここではその境界（どこで展開され、どこで展開されないか）を固定する。

検証する観点:
    - 値タグの解決規則（番号は 1 始まり／slot_id 指定／解決不能はそのまま残す）
    - シナリオ本文・NPC description・PC 配役メモが GM プロンプト上で展開されること
    - 履歴（{history_block}）に紛れ込んだ中括弧は展開されないこと（誤爆防止）
    - 設定テキストに書かれたブロックタグは展開されないこと（自己再帰・多重展開の遮断）
    - intro はセッション開始時に展開され、その本文がターンとして保存されること
"""

from dataclasses import dataclass, field

from backend.services.scenario_chat.prompt_builder import build_gm_system_prompt
from backend.services.scenario_chat.synopsis import build_synopsis_system_prompt
from backend.services.scenario_chat.template_tags import (
    build_tag_context,
    expand_value_tags,
)
from backend.services.scenario_chat.turns import seed_intro_turns

from tests._scenario_sqlite_helpers import _make_npc, _make_scenario, _make_session


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


@dataclass
class FakeScenario:
    """Scenario ORM 風のダミー。値タグ解決に必要な属性だけ持つ。"""

    scenario: str = ""
    pc_slots: list = field(default_factory=list)
    custom_system_prompt: str | None = None


@dataclass
class FakeNpc:
    """ScenarioNpc ORM 風のダミー。name / description だけ持つ。"""

    name: str
    description: str | None = None


# ─── 値タグの解決規則 ────────────────────────────────────────────────────────


class TestValueTagResolution:
    """expand_value_tags の解決規則を単体で検証する。

    ここが全経路（GM プロンプト・intro・PC 配役メモ・あらすじ）の共通土台なので、
    番号の基点・slot_id 指定・解決できなかったときの振る舞いをここで固定しておく。
    上位の経路テストは「この関数が効いているか」だけを見る。
    """

    def test_user_alias_and_narrator_name(self):
        """添字なしの 2 タグがそのまま名前へ置き換わること。"""
        ctx = build_tag_context(user_alias="アリス", narrator_name="語り部")
        assert expand_value_tags("{user_alias} と {narrator_name}", ctx) == "アリス と 語り部"

    def test_pc_name_index_is_one_based(self):
        """{pc_name[n]} は 1 始まりで pc_slots の並び順を指すこと。"""
        ctx = build_tag_context(pc_entries=[
            {"slot_id": "pc1", "name": "アリス"},
            {"slot_id": "pc2", "name": "ボブ"},
        ])
        assert expand_value_tags("{pc_name[1]}/{pc_name[2]}", ctx) == "アリス/ボブ"

    def test_pc_name_accepts_slot_id(self):
        """{pc_name[slot_id]} でも指定でき、大文字小文字は区別しないこと。

        番号指定は枠の並べ替え・追加でずれるため、安定キーである slot_id でも
        指せるようにしてある。
        """
        ctx = build_tag_context(pc_entries=[
            {"slot_id": "PC1", "name": "アリス"},
            {"slot_id": "pc2", "name": "ボブ"},
        ])
        assert expand_value_tags("{pc_name[pc1]}/{pc_name[PC2]}", ctx) == "アリス/ボブ"

    def test_npc_name_index(self):
        """{npc_name[n]} は NPC の並び順（作成順）を 1 始まりで指すこと。"""
        ctx = build_tag_context(npcs=[FakeNpc(name="レイカ"), FakeNpc(name="ガロ")])
        assert expand_value_tags("{npc_name[2]}", ctx) == "ガロ"

    def test_unresolved_tags_are_left_as_written(self):
        """解決できないタグは書かれたまま残ること。

        黙って空文字にすると、タイポで名前が抜けた文章が LLM へ渡っても
        誰も気づけない。範囲外の番号・未定義 slot_id・NPC への非数値指定・
        添字なしの {pc_name}・添字付きの {user_alias} をまとめて確認する。
        """
        ctx = build_tag_context(
            pc_entries=[{"slot_id": "pc1", "name": "アリス"}],
            npcs=[FakeNpc(name="レイカ")],
        )
        text = "{pc_name[9]} {pc_name[unknown]} {npc_name[zzz]} {pc_name} {user_alias[1]}"
        assert expand_value_tags(text, ctx) == text

    def test_slot_without_name_is_unresolved_but_keeps_index(self):
        """名前未入力の枠は未解決扱い。ただし後続の番号はずれないこと。"""
        ctx = build_tag_context(pc_entries=[
            {"slot_id": "pc1", "name": ""},
            {"slot_id": "pc2", "name": "ボブ"},
        ])
        assert expand_value_tags("{pc_name[1]}/{pc_name[2]}", ctx) == "{pc_name[1]}/ボブ"

    def test_expansion_is_single_pass(self):
        """展開結果に含まれる中括弧は再展開されないこと（多重展開の遮断）。"""
        ctx = build_tag_context(
            pc_entries=[{"slot_id": "pc1", "name": "{user_alias}"}],
            user_alias="アリス",
        )
        assert expand_value_tags("{pc_name[1]}", ctx) == "{user_alias}"

    def test_accepts_orm_like_entries(self):
        """pc_slots が dict でなく属性持ちオブジェクトでも解決できること。

        生 JSON（Scenario.pc_slots）で渡る経路と、正規化済み PcSlot / PcAssignment で
        渡る経路の両方があるため、両対応であることを担保する。
        """
        ctx = build_tag_context(pc_entries=[FakeNpc(name="アリス")])
        assert expand_value_tags("{pc_name[1]}", ctx) == "アリス"


# ─── GM システムプロンプト経由 ───────────────────────────────────────────────


class TestGmPromptExpansion:
    """設定テキストが GM プロンプトへ注入される時点で展開されることを検証する。

    シナリオ本文・NPC description・PC 配役メモ（pc_summary）は、いずれも
    ブロックタグ経由で GM プロンプトへ差し込まれる。差し込む前に値タグを
    解決しているか、そして「解決してはいけないもの」を巻き込んでいないかを見る。
    """

    def _scenario(self, **kwargs):
        """PC 枠 1 つ（pc1=アリス）を持つダミーシナリオを作る。"""
        base = {"pc_slots": [{"slot_id": "pc1", "name": "アリス"}]}
        base.update(kwargs)
        return FakeScenario(**base)

    def test_scenario_text_tags_expanded(self):
        """シナリオ本文中の値タグが展開されること。"""
        sc = self._scenario(scenario="{pc_name[1]} は {npc_name[1]} の弟子だ。")
        out = build_gm_system_prompt(
            sc, npcs=[FakeNpc(name="レイカ")], history_text="", user_speaker_name="アリス",
        )
        assert "アリス は レイカ の弟子だ。" in out

    def test_npc_description_tags_expanded(self):
        """NPC の description 中の値タグが NPC 詳細ブロックで展開されること。"""
        sc = self._scenario(scenario="世界の説明")
        npc = FakeNpc(name="レイカ", description="{pc_name[1]} を昔から知っている。")
        out = build_gm_system_prompt(
            sc, npcs=[npc], history_text="", user_speaker_name="アリス",
        )
        assert "アリス を昔から知っている。" in out

    def test_pc_summary_tags_expanded(self):
        """PC 配役一覧（pc_summary）中の値タグが展開されること。

        pc_summary は PC 枠の description から組まれるので、この経路が
        「PC の Description でタグが使える」ことの担保になる。
        """
        sc = self._scenario(scenario="世界の説明")
        out = build_gm_system_prompt(
            sc,
            npcs=[FakeNpc(name="レイカ")],
            history_text="",
            user_speaker_name="アリス",
            pc_summary="@アリス ← PC。{npc_name[1]} の弟子。",
        )
        assert "@アリス ← PC。レイカ の弟子。" in out

    def test_history_braces_are_not_expanded(self):
        """履歴に紛れ込んだ中括弧は展開しないこと。

        履歴はユーザ発話と GM 出力そのもの。ここまで展開対象にすると、
        過去の発話が後から書き換わる（＝ログと送信内容が食い違う）。
        """
        sc = self._scenario(scenario="世界の説明")
        out = build_gm_system_prompt(
            sc,
            npcs=[],
            history_text="<アリス>{user_alias} って書いてみる</アリス>",
            user_speaker_name="アリス",
        )
        assert "{user_alias} って書いてみる" in out

    def test_block_tags_in_scenario_text_are_inert(self):
        """設定テキストに書かれたブロックタグは展開されないこと。

        シナリオ本文に {scenario} と書けたら自己再帰、{history_block} と書けたら
        履歴の多重展開になる。値タグだけを設定テキストに開放した設計を固定する。
        """
        sc = self._scenario(scenario="ここに {scenario} と {history_block} と書く")
        out = build_gm_system_prompt(
            sc, npcs=[], history_text="HISTORY_MARKER", user_speaker_name="アリス",
        )
        assert "ここに {scenario} と {history_block} と書く" in out
        # 履歴自体は本来のブロックへ 1 度だけ入る
        assert out.count("HISTORY_MARKER") == 1

    def test_custom_system_prompt_tags_still_work(self):
        """従来どおり custom_system_prompt 側のタグも展開されること（回帰防止）。"""
        sc = self._scenario(
            scenario="世界の説明",
            custom_system_prompt="語り手は{narrator_name}。PC は {pc_name[pc1]}。\n{scenario}",
        )
        out = build_gm_system_prompt(
            sc, npcs=[], history_text="", user_speaker_name="アリス", narrator_name="語り部",
        )
        assert "語り手は語り部。PC は アリス。" in out
        assert "世界の説明" in out


# ─── intro ────────────────────────────────────────────────────────────────────


class TestIntroExpansion:
    """intro が「セッション開始時に 1 度だけ展開されて保存される」ことを検証する。

    intro はプロンプトへ都度注入されるのではなく、ターン列（scenario_turns）へ
    展開・保存される。したがってタグ解決もその 1 回きりで、保存後の本文に
    タグが残っていないことが期待される。
    """

    def test_intro_tags_expanded_into_saved_turns(self, sqlite_store):
        """保存されたターン本文でタグが解決済みであること。"""
        scenario = _make_scenario(
            sqlite_store,
            user_alias=None,
            pc_slots=[{"slot_id": "pc1", "name": "アリス", "description": ""}],
            intro="@Narrator: {pc_name[1]} は {npc_name[1]} の前に立った。",
        )
        _make_npc(sqlite_store, scenario.id, name="レイカ")
        session = _make_session(sqlite_store, scenario.id)

        saved = seed_intro_turns(sqlite_store, session.id, scenario)

        assert saved == 1
        turns = sqlite_store.list_scenario_turns(session.id)
        assert turns[0].content == "アリス は レイカ の前に立った。"

    def test_unresolved_intro_tag_is_left_as_written(self, sqlite_store):
        """解決できないタグは intro でも書かれたまま残ること。"""
        scenario = _make_scenario(
            sqlite_store,
            user_alias=None,
            pc_slots=[{"slot_id": "pc1", "name": "アリス", "description": ""}],
            intro="@Narrator: {pc_name[5]} が現れた。",
        )
        session = _make_session(sqlite_store, scenario.id)

        seed_intro_turns(sqlite_store, session.id, scenario)

        turns = sqlite_store.list_scenario_turns(session.id)
        assert turns[0].content == "{pc_name[5]} が現れた。"


# ─── あらすじ蒸留プロンプト ──────────────────────────────────────────────────


class TestSynopsisExpansion:
    """あらすじ蒸留プロンプトでもシナリオ本文が展開されることを検証する。

    同じシナリオ本文が GM プロンプトと蒸留プロンプトの両方へ載るため、
    片方だけ生タグが残ると「あらすじ側にだけ {pc_name[1]} が漏れる」ことになる。
    """

    def test_scenario_text_expanded(self):
        """蒸留プロンプトの世界・シナリオ節でタグが解決されること。"""
        sc = FakeScenario(
            scenario="{pc_name[1]} と {npc_name[1]} の物語。",
            pc_slots=[{"slot_id": "pc1", "name": "アリス"}],
        )
        out = build_synopsis_system_prompt(
            sc,
            existing_auto="",
            user_speaker_name="アリス",
            npcs=[FakeNpc(name="レイカ")],
        )
        assert "アリス と レイカ の物語。" in out
