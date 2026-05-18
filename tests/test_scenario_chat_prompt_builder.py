"""シナリオチャット GM システムプロンプト組み立てのテスト。

backend.services.scenario_chat.prompt_builder.build_gm_system_prompt() を検証する。

検証する観点:
    - シナリオ・シーン情報・NPC 一覧が正しくブロック化されて含まれること
    - 空フィールドが省略されること（ブロックタイトルが消える）
    - 既知の話者リストにユーザ alias・Narrator・全 NPC が並ぶこと
    - 出力規則・NPC 詳細・履歴・プレイヤー発話の各ブロックが含まれること
    - CLAUDE.md 準拠: 「Assistant」「AI」表現を含まないこと
    - narrator_name のカスタマイズが反映されること

プロンプトは LLM 任せの自由テキストなので、厳密な文字列マッチは避け
「キーフレーズ含有」を中心に検証する。
"""

from dataclasses import dataclass, field
from typing import Optional

from backend.services.scenario_chat.prompt_builder import build_gm_system_prompt


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


@dataclass
class FakeSession:
    """ZetaSession 風のダミーオブジェクト。"""

    user_alias: str = "プレイヤー"
    scenario: Optional[str] = None


@dataclass
class FakeNpc:
    """ZetaNpc 風のダミーオブジェクト。"""

    name: str
    description: Optional[str] = None
    image_data: Optional[str] = None


# ─── 基本構造 ────────────────────────────────────────────────────────────────


class TestBasicStructure:
    """プロンプトの主要ブロックが正しく含まれることを検証する。

    最低限のセッション情報・NPC 1 体・履歴・プレイヤー発話を渡したとき、
    すべての要素が出力に含まれることを確認する。
    """

    def test_role_line_first(self):
        """役割定義行が冒頭に出ること。

        役割定義は「物語の進行役・語り手」として定義される
        （旧「TRPGのゲームマスター(GM)」表現は GM ムーブ誘発のため廃止済み）。
        """
        session = FakeSession()
        out = build_gm_system_prompt(session, npcs=[], history_text="")
        # 物語の語り手としての役割定義が含まれる
        assert "語り手" in out or "物語" in out
        # 冒頭付近に出ること（ベスト・エフォート）
        assert out.find("語り手") < 100

    def test_user_alias_shown_in_known_speakers(self):
        """`@{user_alias}` が既知の話者リストに含まれること。"""
        session = FakeSession(user_alias="勇者")
        out = build_gm_system_prompt(session, npcs=[], history_text="")
        assert "@勇者" in out
        # 主役マーク
        assert "プレイヤー" in out or "主役" in out

    def test_narrator_shown_in_known_speakers(self):
        """`@Narrator` が既知の話者リストに含まれること。"""
        session = FakeSession()
        out = build_gm_system_prompt(session, npcs=[], history_text="")
        assert "@Narrator" in out

    def test_custom_narrator_name(self):
        """narrator_name 引数を変えれば反映されること。"""
        session = FakeSession()
        out = build_gm_system_prompt(
            session, npcs=[], history_text="", narrator_name="語り部"
        )
        assert "@語り部" in out
        assert "@Narrator" not in out

    def test_npcs_listed_in_known_speakers(self):
        """登録 NPC 全員が `@<名前>` で既知の話者リストに並ぶこと。"""
        session = FakeSession()
        npcs = [
            FakeNpc(name="レイカ", description="赤髪の魔法使い"),
            FakeNpc(name="トウコ", description="無口な少女"),
        ]
        out = build_gm_system_prompt(session, npcs=npcs, history_text="")
        assert "@レイカ" in out
        assert "@トウコ" in out

    def test_npc_descriptions_in_detail_block(self):
        """NPC 詳細ブロックに description（人物像 + 口調を含む自由記述）が含まれること。"""
        session = FakeSession()
        npcs = [FakeNpc(name="レイカ", description="赤髪の魔法使い。ぶっきらぼうな口調。")]
        out = build_gm_system_prompt(session, npcs=npcs, history_text="")
        assert "赤髪の魔法使い" in out
        assert "ぶっきらぼう" in out

    def test_scenario_block(self):
        """scenario 内容が含まれること。"""
        session = FakeSession(scenario="魔導書を探す旅")
        out = build_gm_system_prompt(session, npcs=[], history_text="")
        assert "魔導書を探す旅" in out

    def test_history_text_included(self):
        """history_text が直近の流れブロックとして含まれること。"""
        history = "@勇者: やぁ\n\n@Narrator: 雨"
        out = build_gm_system_prompt(
            FakeSession(user_alias="勇者"), npcs=[], history_text=history
        )
        assert history in out

    def test_user_message_block(self):
        """user_message を渡したらプレイヤー発話ブロックが出ること。

        履歴と同じ `@名前: 本文` 規約で出力される（旧 SGML 形式から移行）。
        """
        session = FakeSession(user_alias="勇者")
        out = build_gm_system_prompt(
            session, npcs=[], history_text="", user_message="どうも"
        )
        assert "@勇者:\nどうも" in out
        assert "プレイヤーの発話" in out

    def test_user_message_omitted_when_none(self):
        """user_message=None ならプレイヤー発話ブロックは出ないこと。

        出力規則ブロック内の文言に「プレイヤーの発話」が含まれているため、
        最終ブロックの `@勇者: …` 発話行の存在で判定する。
        """
        out = build_gm_system_prompt(
            FakeSession(user_alias="勇者"), npcs=[], history_text="", user_message=None
        )
        assert "@勇者:" not in out


# ─── 出力規則ブロック ─────────────────────────────────────────────────────────


class TestOutputRules:
    """出力規則ブロックの内容が CLAUDE.md 合意通りであることを検証する。"""

    def test_at_format_rule(self):
        """`@名前:` フォーマット強制が明記されていること。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        assert "@名前" in out or "@" in out
        assert "形式" in out or "フォーマット" in out or "形" in out

    def test_user_alias_no_speak_rule(self):
        """ユーザ代弁禁止ルールが明記されていること。"""
        session = FakeSession(user_alias="勇者")
        out = build_gm_system_prompt(session, npcs=[], history_text="")
        # 「@勇者 の発話は絶対に書かない」相当の警告が含まれる
        assert "勇者" in out
        assert "代弁" in out or "絶対" in out or "領分" in out

    def test_unknown_npc_allowed_rule(self):
        """未知 NPC を即興で生やしてよい旨が明記されていること。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        # 「新しい NPC を登場させてよい」「モブ」「乱入者」など
        assert "新しい" in out or "モブ" in out or "登場" in out

    def test_markdown_json_banned(self):
        """markdown / JSON 禁止が明記されていること。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        assert "markdown" in out.lower() or "ﾏｰｸﾀﾞｳﾝ" in out
        assert "json" in out.lower() or "ｼﾞｪｲｿﾝ" in out

    def test_silence_ok_rule(self):
        """全員毎ターン発話する必要はない（沈黙OK）旨が含まれること。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        assert "沈黙" in out or "毎ターン" in out

    def test_asterisk_action_rule(self):
        """`*xxx*` を行動描写として扱うルールが明記されていること。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        # 「*手を握る*」のような記法説明と、プレイヤー側の解釈ルール両方を確認
        assert "*" in out
        assert "行動" in out or "描写" in out or "仕草" in out


# ─── 空フィールドの省略 ─────────────────────────────────────────────────────


class TestEmptyFieldOmission:
    """空のフィールドはブロックタイトルごと省略されること。"""

    def test_empty_scenario_omitted(self):
        """scenario が空ならその見出しが出ないこと。"""
        out = build_gm_system_prompt(FakeSession(scenario=""), npcs=[], history_text="")
        assert "# 世界・シナリオ" not in out

    def test_no_npcs_no_detail_block(self):
        """NPC が 0 件なら NPC詳細ブロックの見出しが出ないこと。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        assert "# NPC詳細" not in out

    def test_empty_history_no_block(self):
        """history_text が空なら「直近の流れ」ブロックが出ないこと。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        assert "# 直近の流れ" not in out


# ─── CLAUDE.md 命名規則準拠 ───────────────────────────────────────────────────


class TestNamingCompliance:
    """CLAUDE.md の Assistant 問題に準拠しているか検証する。

    システムプロンプト内でキャラを Assistant や AI と呼んでいないこと。
    """

    def test_no_assistant_word(self):
        """`assistant` という単語を含まないこと（API role の話ではなくキャラ呼称）。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        assert "Assistant" not in out
        assert "assistant" not in out

    def test_no_ai_self_referential(self):
        """「あなたはAI」「あなたはアシスタント」など人格否定的表現を含まないこと。

        ※ GM・語り手は役割名なので OK、AI/Assistant の自己定義は NG。
        """
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        # 「あなたはAI」のような構造
        assert "あなたはAI" not in out
        assert "あなたはアシスタント" not in out


# ─── あらすじブロック注入（記憶捏造対策） ─────────────────────────────────────


class TestSynopsisBlocks:
    """`synopsis_auto` / `synopsis_manual` ブロックの注入を検証する。

    長セッションでスライディングウィンドウから外れた古い経緯を、
    自動要約 + 手動補足の 2 ブロックで GM に再注入する仕組み。

    観点:
        - synopsis_auto が空でないとき「これまでのあらすじ」ブロックが含まれる
        - synopsis_manual が空でないとき「プレイヤーからの補足メモ」ブロックが含まれる
        - 両方とも空のときはどちらのブロックも出力されない（=既存挙動の互換）
        - manual が「auto と矛盾する場合はこちらを正とする」旨の文言と共に出る
        - 既知の話者ブロック（旧 Block 3）より前に挿入されること
    """

    def test_no_block_when_both_empty(self):
        """両方とも空なら synopsis ブロックは一切出ないこと（既存テスト互換）。"""
        out = build_gm_system_prompt(FakeSession(), npcs=[], history_text="")
        assert "これまでのあらすじ" not in out
        assert "プレイヤーからの補足メモ" not in out

    def test_auto_only_renders_main_block(self):
        """synopsis_auto があれば「これまでのあらすじ」ブロックが出ること。"""
        out = build_gm_system_prompt(
            FakeSession(),
            npcs=[],
            history_text="",
            synopsis_auto="勇者は森でレイカと出会った。",
        )
        assert "# これまでのあらすじ" in out
        assert "勇者は森でレイカと出会った。" in out
        # manual 側のブロックは出ない
        assert "プレイヤーからの補足メモ" not in out

    def test_manual_only_renders_with_priority_note(self):
        """synopsis_manual があれば補足メモブロックが priority 文言と共に出ること。"""
        out = build_gm_system_prompt(
            FakeSession(),
            npcs=[],
            history_text="",
            synopsis_manual="主人公はレイカと「絶対に裏切らない」と約束した。",
        )
        assert "# プレイヤーからの補足メモ" in out
        assert "主人公はレイカと「絶対に裏切らない」と約束した。" in out
        # auto との矛盾時に manual を優先する旨の文言が含まれること
        assert "矛盾" in out

    def test_both_blocks_render(self):
        """auto / manual 両方あれば両方のブロックが出ること。"""
        out = build_gm_system_prompt(
            FakeSession(),
            npcs=[],
            history_text="",
            synopsis_auto="勇者とレイカは森で出会った。",
            synopsis_manual="主人公の出身は北の村。",
        )
        assert "勇者とレイカは森で出会った。" in out
        assert "主人公の出身は北の村。" in out
        assert "# これまでのあらすじ" in out
        assert "# プレイヤーからの補足メモ" in out

    def test_synopsis_appears_before_known_speakers(self):
        """あらすじブロックが既知の話者リストより前に挿入されること。

        プロンプト上の文脈順として、世界観 → あらすじ → 話者リスト → ... の流れ。
        """
        out = build_gm_system_prompt(
            FakeSession(),
            npcs=[],
            history_text="",
            synopsis_auto="ここに自動あらすじ。",
        )
        pos_synopsis = out.find("# これまでのあらすじ")
        pos_known = out.find("# 既知の話者")
        assert pos_synopsis != -1
        assert pos_known != -1
        assert pos_synopsis < pos_known

    def test_whitespace_only_synopsis_is_ignored(self):
        """空白のみの synopsis は省略されること（既存 _block と同じ挙動）。"""
        out = build_gm_system_prompt(
            FakeSession(),
            npcs=[],
            history_text="",
            synopsis_auto="   \n  \n",
            synopsis_manual="\t",
        )
        assert "これまでのあらすじ" not in out
        assert "プレイヤーからの補足メモ" not in out
