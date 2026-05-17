"""シナリオチャット用 @名前: ストリーミングパーサのテスト。

ScenarioChatParser の feed() / flush() 動作を網羅的に検証する。

主要な検証観点:
    - 既知 NPC（known_npc_names に登録済み）の正常パース
    - 未知 NPC（GM が即興で登場させた話者）の ephemeral 扱い
    - Narrator 行の処理
    - 行頭でない `@名前` は話者切替とみなさない
    - 行頭に `@` がない冒頭の地の文は Narrator にフォールバック吸収
    - GM がユーザ代弁（@user_alias:）した場合の破棄＋警告ログ
    - `@user_alias` の名前が Narrator 本文中に出てきても切替対象外
    - is_speaker_change フラグが話者切替の最初の delta だけ True
    - チャンク境界（@名前: の途中で分割された場合）の正しい再構築
    - 同一話者の長文ストリーミング（複数 delta に分散）
    - flush() の挙動: 残バッファ・未完`@`を本文として吐き出す
    - 形式破りパターン: JSON 風入力、改行なし末尾、`@:` 空名前など

これらのケースは P1 で確定した「形式破りは未知 NPC か Narrator フォールバック」
ポリシーを反映している。
"""

import pytest

from backend.services.scenario_chat.parser import ScenarioChatParser, UtteranceDelta


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


def _all_text_by_speaker(deltas: list[UtteranceDelta]) -> dict[str, str]:
    """delta 列から話者名 → 連結本文の辞書を作る（テストアサート用）。

    同一話者が複数回登場すれば本文を改行で連結する。
    """
    out: dict[str, list[str]] = {}
    for d in deltas:
        key = d.speaker_name
        if d.is_speaker_change and key in out:
            out[key].append("|||")  # 切替境界マーカー
        out.setdefault(key, []).append(d.content_delta)
    return {k: "".join(v) for k, v in out.items()}


def _ordered_speakers(deltas: list[UtteranceDelta]) -> list[str]:
    """speaker_change=True の話者だけを出現順に並べる。"""
    return [d.speaker_name for d in deltas if d.is_speaker_change]


# ─── 単純なケース ─────────────────────────────────────────────────────────────


class TestBasicParsing:
    """既知 NPC・Narrator・ユーザ代弁を含む基本パターンを検証する。

    最も典型的な 1 ターン分の出力をワンショットで feed して flush する。
    """

    def test_single_known_npc(self):
        """既知 NPC のみの 1 ブロックを正しく抽出すること。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-1"})
        deltas = parser.feed("@レイカ: ……来たんだ\n")
        deltas += parser.flush()
        text_by = _all_text_by_speaker(deltas)
        assert "レイカ" in text_by
        assert "……来たんだ" in text_by["レイカ"]

    def test_known_npc_carries_id(self):
        """既知 NPC delta の speaker_id が登録 ID になっていること。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-XYZ"})
        deltas = parser.feed("@レイカ: hi\n")
        npc_deltas = [d for d in deltas if d.speaker_type == "npc"]
        assert all(d.speaker_id == "id-XYZ" for d in npc_deltas)
        assert all(d.is_known for d in npc_deltas)

    def test_unknown_npc_is_ephemeral(self):
        """未知の話者は speaker_id=None, is_known=False で通すこと。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-1"})
        deltas = parser.feed("@モブ店主: お買い物ですかい？\n")
        deltas += parser.flush()
        npc_deltas = [d for d in deltas if d.speaker_type == "npc"]
        assert npc_deltas, "未知 NPC でも npc delta を返すこと"
        assert all(d.speaker_id is None for d in npc_deltas)
        assert all(not d.is_known for d in npc_deltas)
        assert all(d.speaker_name == "モブ店主" for d in npc_deltas)

    def test_narrator(self):
        """Narrator は speaker_type=narrator, is_known=True で通すこと。"""
        parser = ScenarioChatParser()
        deltas = parser.feed("@Narrator: 雨音が窓を叩いている\n")
        deltas += parser.flush()
        nar_deltas = [d for d in deltas if d.speaker_type == "narrator"]
        assert nar_deltas
        assert all(d.is_known for d in nar_deltas)
        text = "".join(d.content_delta for d in nar_deltas)
        assert "雨音" in text

    def test_multiple_speakers_order(self):
        """複数話者が交互に登場するときの順序が正しいこと。"""
        parser = ScenarioChatParser(
            known_npc_names={"レイカ": "id-r", "トウコ": "id-t"}
        )
        text = (
            "@Narrator: 雨\n"
            "@レイカ: ……来たんだ\n"
            "@トウコ: 遅いですよ、ご主人様\n"
        )
        deltas = parser.feed(text)
        deltas += parser.flush()
        assert _ordered_speakers(deltas) == ["Narrator", "レイカ", "トウコ"]


# ─── ユーザ代弁の破棄 ─────────────────────────────────────────────────────────


class TestUserAliasSuppression:
    """GM がユーザを代弁したブロックが破棄され、警告が記録されることを検証する。

    @user_alias: で始まる行は完全に捨てる。次の真の話者切替まで suppress 状態を維持する。
    """

    def test_user_alias_block_dropped(self):
        """`@<user_alias>:` ブロックは出力に含まれないこと。"""
        parser = ScenarioChatParser(user_alias="プレイヤー")
        deltas = parser.feed("@プレイヤー: 勝手な発話\n@Narrator: でも雨だ\n")
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        assert "プレイヤー" not in speakers
        # Narrator は通る
        narrator_text = "".join(
            d.content_delta for d in deltas if d.speaker_type == "narrator"
        )
        assert "雨" in narrator_text

    def test_user_alias_warning_logged(self):
        """捨てられたユーザ代弁ブロックは warnings にログされること。"""
        parser = ScenarioChatParser(user_alias="プレイヤー")
        parser.feed("@プレイヤー: 勝手\n@Narrator: ok\n")
        parser.flush()
        assert any("user_alias" in w for w in parser.warnings)

    def test_user_alias_in_narrator_text_kept(self):
        """Narrator の地の文中に `@プレイヤー` が現れても捨てないこと。"""
        parser = ScenarioChatParser(user_alias="プレイヤー")
        text = "@Narrator: その時 @プレイヤー は眉をしかめた\n"
        deltas = parser.feed(text)
        deltas += parser.flush()
        narrator_text = "".join(
            d.content_delta for d in deltas if d.speaker_type == "narrator"
        )
        assert "@プレイヤー" in narrator_text
        # speaker_change が Narrator のみで、プレイヤーは出ない
        speakers = _ordered_speakers(deltas)
        assert speakers == ["Narrator"]

    def test_user_alias_followed_by_real_speaker_resumes(self):
        """suppress 後の真の話者切替で出力が復活すること。"""
        parser = ScenarioChatParser(
            user_alias="プレイヤー", known_npc_names={"レイカ": "id-r"}
        )
        deltas = parser.feed(
            "@プレイヤー: 捨てられる\n"
            "@プレイヤー: これも捨てられる\n"
            "@レイカ: 通る\n"
        )
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        assert "プレイヤー" not in speakers
        assert "レイカ" in speakers


# ─── Narrator フォールバック ─────────────────────────────────────────────────


class TestNarratorFallback:
    """話者宣言なしのテキスト・形式破り入力が Narrator に吸収されることを検証する。"""

    def test_leading_text_without_at_goes_to_narrator(self):
        """先頭が `@名前:` でない地の文は Narrator として通ること。"""
        parser = ScenarioChatParser()
        deltas = parser.feed("いきなり地の文から始まる\n@Narrator: ok\n")
        deltas += parser.flush()
        narrator_text = "".join(
            d.content_delta for d in deltas if d.speaker_type == "narrator"
        )
        assert "いきなり地の文" in narrator_text

    def test_json_response_absorbed_to_narrator(self):
        """JSON 風の出力でも Narrator として通ること（形式破りでも止まらない）。"""
        parser = ScenarioChatParser()
        deltas = parser.feed('{"speaker":"レイカ","content":"hi"}\n')
        deltas += parser.flush()
        # 何らかの Narrator 出力があれば OK
        narrator_text = "".join(
            d.content_delta for d in deltas if d.speaker_type == "narrator"
        )
        assert "レイカ" in narrator_text or "hi" in narrator_text or "{" in narrator_text

    def test_empty_speaker_name_is_narrator(self):
        """`@:` のような空名前は Narrator フォールバック。"""
        parser = ScenarioChatParser()
        deltas = parser.feed("@: 空名前のはず\n")
        deltas += parser.flush()
        assert all(d.speaker_type == "narrator" for d in deltas)


# ─── 行頭でない @ は無視 ─────────────────────────────────────────────────────


class TestInlineAtIgnored:
    """行の途中に登場する `@名前` は話者切替に使わないことを検証する。"""

    def test_at_mid_line(self):
        """Narrator の地の文に `@レイカ` が含まれても話者は切替らない。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-r"})
        deltas = parser.feed("@Narrator: そこには @レイカ が立っていた\n")
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        assert speakers == ["Narrator"]
        narrator_text = "".join(
            d.content_delta for d in deltas if d.speaker_type == "narrator"
        )
        assert "@レイカ" in narrator_text


class TestBlankLinesBetweenBlocks:
    """LLM が話者ブロック間に空行を挟むケースで、誤って Narrator にフォールバック切替
    しないことを検証する。

    実際の Claude などの LLM は `@A: ...\\n\\n@B: ...` のように空行で
    話者ブロックを区切る出力をすることが多い。空行で話者切替が発生すると
    UI 上に空の Narrator 吹き出しが挟まり、後続の speaker_end の遅延発火と
    合わさってさらに表示が崩れる原因になる。
    """

    def test_blank_line_between_npc_blocks(self):
        """`@A: ...\\n\\n@B: ...` で間に話者切替が発生しないこと。"""
        parser = ScenarioChatParser(
            known_npc_names={"レイカ": "id-r", "トウコ": "id-t"}
        )
        text = "@レイカ: 一行目\n\n@トウコ: 二行目\n"
        deltas = parser.feed(text)
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        # 切替は 2 回だけ。空Narrator吹き出しが挟まらない。
        assert speakers == ["レイカ", "トウコ"]

    def test_multiple_blank_lines(self):
        """空行が複数連続しても話者切替が発生しないこと。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-r"})
        text = "@Narrator: 雨\n\n\n\n@レイカ: 来た\n"
        deltas = parser.feed(text)
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        assert speakers == ["Narrator", "レイカ"]

    def test_realistic_llm_output(self):
        """実際の LLM 出力に近い 3 話者ブロック + 空行のケース。"""
        parser = ScenarioChatParser(
            known_npc_names={"アリサ": "id-a", "ユイナ": "id-y"}
        )
        text = (
            "@Narrator: 見慣れた宿の朝。窓から差し込む光が埃をきらきらと舞わせている。\n"
            "\n"
            "@アリサ: ……なんだ急に。朝から元気だな、もわ。\n"
            "\n"
            "@ユイナ: いえーい、って……ふふ。何があったの、今日。\n"
        )
        deltas = parser.feed(text)
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        assert speakers == ["Narrator", "アリサ", "ユイナ"]
        # 各話者の本文が他話者に混ざらないこと
        text_by = _all_text_by_speaker(deltas)
        assert "見慣れた" in text_by["Narrator"]
        assert "見慣れた" not in text_by["ユイナ"]
        assert "なんだ急に" in text_by["アリサ"]
        assert "いえーい" in text_by["ユイナ"]


# ─── ストリーミング境界 ──────────────────────────────────────────────────────


class TestChunkBoundaries:
    """チャンクが任意の位置で分割されても正しく再構築できることを検証する。

    LLM ストリームは句切れの良い場所で来るとは限らない。
    `@名前` の途中、`:` の前、本文の途中、改行の前後、いずれで分割されても
    結果が一致することを保証する。
    """

    def test_split_inside_speaker_header(self):
        """`@レ` `イカ:` のように話者ヘッダ途中で分割しても認識すること。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-r"})
        deltas = []
        deltas += parser.feed("@レ")
        deltas += parser.feed("イカ:")
        deltas += parser.feed(" こんにちは\n")
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        assert "レイカ" in speakers
        text = "".join(d.content_delta for d in deltas if d.speaker_name == "レイカ")
        assert "こんにちは" in text

    def test_split_between_at_and_name(self):
        """`@` だけ来てから次に `レイカ:` が来てもOKなこと。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-r"})
        deltas = []
        deltas += parser.feed("@")
        deltas += parser.feed("レイカ: hi\n")
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        assert "レイカ" in speakers

    def test_split_before_colon(self):
        """`@レイカ` 途中で止まって次に `: hi\\n` が来てもOKなこと。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-r"})
        deltas = []
        deltas += parser.feed("@レイカ")
        deltas += parser.feed(": hi\n")
        deltas += parser.flush()
        speakers = _ordered_speakers(deltas)
        assert "レイカ" in speakers

    def test_split_mid_body(self):
        """本文途中で分割しても、同じ話者の delta が複数届くこと。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-r"})
        deltas = []
        deltas += parser.feed("@レイカ: こん")
        deltas += parser.feed("にち")
        deltas += parser.feed("は\n")
        deltas += parser.flush()
        # is_speaker_change=True は最初の delta のみ
        change_count = sum(1 for d in deltas if d.is_speaker_change)
        assert change_count == 1
        text = "".join(d.content_delta for d in deltas if d.speaker_name == "レイカ")
        assert text.replace("\n", "") == "こんにちは"

    def test_char_by_char_stream(self):
        """1 文字ずつ feed しても結果が壊れないこと。"""
        parser = ScenarioChatParser(
            known_npc_names={"レイカ": "id-r", "トウコ": "id-t"}
        )
        full = "@Narrator: 雨\n@レイカ: 来た\n@トウコ: 遅い\n"
        deltas = []
        for c in full:
            deltas += parser.feed(c)
        deltas += parser.flush()
        assert _ordered_speakers(deltas) == ["Narrator", "レイカ", "トウコ"]


# ─── is_speaker_change セマンティクス ────────────────────────────────────────


class TestSpeakerChangeFlag:
    """is_speaker_change が話者切替の最初の delta だけ True であることを検証する。

    SSE で speaker_start イベント発火に直結するフラグなので、誤発火・発火漏れがあると
    UI 上で吹き出しが重複・欠落する。
    """

    def test_first_delta_is_change(self):
        """最初の話者の最初の delta で is_speaker_change=True。"""
        parser = ScenarioChatParser()
        deltas = parser.feed("@Narrator: 一発目\n")
        deltas += parser.flush()
        assert deltas[0].is_speaker_change is True

    def test_subsequent_deltas_not_change(self):
        """同じ話者の 2 個目以降の delta は is_speaker_change=False。"""
        parser = ScenarioChatParser()
        deltas = []
        deltas += parser.feed("@Narrator: aaa")
        deltas += parser.feed(" bbb")
        deltas += parser.feed(" ccc\n")
        deltas += parser.flush()
        narrator_deltas = [d for d in deltas if d.speaker_type == "narrator"]
        change_flags = [d.is_speaker_change for d in narrator_deltas]
        # 最初が True、以降は False
        assert change_flags[0] is True
        assert all(f is False for f in change_flags[1:])

    def test_switch_back_to_same_speaker_still_marks_change(self):
        """A→B→A と切り替わったとき、2 度目の A も is_speaker_change=True。"""
        parser = ScenarioChatParser(
            known_npc_names={"レイカ": "id-r", "トウコ": "id-t"}
        )
        deltas = parser.feed(
            "@レイカ: 一回目\n@トウコ: 割り込み\n@レイカ: 二回目\n"
        )
        deltas += parser.flush()
        layer_count = sum(
            1 for d in deltas if d.is_speaker_change and d.speaker_name == "レイカ"
        )
        assert layer_count == 2


# ─── flush 挙動 ───────────────────────────────────────────────────────────────


class TestFlush:
    """flush() がストリーム終端で未確定バッファを正しく吐くことを検証する。"""

    def test_empty_flush(self):
        """空のままflushすると空リストを返すこと。"""
        parser = ScenarioChatParser()
        assert parser.flush() == []

    def test_flush_after_unterminated_speaker_header(self):
        """`:` が来ないまま終了した場合、本文として吐くこと（エラーにしない）。"""
        parser = ScenarioChatParser()
        parser.feed("@これは未完")
        deltas = parser.flush()
        # 何らかの delta が出る（Narrator フォールバック）
        assert len(deltas) >= 1
        text = "".join(d.content_delta for d in deltas)
        assert "@これは未完" in text

    def test_flush_after_partial_body(self):
        """改行なしで終わった本文も最終的に出力されること（feed と flush の合算）。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-r"})
        # 本文部分は改行を待たずに feed 中に逐次 emit される。
        # 改行なしで終わった場合も flush で残りが吐かれる。
        deltas = parser.feed("@レイカ: 改行なしで終わる")
        deltas += parser.flush()
        text = "".join(d.content_delta for d in deltas if d.speaker_name == "レイカ")
        assert "改行なし" in text


# ─── 状態保持 ───────────────────────────────────────────────────────────────


class TestStateAcrossFeed:
    """複数回の feed をまたいだ話者状態保持を検証する。

    実運用ではチャンクが断片的に来るため、現在話者・suppress 状態が
    feed 間で正しく持続するかが重要。
    """

    def test_speaker_persists_across_feeds(self):
        """同じ話者の連続発話は feed 境界でも 1 つの話者として扱われること。"""
        parser = ScenarioChatParser(known_npc_names={"レイカ": "id-r"})
        parser.feed("@レイカ: 最初\n")
        deltas2 = parser.feed("そのまま2行目に続く\n")
        # 2 行目は前の話者(レイカ)が継続するわけではなく、
        # 行頭`@`が無いので Narrator にフォールバックする。
        # ※ ただし「直前の話者」を引き継ぐ仕様もありうる。
        # 本実装では行頭 `@名前:` が無ければ Narrator として扱う設計。
        # ※ ここでは Narrator になることを期待する。
        assert any(d.speaker_type == "narrator" for d in deltas2)

    def test_suppress_persists_until_next_speaker(self):
        """user_alias の suppress 状態が次の真の話者切替まで維持されること。"""
        parser = ScenarioChatParser(user_alias="プレイヤー")
        deltas = parser.feed("@プレイヤー: 捨てる1\n")
        deltas += parser.feed("@プレイヤー: 捨てる2\n")
        deltas += parser.feed("@Narrator: 通る\n")
        deltas += parser.flush()
        # プレイヤー由来の発話は一切ない
        assert not any("捨てる" in d.content_delta for d in deltas)
        assert any("通る" in d.content_delta for d in deltas)
