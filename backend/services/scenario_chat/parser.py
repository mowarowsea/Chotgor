"""シナリオチャット用 @名前: ストリーミングパーサ。

LLM (GM) からのストリーム出力を逐次的に消費しつつ、`@話者名:` で始まる
発話ブロックに分解する。tag_parser.py の [TAG:body] 形式とは文法が異なるため
独立実装する。

形式破り対応の方針（CLAUDE.md の合意通り）:
    - 行頭 `@known_npc:` 本文        → known な NPC として確定
    - 行頭 `@unknown_name:` 本文     → 未知 NPC（ephemeral）として通す
    - 行頭 `@Narrator:` 本文         → Narrator として確定
    - 行頭 `@{user_alias}:` 本文     → 「GM がユーザ代弁」のため捨てる（警告）
    - 行頭 `@` で始まらない地の文    → 直前話者（初期値 Narrator）に吸収
    - JSON / markdown / その他不定形 → Narrator フォールバックで吸収
    - 任意の地の文中の `@user_alias` は無関係（Narrator の文中描写として通す）

ストリーミング戦略:
    feed() に逐次チャンクを与え、確定した分だけ UtteranceDelta 列を返す。
    話者切替の瞬間に is_speaker_change=True のデルタを 1 つ発行する。
    flush() でストリーム終了時の残バッファを最終発話としてフラッシュする。

注: `@名前:` の検出は「行頭のみ」とする。Narrator の地の文に "@誰々" が
含まれているケースは話者切替とみなさない。これによりキャラ名と地の文の混在に強い。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UtteranceDelta:
    """ストリーミング中の発話差分。

    Attributes:
        speaker_type: "narrator" | "npc"
        speaker_id: known NPC のみ NPC.id が入る。Narrator・未知 NPC は None。
        speaker_name: 表示・履歴整形用の話者名。@タグから抽出する。
        content_delta: この差分で追加されたテキスト断片。
        is_speaker_change: True の場合、この差分は新しい話者の最初の差分。
        is_known: known NPC か Narrator なら True。未知 NPC なら False。
    """

    speaker_type: str
    speaker_id: Optional[str]
    speaker_name: str
    content_delta: str
    is_speaker_change: bool
    is_known: bool


class ScenarioChatParser:
    """GM 出力の `@名前:` を行頭ベースで切り出すストリーミングパーサ。

    使い方:
        parser = ScenarioChatParser(
            known_npc_names={"レイカ": "npc-uuid-1", "トウコ": "npc-uuid-2"},
            user_alias="プレイヤー",
        )
        for chunk in stream:
            for delta in parser.feed(chunk):
                ...
        for delta in parser.flush():
            ...

    フィールド is_speaker_change の使い方:
        - True: SSE で `speaker_start` イベントを発火する
        - False: 同じ話者の content_delta として継ぎ足す
        - 話者の発話末尾は次の話者の最初の delta か flush() で判別する
    """

    # NPC 名として許容する文字（行頭 @<...>: パターン）。
    # 日本語・英数字・空白・記号など、`:` 以外の任意文字を許容する。
    # ※ `:` は半角コロンのみ識別子終端と扱う（全角コロンは話者名内に許す）。

    def __init__(
        self,
        known_npc_names: Optional[dict[str, str]] = None,
        user_alias: str = "ユーザ",
        narrator_name: str = "Narrator",
    ) -> None:
        """パーサを初期化する。

        Args:
            known_npc_names: {NPC名: NPC.id} の辞書。known と未知の判定に使う。
            user_alias: ユーザの @タグ名。これに一致する話者ブロックは捨てる。
            narrator_name: Narrator のタグ名。デフォルト "Narrator"。
        """
        self._known: dict[str, str] = dict(known_npc_names or {})
        self._user_alias = user_alias
        self._narrator_name = narrator_name

        # 内部バッファ。直前の改行以降のテキストを蓄える。
        self._buffer: str = ""
        # 現在の話者状態。初期値は Narrator（行頭`@`なしのテキストは Narrator に吸収）。
        self._cur_type: str = "narrator"
        self._cur_id: Optional[str] = None
        self._cur_name: str = self._narrator_name
        self._cur_is_known: bool = True
        # ユーザ代弁ブロック中は出力を捨てる。
        self._suppress: bool = False
        # 1 度でも発話を出したか（最初の発話で is_speaker_change=True を立てる用）。
        self._emitted_for_current_speaker: bool = False
        # 警告ログ（捨てたユーザ代弁ブロックの件数など）。テストやデバッグで参照する。
        self.warnings: list[str] = []
        # 直前 emit が改行で終わったか（行頭 @ 検出のため必要）。
        # 初期状態は「直前が行頭」（バッファ先頭は行頭扱い）。
        self._at_line_start: bool = True
        # 話者切替直後の単一スペースを除去するフラグ（`@名前: 本文` の慣用対応）。
        # チャンク境界をまたいでも 1 文字目のスペースを 1 度だけ除去できる。
        self._strip_next_space: bool = False
        # `@名前:\n本文` 形式では、次の話者タグまで非タグ行を同じ話者の本文として扱う。
        self._block_mode: bool = False

    # --- パブリックAPI ---

    def feed(self, chunk: str) -> list[UtteranceDelta]:
        """ストリームチャンクを投入し、確定した UtteranceDelta 列を返す。

        改行が来ない限り次の話者切替を確定できないため、改行までは
        バッファに残してしまう実装もありえるが、ここでは「現在の話者の本文」は
        改行を待たずに逐次フラッシュする。`@` 文字が行頭に来た瞬間だけ
        話者識別子終端の `:` を待ってバッファに留める。
        """
        if not chunk:
            return []
        self._buffer += chunk
        return self._drain()

    def flush(self) -> list[UtteranceDelta]:
        """ストリーム終了時にバッファに残ったテキストを最終発話として返す。

        確定できない `@...` で終わった場合（`:` が来なかった等）、
        その文字列も現在話者の本文として吐き出す。これは GM が形式破りで
        終わったケースをカバーする保険。
        """
        deltas: list[UtteranceDelta] = []
        if not self._buffer:
            return deltas
        # 残バッファに完全な行頭@パターンが含まれる可能性がまだあるため、
        # 最後に 1 度 _drain() を実行してから残りを「現在話者の本文」として吐く。
        deltas.extend(self._drain(eof=True))
        if self._buffer:
            # `:` を待ち続けた未完バッファ等。現在話者の本文として吐く。
            emitted = self._emit_text(self._buffer)
            if emitted is not None:
                deltas.append(emitted)
            self._buffer = ""
        return deltas

    # --- 内部 ---

    def _drain(self, eof: bool = False) -> list[UtteranceDelta]:
        """バッファをスキャンして発話 delta 列を出力する。

        ロジック:
            1. バッファ先頭が「行頭 `@`」位置にある場合、`:` を探す
               - `:` が見つかれば話者切替確定（または user_alias で suppress）
               - `:` が見つからず eof=False ならバッファに留めて待機
               - `:` が見つからず eof=True なら現在話者の本文として扱う
            2. それ以外は次の改行までを「本文」として現在話者にエミット
               - 改行を含めて出力した直後、次は行頭扱いになる
        """
        deltas: list[UtteranceDelta] = []

        while self._buffer:
            if self._at_line_start and self._buffer.startswith("@"):
                # `:` を探す（同一行内）
                line_end = self._buffer.find("\n")
                search_end = line_end if line_end != -1 else len(self._buffer)
                colon_pos = self._buffer.find(":", 1, search_end)

                if colon_pos == -1:
                    # 同一行内に `:` がない。
                    # - 改行があるなら：これは話者宣言ではない。@... 文字列として本文扱い。
                    # - 改行がないかつ eof：本文として扱う（保険）。
                    # - 改行がないかつ eof=False：次のチャンクを待つ。
                    if line_end != -1:
                        # 行頭 `@` だが `:` がない → 単なる本文。
                        text = self._buffer[: line_end + 1]
                        self._buffer = self._buffer[line_end + 1 :]
                        emitted = self._emit_text(text)
                        if emitted is not None:
                            deltas.append(emitted)
                        # 次は新しい行頭
                        self._at_line_start = True
                        continue
                    elif eof:
                        # 完全に未完。本文として吐く。
                        emitted = self._emit_text(self._buffer)
                        if emitted is not None:
                            deltas.append(emitted)
                        self._buffer = ""
                        break
                    else:
                        # 次チャンク待ち
                        break

                # `:` が見つかった → 話者宣言確定
                if not eof and line_end == -1 and colon_pos == len(self._buffer) - 1:
                    break
                name = self._buffer[1:colon_pos].strip()
                # 話者宣言ヘッダ部 `@名前:` を捨てる
                self._buffer = self._buffer[colon_pos + 1 :]
                self._switch_speaker(name)
                if self._buffer.startswith("\r\n"):
                    self._buffer = self._buffer[2:]
                    self._strip_next_space = False
                    self._at_line_start = True
                    self._block_mode = True
                elif self._buffer.startswith("\n"):
                    self._buffer = self._buffer[1:]
                    self._strip_next_space = False
                    self._at_line_start = True
                    self._block_mode = True
                else:
                    # 次に出る本文の先頭スペース 1 文字を除去するフラグを立てる
                    self._strip_next_space = True
                    # 話者切替後の続きは「本文」として続行（行頭ではない）
                    self._at_line_start = False
                    self._block_mode = False
                continue

            # 行頭で @ で始まらない地の文は Narrator にフォールバック。
            # suppress 中（@user_alias 後）の地の文も Narrator に戻す。
            # ただし「空白/改行のみの行」は話者切替を発生させない
            # （LLM が `@A: ...\n\n@B: ...` のようにブロック間に空行を挟むケース対策）。
            if self._at_line_start and not self._block_mode and (
                self._suppress or self._cur_name != self._narrator_name
            ):
                nl_pos = self._buffer.find("\n")
                end_of_line = nl_pos + 1 if nl_pos != -1 else len(self._buffer)
                if self._buffer[:end_of_line].strip():
                    self._switch_speaker(self._narrator_name)
                # 空行ならフォールバックしない（現在話者のままで通す or 後で @ が来るのを待つ）

            # 通常本文。改行までをまとめて吐く。
            nl_pos = self._buffer.find("\n")
            if nl_pos == -1:
                # 改行なし。
                if eof:
                    text = self._buffer
                    self._buffer = ""
                    emitted = self._emit_text(text)
                    if emitted is not None:
                        deltas.append(emitted)
                    break
                # 全部本文として出して、次のチャンクで行頭判定する手もあるが、
                # 改行が来ないまま次が行頭 `@` で開始される可能性は無いので
                # 現状全部出してしまって安全。
                text = self._buffer
                self._buffer = ""
                emitted = self._emit_text(text)
                if emitted is not None:
                    deltas.append(emitted)
                self._at_line_start = False
                break

            # 改行までを本文として吐く（改行も含めて）
            text = self._buffer[: nl_pos + 1]
            self._buffer = self._buffer[nl_pos + 1 :]
            emitted = self._emit_text(text)
            if emitted is not None:
                deltas.append(emitted)
            self._at_line_start = True

        return deltas

    def _switch_speaker(self, raw_name: str) -> None:
        """話者切替を反映する。user_alias の場合は suppress フラグを立てる。"""
        name = raw_name.strip()
        if not name:
            # `@:` のような不正系。Narrator フォールバック。
            name = self._narrator_name

        if name == self._user_alias:
            # GM がユーザを代弁しようとしている。捨てる。
            self._suppress = True
            self.warnings.append(f"user_alias 代弁ブロックを破棄: '@{name}:'")
            # 話者状態自体は前のまま保持（次の真の話者切替を待つ）。
            self._emitted_for_current_speaker = False
            return

        self._suppress = False
        if name == self._narrator_name:
            self._cur_type = "narrator"
            self._cur_id = None
            self._cur_name = self._narrator_name
            self._cur_is_known = True
        elif name in self._known:
            self._cur_type = "npc"
            self._cur_id = self._known[name]
            self._cur_name = name
            self._cur_is_known = True
        else:
            # 未知 NPC（ephemeral）
            self._cur_type = "npc"
            self._cur_id = None
            self._cur_name = name
            self._cur_is_known = False
        self._emitted_for_current_speaker = False

    def _emit_text(self, text: str) -> Optional[UtteranceDelta]:
        """現在話者として本文 delta を発行する。空文字や suppress 中は None。"""
        if not text:
            return None
        if self._suppress:
            # ユーザ代弁ブロック中は出力を捨てる
            return None
        # 話者切替直後の先頭スペース 1 文字を除去する（`@名前: 本文` の慣用対応）。
        if self._strip_next_space:
            self._strip_next_space = False
            if text.startswith(" "):
                text = text[1:]
                if not text:
                    return None
        is_speaker_change = not self._emitted_for_current_speaker
        self._emitted_for_current_speaker = True
        return UtteranceDelta(
            speaker_type=self._cur_type,
            speaker_id=self._cur_id,
            speaker_name=self._cur_name,
            content_delta=text,
            is_speaker_change=is_speaker_change,
            is_known=self._cur_is_known,
        )
