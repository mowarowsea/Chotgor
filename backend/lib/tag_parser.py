"""TagParser — ツールタグ [TAG:...] の文字単位パーサー。

LLM応答テキストから [INSCRIBE_MEMORY:...] / [CARVE_NARRATIVE:...] などのツールタグを
正規表現ではなく文字単位でスキャンし、正確に抽出するためのユーティリティ。

特徴:
  - 閉じ括弧 `]` は、本体開始位置以降で**最初に現れる `]`** とする。改行はまたいでよい。
  - バッククォートインラインコード・コードフェンス内はスキャンをスキップする
  - タグ定義は呼び出し側が渡すため、任意のタグ名を自由に追加できる抽象設計
  - タグ名の列挙順を呼び出し側が意識しなくてよいよう、内部で長さ降順ソートする
    (短いタグ名が長いタグ名の接頭辞になっていても、長い方を必ず先に照合できる)

【閉じ括弧を「最初の `]`」に統一した経緯 — 2026-08-04】
かつては `multiline` フラグで挙動を切り替えていた（False=同一行内の最後の `]`、
True=文末までの最初の `]`）。False 側は「本体に `]` を含むタグ」を救うためのもので、
実行時の抽出（anticipator / inscriber / carver）が False、ログ表示だけが True だったため、
**改行を含むタグが本文から切り取られないのに、ログUIの表示だけは出る**という非対称が
生まれていた（うつつの ANTICIPATE_RESPONSE で顕在化）。

「本体に `]` を含む」と「タグの後に普通の本文が続く」はテキスト上で区別できないため、
どちらの規則を採っても一方は壊れる。そこで失敗モードの軽い方へ倒した:
  - 最初の `]`（現行）: 本体が途中で切れ、残骸が本文に**見える形で**残る
  - 最後の `]`（旧 False）: 後続の本文をまるごと飲み込んで**消す**
LLM が本体に `]` を書いた場合の取りこぼしは、形式エラーとして許容する方針。

新規タグを追加するには:
  1. 処理モジュール（inscriber.py 相当）を新規作成し parse_tags() を呼ぶだけでよい
  2. tag_parser.py 本体の変更は不要
"""

from dataclasses import dataclass


@dataclass
class TagMatch:
    """1つのタグマッチ結果。"""

    # タグ名 (例: "INSCRIBE_MEMORY", "CARVE_NARRATIVE", "END_SESSION")
    tag_name: str
    # マッチした元テキスト全体 (例: "[INSCRIBE_MEMORY:contextual|1.0|content]")
    raw: str
    # タグ名コロン以降、末尾 ']' を除いた本体 (例: "contextual|1.0|content")
    # 固定マーカー ([TAG]) の場合は空文字列
    body: str
    # テキスト内の開始インデックス（含む）
    start: int
    # テキスト内の終了インデックス（排他）
    end: int


def _skip_backtick(text: str, pos: int, n: int) -> int:
    """バッククォートコードブロックをスキップし、次のスキャン位置を返す。

    コードフェンス (```) とインラインコード (`) の両方に対応する。

    Args:
        text: スキャン対象テキスト。
        pos: 開きバッククォートの位置。
        n: テキスト長 (= len(text))。

    Returns:
        コードブロック終端の次のインデックス。閉じが見つからない場合は n。
    """
    if text.startswith("```", pos):
        end = text.find("```", pos + 3)
        return (end + 3) if end != -1 else n
    else:
        end = text.find("`", pos + 1)
        return (end + 1) if end != -1 else n


def parse_tags(
    text: str,
    tag_names: list[str],
) -> tuple[str, dict[str, list[TagMatch]]]:
    """テキストからツールタグを文字単位で抽出する。

    文字単位でスキャンし、タグ本体開始以降で最初に現れる ']' を閉じ括弧とする
    （改行をまたぐタグ内容も抽出できる。規則の根拠はモジュール docstring 参照）。
    バッククォートコードブロック内のタグ形式テキストはスキップする。

    タグ名の列挙順は呼び出し側が意識しなくてよい。
    内部で長さ降順ソートするため、プレフィックスが共通するタグ名
    ("SEARCH_RESET" と "SEARCH" など) でも安全に照合できる。

    Args:
        text: パース対象のテキスト。
        tag_names: 認識するタグ名のリスト (例: ["INSCRIBE_MEMORY", "CARVE_NARRATIVE", "END_SESSION"])。
                   "[TAG]" 形式の固定マーカーも "[TAG:...]" 形式のコンテンツタグも両対応。

    Returns:
        tuple:
            clean_text (str): タグをすべて除去したテキスト。前後の空白をトリム済み。
            matches (dict[str, list[TagMatch]]): タグ名 → マッチ結果リスト。
    """
    matches: dict[str, list[TagMatch]] = {name: [] for name in tag_names}
    # i は単調増加するため removed は挿入順でソート済みが保証される
    removed: list[tuple[int, int]] = []

    # プレフィックスが共通するタグ名の誤照合を防ぐため、長さ降順でソートする
    # （短いタグ名が長いタグ名の接頭辞でも、長い方を先に照合できる）
    sorted_names = sorted(tag_names, key=len, reverse=True)

    i = 0
    n = len(text)

    while i < n:
        # バッククォートコードフェンス・インラインコードをスキップ
        if text[i] == "`":
            i = _skip_backtick(text, i, n)
            continue

        if text[i] != "[":
            i += 1
            continue

        # '[' を見つけた。各タグ名と照合する（長さ降順でソート済み）
        matched = False
        for tag_name in sorted_names:
            # 固定マーカー: "[TAGNAME]"
            fixed = f"[{tag_name}]"
            if text[i : i + len(fixed)] == fixed:
                tag_end = i + len(fixed)
                m = TagMatch(tag_name=tag_name, raw=fixed, body="", start=i, end=tag_end)
                matches[tag_name].append(m)
                removed.append((i, tag_end))
                i = tag_end
                matched = True
                break

            # コンテンツタグ: "[TAGNAME:..."
            prefix = f"[{tag_name}:"
            if text[i : i + len(prefix)] == prefix:
                body_start = i + len(prefix)

                # 閉じ括弧は本体開始以降で最初の ']'（改行をまたいでよい）
                j = text.find("]", body_start)

                if j != -1:
                    tag_end = j + 1  # ']' の次
                    raw = text[i:tag_end]
                    body = text[body_start:j]
                    m = TagMatch(tag_name=tag_name, raw=raw, body=body, start=i, end=tag_end)
                    matches[tag_name].append(m)
                    removed.append((i, tag_end))
                    i = tag_end
                    matched = True
                    break

        if not matched:
            i += 1

    # 除外区間を使ってクリーンテキストを構築
    # removed は i の単調増加により挿入順でソート済みのため sort() 不要
    if not removed:
        return text.strip(), matches

    parts: list[str] = []
    prev = 0
    for start, end in removed:
        if start > prev:
            parts.append(text[prev:start])
        prev = end
    if prev < n:
        parts.append(text[prev:])

    return "".join(parts).strip(), matches


class StreamingTagStripper:
    """ストリーミングチャンクからツールタグをリアルタイムで除去するバッファ。

    LLM応答をチャンク単位で受け取り、[TAG:...] 形式のマーカーを除去しながら
    安全な部分だけを逐次返す。マーカーが複数チャンクにまたがっても正しく処理できる。

    使い方:
        stripper = StreamingTagStripper()
        for chunk in stream:
            safe = stripper.feed(chunk)
            if safe:
                yield safe
        remaining = stripper.flush()
        if remaining:
            yield remaining

    除去対象は KNOWN_PREFIXES（全プロバイダー共通）＋ extra_prefixes（呼び出し側固有）。
    """

    # 除去対象マーカーのプレフィックス。
    # 固定マーカー（']' で終わるもの）はプレフィックスがマーカー全体と一致する。
    KNOWN_PREFIXES: list[str] = [
        "[INSCRIBE_MEMORY:",
        "[CARVE_NARRATIVE:",
        "[END_SESSION]",    # 固定マーカー
        "[END_SESSION:",
        "[POWER_RECALL:",  # マーカー前テキストはUIへ流すが、タグ自体は除去する。
        # full_text（生テキスト）にはタグが残るため、ストリーム終了後に Recaller で検出できる。
        "[ANTICIPATE_RESPONSE:",  # 次ターンの予想（期待）タグ。全プロバイダー一律でストリームから除去する。
    ]

    # バッファがこの長さを超えたら強制フラッシュ（無限バッファを防ぐ）
    MAX_BUFFER: int = 1000

    def __init__(self, extra_prefixes: list[str] | None = None) -> None:
        """StreamingTagStripper を初期化する。

        Args:
            extra_prefixes: この呼び出し側でだけ除去したいマーカーのプレフィックス
                （例: シナリオ専用の "[SCENE_CLOSE]"）。全プロバイダー共通ではない
                マーカーを KNOWN_PREFIXES に混ぜずに足すための口。
        """
        self._buffer: str = ""
        self._prefixes: list[str] = list(self.KNOWN_PREFIXES) + list(extra_prefixes or [])

    def feed(self, chunk: str) -> str:
        """チャンクを投入し、マーカーを除去した安全なテキストを返す。

        マーカーが複数チャンクにまたがる場合、完結するまでバッファに保持する。

        Args:
            chunk: LLMから受け取った生テキストチャンク。

        Returns:
            マーカーを除去した、即座に表示可能なテキスト。空文字列の場合もある。
        """
        self._buffer += chunk
        return self._drain()

    def flush(self) -> str:
        """ストリーム終了時に残ったバッファを全て返す。完全なマーカーは除去される。

        Returns:
            残バッファのクリーンテキスト。
        """
        tag_names = []
        for p in self._prefixes:
            name = p.lstrip("[").rstrip(":]")
            if name not in tag_names:
                tag_names.append(name)
        clean, _ = parse_tags(self._buffer, tag_names)
        self._buffer = ""
        return clean

    def _could_be_marker_prefix(self, buf: str) -> bool:
        """buf が既知マーカーのプレフィックスになり得るか判定する。

        buf が既知プレフィックスのいずれかの冒頭部分（部分マッチ）の場合 True。
        例: buf="[INSCRIBE" → "[INSCRIBE_MEMORY:" の冒頭に一致 → True

        Args:
            buf: '[' から始まる未確定バッファ。

        Returns:
            既知マーカーになり得る場合 True。
        """
        for prefix in self._prefixes:
            if prefix.startswith(buf):
                return True
        return False

    def _find_complete_prefix(self, buf: str) -> str | None:
        """buf の先頭に完全マッチする既知プレフィックスを返す。なければ None。

        Args:
            buf: '[' から始まるバッファ。

        Returns:
            マッチしたプレフィックス文字列、またはNone。
        """
        for prefix in self._prefixes:
            if buf.startswith(prefix):
                return prefix
        return None

    def _drain(self) -> str:
        """バッファから安全に流せる部分を取り出して返す。

        '[' を見つけるまでは直接出力し、'[' を見つけたらマーカー判定を行う。
        マーカーが完結したら除去して続きを処理、未確定なら次のチャンクを待つ。

        Returns:
            マーカーを除去した出力可能テキスト。
        """
        output: list[str] = []
        buf = self._buffer

        while buf:
            bt_pos = buf.find("`")
            open_pos = buf.find("[")

            if open_pos == -1 and bt_pos == -1:
                # '[' も '`' もない → 全部流す
                output.append(buf)
                buf = ""
                break

            # '`' が '[' より手前にある場合、バッククォートブロックを処理する
            if bt_pos != -1 and (open_pos == -1 or bt_pos < open_pos):
                n_buf = len(buf)
                next_pos = _skip_backtick(buf, bt_pos, n_buf)
                if next_pos == n_buf:
                    # 閉じバッククォートがまだ来ていない → bt_pos 手前まで流して待つ
                    output.append(buf[:bt_pos])
                    buf = buf[bt_pos:]
                    break
                # 完結したバッククォートブロックをそのまま出力して続ける
                output.append(buf[:next_pos])
                buf = buf[next_pos:]
                continue

            if open_pos == -1:
                # '[' がない → 全部流す
                output.append(buf)
                buf = ""
                break

            # '[' の手前を安全に流す
            if open_pos > 0:
                output.append(buf[:open_pos])
                buf = buf[open_pos:]

            # buf は今 '[' から始まる
            complete_prefix = self._find_complete_prefix(buf)
            if complete_prefix is not None:
                # 既知プレフィックスが確定した
                if complete_prefix.endswith("]"):
                    # 固定マーカー（例: [END_SESSION]）: プレフィックス自体がマーカー全体
                    buf = buf[len(complete_prefix):]
                else:
                    # コンテンツマーカー（例: [INSCRIBE_MEMORY:...]）: ']' を探す
                    close_pos = buf.find("]", len(complete_prefix))
                    if close_pos == -1:
                        # ']' がまだ来ていない → 次のチャンクを待つ
                        break
                    buf = buf[close_pos + 1:]
                continue

            # 既知プレフィックスとの部分マッチが残っているか確認
            if self._could_be_marker_prefix(buf):
                # まだ確定できない → 次のチャンクを待つ。
                # バッファが長くなりすぎた場合（']' が永遠に来ない等）は強制フラッシュする。
                if len(buf) > self.MAX_BUFFER:
                    output.append(buf)
                    buf = ""
                break

            # マーカーではない '[' → そのまま流す
            output.append("[")
            buf = buf[1:]

        self._buffer = buf
        return "".join(output)
