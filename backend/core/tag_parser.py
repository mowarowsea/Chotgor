"""TagParser — ツールタグ [TAG:...] の文字単位パーサー。

LLM応答テキストから [MEMORY:...] / [DRIFT:...] などのツールタグを
正規表現ではなく文字単位でスキャンし、正確に抽出するためのユーティリティ。

特徴:
  - 閉じ括弧 `]` の検出ロジック（2段階）:
    1. ']' の直後が `[UPPERCASE...:` や `[UPPERCASE...]` 形式（タグ形式）で始まれば
       そこをタグ境界として閉じ括弧とみなす。認識済みかどうかは問わない。
       これにより同一行に複数タグが連続する場合や、認識外タグの直前でも正しく分割できる。
    2. 上記に該当する ']' が見つからない場合は rfind フォールバック。
       検索範囲は multiline=False なら同一行内、multiline=True なら文末まで。
       これにより内容テキストに ']' が含まれる場合も正しくパースできる（Issue #49対応）。
  - バッククォートインラインコード・コードフェンス内はスキャンをスキップする
  - タグ定義は呼び出し側が渡すため、任意のタグ名を自由に追加できる抽象設計
  - タグ名の列挙順を呼び出し側が意識しなくてよいよう、内部で長さ降順ソートする
    (例: "SEARCH_RESET" が "SEARCH" より必ず先に照合される)

multiline パラメータ:
  - False（デフォルト）: 改行をまたぐタグ内容は非対応。同一行内の rfind を使用。
  - True: 改行をまたぐタグ内容に対応。文末まで rfind を使用。drifter.py が使用する。

新規タグを追加するには:
  1. 処理モジュール（inscriber.py 相当）を新規作成し parse_tags() を呼ぶだけでよい
  2. tag_parser.py 本体の変更は不要
"""

from dataclasses import dataclass


@dataclass
class TagMatch:
    """1つのタグマッチ結果。"""

    # タグ名 (例: "MEMORY", "DRIFT", "DRIFT_RESET")
    tag_name: str
    # マッチした元テキスト全体 (例: "[MEMORY:fact|1.0|content]")
    raw: str
    # タグ名コロン以降、末尾 ']' を除いた本体 (例: "fact|1.0|content")
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


def _line_end(text: str, pos: int, n: int) -> int:
    """pos 以降の改行位置（排他）を返す。改行がなければ n を返す。

    Args:
        text: スキャン対象テキスト。
        pos: 検索開始位置。
        n: テキスト長 (= len(text))。

    Returns:
        改行文字の位置（改行がなければ n）。
    """
    nl = text.find("\n", pos)
    return nl if nl != -1 else n


def _is_tag_like(text: str, pos: int, n: int) -> bool:
    """pos 以降のテキストがタグ形式 [UPPERCASE...: または [UPPERCASE...] で始まるか判定する。

    '[' に続いて大文字アルファベット・アンダースコア・数字が1文字以上並び、
    ':' または ']' で終わる場合にタグ形式とみなす。
    日本語・記号などはこの条件を満たさないため、通常の括弧内テキストと区別できる。

    Args:
        text: スキャン対象テキスト。
        pos: 判定開始位置。
        n: テキスト長 (= len(text))。

    Returns:
        タグ形式で始まる場合 True。
    """
    if pos >= n or text[pos] != "[":
        return False
    end = pos + 1
    while end < n and (text[end].isupper() or text[end] == "_" or text[end].isdigit()):
        end += 1
    # タグ名が1文字以上あり、':' または ']' で終わること
    if end >= n or end == pos + 1:
        return False
    return text[end] in (":", "]")


def _find_closing_bracket(
    text: str, body_start: int, end_of_search: int, n: int, multiline: bool
) -> int:
    """タグの閉じ括弧 ']' の位置を返す。

    検出ロジック（2段階）:
    1. ']' の直後がタグ形式（[UPPERCASE...: など）で始まれば即タグ境界として返す。
       これにより連続タグが認識外タグも含め正しく分割される。
       例: "[DRIFT:指針A][DRIFT:指針B]" → 1つ目の ']' 直後に "[DRIFT:" → 即返す
       例: "[DRIFT:指針A][MEMORY:...]"  → ']' 直後に "[MEMORY:" → 即返す（認識外でもOK）

    2. 上記に該当しない場合のフォールバック:
       - multiline=False: rfind（最後の ']'）を返す。
         内容中に ']' を含むタグに対応（Issue #49対応）。
         例: "[MEMORY:cat|0.5|内容に]が含まれる]" → 最後の ']' が閉じ括弧。
       - multiline=True: find（最初の ']'）を返す。
         rfind にすると "普通のテキスト" をまたいで次タグの ']' まで飲み込んでしまうため。
         例: "[DRIFT:指針B]普通のテキスト[DRIFT:指針C]" の 2つ目DRIFT解析時、
             最初の ']'（指針B 直後）を正しく閉じ括弧とする。

    Args:
        text: スキャン対象テキスト。
        body_start: タグ本体の開始位置（プレフィックスの直後）。
        end_of_search: 検索の上限位置（排他）。multiline=False なら行末、True なら文末。
        n: テキスト長 (= len(text))。
        multiline: フォールバック挙動の切り替え。

    Returns:
        閉じ括弧 ']' の位置。見つからない場合は -1。
    """
    pos = body_start
    while pos < end_of_search:
        cand = text.find("]", pos, end_of_search)
        if cand == -1:
            break
        # ']' の直後がタグ形式なら即タグ境界として返す
        if _is_tag_like(text, cand + 1, n):
            return cand
        pos = cand + 1

    # フォールバック
    if multiline:
        # multiline=True: 最初の ']' を使う（普通テキストをまたいだ誤飲み込みを防ぐ）
        return text.find("]", body_start, end_of_search)
    else:
        # multiline=False: 最後の ']' を使う（内容中の ']' をスキップするため）
        return text.rfind("]", body_start, end_of_search)


def parse_tags(
    text: str,
    tag_names: list[str],
    *,
    multiline: bool = False,
) -> tuple[str, dict[str, list[TagMatch]]]:
    """テキストからツールタグを文字単位で抽出する。

    文字単位でスキャンし、各タグの閉じ括弧を _find_closing_bracket() で取得する。
    バッククォートコードブロック内のタグ形式テキストはスキップする。

    タグ名の列挙順は呼び出し側が意識しなくてよい。
    内部で長さ降順ソートするため、プレフィックスが共通するタグ名
    ("SEARCH_RESET" と "SEARCH" など) でも安全に照合できる。

    Args:
        text: パース対象のテキスト。
        tag_names: 認識するタグ名のリスト (例: ["MEMORY", "DRIFT", "DRIFT_RESET"])。
                   "[TAG]" 形式の固定マーカーも "[TAG:...]" 形式のコンテンツタグも両対応。
        multiline: True の場合、改行をまたぐタグ内容を許容する。
                   False（デフォルト）の場合、タグは同一行に収まることを前提とする。
                   drifter.py のみ True を使用する。

    Returns:
        tuple:
            clean_text (str): タグをすべて除去したテキスト。前後の空白をトリム済み。
            matches (dict[str, list[TagMatch]]): タグ名 → マッチ結果リスト。
    """
    matches: dict[str, list[TagMatch]] = {name: [] for name in tag_names}
    # i は単調増加するため removed は挿入順でソート済みが保証される
    removed: list[tuple[int, int]] = []

    # プレフィックスが共通するタグ名の誤照合を防ぐため、長さ降順でソートする
    # 例: ["DRIFT", "DRIFT_RESET"] → ["DRIFT_RESET", "DRIFT"]
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

                # multiline=False なら同一行内、True なら文末まで検索する
                end_of_search = n if multiline else _line_end(text, body_start, n)

                j = _find_closing_bracket(text, body_start, end_of_search, n, multiline)

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
