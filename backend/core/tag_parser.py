"""TagParser — ツールタグ [TAG:...] の文字単位パーサー。

LLM応答テキストから [MEMORY:...] / [DRIFT:...] などのツールタグを
正規表現ではなく文字単位でスキャンし、正確に抽出するためのユーティリティ。

特徴:
  - ネストした角括弧 ([TAG:...[INNER:]...]) を正しく処理する
  - バッククォートインラインコード・コードフェンス内はスキャンをスキップする
  - タグ定義は呼び出し側が渡すため、任意のタグ名を自由に追加できる抽象設計
  - タグ名の列挙順を呼び出し側が意識しなくてよいよう、内部で長さ降順ソートする
    (例: "SEARCH_RESET" が "SEARCH" より必ず先に照合される)

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


def parse_tags(text: str, tag_names: list[str]) -> tuple[str, dict[str, list[TagMatch]]]:
    """テキストからツールタグを文字単位で抽出する。

    正規表現の代わりに文字単位でスキャンし、
    ネストした角括弧とバッククォートコードブロックを正しく処理する。

    タグ名の列挙順は呼び出し側が意識しなくてよい。
    内部で長さ降順ソートするため、プレフィックスが共通するタグ名
    ("SEARCH_RESET" と "SEARCH" など) でも安全に照合できる。

    Args:
        text: パース対象のテキスト。
        tag_names: 認識するタグ名のリスト (例: ["MEMORY", "DRIFT", "DRIFT_RESET"])。
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
                # ネスト対応スキャン: '[' でdepth増加、']' でdepth減少
                # i の '[' で depth=1 として開始し、対応する ']' を探す
                j = i + 1  # 開いた '[' の次から
                depth = 1
                while j < n and depth > 0:
                    c = text[j]
                    if c == "`":
                        j = _skip_backtick(text, j, n)
                    elif c == "[":
                        depth += 1
                        j += 1
                    elif c == "]":
                        depth -= 1
                        if depth == 0:
                            break
                        j += 1
                    else:
                        j += 1

                if depth == 0:
                    tag_end = j + 1  # ']' の次
                    raw = text[i:tag_end]
                    body = text[i + len(prefix) : j]
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
