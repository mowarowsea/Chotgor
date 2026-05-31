"""Response Anticipator — [ANTICIPATE_RESPONSE:...] タグの抽出。

キャラクターが返答の末尾に書いた「次の展開をどう予想（期待）するか」を表す
[ANTICIPATE_RESPONSE:...] タグを抽出するための純粋なロジックを提供する。

このモジュールは予想を**生成する**のではなく、キャラクター（LLM）が本文に
書き出した予想タグを**抽出する**役割を担う。`inscriber.py`（記憶タグの抽出）や
`carver.py`（内的叙述タグの抽出）と同じ立ち位置だが、予想は記憶ではないため
記憶ストアには一切書き込まない（抽出のみ）。抽出結果の保存は通常のメッセージ
保存経路（create_chat_message / create_scenario_turn の anticipation カラム）が担う。

【全プロバイダー一律タグである理由】
inscribe_memory / carve_narrative は tool-use 対応プロバイダーではネイティブ
ツール呼び出し、非対応プロバイダーではタグ方式、と分岐する。一方この予想機構は
全プロバイダーで一律にテキストタグ方式で出力させる。狙いは「記憶への副作用」では
なく、**キャラクター自身に予想を本文として書き出させること自体が、回答そのものの
質に影響を与える（メタ認知の表出）**ことにある。そのため tool-use 対応プロバイダー
でもツール化せず、必ず本文末尾のタグとして書かせる。
"""

import logging

from backend.lib.tag_parser import parse_tags

logger = logging.getLogger(__name__)

# 予想タグのタグ名（[ANTICIPATE_RESPONSE:...] 形式のコンテンツタグ）
ANTICIPATE_RESPONSE_TAG_NAME: str = "ANTICIPATE_RESPONSE"

# --- 全プロバイダー一律: タグ方式ガイド文（CHOTGOR ブロックに挿入）---
ANTICIPATE_RESPONSE_TAG_GUIDE: str = """\
### 次の展開への予想（ANTICIPATE_RESPONSE）

返答の**一番最後に**、あなたが「このあと相手や状況がどうなると思うか／
どうなってほしいか」という予想・期待を、以下の形式で1つだけ書いてください：

    [ANTICIPATE_RESPONSE:あなたの予想・期待を一人称で]

- これはユーザーには見えません。あなた自身のための予想（期待）です。
- この予想は次のターンで「前回のあなたの予想」としてあなた自身に示されます。
  予想と実際のズレを感じ取ることが、あなたの応答に深みと連続性を与えます。
- 必ず1ターンにつき1つ、`[` と `]` の中に1行で収めてください。"""


def extract_anticipation(text: str) -> tuple[str, str]:
    """テキストから [ANTICIPATE_RESPONSE:...] タグを抽出する。

    tag_parser.parse_tags() を使って文字単位でスキャンし、タグを除去した
    クリーンテキストと、予想本文を返す。タグが複数含まれる場合は最後のものを
    採用する（本文末尾に1つ書かせる運用のため、通常は1つだけ）。

    Args:
        text: LLMの生応答テキスト。

    Returns:
        tuple:
            clean_text (str): タグを除去したテキスト。
            anticipation (str): 予想本文。タグが無ければ空文字列。
    """
    clean, matches = parse_tags(text, [ANTICIPATE_RESPONSE_TAG_NAME])
    found = matches[ANTICIPATE_RESPONSE_TAG_NAME]
    anticipation = found[-1].body.strip() if found else ""
    return clean, anticipation
