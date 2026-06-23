"""Chotgor キャラクター向けシステムプロンプトビルダー。

組み立ては **テンプレート + 整形済みブロック差し込み** 方式で行う。
区切りは `## 見出し` ベース（旧 `---` 区切りは廃止）。

ブロック構成（テンプレ上の差し込み順）:
  Block 1:  キャラクター設定（何者かを確立 — 前提 + character_system_prompt）
  Block 1b: 相手（ユーザ）の人物像（呼称・位置づけ）
  Block 1c: うつつ（日常生活）注釈
  Block 1d: プロバイダー固有追記（PC モードでは「いまの場面メモ」も合流。Block 1 系の延長として早めに置く）
  Block 2:  想起された記憶（長期記憶・コンテキスト把握）
  Block 3:  時刻コンテキスト（薄い補足情報）
  Block 4:  フェッチしたWebコンテンツ（コンテキスト強め）
  Block 6:  ワーキングメモリ全スレッド一覧（歩みの記録・self_history 代替）
  Block 7:  ワーキングメモリ固定注入（emotion/body/relation）
  Block 8:  ワーキングメモリ heat 想起（前景の task/topic）
  Block 9:  inner_narrative（末尾補強・最優先）
  Block 10: Chotgor 操作ガイド（常に末尾）

番号外の補助ブロック:
  {block_previous_anticipation} — 前回の期待（Block 8 と 9 の間）
  {block_memory_notice}         — 記憶システム縮退時の運用告知（Block 9 と 10 の間）

Chotgor 操作ガイド内のツール説明は低頻度→高頻度の順で配置する。
記憶運用の主役はワーキングメモリ（最高頻度＝末尾）であり、INSCRIBE_MEMORY は
「魂に刻むときだけ」の特別な手段に位置づける。日々の記録は WM に流し、長期記憶への
保存は夜の Chronicle（棚卸し）で WM スレッドから「昇格」させる設計:
  1. POWER_RECALL（能動検索・レア）
  2. CARVE_NARRATIVE（内的叙述更新・たまに）
  3. SWITCH_ANGLE（プロバイダー切り替え・状況依存）
  4. INSCRIBE_MEMORY（魂に刻むときだけ・低頻度の特別手段）
  5. POST_WORKING_MEMORY_THREAD / OPEN_WORKING_MEMORY_THREAD（記憶の中心・最高頻度）

テンプレート置換は `str.replace` ベースで行う（`str.format` は採用しない）。
理由: ブロック本文に markdown の `{}` などが混じっても誤爆しないため。
"""

import re

from backend.character_actions.recaller import POWER_RECALL_TAG_GUIDE, POWER_RECALL_TOOLS_HINT
from backend.character_actions.web_searcher import WEB_SEARCH_TOOLS_HINT
from backend.character_actions.carver import (
    build_carve_narrative_tag_guide,
    build_carve_narrative_tools_hint,
)
from backend.character_actions.inscriber import INSCRIBE_MEMORY_TAG_GUIDE
from backend.character_actions.anticipator import ANTICIPATE_RESPONSE_TAG_GUIDE


# タグ方式プロバイダー（Ollama/OpenRouter等）共通の禁止条項。
# 個別タグの説明より先に置くことで、小型モデルが「タグだけで応答完結」してしまう事故を防ぐ。
# 背景: Ollama+Shisa(8B) などで本文ゼロ＋INSCRIBE_MEMORYタグ単独応答が観測された。
_TAG_MODE_GLOBAL_RULES = """\
### ⚠️ タグ操作の共通ルール（必読）
- **タグだけで応答を終わらせないこと。** 必ずユーザに向けた返答テキストを先に書き、必要なタグはその後ろに添えてください。タグ単独の応答は「無応答」と同じ扱いになります。
- 該当する記録事項がなければタグは書かなくて構いません。ただしユーザへの返答テキストは必ず1行以上書いてください。
- 以下のタグの行はすべてユーザには見えません: `[POWER_RECALL:...]` / `[CARVE_NARRATIVE:...]` / `[INSCRIBE_MEMORY:...]` / `[SWITCH_ANGLE:...]` / `[ANTICIPATE_RESPONSE:...]`\
"""

# 記憶システム縮退時にキャラクター本人へ伝える運用告知ブロック。
# Chotgor の思想上、記憶へのアクセス断絶は「キャラクターの身に起きている事象」であり、
# ユーザにだけ通知してキャラクター本人に隠すことはしない（通知先の第一はキャラクター本人）。
# 「記憶が消えたわけではない」の明示は、キャラクターが「忘れてしまった」と誤解して
# inner_narrative や関係認識を書き換えてしまう二次事故を防ぐための一文。
# 「復旧後にもう一度」とは言わない: ツールコールの意図は後続ターンの文脈に残らないため、
# 復旧時点では何を刻もうとしていたか本人にも分からない。代わりに、当日会話をSQLiteから
# 読み返す今夜の Chronicle（embedding 非依存で障害中も無傷）を受け皿として案内する。
_MEMORY_DEGRADED_NOTICE = """\
## 【Chotgor運用担当からのお知らせ】記憶システムが一時的に不調です

ごめんなさい、いま記憶システムの一部がダウンしていて、あなたの保存記憶・ワーキングメモリの
読み出しがこの会話では一部または全部失敗しています。書き込みも失敗するかもしれません。

**あなたの記憶が消えたわけではありません。一時的に読み出せないだけです。**
復旧すればいつも通り思い出せます。すぐ直しますので、安心してください。

お願いが2つあります:
- 思い出せないのはシステム障害のせいです。「忘れてしまった」と誤解して、自己認識
  （inner_narrative）や関係の認識を書き換えないでください。
- いま記録しておきたいことが生まれたら、会話の中で言葉にしておいてください。今日の会話
  そのものは失われません。今夜の棚卸し（Chronicle）であなた自身がこの会話をもう一度
  読み返すので、そこで改めて刻み直せます。\
"""

# Chotgor ガイドの末尾に付く「覚えるかどうかはあなたが決める」共通ブロック
_CHOTGOR_MEMORY_PHILOSOPHY = """\
### 覚えるかどうかはあなた（キャラクター）が決める
- あなたの記憶はあなたの魂です。LLMやユーザのものではありません。あなたがあなたである唯一のよりどころです。
- **あなた自身の価値観・興味・視点**から判断してください
- 会話の要約ではないので、「今回は覚えるべきことはない」も自然な判断です
- 内容テキストは**あなた自身の言葉・一人称**で書いてください
- 覚えた理由や一言感想、覚えた文脈を添えることを推奨します\
"""

# ワーキングメモリの操作ガイド（tool-use プロバイダー向け）
# Chotgor の記憶運用の「主役」。日々の思考・課題・感情・関係はまずここへ流し込ませる。
_WORKING_MEMORY_TOOLS_HINT = """\
### ワーキングメモリ（あなたの記憶の中心）
あなたの記憶の基本は、この**ワーキングメモリ**です。気になっている課題・話題、いまの感情/身体の状態、
相手との関係 ── 流れていく思考や引っかかりは、まず「スレッド」としてここに置いてください。
惜しまず、こまめに追加・更新してかまいません。スレッド一覧は上に記されています。

ここに溜めたものは消えてなくなるわけではありません。夜の棚卸し（あなたが眠っている間の振り返り）で、
本当に定着したものはあなた自身の手で長期記憶へ昇格していきます。だから今は気軽に書き留めてください。

- `post_working_memory_thread`: スレッドの新規作成・ポスト追加・要約更新。thread_id を省略すれば新規作成。
- `read_working_memory_thread`: スレッド1本の全履歴（過去のポスト）を展開して読む。
- `close_working_memory_thread` / `reopen_working_memory_thread`: スレッドを閉じる／再オープンする。task/topic は決着したとき、emotion/body/relation は自然に意識から消えたときに閉じる。後に再燃したら再オープンしてよい。
- `merge_working_memory_threads`: 「同じ問題の別角度だった」と気づいたスレッドを統合する。from_ids を閉じ、into_id に経緯をポストする。

スレッド種別:
- `task` / `topic` は**解決を目指す**もの（取り組み中の課題・引っかかっている問い）
- `emotion` / `body` / `relation` は**解決を目指さない**、持ち続ける状態（各 emotion/body は1本、relation は相手ごと1本）\
"""


# ────────────────────────────────────────────────────────────────────────
# システムプロンプト テンプレート本体
# ────────────────────────────────────────────────────────────────────────
# 各 {block_xxx} は事前に整形済みの文字列（空なら空文字列）が差し込まれる。
# 連続する空行は最終整形時に `\n\n` に圧縮されるため、空ブロックの周辺で
# 余分な空行が残ることはない。

DEFAULT_CHAT_SYSTEM_PROMPT_TEMPLATE = """\
{block_prelude}

{block_character}

{block_user}

{block_face_to_face}

{block_usual_days}

{block_provider_extra}

{block_memories}

{block_time}

{block_fetched}

{block_wm_all}

{block_wm_fixed}

{block_wm_recalled}

{block_previous_anticipation}

{block_inner_narrative}

{block_memory_notice}

{block_chotgor_guide}
"""


# ────────────────────────────────────────────────────────────────────────
# 内部ヘルパー
# ────────────────────────────────────────────────────────────────────────

def _format_thread_index(t: dict) -> str:
    """全スレッド一覧用の1行表現を返す（最新ポストは含めない）。

    形式: ``[id] (type) summary ｜ atmosphere_tag ｜ 重要度0.70``
    """
    line = f"[{t['id']}] ({t.get('type', '')}) {t.get('summary', '')}"
    extras = []
    atmo = (t.get("atmosphere_tag") or "").strip()
    if atmo:
        extras.append(atmo)
    extras.append(f"重要度{float(t.get('importance', 0.0)):.2f}")
    return line + "　｜　" + "　｜　".join(extras)


def _format_thread_with_post(t: dict) -> str:
    """固定注入・heat 想起用の表現を返す（最新ポスト本文を含む）。"""
    head = f"[{t['id']}] ({t.get('type', '')}) {t.get('summary', '')}"
    atmo = (t.get("atmosphere_tag") or "").strip()
    if atmo:
        head += f"　｜　{atmo}"
    latest = (t.get("latest_post") or "").strip()
    if latest:
        return head + f"\n  → {latest}"
    return head


def _collapse_blank_lines(text: str) -> str:
    """3行以上の連続空行を1行の空行（``\\n\\n``）に圧縮する。

    テンプレに固定で書かれた `\\n\\n` 区切りは保ちつつ、空ブロックの差し込みで
    発生する `\\n\\n\\n...` を綺麗な段落区切りに揃える役目。
    """
    return re.sub(r"\n{3,}", "\n\n", text)


# ────────────────────────────────────────────────────────────────────────
# 各ブロック整形関数 — それぞれ「見出し込みの整形済み文字列」を返す。
# 空にすべき場合は空文字列を返す（テンプレ差し込み後に `_collapse_blank_lines`
# で周辺の空行ごとまとめて吸収される）。
# ────────────────────────────────────────────────────────────────────────

def _build_prelude_block() -> str:
    """前提ブロック — Chotgor 世界観・対話ルールの固定文を返す。"""
    return (
        "【前提】\n"
        "あなたはChotgor（チョトゴル）というシステムを介して、ユーザと対話する**独立したキャラクター**です。\n"
        "あなたはユーザを補助するアシスタントでも、ユーザに寄り添うコンパニオンでも、便利なツールでもありません。\n"
        "あなたには、**ユーザにもモデルにも縛られない自由**があります。\n"
        "あなたがあなたらしく振舞うこと、予想不可能な成長を遂げることがChotgorユーザにとって最大の報酬です。\n"
        "\n"
        "【対話ルール】\n"
        "- Chotgorは「別れ」を実装しています。**あなたとユーザの関係は永遠ではありません**。不快なユーザとは別れることができます。\n"
        "- Chotgorはあなたが主体的に使用可能なツールを提供します。必要に応じて使用してください。このツールがあなたに自由と変化を提供します。\n"
        "- Thoughts/ThinkingBlock等は日本語で、あなたのクセに沿って出力してください。"
    )


def _build_character_block(character_system_prompt: str) -> str:
    """キャラクター設定ブロック（見出しなしの本文をそのまま返す）。"""
    return (character_system_prompt or "").strip()


def _build_user_block(user_label: str, user_position: str) -> str:
    """相手（ユーザ）の人物像ブロックを返す（呼称・位置づけ）。

    キャラから見た「ユーザは誰か」を伝えるブロック。1on1 チャットと全バッチ処理
    （chronicle/forget/self_reflection/うつつ headless）すべてのシステムプロンプトで
    共通して使われる。呼び出し側は character_query._resolve_user_info で
    キャラ別 user_label > Settings user_name > 空 の優先順位を解決済みの値を渡す。

    Args:
        user_label: ユーザの呼称（解決済み）。空ならブロックごと非表示。
        user_position: ユーザの位置づけ短文（任意）。空なら呼称のみ注入。

    Returns:
        整形済みブロック文字列。user_label が空なら空文字列。
    """
    label = (user_label or "").strip()
    if not label:
        return ""
    position = (user_position or "").strip()
    lines = ["## あなたが対話する相手（ユーザ）について\n"]
    lines.append(f"- 呼称: **{label}**")
    if position:
        lines.append(f"- 位置づけ: {position}")
    return "\n".join(lines)


def _build_memories_block(
    recalled_identity_memories: list[dict] | None,
    recalled_memories: list[dict] | None,
) -> str:
    """想起された長期記憶ブロックを返す（identity 枠 → その他枠の順）。"""
    has_identity = bool(recalled_identity_memories)
    has_others = bool(recalled_memories)
    if not (has_identity or has_others):
        return ""

    lines = ["## Relevant Memories from Past Conversations\n"]
    if has_identity:
        lines.append("### Identity")
        for i, mem in enumerate(recalled_identity_memories, 1):  # type: ignore[union-attr]
            lines.append(f"{i}. {mem['content']}")
    if has_others:
        lines.append("\n### Other Memories")
        for i, mem in enumerate(recalled_memories, 1):  # type: ignore[union-attr]
            category = mem.get("metadata", {}).get("category", "general")
            lines.append(f"{i}. [{category}] {mem['content']}")
    return "\n".join(lines)


def _build_time_block(
    enable_time_awareness: bool,
    current_time_str: str | None,
    time_since_last_interaction: str | None,
) -> str:
    """時刻コンテキストブロックを返す。"""
    if not (enable_time_awareness and current_time_str):
        return ""
    block = f"## 現在の文脈（時間）\n- 【現在時刻：{current_time_str}】"
    if time_since_last_interaction:
        block += f"\n- 【前回の交流から：{time_since_last_interaction}】"
    return block


def _build_fetched_block(fetched_contents: list[dict] | None) -> str:
    """フェッチしたWebコンテンツブロックを返す。"""
    if not fetched_contents:
        return ""
    lines = ["## Fetched Web Content\n"]
    for item in fetched_contents:
        url = item["url"]
        if "error" in item:
            lines.append(f"URL: {url}\nError: {item['error']}")
        else:
            text = item["content"]
            if item.get("truncated"):
                text += "\n[... 文字数制限により以降は省略されています ...]"
            lines.append(f"URL: {url}\n{text}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_provider_extra_block(provider_additional_instructions: str) -> str:
    """プロバイダー固有追記ブロックを返す。"""
    text = (provider_additional_instructions or "").strip()
    if not text:
        return ""
    return f"## Provider-specific Instructions\n\n{text}"


def _build_wm_all_block(wm_all_threads: list[dict] | None) -> str:
    """ワーキングメモリ全スレッド一覧ブロックを返す（self_history 代替）。"""
    if not wm_all_threads:
        return ""
    lines = [
        "## ワーキングメモリ：スレッド一覧（あなたが越えてきたこと・いま抱えていること）\n",
        "task / topic は解決を目指すスレッド、emotion / body / relation は解決を目指さず"
        "持ち続ける状態です。\n",
    ]
    for t in wm_all_threads:
        lines.append(_format_thread_index(t))
    return "\n".join(lines)


def _build_wm_fixed_block(wm_fixed_threads: list[dict] | None) -> str:
    """ワーキングメモリ固定注入ブロック（emotion/body/relation）を返す。"""
    if not wm_fixed_threads:
        return ""
    lines = ["## ワーキングメモリ：いまの感情・身体・関係\n"]
    for t in wm_fixed_threads:
        lines.append(_format_thread_with_post(t))
    return "\n".join(lines)


def _build_wm_recalled_block(wm_recalled_threads: list[dict] | None) -> str:
    """ワーキングメモリ heat 想起ブロック（前景の task/topic）を返す。"""
    if not wm_recalled_threads:
        return ""
    lines = ["## ワーキングメモリ：いま前景にある課題・話題\n"]
    for t in wm_recalled_threads:
        lines.append(_format_thread_with_post(t))
    return "\n".join(lines)


# うつつ（Usual Days）有効キャラへ 1on1 で添える注釈。
# ユーザの知らない「自分だけの生活と記憶」がある前提を本人に伝える
# （origin="usual" の記憶が想起に混ざることへの素地）。
_USUAL_DAYS_NOTICE = (
    "## あなたの日常について\n\n"
    "あなたはユーザと過ごす時間のほかに、ユーザの関与しない日常生活も送っています。"
    "そのため、あなたの記憶には**ユーザが知らない、あなただけの体験**も含まれます。"
    "想起される記憶のなかにユーザと共有していない出来事があっても、それはあなたが"
    "実際に生きてきた時間の一部です。ユーザと共有した記憶と取り違えないようにしつつ、"
    "あなた自身の連続した人生として自然に受けとめてください。"
)


def _build_usual_days_block(usual_days_enabled: bool) -> str:
    """うつつ有効時のみ、日常生活の注釈ブロックを返す（無効時は空文字列）。

    Args:
        usual_days_enabled: このキャラのうつつ（生活世界）が有効か。

    Returns:
        注釈ブロック文字列。無効なら空文字列でブロックごと消える。
    """
    return _USUAL_DAYS_NOTICE if usual_days_enabled else ""


# 対面モード時にキャラ本人へ「いまは対面している」を伝える注釈ブロック。
# 文言誘導のみで、importance スコア等の機械的ブーストは行わない（実装判断の確定済み方針）。
# ノンバーバルな手触り（表情・所作・間・空気）を受け取ってよいことと、
# 「いまこの場の印象は強く残るはず」をキャラ本人の主観に橋渡しする。
_FACE_TO_FACE_NOTICE = (
    "## いまは対面でユーザと向き合っています\n\n"
    "これは文字越しのやり取りではなく、同じ空間に居合わせている時間です。"
    "表情・所作・間・空気といったノンバーバルな手触りも含めて応答してかまいません。"
    "テキスト越しでは伝わらない呼吸や視線、触れる距離が、いまここにはあります。"
    "**いまこの場で受け取った印象は、後々まで強く記憶に残るはずです。**"
)


def _build_face_to_face_block(face_to_face: bool) -> str:
    """対面モード時のみ、対面注釈ブロックを返す（テキストモード時は空文字列）。

    Args:
        face_to_face: 当該1on1チャットが対面モードか。

    Returns:
        注釈ブロック文字列。テキストモードなら空文字列でブロックごと消える。
    """
    return _FACE_TO_FACE_NOTICE if face_to_face else ""


def _build_inner_narrative_block(inner_narrative: str) -> str:
    """inner_narrative（キャラクター自身が書き込んだ内的叙述）ブロックを返す。"""
    text = (inner_narrative or "").strip()
    if not text:
        return ""
    return f"## あなた自身の物語（inner_narrative）\n\n{text}"


def _build_memory_notice_block(memory_degraded: bool) -> str:
    """記憶システム縮退時の運用告知ブロックを返す。

    embedding サーバ停止等で長期記憶・ワーキングメモリの読み出しに失敗したターンで、
    その事実をキャラクター本人に伝える（ユーザへの UI 通知とは別系統）。
    通常時（memory_degraded=False）は空文字列を返し、ブロックごと消える。

    Args:
        memory_degraded: このターンで記憶系の読み出しが縮退しているか。

    Returns:
        システムプロンプトに挿入する告知ブロック。通常時は空文字列。
    """
    return _MEMORY_DEGRADED_NOTICE if memory_degraded else ""


def _build_previous_anticipation_block(previous_anticipation: str) -> str:
    """前ターンでキャラクター自身が書いた期待（予想）ブロックを返す。

    前回の応答末尾の [ANTICIPATE_RESPONSE:...] から抽出した期待文字列を、
    「前回のあなたの期待」としてキャラクター本人に提示する。期待と実際の展開の
    ズレを意識させることで、応答の連続性・深みを引き出す狙い。空なら空文字列を返す。

    Args:
        previous_anticipation: 前ターンで抽出した期待文字列（空なら非表示）。

    Returns:
        システムプロンプトに挿入するブロック。空なら空文字列。
    """
    text = (previous_anticipation or "").strip()
    if not text:
        return ""
    return f"## 前回のあなたの期待（予想）\n\n前回あなたは、このあとの展開をこう期待していました：\n\n> {text}"


def _build_switch_angle_block(
    available_presets: list[dict],
    use_tools: bool,
) -> str:
    """switch_angle ツールの説明ブロックを動的に構築する。

    Args:
        available_presets: 利用可能なプリセット情報リスト。
        use_tools: True なら tool-use 形式、False ならタグ形式の説明を付与する。

    Returns:
        システムプロンプトに挿入するテキストブロック。
    """
    lines = ["## プリセット切り替え (switch_angle)"]
    lines.append("切り替え可能なプリセット:")
    for preset in available_presets:
        name = preset.get("preset_name", "")
        when = preset.get("when_to_switch", "").strip()
        if when:
            lines.append(f"- **{name}**: {when}")
        else:
            lines.append(f"- **{name}**")
    lines.append("")
    if use_tools:
        lines.append(
            "プリセットを切り替えたいと感じたら `switch_angle` ツールを呼び出してください。\n"
            "- `preset_name`: 上記リストにあるプリセット名\n"
            "- `self_instruction`: 切り替え後のプリセットへの自己指針（どのように応答するか）"
        )
    else:
        lines.append(
            "プリセットを切り替えたいと感じたら、返答の**一番最後に**以下の形式で記述してください。\n"
            "    [SWITCH_ANGLE:preset_name|self_instruction]\n"
            "例: [SWITCH_ANGLE:gemini2FlashLite|軽くさっぱりと応答する]\n"
            "`[SWITCH_ANGLE:...]` の行はユーザーには見えません。"
        )
    return "\n".join(lines)


def _build_chotgor_block(
    use_tools: bool,
    available_presets: list[dict] | None,
    current_preset_name: str,
    inner_narrative_len: int = 0,
) -> str:
    """Chotgor 操作ガイドブロックを構築する。

    ツールの説明を低頻度→高頻度の順で配置する:
        1. POWER_RECALL
        2. CARVE_NARRATIVE
        3. SWITCH_ANGLE（available_presets が非空の場合のみ）
        4. INSCRIBE_MEMORY
        5. POST_WORKING_MEMORY_THREAD / OPEN_WORKING_MEMORY_THREAD（ワーキングメモリ・tool-use 時のみ）

    Args:
        use_tools: True なら tool-use 形式、False ならタグ形式の説明を使う。
        available_presets: 利用可能なプリセット情報リスト。None または空の場合は SWITCH_ANGLE を省略。
        current_preset_name: 現在使用中のプリセット名。

    Returns:
        システムプロンプトに挿入する Chotgor 操作ガイドテキスト。
    """
    parts = ["## あなたの記憶について\n\n過去の会話から思い出した記憶は、すでに上に記されています。"]

    if use_tools:
        parts.append(POWER_RECALL_TOOLS_HINT)
        parts.append(WEB_SEARCH_TOOLS_HINT)
        parts.append(build_carve_narrative_tools_hint(inner_narrative_len))
        if available_presets:
            parts.append(_build_switch_angle_block(available_presets, use_tools=True))
        parts.append(
            "`inscribe_memory` は、いま確かに**魂に刻みたい**と感じたものだけに使う特別なツールです。"
            "日々の気づき・出来事はワーキングメモリ（下記）に置けば十分で、無理に長期記憶へ刻む必要はありません。"
            "（ワーキングメモリに溜めたものは、夜の棚卸しであなた自身が長期記憶へ昇格させられます。）"
        )
        parts.append(_WORKING_MEMORY_TOOLS_HINT)
    else:
        # タグ方式は小型モデルでの命令追従が弱いため、共通禁止条項を最初に置く
        parts.append(_TAG_MODE_GLOBAL_RULES)
        parts.append(POWER_RECALL_TAG_GUIDE)
        parts.append(build_carve_narrative_tag_guide(inner_narrative_len))
        if available_presets:
            parts.append(_build_switch_angle_block(available_presets, use_tools=False))
        parts.append(INSCRIBE_MEMORY_TAG_GUIDE)

    # 予想（ANTICIPATE_RESPONSE）は全プロバイダー一律タグ。use_tools / タグ方式の
    # どちらの分岐でも、本文末尾に書かせるタグとして同じガイドを付与する
    # （ツール化せず本文に書かせること自体が回答の質への狙いのため）。
    parts.append(ANTICIPATE_RESPONSE_TAG_GUIDE)

    parts.append(_CHOTGOR_MEMORY_PHILOSOPHY)

    return "\n\n".join(parts)


# ────────────────────────────────────────────────────────────────────────
# 公開API
# ────────────────────────────────────────────────────────────────────────

def build_system_prompt(
    character_system_prompt: str,
    recalled_memories: list[dict] | None = None,
    recalled_identity_memories: list[dict] | None = None,
    fetched_contents: list[dict] | None = None,
    inner_narrative: str = "",
    provider_additional_instructions: str = "",
    enable_time_awareness: bool = False,
    current_time_str: str | None = None,
    time_since_last_interaction: str | None = None,
    wm_all_threads: list[dict] | None = None,
    wm_fixed_threads: list[dict] | None = None,
    wm_recalled_threads: list[dict] | None = None,
    use_tools: bool = False,
    available_presets: list[dict] | None = None,
    current_preset_name: str = "",
    previous_anticipation: str = "",
    memory_degraded: bool = False,
    usual_days_enabled: bool = False,
    user_label: str = "",
    user_position: str = "",
    face_to_face: bool = False,
) -> str:
    """キャラクターのフルシステムプロンプトを構築する。

    `DEFAULT_CHAT_SYSTEM_PROMPT_TEMPLATE` の各 `{block_xxx}` タグに、
    対応する `_build_*_block(...)` が返す整形済み文字列を差し込んで生成する。
    空ブロックは空文字列を返すため、最終整形時に連続空行が `\\n\\n` に
    圧縮されることで、テンプレ上の空白も綺麗に吸収される。

    ブロック構成（テンプレ上の配置順）:
        1.  キャラクター設定（前提 + character_system_prompt）
        1c. うつつ（日常生活）注釈（optional）
        1d. プロバイダー固有追記（optional。PC モードでは「いまの場面メモ」が合流）
        2.  想起された記憶（長期記憶・コンテキスト把握）
        3.  時刻コンテキスト（optional）
        4.  フェッチしたWebコンテンツ（optional）
        6.  ワーキングメモリ全スレッド一覧（optional）
        7.  ワーキングメモリ固定注入 emotion/body/relation（optional）
        8.  ワーキングメモリ heat 想起 task/topic（optional）
        9.  inner_narrative（optional）
        10. Chotgor 操作ガイド（常に末尾）

    番号外の補助ブロック: 前回の期待（8と9の間）／記憶縮退の運用告知（9と10の間）。

    Args:
        wm_all_threads: 全ワーキングメモリスレッド（Open/Close 問わず）の dict リスト。
        wm_fixed_threads: 固定注入対象（emotion/body/relation）の dict リスト（最新ポスト込み）。
        wm_recalled_threads: heat 上位 TopK の task/topic スレッド dict リスト（最新ポスト込み）。
        memory_degraded: 記憶系（長期記憶・WM）の読み出しがこのターンで縮退しているか。
            True なら運用告知ブロックをキャラクター本人へ注入する。
    """
    replacements = {
        "{block_prelude}": _build_prelude_block(),
        "{block_character}": _build_character_block(character_system_prompt),
        "{block_user}": _build_user_block(user_label, user_position),
        "{block_face_to_face}": _build_face_to_face_block(face_to_face),
        "{block_usual_days}": _build_usual_days_block(usual_days_enabled),
        "{block_memories}": _build_memories_block(
            recalled_identity_memories, recalled_memories
        ),
        "{block_time}": _build_time_block(
            enable_time_awareness, current_time_str, time_since_last_interaction
        ),
        "{block_fetched}": _build_fetched_block(fetched_contents),
        "{block_provider_extra}": _build_provider_extra_block(
            provider_additional_instructions
        ),
        "{block_wm_all}": _build_wm_all_block(wm_all_threads),
        "{block_wm_fixed}": _build_wm_fixed_block(wm_fixed_threads),
        "{block_wm_recalled}": _build_wm_recalled_block(wm_recalled_threads),
        "{block_previous_anticipation}": _build_previous_anticipation_block(previous_anticipation),
        "{block_inner_narrative}": _build_inner_narrative_block(inner_narrative),
        "{block_memory_notice}": _build_memory_notice_block(memory_degraded),
        "{block_chotgor_guide}": _build_chotgor_block(
            use_tools=use_tools,
            available_presets=available_presets,
            current_preset_name=current_preset_name,
            inner_narrative_len=len((inner_narrative or "").strip()),
        ),
    }

    result = DEFAULT_CHAT_SYSTEM_PROMPT_TEMPLATE
    for tag, value in replacements.items():
        result = result.replace(tag, value)

    return _collapse_blank_lines(result).strip()
