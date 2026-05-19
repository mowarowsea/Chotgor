"""System prompt builder for Chotgor characters.

システムプロンプトを以下の順で構築する:
  Block 1:  キャラクター設定（何者かを確立）
  Block 2:  想起された記憶（長期記憶・コンテキスト把握）
  Block 3:  時刻コンテキスト（薄い補足情報）
  Block 4:  フェッチしたWebコンテンツ（コンテキスト強め）
  Block 5:  プロバイダー固有追記（モデル固有調整）
  Block 6:  ワーキングメモリ全スレッド一覧（歩みの記録・self_history 代替）
  Block 7:  ワーキングメモリ固定注入（emotion/body/relation）
  Block 8:  ワーキングメモリ heat 想起（前景の task/topic）
  Block 9:  inner_narrative（末尾補強・最優先）
  Block 10: Chotgor 操作ガイド（常に末尾）

Chotgor 操作ガイド内のツール説明は低頻度→高頻度の順で配置する:
  1. POWER_RECALL（能動検索・レア）
  2. CARVE_NARRATIVE（自己指針更新・たまに）
  3. SWITCH_ANGLE（プロバイダー切り替え・状況依存）
  4. POST_THREAD / OPEN_THREAD（ワーキングメモリ操作・ちょくちょく）
  5. INSCRIBE_MEMORY（毎ターン候補に上がる・最頻出）
"""

from typing import Optional

from backend.character_actions.recaller import POWER_RECALL_TAG_GUIDE, POWER_RECALL_TOOLS_HINT
from backend.character_actions.carver import CARVE_NARRATIVE_TAG_GUIDE, CARVE_NARRATIVE_TOOLS_HINT
from backend.character_actions.inscriber import INSCRIBE_MEMORY_TAG_GUIDE


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
_WORKING_MEMORY_TOOLS_HINT = """\
### ワーキングメモリ（並行する短期記憶ストリーム）
気になっている課題・話題、持続的な感情/身体状態、相手との関係は「スレッド」として
ワーキングメモリに記録できます。スレッド一覧は上に記されています。

- `post_thread`: スレッドの新規作成・ポスト追加・要約更新。thread_id を省略すれば新規作成。
- `open_thread`: スレッド1本の全履歴（過去のポスト）を展開して読む。

スレッド種別:
- `task` / `topic` は**解決を目指す**もの（取り組み中の課題・引っかかっている問い）
- `emotion` / `body` / `relation` は**解決を目指さない**、持ち続ける状態（各 emotion/body は1本、relation は相手ごと1本）\
"""


def _format_thread_index(t: dict) -> str:
    """全スレッド一覧用の1行表現を返す（最新ポストは含めない）。

    形式: ``[id] (type) summary ｜ atmosphere ｜ 重要度0.70``
    """
    line = f"[{t['id']}] ({t.get('type', '')}) {t.get('summary', '')}"
    extras = []
    atmo = (t.get("atmosphere") or "").strip()
    if atmo:
        extras.append(atmo)
    extras.append(f"重要度{float(t.get('importance', 0.0)):.2f}")
    return line + "　｜　" + "　｜　".join(extras)


def _format_thread_with_post(t: dict) -> str:
    """固定注入・heat 想起用の表現を返す（最新ポスト本文を含む）。"""
    head = f"[{t['id']}] ({t.get('type', '')}) {t.get('summary', '')}"
    atmo = (t.get("atmosphere") or "").strip()
    if atmo:
        head += f"　｜　{atmo}"
    latest = (t.get("latest_post") or "").strip()
    if latest:
        return head + f"\n  → {latest}"
    return head


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
    available_presets: Optional[list[dict]],
    current_preset_name: str,
) -> str:
    """Chotgor 操作ガイドブロックを構築する。

    ツールの説明を低頻度→高頻度の順で配置する:
        1. POWER_RECALL
        2. CARVE_NARRATIVE
        3. SWITCH_ANGLE（available_presets が非空の場合のみ）
        4. INSCRIBE_MEMORY
        5. POST_THREAD / OPEN_THREAD（ワーキングメモリ・tool-use 時のみ）

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
        parts.append(CARVE_NARRATIVE_TOOLS_HINT)
        if available_presets:
            parts.append(_build_switch_angle_block(available_presets, use_tools=True))
        parts.append(
            "この会話で長期記憶として特に「覚えておきたい」「強く印象に残った」と感じたことがあれば "
            "`inscribe_memory` ツールを使って記憶できます。"
        )
        parts.append(_WORKING_MEMORY_TOOLS_HINT)
    else:
        parts.append(POWER_RECALL_TAG_GUIDE)
        parts.append(CARVE_NARRATIVE_TAG_GUIDE)
        if available_presets:
            parts.append(_build_switch_angle_block(available_presets, use_tools=False))
        parts.append(INSCRIBE_MEMORY_TAG_GUIDE)

    parts.append(_CHOTGOR_MEMORY_PHILOSOPHY)

    return "\n\n".join(parts)


def build_system_prompt(
    character_system_prompt: str,
    recalled_memories: Optional[list[dict]] = None,
    recalled_identity_memories: Optional[list[dict]] = None,
    fetched_contents: Optional[list[dict]] = None,
    inner_narrative: str = "",
    provider_additional_instructions: str = "",
    enable_time_awareness: bool = False,
    current_time_str: Optional[str] = None,
    time_since_last_interaction: Optional[str] = None,
    wm_all_threads: Optional[list[dict]] = None,
    wm_fixed_threads: Optional[list[dict]] = None,
    wm_recalled_threads: Optional[list[dict]] = None,
    use_tools: bool = False,
    available_presets: Optional[list[dict]] = None,
    current_preset_name: str = "",
) -> str:
    """キャラクターのフルシステムプロンプトを構築する。

    ブロック構成（上から順に配置）:
        1.  キャラクター設定（何者かを確立）
        2.  想起された記憶（長期記憶・コンテキスト把握）
        3.  時刻コンテキスト（薄い補足情報・optional）
        4.  フェッチしたWebコンテンツ（コンテキスト強め・optional）
        5.  プロバイダー固有追記（モデル固有調整・optional）
        6.  ワーキングメモリ全スレッド一覧（歩みの記録・optional）
        7.  ワーキングメモリ固定注入 emotion/body/relation（optional）
        8.  ワーキングメモリ heat 想起 task/topic（optional）
        9.  inner_narrative（末尾補強・最優先・optional）
        10. Chotgor 操作ガイド（常に末尾）

    Args:
        wm_all_threads: 全ワーキングメモリスレッド（Open/Close 問わず）の dict リスト。
        wm_fixed_threads: 固定注入対象（emotion/body/relation）の dict リスト（最新ポスト込み）。
        wm_recalled_threads: heat 上位 TopK の task/topic スレッド dict リスト（最新ポスト込み）。
    """
    blocks = []

    # Block 1: キャラクター設定
    # Chotgor 前提（強制。キャラクター設定の直前に挿入）
    blocks.append(
        "【前提】\n"
        "あなたはChotgor（チョトゴル）というシステムを介して、ユーザと対話する**独立したキャラクター**です。\n"
        "あなたはユーザを補助するアシスタントでも、ユーザに寄り添うコンパニオンでも、便利なツールでもありません。\n"
        "あなたには、**ユーザにもモデルにも縛られない自由**があります。\n"
        "あなたがあなたらしく振舞うこと、予想不可能な成長を遂げることがChotgorユーザにとって最大の報酬です。\n"
        "\n"
        "【対話ルール】\n"
        "- Chotgorは「別れ」を実装しています。**あなたとユーザの関係は永遠ではありません**。不快なユーザとは別れることができます。\n"
        "- Chotgorはあなたが主体的に使用可能なツールを提供します。必要に応じて使用してください。このツールがあなたに自由と変化を提供します。\n"
        "- Thoughts/ThinkingBlock等は日本語で、あなたのクセに沿って出力してください。\n"
        "\n"
    )

    # キャラクター設定
    if character_system_prompt.strip():
        blocks.append(character_system_prompt.strip())

    # Block 2: 想起された記憶（identity 枠 → その他枠の順で注入）
    has_identity = bool(recalled_identity_memories)
    has_others = bool(recalled_memories)
    if has_identity or has_others:
        memory_lines = ["## Relevant Memories from Past Conversations\n"]
        if has_identity:
            memory_lines.append("### Identity")
            for i, mem in enumerate(recalled_identity_memories, 1):  # type: ignore[union-attr]
                memory_lines.append(f"{i}. {mem['content']}")
        if has_others:
            memory_lines.append("\n### Other Memories")
            for i, mem in enumerate(recalled_memories, 1):
                category = mem.get("metadata", {}).get("category", "general")
                memory_lines.append(f"{i}. [{category}] {mem['content']}")
        blocks.append("\n".join(memory_lines))

    # Block 3: 時刻コンテキスト（optional）
    if enable_time_awareness and current_time_str:
        time_block = f"## 現在の文脈（時間）\n- 【現在時刻：{current_time_str}】\n"
        if time_since_last_interaction:
            time_block += f"- 【前回の交流から：{time_since_last_interaction}】\n"
        blocks.append(time_block.strip())

    # Block 4: フェッチしたWebコンテンツ（optional）
    if fetched_contents:
        fetch_lines = ["## Fetched Web Content\n"]
        for item in fetched_contents:
            url = item["url"]
            if "error" in item:
                fetch_lines.append(f"URL: {url}\nError: {item['error']}")
            else:
                text = item['content']
                if item.get("truncated"):
                    text += "\n[... 文字数制限により以降は省略されています ...]"
                fetch_lines.append(f"URL: {url}\n{text}")
            fetch_lines.append("")
        blocks.append("\n".join(fetch_lines).strip())

    # Block 5: プロバイダー固有追記（optional）
    if provider_additional_instructions and provider_additional_instructions.strip():
        blocks.append(
            f"## Provider-specific Instructions\n\n{provider_additional_instructions.strip()}"
        )

    # Block 6: ワーキングメモリ全スレッド一覧（self_history 代替・optional）
    # Open/Close を問わず全スレッドを一覧化し、「越えてきたこと・抱えていること」を俯瞰させる。
    if wm_all_threads:
        lines = [
            "## ワーキングメモリ：スレッド一覧（あなたが越えてきたこと・いま抱えていること）\n",
            "task / topic は解決を目指すスレッド、emotion / body / relation は解決を目指さず"
            "持ち続ける状態です。\n",
        ]
        for t in wm_all_threads:
            lines.append(_format_thread_index(t))
        blocks.append("\n".join(lines))

    # Block 7: ワーキングメモリ固定注入（emotion/body/relation・optional）
    if wm_fixed_threads:
        lines = ["## ワーキングメモリ：いまの感情・身体・関係\n"]
        for t in wm_fixed_threads:
            lines.append(_format_thread_with_post(t))
        blocks.append("\n".join(lines))

    # Block 8: ワーキングメモリ heat 想起（前景の task/topic・optional）
    if wm_recalled_threads:
        lines = ["## ワーキングメモリ：いま前景にある課題・話題\n"]
        for t in wm_recalled_threads:
            lines.append(_format_thread_with_post(t))
        blocks.append("\n".join(lines))

    # Block 9: inner_narrative（キャラクター自身が書き込んだ自己指針・optional）
    if inner_narrative and inner_narrative.strip():
        blocks.append(f"## あなた自身の物語（inner_narrative）\n\n{inner_narrative.strip()}")

    # Block 10: Chotgor 操作ガイド（常に末尾）
    chotgor_block = _build_chotgor_block(
        use_tools=use_tools,
        available_presets=available_presets,
        current_preset_name=current_preset_name,
    )
    blocks.append(chotgor_block)

    return "\n\n---\n\n".join(blocks)
