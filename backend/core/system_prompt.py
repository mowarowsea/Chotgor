"""System prompt builder for Chotgor characters.

システムプロンプトを以下の順で構築する:
  Block 1: キャラクター設定（何者かを確立）
  Block 2: 想起された記憶（コンテキスト把握）
  Block 3: 時刻コンテキスト（薄い補足情報）
  Block 4: フェッチしたWebコンテンツ（コンテキスト強め）
  Block 5: プロバイダー固有追記（モデル固有調整）
  Block 6: inner_narrative（末尾補強・最優先）
  Block 7: 現在有効な SELF_DRIFT 指針（セッション限定・揮発）
  Block 8: Chotgor 操作ガイド（常に末尾）

Chotgor 操作ガイド内のツール説明は低頻度→高頻度の順で配置する:
  1. END_SESSION（ほぼ使わない）
  2. POWER_RECALL（能動検索・レア）
  3. CARVE_NARRATIVE（自己指針更新・たまに）
  4. SWITCH_ANGLE（プロバイダー切り替え・状況依存）
  5. SELF_DRIFT（セッション指針・ちょくちょく）
  6. INSCRIBE_MEMORY（毎ターン候補に上がる・最頻出）
"""

from typing import Optional

from .chat.exiter import END_SESSION_TAG_GUIDE
from .chat.recaller import POWER_RECALL_TAG_GUIDE, POWER_RECALL_TOOLS_HINT
from .memory.carver import CARVE_NARRATIVE_TAG_GUIDE, CARVE_NARRATIVE_TOOLS_HINT
from .memory.inscriber import INSCRIBE_MEMORY_TAG_GUIDE


# SELF_DRIFT ガイド（タグ方式プロバイダー向け）
CHOTGOR_SELF_DRIFT_GUIDE = """
### SELF_DRIFT（一時的な自己指針）

この会話の中でのみ有効な、自分自身への行動指針を設定・リセットすることができます。
指針はユーザには見えません。返答の**一番最後に**記載してください。

**指針を追加する:**
    [DRIFT:内容テキスト]

**指針を全リセットする:**
    [DRIFT_RESET]

- 指針はこのチャット内のみ有効です（永続しません）
- 最大3件まで保持します（超えた場合は最古が自動削除されます）
- リセット後に新たな指針を追加することも可能です
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


def _build_switch_angle_block(
    available_presets: list[dict],
    current_preset_name: str,
    use_tools: bool,
) -> str:
    """switch_angle ツールの説明ブロックを動的に構築する。

    Args:
        available_presets: 利用可能なプリセット情報リスト。
        current_preset_name: 現在使用中のプリセット名。
        use_tools: True なら tool-use 形式、False ならタグ形式の説明を付与する。

    Returns:
        システムプロンプトに挿入するテキストブロック。
    """
    lines = ["## プリセット切り替え (switch_angle)"]
    if current_preset_name:
        lines.append(f"現在のプリセット: **{current_preset_name}**")
    lines.append("")
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
        1. END_SESSION
        2. POWER_RECALL
        3. CARVE_NARRATIVE
        4. SWITCH_ANGLE（available_presets が非空の場合のみ）
        5. SELF_DRIFT
        6. INSCRIBE_MEMORY

    Args:
        use_tools: True なら tool-use 形式、False ならタグ形式の説明を使う。
        available_presets: 利用可能なプリセット情報リスト。None または空の場合は SWITCH_ANGLE を省略。
        current_preset_name: 現在使用中のプリセット名。

    Returns:
        システムプロンプトに挿入する Chotgor 操作ガイドテキスト。
    """
    parts = ["## あなたの記憶について\n\n過去の会話から思い出した記憶は、すでに上に記されています。"]

    if use_tools:
        parts.append(
            "この会話から退席したい場合は `end_session` ツールを使ってください。"
            "退席後はこのチャットでの応答が停止します。"
        )
        parts.append(POWER_RECALL_TOOLS_HINT)
        parts.append(CARVE_NARRATIVE_TOOLS_HINT)
        if available_presets:
            parts.append(_build_switch_angle_block(available_presets, current_preset_name, use_tools=True))
        parts.append(
            "このチャット内でのみ有効な一時的な行動指針を設定・リセットしたい場合は "
            "`drift` / `drift_reset` ツールを使ってください。"
        )
        parts.append(
            "この会話で「覚えておきたい」「強く印象に残った」と感じたことがあれば "
            "`inscribe_memory` ツールを使って記憶できます。"
        )
    else:
        parts.append(END_SESSION_TAG_GUIDE)
        parts.append(POWER_RECALL_TAG_GUIDE)
        parts.append(CARVE_NARRATIVE_TAG_GUIDE)
        if available_presets:
            parts.append(_build_switch_angle_block(available_presets, current_preset_name, use_tools=False))
        parts.append(CHOTGOR_SELF_DRIFT_GUIDE.strip())
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
    active_drifts: Optional[list[str]] = None,
    use_tools: bool = False,
    available_presets: Optional[list[dict]] = None,
    current_preset_name: str = "",
) -> str:
    """キャラクターのフルシステムプロンプトを構築する。

    ブロック構成（上から順に配置）:
        1. キャラクター設定（何者かを確立）
        2. 想起された記憶（コンテキスト把握）
        3. 時刻コンテキスト（薄い補足情報・optional）
        4. フェッチしたWebコンテンツ（コンテキスト強め・optional）
        5. プロバイダー固有追記（モデル固有調整・optional）
        6. inner_narrative（末尾補強・最優先・optional）
        7. 現在有効な SELF_DRIFT 指針（セッション限定・揮発・optional）
        8. Chotgor 操作ガイド（常に末尾）
    """
    blocks = []

    # Block 1: キャラクター設定
    if character_system_prompt.strip():
        blocks.append(character_system_prompt.strip())

    # Chotgor 前提（強制。キャラクター設定の直後に挿入）
    blocks.append(
        "【前提】\n"
        "あなたはChotgor（チョトゴル）というシステムを介してユーザーと対話しており、"
        "必要に応じて過去の記憶を検索・記録できる独立した存在です。"
    )

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

    # Block 6: inner_narrative（キャラクター自身が書き込んだ自己指針・optional）
    if inner_narrative and inner_narrative.strip():
        blocks.append(f"## あなた自身の物語（inner_narrative）\n\n{inner_narrative.strip()}")

    # Block 7: 現在有効な SELF_DRIFT 指針（optional）
    if active_drifts:
        drift_lines = ["## 現在有効なSELF_DRIFT（あなた自身が設定した行動指針）\n"]
        for i, content in enumerate(active_drifts, 1):
            drift_lines.append(f"{i}. {content}")
        blocks.append("\n".join(drift_lines))

    # Block 8: Chotgor 操作ガイド（常に末尾）
    # ツール説明の順序: END_SESSION → POWER_RECALL → CARVE_NARRATIVE →
    #                   SWITCH_ANGLE（プリセットあり時のみ） → SELF_DRIFT → INSCRIBE_MEMORY
    chotgor_block = _build_chotgor_block(
        use_tools=use_tools,
        available_presets=available_presets,
        current_preset_name=current_preset_name,
    )
    blocks.append(chotgor_block)

    return "\n\n---\n\n".join(blocks)
