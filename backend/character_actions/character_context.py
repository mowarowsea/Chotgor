"""キャラクターに何かを問い合わせる際の共通コンテキストブロック構築ユーティリティ。

通常チャット以外（自己参照・バッチ処理など）でLLMに
「キャラクターとして」問い合わせる際のシステムプロンプトを構築する。
"""


def build_character_context(
    character_system_prompt: str,
    inner_narrative: str = "",
    self_history: str = "",
    relationship_state: str = "",
    suffix: str = "",
) -> str:
    """キャラクター設定テキストを連結してコンテキストブロックを構築する。

    空のフィールドは省略する。すべて空の場合は空文字列を返す。
    通常チャットの build_system_prompt() と異なり、記憶想起・時刻・Chotgorガイド等の
    付随ブロックは含まない。「キャラクターに聞く」最小限の文脈のみ組み立てる。

    Args:
        character_system_prompt: キャラクター基本設定テキスト。
        inner_narrative: キャラクターが自己記述した inner_narrative。
        self_history: キャラクターの歴史・経緯（chronicle で更新）。
        relationship_state: ユーザ・他キャラとの現在の関係（chronicle で更新）。
        suffix: 非空のコンテキストブロック末尾に付与する文字列。

    Returns:
        組み立てたコンテキスト文字列（suffix 含む）。すべて空の場合は空文字列。
    """
    parts = []
    if character_system_prompt.strip():
        parts.append(character_system_prompt.strip())
    if self_history.strip():
        parts.append(f"## あなたの歩み（self_history）\n\n{self_history.strip()}")
    if relationship_state.strip():
        parts.append(f"## 今の関係（relationship_state）\n\n{relationship_state.strip()}")
    if inner_narrative.strip():
        parts.append(f"## あなた自身の物語（inner_narrative）\n\n{inner_narrative.strip()}")
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts) + suffix
