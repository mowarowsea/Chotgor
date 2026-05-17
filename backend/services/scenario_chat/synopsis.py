"""シナリオチャット用 「あらすじ」自動要約モジュール。

ScenarioChat は長く続けると、スライディングウィンドウで履歴が切り捨てられて
GM (LLM) が経緯を忘れ、存在しない過去を捏造してしまう問題がある。
これへの対策として、本モジュールは「これから LLM に渡らなくなるターン群」を
LLM 自身に要約させ、`zeta_sessions.synopsis_auto` に**追記**していく。

設計方針:
    - 自動更新は `synopsis_auto` のみ。`synopsis_manual` には一切触らない。
    - `synopsis_auto` は**追記モード**で書き込む（既存テキスト + 新規要約）。
      ユーザが手編集した auto の既存記述を破壊しない。
    - 既存 auto は要約 LLM に「コンテキスト」として渡すが、それを書き換えるよう
      指示しない（あくまで「続きを書く」発想）。
    - 失敗時は None を返し、呼び出し元で best-effort 扱い（チャット本体は継続）。
"""

from typing import Any, Awaitable, Callable, Optional

from backend.providers.registry import create_provider
from backend.services.scenario_chat.context import format_history_for_gm


def build_synopsis_system_prompt(
    scenario: Any,
    existing_auto: str,
    narrator_name: str = "Narrator",
) -> str:
    """あらすじ自動要約用の system prompt を組み立てる。

    既存の synopsis_auto をコンテキストとして渡し、その「続きとして」新ターン分の
    要約だけを生成させる方針のプロンプト。既存文の書き換えは禁止する。

    Args:
        scenario: ZetaScenario ORM 風オブジェクト（user_alias / scenario を使う）。
        existing_auto: 既存の synopsis_auto テキスト（空文字列も可）。
        narrator_name: Narrator のタグ名。

    Returns:
        system prompt 文字列。
    """
    user_alias = getattr(scenario, "user_alias", "ユーザ")
    scenario_text = (getattr(scenario, "scenario", "") or "").strip()

    parts = [
        "あなたは物語の語り手が過去のセッション流れを記録するための「あらすじ作成役」です。",
        "与えられた発話履歴を、誰が何を言い・どう動き・何が決まったかが分かる",
        "簡潔なあらすじ文に圧縮してください。",
        "",
        "規則:",
        f"- 主役は @{user_alias}（プレイヤー）。プレイヤーの意思決定と行動を中心に描写",
        "- NPCの重要な台詞・提案、状況変化、決定的な情報の開示は漏らさず含める",
        "- 履歴に無い情報を補わない（推測・創作禁止）",
        "- 簡潔な平文で。箇条書きや markdown 記法は使わない",
        "- 200〜500文字を目安。長さは内容次第で前後してよい",
        "- メタ言及（「以上をまとめると」等）禁止",
    ]
    if scenario_text:
        parts.append("")
        parts.append("# 世界・シナリオ")
        parts.append(scenario_text)

    if existing_auto and existing_auto.strip():
        parts.append("")
        parts.append("# これまでのあらすじ（既存。書き換え禁止・繰り返し禁止）")
        parts.append(existing_auto.strip())
        parts.append("")
        parts.append(
            "上記の「これまでのあらすじ」の続きとして、"
            "次に与えられる新ターン分だけのあらすじを生成してください。"
            "既存の文章を繰り返したり書き直したりしないこと。"
        )
    return "\n".join(parts)


def append_auto_synopsis(existing_auto: str, new_summary: str) -> str:
    """既存 synopsis_auto に新規要約を追記する。

    既存が空ならそのまま、内容があれば空行で区切って末尾に連結する。
    ユーザの手編集記述を保護するため、上書きは絶対に行わない。
    """
    new_summary = (new_summary or "").strip()
    if not new_summary:
        return existing_auto
    existing = (existing_auto or "").rstrip()
    if not existing:
        return new_summary
    return f"{existing}\n\n{new_summary}"


async def update_auto_synopsis(
    *,
    scenario: Any,
    dropped_turns: list[Any],
    existing_auto: str,
    settings: dict,
    preset_loader: Callable[[str], Any],
    provider_factory: Callable[..., Any] = create_provider,
    narrator_name: str = "Narrator",
) -> Optional[str]:
    """`dropped_turns` を LLM で要約し、既存 `existing_auto` に追記した結果を返す。

    呼び出し元 (engine.py) は本関数の戻り値（None でなければ）を
    `update_zeta_session_synopsis(..., auto=戻り値, last_turn_index=...)` で
    SQLite に書き込む。

    Args:
        scenario: ZetaScenario ORM。user_alias / scenario / gm_preset_id を使う。
        dropped_turns: 今回新たに送信対象外となったターン群（時系列昇順）。
        existing_auto: 既存の synopsis_auto テキスト。
        settings: グローバル設定辞書（API キー等）。
        preset_loader: preset_id を受け取り preset ORM 風オブジェクトを返す関数。
        provider_factory: プロバイダ生成関数（デフォルト registry.create_provider）。
        narrator_name: Narrator のタグ名。

    Returns:
        追記後の synopsis_auto 文字列。dropped_turns が空、または要約生成に
        失敗した場合は None を返す（その場合は SQLite を更新しないこと）。
    """
    if not dropped_turns:
        return None

    preset = preset_loader(scenario.gm_preset_id)
    if preset is None:
        return None

    try:
        provider = provider_factory(
            preset.provider,
            model=preset.model_id,
            settings=settings,
            preset_name=preset.name,
        )
    except Exception:
        return None

    system_prompt = build_synopsis_system_prompt(
        scenario=scenario,
        existing_auto=existing_auto or "",
        narrator_name=narrator_name,
    )

    history_text = format_history_for_gm(
        dropped_turns,
        user_alias=getattr(scenario, "user_alias", "ユーザ"),
        narrator_name=narrator_name,
    )
    user_content = (
        "以下が、今回新たにあらすじに加えるべき発話履歴です。\n"
        "この区間だけを要約してください（既存あらすじは繰り返さない）。\n\n"
        f"{history_text}"
    )
    messages = [{"role": "user", "content": user_content}]

    chunks: list[str] = []
    try:
        async for chunk_type, content in provider.generate_stream_typed(
            system_prompt, messages
        ):
            if chunk_type != "text" or not content:
                continue
            chunks.append(content)
    except Exception:
        return None

    new_summary = "".join(chunks).strip()
    if not new_summary:
        return None

    return append_auto_synopsis(existing_auto or "", new_summary)
