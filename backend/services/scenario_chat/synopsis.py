"""シナリオチャット用 「あらすじ」自動蒸留モジュール。

ScenarioChat は長く続けると、スライディングウィンドウで履歴が切り捨てられて
GM (LLM) が経緯を忘れ、存在しない過去を捏造してしまう問題がある。
これへの対策として、本モジュールは「これから LLM に渡らなくなるターン群」を
既存のあらすじと統合し、物語全体を 1 つのあらすじへ**再蒸留**する。

設計方針:
    - 自動更新は `synopsis_auto` のみ。`synopsis_manual`（プレイヤー手書きメモ）
      には一切触らない。プレイヤーの補足はそちらで保持される。
    - `synopsis_auto` は**全体再蒸留モード**で書き換える。単純な追記ではなく、
      既存あらすじ + 新ターンを LLM に渡し、全体を蒸留し直した結果で置き換える。
      これにより追記による肥大化を防ぐ。
    - 蒸留は「直近を厚く・古い経緯は薄く」。ただし古い経緯でも
      「誰が・何をして・何が確定したか」の事実関係は省かない（薄くしても消さない）。
    - 失敗時は None を返し、呼び出し元で best-effort 扱い（チャット本体は継続）。
"""

import logging
from typing import Any, Callable

from backend.providers.registry import create_provider
from backend.services.scenario_chat.context import format_history_for_gm

logger = logging.getLogger(__name__)


def build_synopsis_system_prompt(
    scenario: Any,
    existing_auto: str,
    narrator_name: str = "Narrator",
) -> str:
    """あらすじ蒸留用の system prompt を組み立てる。

    既存のあらすじ（前回までの蒸留結果）と新しい発話履歴を統合し、
    物語全体を 1 つのあらすじへ再蒸留させる方針のプロンプト。
    単純な追記ではなく、古い経緯を圧縮しつつ全体を作り直させる。

    Args:
        scenario: Scenario ORM 風オブジェクト（user_alias / scenario を使う）。
        existing_auto: 既存の synopsis_auto テキスト（空文字列も可）。
        narrator_name: Narrator のタグ名。

    Returns:
        system prompt 文字列。
    """
    user_alias = getattr(scenario, "user_alias", "ユーザ")
    scenario_text = (getattr(scenario, "scenario", "") or "").strip()

    parts = [
        "あなたは物語の経緯を「あらすじ」として蒸留・更新する記録役です。",
        "「これまでのあらすじ」と「新しく加わった発話履歴」が与えられます。",
        "両者を統合し、物語全体を 1 つのあらすじへ作り直してください。",
        "既存あらすじを丸ごとコピーして末尾に足すのではなく、",
        "全体を 1 本のあらすじとして蒸留し直すこと。",
        "",
        "規則:",
        f"- 主役は @{user_alias}（プレイヤー）。プレイヤーの意思決定と行動を軸に描写",
        "- 直近の展開は具体的に、古い経緯は要点だけに圧縮する（新しいほど厚く）",
        "- ただし古い経緯でも「誰が・何をして・何が確定したか」の",
        "  事実関係は省かない（薄くはしても消さない）",
        "- NPC の重要な決定・約束・状況変化・決定的な情報の開示は事実として残す",
        "- 履歴・既存あらすじに無い情報を補わない（推測・創作禁止）",
        "- 時系列順の簡潔な箇条書きでよい（地の文でも可）",
        "- メタ言及（「以上をまとめると」等）禁止",
        "- 全体が長くなってきたら古い経緯をさらに圧縮し、肥大化させないこと",
    ]
    if scenario_text:
        parts.append("")
        parts.append("# 世界・シナリオ")
        parts.append(scenario_text)

    if existing_auto and existing_auto.strip():
        parts.append("")
        parts.append("# これまでのあらすじ（前回までの蒸留結果。これも再蒸留の対象）")
        parts.append(existing_auto.strip())
    return "\n".join(parts)


async def update_auto_synopsis(
    *,
    scenario: Any,
    new_turns: list[Any],
    existing_auto: str,
    settings: dict,
    preset_loader: Callable[[str], Any],
    synopsis_preset_id: str,
    provider_factory: Callable[..., Any] = create_provider,
    narrator_name: str = "Narrator",
) -> str | None:
    """既存 `existing_auto` と `new_turns` を統合し、全体を再蒸留した結果を返す。

    呼び出し元 (service.py) は本関数の戻り値（None でなければ）を
    `update_scenario_session_synopsis(..., auto=戻り値, last_turn_index=...)` で
    SQLite に書き込む。戻り値は既存 auto への追記ではなく、全体の置き換え版。

    Args:
        scenario: Scenario ORM。user_alias / scenario を使う。
        new_turns: 前回蒸留以降の新規ターン群（時系列昇順）。生ログから押し出された
                   ものに限らず、まだ生ログに残っているターンも含めて渡してよい。
                   これによりあらすじは常に生ログ右端まで（または超えて）カバーされ、
                   「最近のことを GM が忘却している」というギャップが発生しない。
        existing_auto: 既存の synopsis_auto テキスト。
        settings: グローバル設定辞書（API キー等）。
        preset_loader: preset_id を受け取り preset ORM 風オブジェクトを返す関数。
        synopsis_preset_id: あらすじ蒸留に使う LLM プリセット ID
                            （ScenarioSession.synopsis_preset_id）。
                            GM 用とは独立に指定でき、レートリミット節約用に
                            軽量モデルを割り当てられる。
        provider_factory: プロバイダ生成関数（デフォルト registry.create_provider）。
        narrator_name: Narrator のタグ名。

    Returns:
        再蒸留後の synopsis_auto 文字列（全体置き換え版）。new_turns が空、
        または蒸留生成に失敗した場合は None を返す（その場合は SQLite を更新しない）。
    """
    if not new_turns:
        logger.debug("synopsis 蒸留 skip 理由=new_turns空")
        return None

    preset = preset_loader(synopsis_preset_id)
    if preset is None:
        logger.warning(
            "synopsis 蒸留 skip 理由=preset_loader が None を返却 synopsis_preset_id=%s",
            synopsis_preset_id,
        )
        return None

    try:
        provider = provider_factory(
            preset.provider,
            model=preset.model_id,
            settings=settings,
            preset_name=preset.name,
            timeout_seconds=getattr(preset, "timeout_seconds", 300),
        )
    except Exception:
        logger.exception(
            "synopsis 蒸留 skip 理由=プロバイダー生成失敗 provider=%s model=%s",
            getattr(preset, "provider", None),
            getattr(preset, "model_id", None),
        )
        return None

    system_prompt = build_synopsis_system_prompt(
        scenario=scenario,
        existing_auto=existing_auto or "",
        narrator_name=narrator_name,
    )

    history_text = format_history_for_gm(
        new_turns,
        user_alias=getattr(scenario, "user_alias", "ユーザ"),
        narrator_name=narrator_name,
    )
    user_content = (
        "以下が、今回新しくあらすじへ統合すべき発話履歴です。\n"
        "「これまでのあらすじ」とこの履歴を統合し、"
        "物語全体のあらすじを蒸留し直してください。\n\n"
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
        logger.exception(
            "synopsis 蒸留 skip 理由=ストリーム生成中に例外 provider=%s model=%s",
            getattr(preset, "provider", None),
            getattr(preset, "model_id", None),
        )
        return None

    new_synopsis = "".join(chunks).strip()
    if not new_synopsis:
        logger.warning(
            "synopsis 蒸留 skip 理由=LLM 出力が空 provider=%s model=%s (refusal/思考のみ等の可能性)",
            getattr(preset, "provider", None),
            getattr(preset, "model_id", None),
        )
        return None

    return new_synopsis
