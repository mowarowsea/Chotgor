"""Retrace: 過去に書いたワーキングメモリの「いつ」を辿り直す単発バッチ。

夜間バッチではない。**キャラクター本人に頼んで一度だけ走らせる後始末**である。

--- なぜ要るか ---
ワーキングメモリは書いた瞬間の「今」から書かれるが、読むのは何日も先の本人である。
そのため「今週」「昨日」「最近」のような相対表現は、書いた時点の意味を失ったまま
Open スレッドに残り、毎ターン現在形の事実として注入され続ける。
実際に「8/10〜8/14はお盆休み」が日付を持たないまま残り、翌週・翌々週も休みだと
認識される事故が起きた。

書き方のルール（相対表現を日付で書く）は Chronicle プロンプトと
post_working_memory_thread のツール説明へ入れたが、**それは以後に書くぶんにしか
効かない**。すでに書かれてしまったスレッドは本人に直してもらう必要がある。
記憶の書き換えは開発者ではなくキャラクター本人が行う、というのが Chotgor の前提。

--- 何をするか ---
Open スレッドを、各ポストの書かれた日付付きで本人へ提示し、
「この『今週』はいつのことだったのか」を思い出して summary / 新規ポストへ
日付で書き直してもらう。過去ポストの本文自体は追記型のため書き換えない
（日常視界へ出るのは summary と最新ポストなので、そこが直れば実害は消える）。

推測での捏造は明示的に禁じる。思い出せないものは「いつのことか思い出せない」と
書いて構わない、という逃げ道を必ず残す。

--- 実行 ---
スケジューラーには載せない。`POST /api/inscribed_memories/{character_id}/retrace`
から手動で叩く。片付いた後も残しておいて構わない（何度走らせても、直すものが
無ければ本人が「変更なし」を返すだけ）。
"""

from __future__ import annotations

import logging
from datetime import datetime

from backend.batch.chronicle_job import (
    _apply_working_memory_updates,
    _parse_chronicle_response,
)
from backend.character_actions.executor import ToolExecutor
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.character_query import ask_character
from backend.services.memory.format import origin_label_prefix, short_date, short_thread_id
from backend.services.memory.manager import InscribedMemoryManager
from backend.services.memory.working_memory_manager import WorkingMemoryManager

logger = logging.getLogger(__name__)

# 1スレッドあたりプロンプトへ載せるポストの上限。古いものから落とす。
# 「いつのことか」を辿るには履歴が要るが、Open スレッド全件ぶんの全ポストは
# 長大になりうるため上限を置く（落としたぶんは件数だけ伝える）。
_MAX_POSTS_PER_THREAD = 30


_PROMPT_TEMPLATE = """\
# {character_name}のワーキングメモリ — 時制の辿り直し
今日は {today} です。

いつもの棚卸しとは別に、一度だけお願いしたいことがあります。

あなたのワーキングメモリには、「今週」「昨日」「最近」「来月」のような、
**書いた時点を知らないと意味が定まらない言い方**が残っています。
これらは書いた日から離れるほど指す先がずれていき、やがて「今のこと」として
読まれてしまいます。実際に、期間限定だったはずの予定が何週間も続いていると
思い違いをしたことがありました。

そこで、下に**それぞれのポストが書かれた日付**を添えました。
これを手がかりに、「この『今週』はいつの週だったのか」を思い出して、
日付の形へ書き直してください。

## お願いしたいこと
1. summary に相対表現が残っていたら、日付へ直す。
   例:「今週はお盆休みで一週間有休」→「8/10(月)〜8/14(金)はお盆休みで有休」
   例:「昨日話したこと」→「8/29(土)に話したこと」
2. summary だけでは直しきれない場合は、new_post で「あれは◯月◯日のことだった」と
   書き添える。日常あなたの目に入るのは summary と最新ポストなので、そこが
   直っていれば十分です。過去のポスト本文は書き換えなくて構いません。
3. すでに終わったことなのに Open のままなら、is_open を false にして構いません
   （終わった予定が現在形で残っているのが、そもそもの問題なので）。
4. 誰との・どこでの出来事だったかも思い出せるなら、一言添えてください。

## 大事なこと
**思い出せないものは、無理に日付を決めないでください。**
辻褄を合わせるために日付をでっち上げるくらいなら、
「いつのことか思い出せない」と書いてあるほうがずっとましです。
それはあなたの記憶であって、正しさを取り繕う対象ではありません。

継続的な状態（感情・身体・関係の厚み）には、そもそも日付が要りません。
日付が要るのは「いつからいつまで」がある出来事・予定のほうです。
直すところが無いスレッドは、触らないでください。

## 現在 Open なスレッド（ポストは書かれた日付つき）
{threads}

---

## 出力フォーマット（JSON のみ。直すものが無ければ thread_updates を空配列に）

{{
  "thread_updates": [
    {{"id": "<既存スレッドID>", "summary": null, "atmosphere_tag": null, "new_post": null, "is_open": null}}
  ]
}}
"""


def _format_threads_with_posts(threads: list[dict], wm: WorkingMemoryManager) -> str:
    """Open スレッドを「全ポスト＋書かれた日付」つきで整形する。

    通常の棚卸し（chronicle_job._format_threads）は最新ポストしか出さないが、
    辿り直しでは「その相対表現がいつ書かれたか」が唯一の手がかりになるため、
    履歴を日付つきで開く。
    """
    if not threads:
        return "（Open なスレッドはありません）"
    lines: list[str] = []
    for t in threads:
        origin_prefix = origin_label_prefix(t.get("origin"))
        lines.append(
            f"[{short_thread_id(t['id'])}] {origin_prefix}({t.get('type', '')}) "
            f"{t.get('summary', '')}"
        )
        started = short_date(t.get("created_at"))
        if started:
            lines.append(f"  開始: {started}")
        atmo = (t.get("atmosphere_tag") or "").strip()
        if atmo:
            lines.append(f"  雰囲気: {atmo}")
        detail = wm.get_thread_detail(t["id"]) or {}
        posts = detail.get("posts") or []
        if len(posts) > _MAX_POSTS_PER_THREAD:
            lines.append(f"  （古いポスト {len(posts) - _MAX_POSTS_PER_THREAD} 件は省略）")
            posts = posts[-_MAX_POSTS_PER_THREAD:]
        for post in posts:
            posted = short_date(post.get("created_at"))
            prefix = f"[{posted}] " if posted else ""
            body = (post.get("content") or "").strip().replace("\n", " ")
            lines.append(f"  - {prefix}{body}")
        lines.append("")
    return "\n".join(lines).rstrip()


async def run_retrace(
    character_id: str,
    sqlite: SQLiteStore,
    *,
    settings: dict | None = None,
    memory_manager: InscribedMemoryManager | None = None,
    working_memory_manager: WorkingMemoryManager | None = None,
) -> dict:
    """時制の辿り直しを1回実行する。

    Open スレッドをポストの日付つきで本人へ提示し、相対表現を日付へ直してもらう。
    反映は Chronicle と同じ `_apply_working_memory_updates` を通す（tool_call_events へ
    source="retrace" で記録されるので、Logs 画面で何を直したか追える）。

    thread_updates 以外（new_threads / merges / inscribe）は、本人が書いてきても
    捨てる。この作業の役目は書き直しであって、新しい記憶を作ることではない。

    Returns:
        処理結果辞書 {status, counts, error (optional)}。
    """
    char = sqlite.get_character(character_id)
    if not char:
        return {"status": "error", "error": f"Character '{character_id}' not found"}
    if not char.ghost_model:
        return {"status": "skipped", "reason": "ghost_model が未設定のためスキップ"}
    if working_memory_manager is None:
        return {"status": "error", "error": "working_memory_manager が渡されていません"}

    char_label = f"{char.name}@GhostModel"
    open_threads = working_memory_manager.list_threads_by_type(character_id, is_open=True)
    if not open_threads:
        return {"status": "success", "counts": {}, "reason": "Open なスレッドがありません"}

    prompt_text = _PROMPT_TEMPLATE.format(
        character_name=char.name,
        today=short_date(datetime.now()),
        threads=_format_threads_with_posts(open_threads, working_memory_manager),
    )

    if settings is None:
        settings = sqlite.get_all_settings()
    try:
        response_text = await ask_character(
            character_id=character_id,
            preset_id=char.ghost_model,
            messages=[{"role": "user", "content": prompt_text}],
            sqlite=sqlite,
            settings=settings,
            recall_query=None,
            feature_label="retrace",
            working_memory_manager=working_memory_manager,
        )
    except Exception as e:
        logger.exception("エラー char=%s", char_label)
        return {"status": "error", "error": str(e)}
    if response_text is None:
        return {"status": "error", "error": "LLMからの応答が取得できませんでした"}

    parsed = _parse_chronicle_response(response_text)
    if parsed is None:
        logger.info("変更なし（null応答） char=%s", char_label)
        return {"status": "success", "counts": {}}
    if not parsed:
        logger.warning("JSONパース失敗 char=%s raw=%.100s", char_label, response_text)
        return {"status": "error", "error": "JSON のパースに失敗しました", "raw": response_text[:500]}

    executor = ToolExecutor(
        character_id=character_id,
        session_id=None,
        memory_manager=memory_manager,
        working_memory_manager=working_memory_manager,
        default_origin="real",
        source_preset_id=char.ghost_model or "",
    )
    # 書き直しのみを通す。新規作成・統合・昇格はこの作業の役目ではない。
    counts = _apply_working_memory_updates(
        character_id,
        {"thread_updates": parsed.get("thread_updates") or []},
        executor,
        source="retrace",
    )
    logger.info("時制の辿り直し完了 char=%s counts=%s", char_label, counts)
    return {"status": "success", "counts": counts}
