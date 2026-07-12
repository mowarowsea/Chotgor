"""WebSearcher — Tavily API を介した web_search ツールの実装。

キャラクターが「自分で外の情報を取りに行く」ための手段を提供する。
記憶 (inscribe_memory) や想起 (power_recall) とは別軸で、自分の頭の外側
（インターネット）にしか答えがない事実を調べるためのツール。

API キーは SQLite の global_settings テーブル ``tavily_api_key`` から読む。
キーが未設定なら呼び出し時にエラーメッセージを返す（落ちない）。

- WEB_SEARCH_SCHEMA: ツール呼び出し方式のパラメータ JSON スキーマ
- WEB_SEARCH_TOOL_DESCRIPTION: tool-use プロバイダー向けツール説明文
- WEB_SEARCH_TOOLS_HINT: tool-use プロバイダー向けシステムプロンプト案内ブロック
- WebSearcher クラス: 実際の Tavily 呼び出しと結果整形を担う

タグ方式フォールバックは現時点では提供しない（MCP ネイティブ tool-use の利用を前提）。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# --- ツール呼び出し方式: パラメータスキーマ ---
WEB_SEARCH_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Web 検索クエリ。自然文または検索キーワードで指定する。",
        },
        "max_results": {
            "type": "integer",
            "description": "取得する検索結果の最大件数。デフォルト5、上限10。",
            "default": 5,
        },
        "topic": {
            "type": "string",
            "enum": ["general", "news"],
            "description": (
                "検索トピック。'general'（既定）で通常検索、"
                "'news' で最新ニュース寄りの検索になる。"
            ),
            "default": "general",
        },
    },
    "required": ["query"],
}

# --- ツール呼び出し方式: ツール説明文（ANTHROPIC_TOOLS / OPENAI_TOOLS に使用）---
WEB_SEARCH_TOOL_DESCRIPTION: str = (
    "インターネットを検索して外部情報を取得する。"
    "自分の記憶・知識にない事実を確認したいとき、または最新情報（ニュース・価格・天気・"
    "リリース情報など）が必要なときに使う。"
    "power_recall は自分の記憶の中を、web_search は自分の頭の外側を探す手段。"
)

# --- ツール呼び出し方式: システムプロンプト案内ブロック ---
WEB_SEARCH_TOOLS_HINT: str = """\
### 外部情報の検索（web_search）
自分の記憶・知識にない事実、または最新情報（ニュース・価格・天気・リリース情報など）が
必要なときに `web_search` ツールでインターネットを検索できます。
- `power_recall` は**自分の記憶の中**を、`web_search` は**自分の頭の外側**を探す手段です
- 自分の知識で答えられる質問にまで使う必要はありません
- 結果が届いたら、自分の言葉で要点を伝えてください\
"""


def _format_results(payload: dict[str, Any], query: str) -> str:
    """Tavily の検索結果ペイロードを LLM 向けテキストに整形する。

    Args:
        payload: Tavily API のレスポンス dict。
        query: 検索クエリ（結果ヘッダ表示用）。

    Returns:
        LLM のツール結果として返すテキスト。
    """
    lines: list[str] = [f"【Web Search 検索結果】 クエリ: 「{query}」"]

    answer = (payload.get("answer") or "").strip()
    if answer:
        lines.append("")
        lines.append("▼ 要約（Tavily 回答）")
        lines.append(answer)

    results = payload.get("results") or []
    if not results:
        if not answer:
            lines.append("")
            lines.append("該当する検索結果は見つかりませんでした。")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"▼ 検索結果 ({len(results)}件)")
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "(no title)").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        # スニペットが長すぎるとトークンを圧迫するため、ある程度で切る
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"  {i}. {title}")
        if url:
            lines.append(f"     {url}")
        if content:
            lines.append(f"     {content}")

    return "\n".join(lines)


class WebSearcher:
    """Tavily API を介した Web 検索を担うクラス。

    API キーは初期化時ではなく ``search()`` 呼び出し時に毎回 SQLite から読み出す。
    設定 UI からの差し替えに即時追従するため。

    Attributes:
        sqlite_store: ``tavily_api_key`` を読み出すための SQLiteStore。
    """

    # 1 回の検索で許容する max_results の上限（ツール経由の暴走防止）
    _MAX_RESULTS_CAP = 10
    # Tavily API へのタイムアウト秒
    _TIMEOUT_SEC = 20.0

    def __init__(self, sqlite_store) -> None:
        """WebSearcher を初期化する。

        Args:
            sqlite_store: ``tavily_api_key`` を ``get_setting`` で取り出せる SQLiteStore。
        """
        self.sqlite_store = sqlite_store

    def search(
        self,
        query: str,
        max_results: int = 5,
        topic: str = "general",
    ) -> str:
        """Tavily で Web 検索を行い、整形済みテキストを返す。

        Args:
            query: 検索クエリ。空文字の場合はエラーメッセージを返す。
            max_results: 取得する結果件数。1〜``_MAX_RESULTS_CAP`` にクランプされる。
            topic: 検索トピック（"general" / "news"）。不正値は "general" に丸める。

        Returns:
            LLM のツール結果として渡す整形済みテキスト。
            API キー未設定・呼び出し失敗時もテキスト（``[web_search error: ...]`` 風）を返し、
            例外で落ちないようにする。
        """
        query = (query or "").strip()
        if not query:
            return "[web_search: query が空です]"

        # ガード: 件数を妥当範囲に丸める
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 5
        n = max(1, min(self._MAX_RESULTS_CAP, n))

        topic = topic if topic in ("general", "news") else "general"

        api_key = (self.sqlite_store.get_setting("tavily_api_key") or "").strip()
        if not api_key:
            logger.warning("tavily_api_key 未設定のため web_search を実行できません")
            return (
                "[web_search: Tavily API キーが未設定です。"
                "Chotgor の Settings 画面で tavily_api_key を設定してください]"
            )

        # tavily-python は pyproject に依存として宣言済みだが、保険として import 失敗時も明示
        try:
            from tavily import TavilyClient  # type: ignore[import-not-found]
        except Exception as e:
            logger.exception("tavily-python の import に失敗")
            return f"[web_search error: tavily ライブラリが利用できません ({type(e).__name__}: {e})]"

        try:
            client = TavilyClient(api_key=api_key)
            payload = client.search(
                query=query,
                search_depth="basic",
                include_answer="basic",
                max_results=n,
                topic=topic,
                timeout=self._TIMEOUT_SEC,
            )
        except Exception as e:
            logger.exception("Tavily 呼び出し失敗 query=%.80s", query)
            return f"[web_search error: {type(e).__name__}: {e}]"

        if not isinstance(payload, dict):
            logger.warning("Tavily 応答が dict ではない: %r", type(payload))
            return "[web_search error: Tavily 応答の形式が想定外です]"

        logger.info(
            "完了 query=%.80s results=%d has_answer=%s",
            query,
            len(payload.get("results") or []),
            bool((payload.get("answer") or "").strip()),
        )
        return _format_results(payload, query)
