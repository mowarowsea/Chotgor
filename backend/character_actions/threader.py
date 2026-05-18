"""Threader — ワーキングメモリスレッド操作ツール（post_thread / open_thread）。

Threader クラスと関連定数を一元管理する。
- POST_THREAD_SCHEMA / POST_THREAD_TOOL_DESCRIPTION: スレッド作成・更新ツール
- OPEN_THREAD_SCHEMA / OPEN_THREAD_TOOL_DESCRIPTION: スレッド詳細展開ツール
- Threader クラス: WorkingMemoryManager に委譲してスレッド・ポストを操作する

post_thread は「新規作成」「ポスト追加」「summary/atmosphere/importance 更新」を
1ツールで兼ねる統合設計。thread_id を省略すると新規作成、指定すると既存スレッドの更新になる。
inscriber.py / drifter.py と対称的な構成。
"""

import json
import logging

logger = logging.getLogger(__name__)

# --- ツール呼び出し方式: post_thread パラメータスキーマ ---
POST_THREAD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thread_id": {
            "type": "string",
            "description": (
                "更新対象スレッドのID。省略すると新規スレッドを作成する。"
                "既存スレッドへのポスト追加・要約更新時は、ワーキングメモリ一覧に"
                "表示されているIDを指定する。"
            ),
        },
        "type": {
            "type": "string",
            "enum": ["emotion", "body", "task", "topic", "relation"],
            "description": (
                "新規作成時のスレッド種別（thread_id 省略時は必須）。"
                "task: 取り組み中の課題（解決を目指す） / "
                "topic: 引っかかっている話題・問い（解決を目指す） / "
                "emotion: 持続的な感情状態のサマリ（1本のみ・解決を目指さない） / "
                "body: 持続的な身体状態のサマリ（1本のみ・解決を目指さない） / "
                "relation: 特定の相手との関係（相手ごと1本・解決を目指さない）"
            ),
        },
        "summary": {
            "type": "string",
            "description": "スレッドのタイトル・要約。新規作成時は必須。更新時は指定すると上書きする。",
        },
        "atmosphere": {
            "type": "string",
            "description": (
                "スレッドの質感の自由記述。アクティブなら今の温度感、"
                "閉じる段階ならどんな終わり方をしたか。"
            ),
        },
        "importance": {
            "type": "number",
            "description": "重要度 0.0〜1.0。前景に置きたいほど高く。",
        },
        "content": {
            "type": "string",
            "description": "スレッドに追加する新しいポストの本文。時系列の書き込みとして連なる。",
        },
        "relation_target": {
            "type": "string",
            "description": "relation 型スレッド新規作成時の相手の名前（ユーザ・他キャラクター）。",
        },
    },
    "required": [],
}

# --- ツール呼び出し方式: open_thread パラメータスキーマ ---
OPEN_THREAD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thread_id": {
            "type": "string",
            "description": "詳細を読みたいスレッドのID。",
        },
    },
    "required": ["thread_id"],
}

# --- ツール呼び出し方式: ツール説明文 ---
POST_THREAD_TOOL_DESCRIPTION: str = (
    "ワーキングメモリのスレッドを作成・更新する。"
    "気になっている課題や話題、持続的な感情・身体状態、相手との関係を、"
    "並行する認知ストリーム（スレッド）として記録する。"
    "thread_id を省略すれば新規スレッド作成、指定すれば既存スレッドへのポスト追加や要約更新になる。"
)

OPEN_THREAD_TOOL_DESCRIPTION: str = (
    "ワーキングメモリのスレッド1本の詳細（全ポストの履歴）を展開して読む。"
    "経緯を詳しく思い出したいときに使う。"
)


class Threader:
    """ワーキングメモリスレッドの操作を担うクラス（ツール呼び出し方式）。

    post_thread / open_thread を WorkingMemoryManager に委譲する。

    Attributes:
        character_id: 操作対象のキャラクターID。
        working_memory_manager: ワーキングメモリの読み書きを担うマネージャー。
    """

    def __init__(self, character_id: str, working_memory_manager) -> None:
        """Threader を初期化する。

        Args:
            character_id: 操作対象のキャラクターID。
            working_memory_manager: WorkingMemoryManager インスタンス（None ならスキップ）。
        """
        self.character_id = character_id
        self.working_memory_manager = working_memory_manager

    def post_thread(
        self,
        thread_id: str = "",
        type: str = "",
        summary: str = "",
        atmosphere: str = "",
        importance: float | None = None,
        content: str = "",
        relation_target: str = "",
    ) -> str:
        """スレッドを作成・更新する（ポスト追加・要約更新を兼ねる）。

        thread_id が空なら新規作成、指定があれば既存スレッドの更新。

        Returns:
            実行結果メッセージ。新規作成時は作成されたスレッドIDを含む。
        """
        wm = self.working_memory_manager
        if wm is None:
            return "ワーキングメモリは利用できない。"
        try:
            if not thread_id:
                # 新規作成
                if not type:
                    return "[post_thread error: 新規作成には type が必要です]"
                if not summary:
                    return "[post_thread error: 新規作成には summary が必要です]"
                thread = wm.create_thread(
                    character_id=self.character_id,
                    type=type,
                    summary=summary,
                    atmosphere=atmosphere,
                    importance=importance if importance is not None else 0.5,
                    relation_target=relation_target or None,
                    content=content or None,
                )
                return f"スレッドを作成した（id={thread['id']}）。"

            # 既存スレッドの更新
            updated_fields = []
            if summary or atmosphere or importance is not None:
                wm.update_thread(
                    thread_id,
                    summary=summary or None,
                    atmosphere=atmosphere or None,
                    importance=importance,
                )
                updated_fields.append("内容")
            if content:
                wm.add_post(thread_id, content)
                updated_fields.append("ポスト")
            if not updated_fields:
                return "[post_thread error: 更新する内容（summary/atmosphere/importance/content）がありません]"
            return f"スレッド（id={thread_id}）を更新した。"
        except ValueError as e:
            # 本数制約違反・スレッド不在など、キャラクターに伝えるべきエラー
            return f"[post_thread error: {e}]"
        except Exception as e:
            logger.exception("post_thread 失敗 char=%s", self.character_id)
            return f"[post_thread error: {e}]"

    def open_thread(self, thread_id: str) -> str:
        """スレッド1本の詳細（全ポスト）を JSON テキストで返す。

        Returns:
            Thread + Posts を表す JSON 文字列。存在しなければエラーメッセージ。
        """
        wm = self.working_memory_manager
        if wm is None:
            return "ワーキングメモリは利用できない。"
        if not thread_id:
            return "[open_thread error: thread_id が空です]"
        try:
            detail = wm.get_thread_detail(thread_id)
        except Exception as e:
            logger.exception("open_thread 失敗 char=%s", self.character_id)
            return f"[open_thread error: {e}]"
        if detail is None:
            return f"[open_thread: スレッド '{thread_id}' が見つかりません]"
        return json.dumps(detail, ensure_ascii=False, indent=2)
