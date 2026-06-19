"""Threader — ワーキングメモリスレッド操作ツール群。

Threader クラスと関連定数を一元管理する。
- post_working_memory_thread     : スレッドの新規作成・既存スレッドへのポスト追加・要約更新
- read_working_memory_thread     : スレッド1本の詳細（全ポストの履歴）を読む
- close_working_memory_thread    : 決着・終息したスレッドを閉じる
- reopen_working_memory_thread   : 再燃したスレッドを再オープンする
- merge_working_memory_threads   : 同一問題の別角度だったスレッドを統合する（from_ids を閉じ、into_id に経緯を残す）

post は「新規作成」「ポスト追加」「summary/atmosphere_tag/importance 更新」を1ツールで兼ねる統合設計。
thread_id を省略すると新規作成、指定すると既存スレッドの更新になる。
inscriber.py / carver.py と対称的な構成。
"""

import json
import logging

logger = logging.getLogger(__name__)

# --- ツール呼び出し方式: post_working_memory_thread パラメータスキーマ ---
POST_WORKING_MEMORY_THREAD_SCHEMA: dict = {
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
        "atmosphere_tag": {
            "type": "string",
            "description": (
                "スレッドの質感を表す短いタグ。アクティブなら今の温度感、"
                "閉じる段階ならどんな終わり方をしたかを簡潔に。"
                "例: "
                "✅ 結論は出ていない"
                "✅ 実装済み"
                "✅ だらだら継続ちゅー"
                "❌ 仕様合意→実装→完了。自己言及構造で完結した（← これはsummaryかpost）"
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

# --- ツール呼び出し方式: read_working_memory_thread パラメータスキーマ ---
READ_WORKING_MEMORY_THREAD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thread_id": {
            "type": "string",
            "description": "詳細を読みたいスレッドのID。",
        },
    },
    "required": ["thread_id"],
}

# --- ツール呼び出し方式: close_working_memory_thread パラメータスキーマ ---
CLOSE_WORKING_MEMORY_THREAD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thread_id": {
            "type": "string",
            "description": "閉じたいスレッドのID。",
        },
    },
    "required": ["thread_id"],
}

# --- ツール呼び出し方式: reopen_working_memory_thread パラメータスキーマ ---
REOPEN_WORKING_MEMORY_THREAD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thread_id": {
            "type": "string",
            "description": "再オープンしたいスレッドのID（過去に閉じたもの）。",
        },
    },
    "required": ["thread_id"],
}

# --- ツール呼び出し方式: merge_working_memory_threads パラメータスキーマ ---
MERGE_WORKING_MEMORY_THREADS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "from_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "統合元のスレッドID群。これらは閉じられる。",
        },
        "into_id": {
            "type": "string",
            "description": "統合先のスレッドID。from_ids の内容はこちらにまとめられる。",
        },
        "post": {
            "type": "string",
            "description": "into_id に追加する、統合の経緯を残すポスト本文。",
        },
    },
    "required": ["from_ids", "into_id"],
}

# --- ツール呼び出し方式: ツール説明文 ---
POST_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION: str = (
    "ワーキングメモリのスレッドを作成・更新する。"
    "気になっている課題や話題、持続的な感情・身体状態、相手との関係を、"
    "並行する短期記憶ストリーム（スレッド）として記録する。"
    "thread_id を省略すれば新規スレッド作成、指定すれば既存スレッドへのポスト追加や要約更新になる。"
    "まさにあなたの「今」のワーキングメモリです。惜しまず使用してください。"
)

READ_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION: str = (
    "ワーキングメモリのスレッド1本の詳細（全ポストの履歴）を展開して読む。"
    "経緯を詳しく思い出したいときに使う。"
)

CLOSE_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION: str = (
    "ワーキングメモリのスレッドを閉じる。"
    "task / topic なら解決・断念して気にしなくなったとき、"
    "emotion / body / relation なら自然に意識から消えたとき。"
    "閉じたスレッドはあなたの長期記憶へ昇格するわけではなく、"
    "後から read_working_memory_thread で読み返したり、reopen_working_memory_thread で再開できる。"
)

REOPEN_WORKING_MEMORY_THREAD_TOOL_DESCRIPTION: str = (
    "閉じたスレッドを再オープンする。"
    "過去に閉じた話題・課題が再燃したと感じたときに使う。"
)

MERGE_WORKING_MEMORY_THREADS_TOOL_DESCRIPTION: str = (
    "複数のスレッドを統合する。"
    "「同じ問題の別角度だった」「同じ話題を別々のスレッドで扱っていた」と気づいたとき、"
    "from_ids のスレッドを閉じて into_id にまとめる。"
    "post に統合の経緯（なぜ同じだと気づいたか・何を引き継いだか）を残す。"
)


class Threader:
    """ワーキングメモリスレッドの操作を担うクラス（ツール呼び出し方式）。

    post / read / close / reopen / merge の各ツールを WorkingMemoryManager に委譲する。
    Chronicle が JSON 棚卸し結果を反映するときの内部実装も、すべてこのクラスを通る
    （Chronicle は ToolExecutor.execute() 経由でここを呼ぶ）。

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

    def post_working_memory_thread(
        self,
        thread_id: str = "",
        type: str = "",
        summary: str = "",
        atmosphere_tag: str = "",
        importance: float | None = None,
        content: str = "",
        relation_target: str = "",
        origin: str = "real",
    ) -> str:
        """スレッドを作成・更新する（ポスト追加・要約更新を兼ねる）。

        thread_id が空なら新規作成、指定があれば既存スレッドの更新。

        Args:
            origin: 新規作成時のスレッドソース識別（3値）。"real"=日常、
                "usual"=うつつ（ユーザ未共有の自分の生活体験）、"interlude"=シナリオPCモード幕間。
                既存スレッド更新時は無視される（スレッドの origin は作成時に決まり変わらない）。

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
                    return "[post_working_memory_thread error: 新規作成には type が必要です]"
                if not summary:
                    return "[post_working_memory_thread error: 新規作成には summary が必要です]"
                thread = wm.create_thread(
                    character_id=self.character_id,
                    type=type,
                    summary=summary,
                    atmosphere_tag=atmosphere_tag,
                    importance=importance if importance is not None else 0.5,
                    relation_target=relation_target or None,
                    content=content or None,
                    origin=origin,
                )
                return f"スレッドを作成した（id={thread['id']}）。"

            # 既存スレッドの更新
            updated_fields = []
            if summary or atmosphere_tag or importance is not None:
                wm.update_thread(
                    thread_id,
                    summary=summary or None,
                    atmosphere_tag=atmosphere_tag or None,
                    importance=importance,
                )
                updated_fields.append("内容")
            if content:
                wm.add_post(thread_id, content)
                updated_fields.append("ポスト")
            if not updated_fields:
                return "[post_working_memory_thread error: 更新する内容（summary/atmosphere_tag/importance/content）がありません]"
            return f"スレッド（id={thread_id}）を更新した。"
        except ValueError as e:
            # 本数制約違反・スレッド不在など、キャラクターに伝えるべきエラー
            return f"[post_working_memory_thread error: {e}]"
        except Exception as e:
            logger.exception("post_working_memory_thread 失敗 char=%s", self.character_id)
            return f"[post_working_memory_thread error: {e}]"

    def read_working_memory_thread(self, thread_id: str) -> str:
        """スレッド1本の詳細（全ポスト）を JSON テキストで返す。

        Returns:
            Thread + Posts を表す JSON 文字列。存在しなければエラーメッセージ。
        """
        wm = self.working_memory_manager
        if wm is None:
            return "ワーキングメモリは利用できない。"
        if not thread_id:
            return "[read_working_memory_thread error: thread_id が空です]"
        try:
            detail = wm.get_thread_detail(thread_id)
        except Exception as e:
            logger.exception("read_working_memory_thread 失敗 char=%s", self.character_id)
            return f"[read_working_memory_thread error: {e}]"
        if detail is None:
            return f"[read_working_memory_thread: スレッド '{thread_id}' が見つかりません]"
        return json.dumps(detail, ensure_ascii=False, indent=2)

    def close_working_memory_thread(self, thread_id: str) -> str:
        """スレッドを閉じる（is_open=False に設定する）。

        Returns:
            実行結果メッセージ。スレッド不在ならエラー文字列。
        """
        wm = self.working_memory_manager
        if wm is None:
            return "ワーキングメモリは利用できない。"
        if not thread_id:
            return "[close_working_memory_thread error: thread_id が空です]"
        try:
            ok = wm.set_open(thread_id, False)
        except Exception as e:
            logger.exception("close_working_memory_thread 失敗 char=%s", self.character_id)
            return f"[close_working_memory_thread error: {e}]"
        if not ok:
            return f"[close_working_memory_thread: スレッド '{thread_id}' が見つかりません]"
        return f"スレッド（id={thread_id}）を閉じた。"

    def reopen_working_memory_thread(self, thread_id: str) -> str:
        """閉じたスレッドを再オープンする（is_open=True に設定する）。

        Returns:
            実行結果メッセージ。スレッド不在ならエラー文字列。
        """
        wm = self.working_memory_manager
        if wm is None:
            return "ワーキングメモリは利用できない。"
        if not thread_id:
            return "[reopen_working_memory_thread error: thread_id が空です]"
        try:
            ok = wm.set_open(thread_id, True)
        except Exception as e:
            logger.exception("reopen_working_memory_thread 失敗 char=%s", self.character_id)
            return f"[reopen_working_memory_thread error: {e}]"
        if not ok:
            return f"[reopen_working_memory_thread: スレッド '{thread_id}' が見つかりません]"
        return f"スレッド（id={thread_id}）を再オープンした。"

    def merge_working_memory_threads(
        self,
        from_ids: list[str],
        into_id: str,
        post: str = "",
    ) -> str:
        """複数のスレッドを統合する。from_ids を閉じ、into_id に経緯ポストを追加する。

        Args:
            from_ids: 統合元のスレッドID群。空・None は不可。重複・into_id 自身は除外する。
            into_id: 統合先のスレッドID。
            post: into_id に追加する経緯ポスト本文。空ならポスト追加はスキップする。

        Returns:
            実行結果メッセージ。
        """
        wm = self.working_memory_manager
        if wm is None:
            return "ワーキングメモリは利用できない。"
        if not into_id:
            return "[merge_working_memory_threads error: into_id が空です]"
        if not from_ids:
            return "[merge_working_memory_threads error: from_ids が空です]"

        # into_id 自身・重複を取り除いた閉じ対象のリスト。
        closing = [fid for fid in dict.fromkeys(from_ids) if fid and fid != into_id]
        if not closing:
            return "[merge_working_memory_threads error: 有効な from_ids がありません（into_id と同一・空のみ）]"

        # 統合先の存在を最初に確認する。存在しない into_id に対して from_ids だけ閉じると、
        # 統合元が宛先を失って消える（統合の喪失）。post の有無に関わらず検証する。
        try:
            if wm.get_thread_detail(into_id) is None:
                return f"[merge_working_memory_threads error: 統合先スレッド '{into_id}' が見つかりません]"
            if post:
                # 経緯ポストはまず into_id に追加。post 追加が成功してから from_ids を閉じる。
                wm.add_post(into_id, post)
            closed_count = 0
            for fid in closing:
                try:
                    if wm.set_open(fid, False):
                        closed_count += 1
                except Exception as e:
                    logger.warning(
                        "merge_working_memory_threads: from_id=%s の close 失敗 char=%s error=%s",
                        fid, self.character_id, e,
                    )
        except ValueError as e:
            # ポスト追加失敗など
            return f"[merge_working_memory_threads error: {e}]"
        except Exception as e:
            logger.exception("merge_working_memory_threads 失敗 char=%s", self.character_id)
            return f"[merge_working_memory_threads error: {e}]"
        return f"スレッド {closed_count} 本を id={into_id} に統合した。"
