"""デバッグログエントリの永続化ストア。

`debug_log_entries` テーブルに対する CRUD 操作を提供する Mixin クラス。
1リクエスト = 1行ではなく、1LLM呼び出し単位で行を作成する。
同一 request_id を持つ行が複数ある場合は同一ユーザーリクエスト内の複数呼び出し
（chat / farewell / trigger 等）またはシナリオの再生成ターンを表す。
"""

import json
import logging
from datetime import datetime
from typing import Optional

_log = logging.getLogger(__name__)


class DebugLogStoreMixin:
    """debug_log_entries テーブルの CRUD を提供する Mixin クラス。"""

    def insert_debug_log_entry(
        self,
        *,
        request_id: str,
        source_type: str,
        session_id: Optional[str] = None,
        turn_sequence: Optional[int] = None,
        target: Optional[str] = None,
        preset: Optional[str] = None,
        user_message: Optional[str] = None,
        response: Optional[str] = None,
        reasoning: Optional[str] = None,
        mcp_calls_json: Optional[str] = None,
        has_error: bool = False,
        warn_reason: Optional[str] = None,
        raw_dir: Optional[str] = None,
    ) -> int:
        """デバッグログエントリを新規 INSERT して返した主キー id を返す。

        Args:
            request_id: リクエスト識別子（8桁hex）。同一ユーザーリクエスト内の複数行で共有。
            source_type: 呼び出し種別（'chat'/'scenario'/'farewell'/'trigger'/'batch' 等）。
            session_id: セッション ID（chat_sessions.id または scenario_sessions.id）。
            turn_sequence: セッション内のターン番号（シナリオ等で使用）。
            target: リクエスト先の識別名（キャラ名 / シナリオ名 / バッチ対象名）。
            preset: 使用したプリセット名。
            user_message: ユーザーの発言テキスト本文（source_type='chat'/'scenario' のみ）。
            response: 応答テキスト本文。
            reasoning: 推論テキスト（思考ブロック等）。
            mcp_calls_json: MCPツール呼び出しのJSON配列文字列。
            has_error: エラーが発生した場合 True。
            warn_reason: 警告・エラーの人間可読な理由文。
            raw_dir: 生ファイルフォルダのパス（null=生ファイルなし）。

        Returns:
            挿入されたレコードの主キー id。
        """
        from backend.repositories.sqlite.store import DebugLogEntry
        with self.get_session() as sess:
            entry = DebugLogEntry(
                request_id=request_id,
                created_at=datetime.now(),
                source_type=source_type,
                session_id=session_id,
                turn_sequence=turn_sequence,
                target=target,
                preset=preset,
                user_message=user_message,
                response=response,
                reasoning=reasoning,
                mcp_calls_json=mcp_calls_json,
                has_error=1 if has_error else 0,
                warn_reason=warn_reason,
                raw_dir=raw_dir,
            )
            sess.add(entry)
            sess.flush()
            entry_id = entry.id
            sess.commit()
            return entry_id

    def update_debug_log_entry(
        self,
        entry_id: int,
        *,
        response: Optional[str] = None,
        reasoning: Optional[str] = None,
        mcp_calls_json: Optional[str] = None,
        has_error: Optional[bool] = None,
        warn_reason: Optional[str] = None,
    ) -> None:
        """指定 id のデバッグログエントリを部分更新する。

        None を渡したフィールドは更新しない。

        Args:
            entry_id: 更新対象のレコード主キー。
            response: 応答テキスト本文。
            reasoning: 推論テキスト。
            mcp_calls_json: MCPツール呼び出しのJSON配列文字列。
            has_error: エラーフラグ。
            warn_reason: 警告・エラーの理由文。
        """
        from backend.repositories.sqlite.store import DebugLogEntry
        with self.get_session() as sess:
            entry = sess.get(DebugLogEntry, entry_id)
            if entry is None:
                return
            if response is not None:
                entry.response = response
            if reasoning is not None:
                entry.reasoning = reasoning
            if mcp_calls_json is not None:
                entry.mcp_calls_json = mcp_calls_json
            if has_error is not None:
                entry.has_error = 1 if has_error else 0
            if warn_reason is not None:
                entry.warn_reason = warn_reason
            sess.commit()

    def get_debug_log_entries_by_request_id(self, request_id: str) -> list[dict]:
        """指定 request_id の全エントリを作成日時昇順で返す。

        Args:
            request_id: リクエスト識別子（8桁hex）。

        Returns:
            エントリ辞書のリスト（作成日時昇順）。
        """
        from backend.repositories.sqlite.store import DebugLogEntry
        from sqlalchemy import asc
        with self.get_session() as sess:
            rows = (
                sess.query(DebugLogEntry)
                .filter(DebugLogEntry.request_id == request_id)
                .order_by(asc(DebugLogEntry.created_at))
                .all()
            )
            return [_entry_to_dict(r) for r in rows]

    def get_debug_log_entries_by_request_ids(
        self, request_ids: list[str]
    ) -> dict[str, list[dict]]:
        """複数の request_id のエントリを IN 句で一括取得して返す。

        N+1 クエリを避けるための一括取得メソッド。
        結果は {request_id: [エントリ辞書, ...]} の形式で、各リストは作成日時昇順。

        Args:
            request_ids: リクエスト識別子のリスト。

        Returns:
            {request_id: エントリ辞書のリスト} の辞書。
        """
        from backend.repositories.sqlite.store import DebugLogEntry
        from sqlalchemy import asc
        if not request_ids:
            return {}
        with self.get_session() as sess:
            rows = (
                sess.query(DebugLogEntry)
                .filter(DebugLogEntry.request_id.in_(request_ids))
                .order_by(asc(DebugLogEntry.created_at))
                .all()
            )
            # request_ids の順序を保持するため先に空リストを用意する
            result: dict[str, list[dict]] = {rid: [] for rid in request_ids}
            for r in rows:
                if r.request_id in result:
                    result[r.request_id].append(_entry_to_dict(r))
            return result

    def get_debug_log_request_ids_paged(
        self, page: int = 1, per_page: int = 50, request_type: Optional[str] = None
    ) -> tuple[list[str], int]:
        """ユニークな request_id を最新順でページネーションして返す。

        同一 request_id の複数行はまとめて1件として扱い、
        その中で最も新しい created_at を代表値としてソートする。

        Args:
            page: ページ番号（1始まり）。
            per_page: 1ページあたりの件数。
            request_type: フィルタ種別。None/'all'=全件, 'chat'=チャット系,
                          'scenario'=シナリオ系, 'batch'=それ以外。

        Returns:
            (request_id のリスト, 総ユニーク件数) のタプル。
        """
        from backend.repositories.sqlite.store import DebugLogEntry
        from sqlalchemy import func, desc

        # タイプ別 source_type セット定義
        _CHAT_TYPES = ("chat", "group_chat", "farewell", "trigger")
        _SCENARIO_TYPES = ("scenario", "scenario_chat")
        _ALL_TYPED = _CHAT_TYPES + _SCENARIO_TYPES

        with self.get_session() as sess:
            base_q = sess.query(DebugLogEntry.request_id)

            if request_type == "chat":
                # chat/group_chat/farewell/trigger のいずれかを含む request_id
                matched_ids = (
                    base_q.filter(DebugLogEntry.source_type.in_(_CHAT_TYPES))
                    .distinct()
                    .subquery()
                )
                filter_subq = matched_ids
            elif request_type == "scenario":
                # scenario/scenario_chat のいずれかを含む request_id
                matched_ids = (
                    base_q.filter(DebugLogEntry.source_type.in_(_SCENARIO_TYPES))
                    .distinct()
                    .subquery()
                )
                filter_subq = matched_ids
            elif request_type == "batch":
                # chat系・scenario系のどちらにも属さない request_id
                typed_ids = (
                    base_q.filter(DebugLogEntry.source_type.in_(_ALL_TYPED))
                    .distinct()
                    .subquery()
                )
                untyped_ids = (
                    base_q.filter(
                        ~DebugLogEntry.request_id.in_(
                            sess.query(typed_ids.c.request_id)
                        )
                    )
                    .distinct()
                    .subquery()
                )
                filter_subq = untyped_ids
            else:
                filter_subq = None

            # ユニーク request_id ごとの最大 created_at を取得
            if filter_subq is not None:
                subq = (
                    sess.query(
                        DebugLogEntry.request_id,
                        func.max(DebugLogEntry.created_at).label("latest_at"),
                    )
                    .filter(
                        DebugLogEntry.request_id.in_(
                            sess.query(filter_subq.c.request_id)
                        )
                    )
                    .group_by(DebugLogEntry.request_id)
                    .subquery()
                )
            else:
                subq = (
                    sess.query(
                        DebugLogEntry.request_id,
                        func.max(DebugLogEntry.created_at).label("latest_at"),
                    )
                    .group_by(DebugLogEntry.request_id)
                    .subquery()
                )

            total = sess.query(func.count()).select_from(subq).scalar() or 0
            offset = (page - 1) * per_page
            rows = (
                sess.query(subq.c.request_id)
                .order_by(desc(subq.c.latest_at))
                .offset(offset)
                .limit(per_page)
                .all()
            )
            ids = [r[0] for r in rows]
            return ids, total


def _entry_to_dict(entry) -> dict:
    """DebugLogEntry ORM オブジェクトを辞書に変換する。"""
    mcp_calls = []
    if entry.mcp_calls_json:
        try:
            mcp_calls = json.loads(entry.mcp_calls_json)
        except (json.JSONDecodeError, ValueError):
            pass
    return {
        "id": entry.id,
        "request_id": entry.request_id,
        "created_at": entry.created_at,
        "source_type": entry.source_type,
        "session_id": entry.session_id,
        "turn_sequence": entry.turn_sequence,
        "target": entry.target,
        "preset": entry.preset,
        "user_message": entry.user_message or "",
        "response": entry.response or "",
        "reasoning": entry.reasoning or "",
        "mcp_calls": mcp_calls,
        "has_error": bool(entry.has_error),
        "warn_reason": entry.warn_reason or "",
        "raw_dir": entry.raw_dir,
    }
