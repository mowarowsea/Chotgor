"""ツール実行イベントの永続化ストア（ToolEventStoreMixin）。

tool_event_recorder（backend/lib/tool_event_recorder.py）がツール実行1回ごとに
add_tool_call_event() で行を追加し、Logs 画面（backend/api/logs_ui/）が
dir_id 単位の取得クエリでツール使用表示を組み立てる。

かつてツール使用表示は debug/ の生ログを表示時に逆解析していたが、
実行時記録方式への移行（2026-06-11）により本テーブルが新方式の source of truth となった。
イベントが存在しない過去ログのみ、従来の生ログ解析（tag_extract.py）に
フォールバックする。
"""

import json
import logging
from datetime import datetime

_log = logging.getLogger(__name__)


class ToolEventStoreMixin:
    """tool_call_events テーブルへの CRUD を提供する Mixin。"""

    def add_tool_call_event(
        self,
        *,
        tool_name: str,
        arguments_json: str | None = None,
        status: str = "ok",
        error_message: str | None = None,
        source: str = "tool_use",
        request_id: str | None = None,
        dir_id: str | None = None,
        target: str | None = None,
        feature: str | None = None,
    ) -> None:
        """ツール実行イベントを1行追加する。

        Args:
            tool_name: ツール名（inscribe_memory / carve_narrative / anticipate_response 等）。
            arguments_json: ツール引数 dict を JSON 文字列化したもの。
            status: 実行結果（"ok" / "error"）。
            error_message: status="error" 時の詳細メッセージ。
            source: 記録経路（"tool_use" / "tag" / "anticipation"）。
            request_id: debug_log_entries.request_id と同じ8桁hex（再生成で共有）。
            dir_id: debug フォルダ名と同じ8桁hex（試行ごとに fresh）。
            target: キャラ名/シナリオ名/バッチ対象名。
            feature: chat / scenario / chronicle 等の機能名。
        """
        from backend.repositories.sqlite.models import ToolCallEvent

        with self.get_session() as session:
            session.add(ToolCallEvent(
                created_at=datetime.now(),
                request_id=request_id,
                dir_id=dir_id,
                target=target,
                feature=feature,
                source=source,
                tool_name=tool_name,
                arguments_json=arguments_json,
                status=status,
                error_message=error_message,
            ))
            session.commit()

    def get_tool_call_events_by_dir_ids(
        self, dir_ids: list[str]
    ) -> dict[str, list[dict]]:
        """複数の dir_id のイベントを IN 句で一括取得して返す。

        Logs 画面の試行アコーディオンが N+1 クエリを避けて
        試行（= debug フォルダ）単位でツール使用を表示するための取得メソッド。
        結果は {dir_id: [イベント辞書, ...]} の形式で、各リストは作成日時昇順
        （= 実行順）。arguments_json はパース済みの dict として "arguments" キーで返す。

        Args:
            dir_ids: debug フォルダ名（8桁hex）のリスト。

        Returns:
            {dir_id: イベント辞書のリスト} の辞書。イベントが無い dir_id は空リスト。
        """
        from backend.repositories.sqlite.models import ToolCallEvent
        from sqlalchemy import asc

        if not dir_ids:
            return {}
        with self.get_session() as session:
            rows = (
                session.query(ToolCallEvent)
                .filter(ToolCallEvent.dir_id.in_(dir_ids))
                .order_by(asc(ToolCallEvent.id))
                .all()
            )
            result: dict[str, list[dict]] = {d: [] for d in dir_ids}
            for r in rows:
                if r.dir_id in result:
                    result[r.dir_id].append(_event_to_dict(r))
            return result


def _event_to_dict(event) -> dict:
    """ToolCallEvent ORM オブジェクトを辞書に変換する。

    arguments_json はここで dict にパースして "arguments" キーに格納する
    （パース失敗時は空 dict にフェイルセーフ）。呼び出し側が表示変換
    （tool_tags.tool_call_to_structured_tag）へそのまま渡せる形にする。
    """
    arguments: dict = {}
    if event.arguments_json:
        try:
            parsed = json.loads(event.arguments_json)
            if isinstance(parsed, dict):
                arguments = parsed
        except (json.JSONDecodeError, ValueError):
            _log.warning("arguments_json のパースに失敗 id=%s", event.id)
    return {
        "id": event.id,
        "created_at": event.created_at,
        "request_id": event.request_id or "",
        "dir_id": event.dir_id or "",
        "target": event.target or "",
        "feature": event.feature or "",
        "source": event.source,
        "tool_name": event.tool_name,
        "arguments": arguments,
        "status": event.status,
        "error_message": event.error_message or "",
    }
