"""Chotgorのログ出力を一元管理するモジュール。

ファイルログ（CHOTGOR_DEBUG=1 時のみ debug/{message_id}/ フォルダに書き込み）と
SQLite への debug_log_entries 書き込み、および標準 logging モジュール経由の
ログ出力の3系統を扱う。
モジュールレベルのシングルトン `logger` を通じて利用すること。
"""

import json
import logging as _logging
import os
from typing import Any, Optional


class ChotgorLogger:
    """デバッグファイルログ・DBログ・コンソールログを一元管理するクラス。

    ファイルログ: CHOTGOR_DEBUG=1 の環境変数が設定された時のみ debug/ ディレクトリに出力。
    DBログ: set_store() で SQLiteStore をセットした後、各リクエストの開始〜完了時に書き込む。
    コンソールログ: 常時 stdout に出力。
    """

    DEBUG_DIR = "debug"

    def __init__(self) -> None:
        """ロガーを初期化する。"""
        self._sqlite = None  # アプリ起動時に set_store() でセットされる

    def set_store(self, sqlite) -> None:
        """DB書き込みに使用する SQLiteStore をセットする。

        main.py の lifespan 関数から起動時に1度だけ呼ぶ。

        Args:
            sqlite: SQLiteStore インスタンス。
        """
        self._sqlite = sqlite

    def is_debug_enabled(self) -> bool:
        """CHOTGOR_DEBUG=1 の場合にデバッグファイルログが有効になる。"""
        return os.getenv("CHOTGOR_DEBUG") == "1"

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """json.dumps で扱えない型を変換するデフォルトシリアライザ。

        model_dump() が dict を返す Pydantic モデルはそのまま展開する。
        model_dump() が dict 以外を返す場合（MagicMock等）は str() にフォールバックして
        json.dumps の default コールバックが無限ループするのを防ぐ。
        """
        if hasattr(obj, "model_dump"):
            result = obj.model_dump()
            if isinstance(result, dict):
                return result
            return str(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    def _unescape_text(self, text: str) -> str:
        """エスケープされた改行・タブを実際の制御文字に展開して可読性を高める。"""
        return text.replace("\\n", "\n").replace("\\t", "\t")

    def _format_json(self, data: Any, *, default: Any = None) -> str:
        """JSON整形 + 文字列値内のエスケープ改行を展開する。

        Args:
            data: JSON化するオブジェクト
            default: json.dumps の default 引数。省略時は _json_serializer を使用

        Returns:
            整形済みJSON文字列（改行コードが展開済み）
        """
        serializer = default if default is not None else self._json_serializer
        raw = json.dumps(data, ensure_ascii=False, indent=2, default=serializer)
        return self._unescape_text(raw)

    def _write_log(self, label: str, content: str) -> None:
        """デバッグログをファイルに書き出す。CHOTGOR_DEBUG=1 の時のみ有効。

        出力先: debug/{message_id}/{NN}_{label}.log
        NN はリクエスト内の通し番号（時系列順）。
        message_id / 通し番号は ContextVar から取得する。
        """
        if not self.is_debug_enabled():
            return
        from .log_context import current_message_id, next_log_index
        msg_id = current_message_id.get()
        idx = next_log_index()
        folder = os.path.join(self.DEBUG_DIR, msg_id)
        os.makedirs(folder, exist_ok=True)
        filename = f"{idx:02d}_{label}.log"
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def _raw_dir(self) -> Optional[str]:
        """現在のリクエストの生ファイルフォルダパスを返す。

        CHOTGOR_DEBUG=1 が無効な場合は None を返す。

        Returns:
            "debug/{message_id}" 形式のパス、またはデバッグ無効時は None。
        """
        if not self.is_debug_enabled():
            return None
        from .log_context import current_message_id
        return f"{self.DEBUG_DIR}/{current_message_id.get()}"

    # --- ファイルログメソッド ---

    def log_front_input(self, data: Any) -> None:
        """フロント(OpenWebUI)からの入力リクエストを記録する。

        ファイル: {NN}_FrontInput.log
        DB: source_type をContextVarから取得してメインDB行を INSERT し、
            returned id を current_log_db_entry_id にセットする。

        Args:
            data: フロントからの入力リクエストデータ。
        """
        self._write_log("FrontInput", self._format_json(data))
        # ユーザーメッセージ本文をContextVarに保存
        if isinstance(data, dict):
            from .log_context import current_log_user_message
            current_log_user_message.set(data.get("content") or "")
        # メインDB行を INSERT
        self._insert_main_entry()

    def _insert_main_entry(self) -> None:
        """現在のリクエストのメインDB行（chat/scenario）を INSERT する。

        source_type は current_log_feature の値を使用する。
        INSERT した行の id を current_log_db_entry_id にセットする。
        """
        if not self._sqlite:
            return
        from .log_context import (
            current_message_id,
            current_log_feature,
            current_log_session_id,
            current_log_target,
            current_log_turn_sequence,
            current_log_user_message,
            current_log_db_entry_id,
        )
        try:
            entry_id = self._sqlite.insert_debug_log_entry(
                request_id=current_message_id.get(),
                source_type=current_log_feature.get(),
                session_id=current_log_session_id.get(),
                turn_sequence=current_log_turn_sequence.get(),
                target=current_log_target.get(),
                preset=None,  # プリセットは log_provider_request 時点では未確定
                user_message=current_log_user_message.get(),
                raw_dir=self._raw_dir(),
            )
            current_log_db_entry_id.set(entry_id)
        except Exception as e:
            _logging.getLogger(__name__).warning("debug_log_entries INSERT 失敗: %s", e)

    def log_front_output(self, text: str) -> None:
        """フロントへ連携する最終的な出力テキストを記録する。

        ファイル: {NN}_FrontOutput.log
        DB: current_log_db_entry_id の行に response をセットする。

        Args:
            text: フロントへの最終出力テキスト。
        """
        self._write_log("FrontOutput", self._unescape_text(text))
        self._update_main_entry(response=text)

    def _update_main_entry(self, **kwargs) -> None:
        """current_log_db_entry_id の行を部分更新する。

        Args:
            **kwargs: update_debug_log_entry に渡すフィールド。
        """
        if not self._sqlite:
            return
        from .log_context import current_log_db_entry_id
        entry_id = current_log_db_entry_id.get()
        if entry_id is None:
            return
        try:
            self._sqlite.update_debug_log_entry(entry_id, **kwargs)
        except Exception as e:
            _logging.getLogger(__name__).warning("debug_log_entries UPDATE 失敗: %s", e)

    def log_reasoning(self, text: str) -> None:
        """思考ブロック・想起記憶の累積テキストをファイルに記録する。

        ファイル: {NN}_Reasoning.log
        DB: current_log_db_entry_id の行に reasoning をセットする。

        Args:
            text: 思考ブロックと想起記憶を結合した累積テキスト。
        """
        self._write_log("Reasoning", text)
        self._update_main_entry(reasoning=text)

    def log_provider_request(self, preset_name: str, params: Any) -> None:
        """プロバイダーAPIへ送るリクエストパラメータを記録する。

        ファイル: {NN}_{feature}_Request_{preset_name}.log
        feature は current_log_feature ContextVar から取得する。

        Args:
            preset_name: プリセット名（またはプロバイダーID）。
            params: リクエストパラメータ。
        """
        from .log_context import current_log_feature
        feature = current_log_feature.get()
        self._write_log(f"{feature}_Request_{preset_name}", self._format_json(params))

    def log_provider_response(self, preset_name: str, data: Any) -> None:
        """プロバイダーAPIからの生レスポンスを記録する。

        ファイル: {NN}_{feature}_Response_{preset_name}.log
        DB: feature が 'chat'/'scenario' 以外（farewell/trigger等）の場合は
            サブDB行を INSERT する。'chat'/'scenario' の場合はプリセット名のみ更新する。

        Args:
            preset_name: プリセット名（またはプロバイダーID）。
            data: レスポンスデータ。
        """
        from .log_context import current_log_feature
        feature = current_log_feature.get()
        if isinstance(data, str):
            content = self._unescape_text(data)
        else:
            content = self._format_json(data, default=str)
        self._write_log(f"{feature}_Response_{preset_name}", content)

        # メイン行（chat/scenario）はプリセット名だけ更新する
        _MAIN_SOURCE_TYPES = {"chat", "scenario", "scenario_chat"}
        if feature in _MAIN_SOURCE_TYPES:
            self._update_main_entry_preset(preset_name)
        else:
            # farewell / trigger / reflection 等はサブ行を INSERT する
            self._insert_sub_entry(feature, preset_name)

    def _update_main_entry_preset(self, preset_name: str) -> None:
        """メインDB行にプリセット名を設定する。

        Args:
            preset_name: 設定するプリセット名。
        """
        if not self._sqlite:
            return
        from .log_context import current_log_db_entry_id
        entry_id = current_log_db_entry_id.get()
        if entry_id is None:
            return
        try:
            from backend.repositories.sqlite.store import DebugLogEntry
            with self._sqlite.get_session() as sess:
                entry = sess.get(DebugLogEntry, entry_id)
                if entry and not entry.preset:
                    entry.preset = preset_name
                    sess.commit()
        except Exception as e:
            _logging.getLogger(__name__).warning("debug_log_entries preset 更新失敗: %s", e)

    def _insert_sub_entry(self, source_type: str, preset_name: str) -> None:
        """farewell/trigger 等のサブDB行を INSERT する。

        Args:
            source_type: 呼び出し種別（'farewell'/'trigger'等）。
            preset_name: 使用したプリセット名。
        """
        if not self._sqlite:
            return
        from .log_context import (
            current_message_id,
            current_log_session_id,
            current_log_target,
            current_log_turn_sequence,
        )
        try:
            self._sqlite.insert_debug_log_entry(
                request_id=current_message_id.get(),
                source_type=source_type,
                session_id=current_log_session_id.get(),
                turn_sequence=current_log_turn_sequence.get(),
                target=current_log_target.get(),
                preset=preset_name,
                raw_dir=self._raw_dir(),
            )
        except Exception as e:
            _logging.getLogger(__name__).warning(
                "debug_log_entries サブ行 INSERT 失敗 source_type=%s: %s", source_type, e
            )

    # --- コンソールログメソッド ---

    def log_warning(self, tag: str, message: str) -> None:
        """警告メッセージを標準 logging とデバッグファイルの両方に出力する。

        コンソール: 常時 logging.warning() で出力。
        ファイル: CHOTGOR_DEBUG=1 時のみ {NN}_Warning_{tag}.log に書き出す（ログ一覧UIで閲覧可能）。
        DB: current_log_db_entry_id の行に has_error=True と warn_reason をセットする。

        Args:
            tag: ログタグ（例: "context_window"）
            message: 警告メッセージ
        """
        _logging.getLogger("backend.core.debug_logger").warning("[%s] %s", tag, message)
        self._write_log(f"Warning_{tag}", message)
        self._update_main_entry(has_error=True, warn_reason=f"[{tag}] {message}"[:500])


# モジュールレベルのシングルトン
logger = ChotgorLogger()
