"""Chotgorのログ出力を一元管理するモジュール。

ファイルログ（CHOTGOR_DEBUG=1 時のみ debug/{message_id}/ フォルダに書き込み）と
標準 logging モジュール経由のログ出力の両方を扱う。
モジュールレベルのシングルトン `logger` を通じて利用すること。
"""

import json
import logging as _logging
import os
from typing import Any


class ChotgorLogger:
    """デバッグファイルログとコンソールログを一元管理するクラス。

    ファイルログ: CHOTGOR_DEBUG=1 の環境変数が設定された時のみ debug/ ディレクトリに出力。
    コンソールログ: 常時 stdout に出力。
    """

    DEBUG_DIR = "debug"

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
            # model_dump() が dict 以外を返した（MagicMock 等） → str にフォールバック
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

    # --- ファイルログメソッド ---

    def log_front_input(self, data: Any) -> None:
        """フロント(OpenWebUI)からの入力リクエストを記録する。

        ファイル: {NN}_FrontInput.log
        """
        self._write_log("FrontInput", self._format_json(data))

    def log_front_output(self, text: str) -> None:
        """フロントへ連携する最終的な出力テキストを記録する。

        ファイル: {NN}_FrontOutput.log
        """
        self._write_log("FrontOutput", self._unescape_text(text))

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

        テキストの場合はそのまま、それ以外はJSON整形して出力。
        ファイル: {NN}_{feature}_Response_{preset_name}.log
        feature は current_log_feature ContextVar から取得する。

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

    # --- コンソールログメソッド ---

    def log_warning(self, tag: str, message: str) -> None:
        """警告メッセージを標準 logging で出力する。

        print() を使わず logging.warning() に変更。

        Args:
            tag: ログタグ（例: "PowerRecall"）
            message: 警告メッセージ
        """
        _logging.getLogger("backend.core.debug_logger").warning("[%s] %s", tag, message)


# モジュールレベルのシングルトン
logger = ChotgorLogger()
