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
        """json.dumps で扱えない型を変換するデフォルトシリアライザ。"""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
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

    def _write_log(self, prefix: str, content: str) -> None:
        """デバッグログをファイルに書き出す。CHOTGOR_DEBUG=1 の時のみ有効。

        出力先: debug/{message_id}/{prefix}.log
        message_id は ContextVar（current_message_id）から取得する。
        """
        if not self.is_debug_enabled():
            return
        from .log_context import current_message_id
        msg_id = current_message_id.get()
        folder = os.path.join(self.DEBUG_DIR, msg_id)
        os.makedirs(folder, exist_ok=True)
        filename = f"{prefix}.log"
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    # --- ファイルログメソッド ---

    def log_front_input(self, data: Any) -> None:
        """フロント(OpenWebUI)からの入力リクエストを記録する。

        ファイル: 01_FrontInput.log
        """
        self._write_log("01_FrontInput", self._format_json(data))

    def log_front_output(self, text: str) -> None:
        """フロントへ連携する最終的な出力テキストを記録する。

        ファイル: 04_FrontOutput.log
        """
        self._write_log("04_FrontOutput", self._unescape_text(text))

    def log_provider_request(self, provider: str, params: Any) -> None:
        """プロバイダーAPIへ送るリクエストパラメータを記録する。

        ファイル: 02_Request_{provider}.log
        """
        self._write_log(f"02_Request_{provider}", self._format_json(params))

    def log_provider_response(self, provider: str, data: Any) -> None:
        """プロバイダーAPIからの生レスポンスを記録する。

        テキストの場合はそのまま、それ以外はJSON整形して出力。
        ファイル: 03_Response_{provider}.log
        """
        if isinstance(data, str):
            content = self._unescape_text(data)
        else:
            content = self._format_json(data, default=str)
        self._write_log(f"03_Response_{provider}", content)

    # --- コンソールログメソッド ---

    def log_chat_debug(
        self,
        label: str,
        character_id: str,
        provider: str,
        model: str | None,
        message_summaries: list[tuple[str, str]],
        clean_text: str,
    ) -> None:
        """LLM呼び出しの操作ログを出力する（内容なし・メタ情報のみ）。

        「いつ・だれに・何した」のみ記録し、メッセージ内容・レスポンス内容は含めない。
        内容の詳細は CHOTGOR_DEBUG=1 時の debug/{message_id}/ フォルダを参照。

        Args:
            label: ログラベル（"CHAT" / "CHAT stream" など）
            character_id: キャラクターID（またはキャラクター名@プリセット名）
            provider: プロバイダー名
            model: モデル名（省略可）
            message_summaries: (role, content) のリスト（件数カウントのみ使用）
            clean_text: キャラクターの最終応答テキスト（文字数カウントのみ使用）
        """
        log = _logging.getLogger("backend.core.debug_logger")
        log.info(
            "%s char=%s provider=%s model=%s messages=%d response_chars=%d",
            label,
            character_id,
            provider,
            model or "(default)",
            len(message_summaries),
            len(clean_text),
        )

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
