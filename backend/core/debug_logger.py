"""Chotgorのログ出力を一元管理するモジュール。

ファイルログ（CHOTGOR_DEBUG=1 時のみ）とコンソールログの両方を扱う。
モジュールレベルのシングルトン `logger` を通じて利用すること。
"""

import json
import os
from datetime import datetime
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

        ファイル名形式: yyyyMMddhhmmssffff_{prefix}.log
        """
        if not self.is_debug_enabled():
            return
        os.makedirs(self.DEBUG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:18]
        filename = f"{timestamp}_{prefix}.log"
        filepath = os.path.join(self.DEBUG_DIR, filename)
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
        """チャット入出力のデバッグサマリーをコンソールに出力する。

        Args:
            label: ログラベル（"CHAT" / "CHAT stream" など）
            character_id: キャラクターID
            provider: プロバイダー名
            model: モデル名（省略可）
            message_summaries: (role, content) のリスト
            clean_text: キャラクターの最終応答テキスト
        """
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"[{label}] character={character_id} provider={provider} model={model or '(default)'}")
        for role, content in message_summaries:
            print(f"  [{role}] {content[:300]}{'...' if len(content) > 300 else ''}")
        print(f"  [RESPONSE] {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")
        print(sep, flush=True)

    def log_warning(self, tag: str, message: str) -> None:
        """警告メッセージをコンソールに出力する。

        Args:
            tag: ログタグ（例: "PowerRecall"）
            message: 警告メッセージ
        """
        print(f"[{tag}] {message}", flush=True)


# モジュールレベルのシングルトン
logger = ChotgorLogger()
