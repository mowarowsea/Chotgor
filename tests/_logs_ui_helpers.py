"""logs_ui テスト群の共有ヘルパー。

debug/ ディレクトリ構造（1リクエスト=1フォルダ＋ログファイル群）をテスト用に
組み立てる関数を提供する。test_logs_ui_*.py の各分割ファイルから共通利用する。
ファイル名先頭がアンダースコアのため pytest のテスト収集対象にはならない。
"""

import json
from pathlib import Path

def _make_debug_dir(base: Path, msg_id: str) -> Path:
    """指定 base 配下に debug フォルダを作成して返す。"""
    folder = base / msg_id
    folder.mkdir(parents=True)
    return folder


def _write_front_input(folder: Path, model_id: str, content: str) -> None:
    """01_FrontInput.log をフォルダに書き込む。"""
    data = {"content": content, "image_ids": None, "model_id": model_id}
    (folder / "01_FrontInput.log").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_front_input_with_newlines(folder: Path, model_id: str, content: str) -> None:
    """debug_logger の _unescape_text() 相当の処理を再現して FrontInput.log を書き込む。

    debug_logger は JSON の文字列内 \\n を実際の改行に変換するため、
    厳密な JSON としては無効なファイルが生成される。
    """
    data = {"content": content, "image_ids": None, "model_id": model_id}
    raw = json.dumps(data, ensure_ascii=False, indent=2)
    # debug_logger._unescape_text() の再現: エスケープ済み \\n → 実際の改行
    raw = raw.replace("\\n", "\n").replace("\\t", "\t")
    (folder / "01_FrontInput.log").write_text(raw, encoding="utf-8")


def _write_response(folder: Path, filename: str, content: str) -> None:
    """指定ファイル名でレスポンスログを書き込む。"""
    (folder / filename).write_text(content, encoding="utf-8")

