"""ログエントリの構築と一覧ロード。

debug/ ディレクトリの1リクエストフォルダ、または debug_log_entries テーブルの行から
ログ閲覧UIが表示するエントリ辞書を組み立てる。
パス類は monkeypatch で差し替えられるよう `config.DEBUG_BASE` 等を属性アクセスで参照する。
"""

import re
from collections import deque
from datetime import datetime
from pathlib import Path

from backend.api.logs_ui import config
from backend.api.logs_ui.tag_extract import _extract_tags_from_file, _parse_json_safely

# chotgor.log から char= を抽出するパターン
# 例: "... [a25e00da] ... char=はる@ClaudeCode ..."
_LOG_CHAR_RE = re.compile(r"\[([0-9a-f]{8})\].*?\bchar=(\S+)")

# ファイル名パターン: "02_chat_Request_ClaudeCode.log"
_FILE_PATTERN = re.compile(r"^\d+_(.+?)_(Request|Response)_(.+)\.log$")
# 警告ファイルパターン: "03_Warning_context_window.log"
_FILE_PATTERN_WARNING = re.compile(r"^\d+_Warning_(.+)\.log$")
# お知らせファイルパターン: "03_Notice_context_window.log"（has_error にはしない情報レベル通知）
_FILE_PATTERN_NOTICE = re.compile(r"^\d+_Notice_(.+)\.log$")


def _build_char_index() -> dict[str, str]:
    """chotgor.log を走査して {msg_id: char_label} の辞書を返す。

    chronicle / forget などバッチ処理の宛先キャラクター名を解決するために使用する。
    同一 msg_id で複数行マッチした場合は最後の値を採用する（メインの完了ログを優先）。
    ファイルが存在しない・読み取り不可の場合は空辞書を返す。

    Returns:
        {8桁hex文字列: "キャラクター名@プリセット名"} の辞書。
    """
    if not config.CHOTGOR_LOG.exists():
        return {}
    index: dict[str, str] = {}
    try:
        for line in config.CHOTGOR_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _LOG_CHAR_RE.search(line)
            if m:
                index[m.group(1)] = m.group(2)
    except Exception:
        pass
    return index


def _tc_label(tc: dict) -> str:
    """tool_call を人間可読なラベル（feature/preset）に整形する。

    warn_reason の組み立てなど、ログ表示用の識別名として使う。

    Args:
        tc: tool_call 辞書（feature / preset キーを持つ）。

    Returns:
        "feature/preset" 形式の文字列。どちらか欠ける場合は存在する方を、
        いずれも無ければ "不明な処理" を返す。
    """
    feature = tc.get("feature") or ""
    preset = tc.get("preset") or ""
    if feature and preset:
        return f"{feature}/{preset}"
    return feature or preset or "不明な処理"


def _parse_entry(msg_id: str, folder: Path, char_index: dict[str, str]) -> dict:
    """1リクエストフォルダの内容を解析してログエントリ辞書を返す。

    Args:
        msg_id: リクエストID（フォルダ名）。
        folder: フォルダのPathオブジェクト。
        char_index: _build_char_index() で構築した {msg_id: char_label} 辞書。
                    FrontInput がない batch エントリの宛先解決に使用する。

    Returns:
        ログエントリ辞書。以下のキーを含む:
            message_id, dt, dt_str, character, preset, model_id,
            source, user_message, character_response, reasoning_text,
            tool_calls, warnings, files, has_error, warn_reason
    """
    try:
        dt = datetime.fromtimestamp(folder.stat().st_mtime)
    except OSError:
        dt = datetime(1970, 1, 1)

    files = sorted(folder.iterdir(), key=lambda p: p.name)
    file_names = [p.name for p in files]

    # --- FrontInput を解析 ---
    character = ""
    preset = ""
    model_id = ""
    user_message = ""
    source = "system"
    front_input_path = folder / "01_FrontInput.log"
    if front_input_path.exists():
        try:
            raw = front_input_path.read_text(encoding="utf-8")
            data = _parse_json_safely(raw)
            model_id = data.get("model_id", "")
            if "@" in model_id:
                character, preset = model_id.rsplit("@", 1)
            else:
                character = model_id
            user_message = data.get("content", "")
            source = "ユーザ"
        except Exception:
            pass

    # --- FrontInput がない場合（batch 処理）は chotgor.log インデックスで補完 ---
    if not model_id and msg_id in char_index:
        model_id = char_index[msg_id]
        if "@" in model_id:
            character, preset = model_id.rsplit("@", 1)
        else:
            character = model_id

    # --- FrontOutput を探す ---
    character_response = ""
    for f in files:
        if "FrontOutput" in f.name:
            try:
                character_response = f.read_text(encoding="utf-8")
            except Exception:
                pass
            break

    # --- Reasoning ログを探す（翻訳ボタン用） ---
    reasoning_text = ""
    for f in files:
        if "Reasoning" in f.name:
            try:
                reasoning_text = f.read_text(encoding="utf-8")
            except Exception:
                pass
            break

    # --- ツール呼び出しを Request/Response ファイルペアから構築 ---
    # 同一 feature/preset でも複数ラウンドトリップが発生する（tool-use時）ため、
    # dict でのキー管理はせず FIFO キューで「先に来た Request に先の Response を対応付ける」。
    tool_calls: list[dict] = []
    # (feature, preset) → 未マッチ Request エントリのキュー（到着順）
    pending: dict[tuple[str, str], deque[dict]] = {}

    for f in files:
        m = _FILE_PATTERN.match(f.name)
        if not m:
            continue
        feature, kind, preset_name = m.group(1), m.group(2), m.group(3)

        pair_key = (feature, preset_name)
        if kind == "Request":
            entry = {"feature": feature, "preset": preset_name,
                     "request_file": f.name, "response_file": None, "tags": []}
            tool_calls.append(entry)
            pending.setdefault(pair_key, deque()).append(entry)
        else:  # Response
            queue = pending.get(pair_key)
            if queue:
                queue.popleft()["response_file"] = f.name
                if not queue:
                    del pending[pair_key]
            else:
                # 対応する Request がない Response（異常ケース）
                tool_calls.append({"feature": feature, "preset": preset_name,
                                   "request_file": None, "response_file": f.name, "tags": []})

    # Response ファイルからタグを抽出して tool_calls に追加
    for tc in tool_calls:
        if tc["response_file"]:
            tc["tags"] = _extract_tags_from_file(folder / tc["response_file"])

    # --- Warning ファイルを収集 ---
    # log_warning() が書き出した {NN}_Warning_{tag}.log を読み込む。
    warnings: list[dict] = []
    for f in files:
        m = _FILE_PATTERN_WARNING.match(f.name)
        if m:
            tag = m.group(1)
            try:
                msg = f.read_text(encoding="utf-8").strip()
            except Exception:
                msg = ""
            warnings.append({"tag": tag, "message": msg, "file": f.name})

    # --- お知らせ（Notice）ファイルを収集 ---
    # log_notice() が書き出した {NN}_Notice_{tag}.log を読み込む。
    # WARN とは別枠の情報レベル通知で、has_error には影響しない。
    notices: list[dict] = []
    for f in files:
        m = _FILE_PATTERN_NOTICE.match(f.name)
        if m:
            tag = m.group(1)
            try:
                msg = f.read_text(encoding="utf-8").strip()
            except Exception:
                msg = ""
            notices.append({"tag": tag, "message": msg, "file": f.name})

    # --- 警告（WARN）判定 ---
    # いずれかの tool_call で Response が存在しない、または Response がエラー文字列の場合は
    # 警告扱いとし、その理由を warn_reason に人間可読な文言で残す（Logs 画面で表示）。
    has_error = False
    warn_reason = ""
    for tc in tool_calls:
        if tc["request_file"] and not tc["response_file"]:
            # Request はあるが Response が存在しない = エラーで中断したと判断
            has_error = True
            warn_reason = f"{_tc_label(tc)} の応答が生成されず、処理が途中で中断しました"
            break
        if tc["response_file"]:
            # Response の先頭がエラー文字列パターンに一致する場合
            resp_path = folder / tc["response_file"]
            try:
                head = resp_path.read_text(encoding="utf-8", errors="replace")[:120]
                if head.lstrip().startswith("[") and ("error" in head.lower() or "Error" in head):
                    has_error = True
                    warn_reason = (
                        f"{_tc_label(tc)} の応答がエラーを返しました"
                        f"（{tc['response_file']} を参照）"
                    )
                    break
            except Exception:
                pass

    return {
        "message_id": msg_id,
        "dt": dt,
        "dt_str": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "character": character,
        "preset": preset,
        "model_id": model_id,
        "source": source,
        "user_message": user_message,
        "character_response": character_response,
        "reasoning_text": reasoning_text,
        "tool_calls": tool_calls,
        "warnings": warnings,
        "notices": notices,
        "files": file_names,
        "has_error": has_error,
        "warn_reason": warn_reason,
    }


def _build_attempt_detail(r: dict, index: int, char_index: dict, skip_files: bool = False) -> dict:
    """DB の1行（1試行）から試行アコーディオン表示用の詳細データを構築する。

    skip_files=True のときはファイルシステムへのアクセスをスキップし、
    DB の値のみで構成する（一覧の高速表示用）。
    skip_files=False のときは raw_dir を読んでツール呼び出し・警告・ファイル一覧を組み立てる。

    Args:
        r: get_debug_log_entries_by_request_id() が返す1行の辞書。
        index: 試行番号（0始まり）。表示は +1 して使う。
        char_index: chotgor.log から構築した {msg_id: char_label} 辞書。
        skip_files: True のときファイルI/Oをスキップする。

    Returns:
        試行アコーディオンが期待するデータ辞書。
    """
    raw_dir = r.get("raw_dir") or ""
    dir_id = Path(raw_dir).name if raw_dir else ""
    tool_calls: list[dict] = []
    warnings: list[dict] = []
    notices: list[dict] = []
    file_names: list[str] = []
    has_error = bool(r.get("has_error"))
    warn_reason = r.get("warn_reason") or ""

    if not skip_files and raw_dir:
        raw_path = Path(raw_dir)
        if raw_path.exists():
            parsed = _parse_entry(dir_id, raw_path, char_index)
            tool_calls = parsed.get("tool_calls", [])
            warnings_parsed = parsed.get("warnings", [])
            if warnings_parsed:
                warnings = warnings_parsed
            notices = parsed.get("notices", [])
            if not has_error:
                has_error = parsed.get("has_error", False)
            if not warn_reason:
                warn_reason = parsed.get("warn_reason", "")
            file_names = sorted(p.name for p in raw_path.iterdir() if p.is_file())

    return {
        "index": index + 1,
        "preset": r.get("preset") or "",
        "response": r.get("response") or "",
        "reasoning": r.get("reasoning") or "",
        "dt_str": r["created_at"].strftime("%H:%M:%S") if r.get("created_at") else "",
        "has_error": has_error,
        "warn_reason": warn_reason,
        "tool_calls": tool_calls,
        "warnings": warnings,
        "notices": notices,
        "files": file_names,
        "dir_id": dir_id,
    }


def _build_entry_from_db_rows(rows: list[dict], skip_files: bool = True) -> dict:
    """同一 request_id の DB 行リストからログ一覧表示用エントリを組み立てる。

    メイン行（chat/scenario 等）は試行ごとに1行存在し、`attempts` リストに
    完全な詳細データを格納する。再生成で複数試行ある場合は attempt_count > 1 になる。
    非メイン行（farewell/trigger/chronicle 等）は tool_calls を top-level で保持する。

    Args:
        rows: get_debug_log_entries_by_request_id() が返すエントリ辞書のリスト（昇順）。
        skip_files: True のときファイルI/Oをスキップする（一覧高速表示用）。

    Returns:
        ログ一覧 UI が期待するエントリ辞書。
    """
    if not rows:
        return {}

    _MAIN_SOURCE_TYPES = {"chat", "scenario", "scenario_chat", "scenario_chat_pc", "group_chat"}

    main_row = next((r for r in rows if r["source_type"] in _MAIN_SOURCE_TYPES), rows[0])
    main_rows = [r for r in rows if r["source_type"] in _MAIN_SOURCE_TYPES]
    latest_main = main_rows[-1] if main_rows else rows[-1]

    request_id = main_row["request_id"]
    source_type = main_row["source_type"]
    dt = latest_main["created_at"] or main_row["created_at"]

    seen: set[str] = set()
    source_types: list[str] = []
    for r in rows:
        if r["source_type"] not in seen:
            seen.add(r["source_type"])
            source_types.append(r["source_type"])

    # コンテキスト圧縮は WARN ではなくお知らせ扱いに変更したため、
    # 旧仕様で has_error=True / warn_reason=[context_window] と記録された
    # 既存エントリも表示時に WARN から降格させる（過去ログにも新挙動を反映）。
    def _is_demoted(reason: str) -> bool:
        return (reason or "").startswith("[context_window]")

    has_error = any(
        r["has_error"] and not _is_demoted(r.get("warn_reason") or "") for r in rows
    )
    warn_reason = next(
        (r["warn_reason"] for r in rows if r["warn_reason"] and not _is_demoted(r["warn_reason"])),
        "",
    )
    # 降格された context_window のお知らせを一覧サマリーでも見えるよう notices に拾う
    demoted_notices = [
        {"tag": "context_window", "message": r["warn_reason"], "file": ""}
        for r in rows
        if r.get("warn_reason") and _is_demoted(r["warn_reason"])
    ]
    target = main_row.get("target") or latest_main.get("target") or ""
    preset = latest_main.get("preset") or main_row.get("preset") or ""
    model_id = f"{target}@{preset}" if target and preset else target or preset

    # ファイルI/Oが必要なときだけ char_index を構築する
    needs_files = not skip_files and any(r.get("raw_dir") for r in rows)
    char_index = _build_char_index() if needs_files else {}

    # メイン行を試行ごとに詳細データへ変換（skip_files=True なら DB の値のみ）
    attempts = [
        _build_attempt_detail(r, i, char_index, skip_files=skip_files)
        for i, r in enumerate(main_rows)
    ]

    # 非メイン行（chronicle/forget 等）向けの top-level tool_calls（旧来互換）
    top_tool_calls: list[dict] = []
    top_warnings: list[dict] = []
    top_notices: list[dict] = []
    top_file_names: list[str] = []
    if not main_rows and not skip_files:
        # メイン行が無いエントリは従来通り最初の行の raw_dir を使う
        raw_dir = rows[0].get("raw_dir") or ""
        if raw_dir:
            raw_path = Path(raw_dir)
            dir_id = raw_path.name
            if raw_path.exists():
                parsed = _parse_entry(dir_id, raw_path, char_index)
                top_tool_calls = parsed.get("tool_calls", [])
                top_warnings = parsed.get("warnings", [])
                top_notices = parsed.get("notices", [])
                if not has_error:
                    has_error = parsed.get("has_error", False)
                if not warn_reason:
                    warn_reason = parsed.get("warn_reason", "")
                top_file_names = sorted(
                    p.name for p in raw_path.iterdir() if p.is_file()
                )

    # top-level の dir_id（旧来の ID/Files 表示と JSON API 用）
    top_raw_dir = latest_main.get("raw_dir") or main_row.get("raw_dir") or ""
    top_dir_id = Path(top_raw_dir).name if top_raw_dir else request_id

    return {
        "message_id": request_id,
        "dt": dt,
        "dt_str": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "",
        "character": target,
        "preset": preset,
        "model_id": model_id,
        "source_type": source_type,
        "source_types": source_types,
        "source": "system" if source_type not in _MAIN_SOURCE_TYPES else "ユーザ",
        "user_message": main_row.get("user_message") or "",
        # 後方互換（JSON API・サマリー表示用）
        "character_response": latest_main.get("response") or "",
        "reasoning_text": latest_main.get("reasoning") or "",
        # 非メイン行エントリ向け（attempts が空の場合のみ HTML で使う）
        "tool_calls": top_tool_calls,
        "warnings": top_warnings,
        # 降格お知らせ（DB由来・一覧でも有効）＋ ファイル由来お知らせ
        "notices": demoted_notices + top_notices,
        "files": top_file_names,
        "dir_id": top_dir_id,
        "has_error": has_error,
        "warn_reason": warn_reason,
        "attempt_count": len(main_rows),
        "attempts": attempts,
    }


def _load_entries(
    page: int = 1, per_page: int = 50, request_type: str = "chat"
) -> tuple[list[dict], int]:
    """デバッグログエントリを読み込んでページネーションして返す。

    SQLiteStore が利用可能な場合は DB から読み込む。
    未セット（テスト・起動直後等）の場合はファイルシステムにフォールバックする。

    Args:
        page: ページ番号（1始まり）。
        per_page: 1ページあたりの件数。
        request_type: フィルタ種別（'chat'/'scenario'/'batch'/'all'）。

    Returns:
        (エントリリスト, 総件数) のタプル。
    """
    sqlite = config.get_sqlite_store()
    if sqlite is not None:
        return _load_entries_from_db(sqlite, page, per_page, request_type)
    return _load_entries_from_files(page, per_page)


def _load_entries_from_db(
    sqlite, page: int, per_page: int, request_type: str = "chat"
) -> tuple[list[dict], int]:
    """DB からログエントリをページネーションして返す。

    IN 句で全件を一括取得してN+1クエリを回避する。
    一覧表示ではファイルI/Oをスキップして高速化する。

    Args:
        sqlite: SQLiteStore インスタンス。
        page: ページ番号（1始まり）。
        per_page: 1ページあたりの件数。
        request_type: フィルタ種別（'chat'/'scenario'/'batch'/'all'）。

    Returns:
        (エントリリスト, 総件数) のタプル。
    """
    request_ids, total = sqlite.get_debug_log_request_ids_paged(
        page=page, per_page=per_page, request_type=request_type
    )
    # IN 句で一括取得（N+1 解消）
    rows_map = sqlite.get_debug_log_entries_by_request_ids(request_ids)
    entries = []
    for req_id in request_ids:
        rows = rows_map.get(req_id, [])
        if rows:
            # 一覧ではファイルI/Oをスキップして高速化
            entries.append(_build_entry_from_db_rows(rows, skip_files=True))
    return entries, total


def _load_entries_from_files(page: int, per_page: int) -> tuple[list[dict], int]:
    """ファイルシステムからログエントリを読み込む（フォールバック）。

    DB 未セット時（テスト等）に使用する旧来の実装。

    Args:
        page: ページ番号（1始まり）。
        per_page: 1ページあたりの件数。

    Returns:
        (エントリリスト, 総件数) のタプル。
    """
    if not config.DEBUG_BASE.exists():
        return [], 0

    folders = []
    for item in config.DEBUG_BASE.iterdir():
        if item.is_dir() and item.name != "trush":
            try:
                mtime = item.stat().st_mtime
                folders.append((mtime, item.name, item))
            except OSError:
                pass
    folders.sort(reverse=True, key=lambda x: x[0])

    total = len(folders)
    start = (page - 1) * per_page
    end = start + per_page
    page_folders = folders[start:end]

    char_index = _build_char_index()
    entries = [_parse_entry(name, folder, char_index) for _, name, folder in page_folders]
    return entries, total
